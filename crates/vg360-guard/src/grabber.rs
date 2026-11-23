//! RTSP reader: runs `ffmpeg` and reads raw BGR frames from its stdout on a background
//! thread, so callers always get the newest frame. Reconnects on drop.
//! Same behaviour as camera_guard's grabber; frames are plain [`Bgr`] buffers instead of OpenCV Mats.
//! The URL (with credentials) comes from the caller — read it from `VG360_CAM_<ID>_URL`.
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use vg360_vision::img::Bgr;

pub struct Grabber {
    pub width: usize,
    pub height: usize,
    rotate_180: bool,
    latest: Mutex<Option<Arc<Vec<u8>>>>,
    pub connected: AtomicBool,
    running: AtomicBool,
    child: Mutex<Option<Child>>,
    count: std::sync::atomic::AtomicU64,
}

impl Grabber {
    pub fn start(url: &str, width: usize, height: usize, rotate_180: bool) -> Arc<Self> {
        let g = Arc::new(Self {
            width, height, rotate_180,
            latest: Mutex::new(None),
            connected: AtomicBool::new(false),
            running: AtomicBool::new(true),
            child: Mutex::new(None),
            count: std::sync::atomic::AtomicU64::new(0),
        });
        let (gc, url) = (g.clone(), url.to_string());
        thread::spawn(move || gc.run(&url));
        g
    }

    fn run(&self, url: &str) {
        let ffmpeg = std::env::var("FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
        let frame_len = self.width * self.height * 3;
        while self.running.load(Ordering::Relaxed) {
            // RTSP cameras: low-latency TCP. Anything else (a recorded file) is replayed in real time
            // and looped, so it behaves like a live camera (demo / soak testing).
            let mut args: Vec<String> = vec!["-loglevel".into(), "error".into()];
            if url.starts_with("rtsp://") || url.starts_with("rtsps://") {
                args.extend(["-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay"].map(String::from));
            } else {
                args.extend(["-re", "-stream_loop", "-1"].map(String::from));
            }
            args.extend(["-i".into(), url.into(), "-an".into(), "-vf".into(), format!("scale={}:{}", self.width, self.height),
                         "-f".into(), "rawvideo".into(), "-pix_fmt".into(), "bgr24".into(), "-".into()]);
            let spawned = Command::new(&ffmpeg)
                .args(&args)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(if std::env::var_os("VG360_GUARD_DEBUG").is_some() { Stdio::inherit() } else { Stdio::null() })
                .spawn();
            let mut child = match spawned {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("cannot start ffmpeg ({ffmpeg}): {e}"); // never print the URL: it carries credentials
                    thread::sleep(Duration::from_secs(2));
                    continue;
                }
            };
            let mut out = child.stdout.take().expect("ffmpeg stdout");
            *self.child.lock().unwrap() = Some(child);
            loop {
                let mut buf = vec![0u8; frame_len];
                if out.read_exact(&mut buf).is_err() || !self.running.load(Ordering::Relaxed) { break; }
                self.connected.store(true, Ordering::Relaxed);
                *self.latest.lock().unwrap() = Some(Arc::new(buf));
                self.count.fetch_add(1, Ordering::Relaxed);
            }
            self.connected.store(false, Ordering::Relaxed);
            if let Some(mut c) = self.child.lock().unwrap().take() {
                let _ = c.kill();
                let _ = c.wait();
            }
            thread::sleep(Duration::from_secs(1)); // reconnect
        }
    }

    /// Frames received since start (a counter, so callers can tell a new frame from the last one).
    pub fn frames(&self) -> u64 { self.count.load(Ordering::Relaxed) }

    /// Newest frame (rotated 180° if configured).
    pub fn read(&self) -> Option<Bgr> {
        let buf = self.latest.lock().unwrap().clone()?;
        let mut f = Bgr::from_raw(self.width, self.height, buf.to_vec());
        if self.rotate_180 { f.rotate_180(); }
        Some(f)
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(mut c) = self.child.lock().unwrap().take() { let _ = c.kill(); }
    }
}
