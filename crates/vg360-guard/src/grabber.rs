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
}

impl Grabber {
    pub fn start(url: &str, width: usize, height: usize, rotate_180: bool) -> Arc<Self> {
        let g = Arc::new(Self {
            width, height, rotate_180,
            latest: Mutex::new(None),
            connected: AtomicBool::new(false),
            running: AtomicBool::new(true),
            child: Mutex::new(None),
        });
        let (gc, url) = (g.clone(), url.to_string());
        thread::spawn(move || gc.run(&url));
        g
    }

    fn run(&self, url: &str) {
        let ffmpeg = std::env::var("FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
        let frame_len = self.width * self.height * 3;
        while self.running.load(Ordering::Relaxed) {
            let spawned = Command::new(&ffmpeg)
                .args(["-loglevel", "error", "-rtsp_transport", "tcp", "-fflags", "nobuffer",
                       "-flags", "low_delay", "-i", url, "-an",
                       "-vf", &format!("scale={}:{}", self.width, self.height),
                       "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
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
            }
            self.connected.store(false, Ordering::Relaxed);
            if let Some(mut c) = self.child.lock().unwrap().take() {
                let _ = c.kill();
                let _ = c.wait();
            }
            thread::sleep(Duration::from_secs(1)); // reconnect
        }
    }

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
