//! RTSP reader: runs `ffmpeg` and reads raw BGR frames from its stdout on a background
//! thread, so callers always get the newest frame. Reconnects on drop.
//! Same behaviour as camera_guard's grabber; frames are plain [`Bgr`] buffers instead of OpenCV Mats.
//! The URL (with credentials) comes from the caller — read it from `VG360_CAM_<ID>_URL`.
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use vg360_vision::img::Bgr;

/// A connected stream whose newest complete frame is older than this is treated as stalled:
/// `read()` returns None rather than serving a stale frame as fresh.
const STALL_S: f64 = 10.0;
/// Socket I/O timeout handed to ffmpeg for network URLs (µs): a camera that goes silent without
/// closing the socket would otherwise block ffmpeg (and so the reconnect) forever.
const IO_TIMEOUT_US: &str = "10000000";

/// Replace `user:pass@` in any `scheme://user:pass@host...` with `***@` so ffmpeg's debug output
/// never leaks camera credentials.
pub fn redact(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(i) = rest.find("://") {
        let (head, tail) = rest.split_at(i + 3);
        out.push_str(head);
        let seg_end = tail.find(char::is_whitespace).unwrap_or(tail.len());
        match tail[..seg_end].rfind('@') {
            Some(at) => { out.push_str("***"); rest = &tail[at..]; }
            None => rest = tail,
        }
    }
    out.push_str(rest);
    out
}

pub struct Grabber {
    pub width: usize,
    pub height: usize,
    rotate_180: bool,
    latest: Mutex<Option<Arc<Vec<u8>>>>,
    pub connected: AtomicBool,
    running: AtomicBool,
    child: Mutex<Option<Child>>,
    count: std::sync::atomic::AtomicU64,
    last_frame: Mutex<Option<Instant>>,
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
            last_frame: Mutex::new(None),
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
                // rtsp demuxer option `-timeout` (µs, socket I/O); the generic `-rw_timeout` is not honoured by it
                args.extend(["-rtsp_transport", "tcp", "-timeout", IO_TIMEOUT_US, "-fflags", "nobuffer", "-flags", "low_delay"].map(String::from));
            } else {
                args.extend(["-re", "-stream_loop", "-1"].map(String::from));
                if url.starts_with("http://") || url.starts_with("https://") {
                    args.extend(["-rw_timeout", IO_TIMEOUT_US].map(String::from));
                }
            }
            args.extend(["-i".into(), url.into(), "-an".into(), "-vf".into(), format!("scale={}:{}", self.width, self.height),
                         "-f".into(), "rawvideo".into(), "-pix_fmt".into(), "bgr24".into(), "-".into()]);
            let debug = std::env::var_os("VG360_GUARD_DEBUG").is_some();
            let spawned = Command::new(&ffmpeg)
                .args(&args)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(if debug { Stdio::piped() } else { Stdio::null() })
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
            if let Some(err) = child.stderr.take() {
                // debug only: pass ffmpeg's stderr through with credentials redacted
                thread::spawn(move || {
                    use std::io::BufRead;
                    for line in std::io::BufReader::new(err).lines().map_while(Result::ok) {
                        eprintln!("ffmpeg: {}", redact(&line));
                    }
                });
            }
            *self.child.lock().unwrap() = Some(child);
            loop {
                let mut buf = vec![0u8; frame_len];
                if out.read_exact(&mut buf).is_err() || !self.running.load(Ordering::Relaxed) { break; }
                self.connected.store(true, Ordering::Relaxed);
                *self.last_frame.lock().unwrap() = Some(Instant::now());
                *self.latest.lock().unwrap() = Some(Arc::new(buf));
                self.count.fetch_add(1, Ordering::Relaxed);
            }
            self.connected.store(false, Ordering::Relaxed);
            *self.latest.lock().unwrap() = None; // never serve a frame from a dead connection
            *self.last_frame.lock().unwrap() = None;
            if let Some(mut c) = self.child.lock().unwrap().take() {
                let _ = c.kill();
                let _ = c.wait();
            }
            thread::sleep(Duration::from_secs(1)); // reconnect
        }
    }

    /// Frames received since start (a counter, so callers can tell a new frame from the last one).
    pub fn frames(&self) -> u64 { self.count.load(Ordering::Relaxed) }

    /// Seconds since the last complete frame arrived (None if none on this connection).
    pub fn age_s(&self) -> Option<f64> {
        self.last_frame.lock().unwrap().map(|t| t.elapsed().as_secs_f64())
    }

    /// Newest frame (rotated 180° if configured). None while disconnected, or when the newest
    /// frame is older than `STALL_S` seconds (a stalled stream is never served as fresh).
    pub fn read(&self) -> Option<Bgr> {
        if !self.connected.load(Ordering::Relaxed) || self.age_s()? > STALL_S { return None; }
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

#[cfg(test)]
mod tests {
    use super::redact;

    #[test]
    fn redacts_credentials() {
        assert_eq!(redact("Opening rtsp://admin:s3cr3t@10.0.0.5:554/stream1 for reading"), "Opening rtsp://***@10.0.0.5:554/stream1 for reading");
        assert_eq!(redact("http://u:p@h/a and rtsp://x:y@z/"), "http://***@h/a and rtsp://***@z/");
        assert_eq!(redact("rtsp://admin:p@ss@cam/x"), "rtsp://***@cam/x");
        assert_eq!(redact("rtsp://10.0.0.5/stream and nothing else"), "rtsp://10.0.0.5/stream and nothing else");
        assert_eq!(redact("no url here"), "no url here");
    }
}
