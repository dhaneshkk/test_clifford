//! End-to-end: start the console with `VG360_DEMO=1` (the simulated rig clip replayed as a camera) and check
//! the live pipeline, the SSE stream, snapshots, site signals and streaming chat completions.
//! Needs ffmpeg and the models (`scripts/fetch_models.sh`); skipped otherwise.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Server(Child);
impl Drop for Server { fn drop(&mut self) { let _ = self.0.kill(); } }

fn http(port: u16, method: &str, path: &str, body: Option<&str>, max_wait: Duration) -> (u16, String, Vec<u8>) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(max_wait)).unwrap();
    let b = body.unwrap_or("");
    write!(s, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len()).unwrap();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let t0 = Instant::now();
    while t0.elapsed() < max_wait {
        match s.read(&mut chunk) { Ok(0) => break, Ok(n) => buf.extend_from_slice(&chunk[..n]), Err(_) => break }
    }
    let split = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap_or(buf.len());
    let head = String::from_utf8_lossy(&buf[..split]).to_string();
    let code = head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    (code, head, buf[(split + 4).min(buf.len())..].to_vec())
}

#[test]
fn live_demo_end_to_end() {
    let root = env!("CARGO_MANIFEST_DIR");
    if !std::path::Path::new(&format!("{root}/../../models/yolox_nano.onnx")).exists()
        || Command::new("ffmpeg").arg("-version").stdout(Stdio::null()).status().is_err() {
        eprintln!("skipped: needs ffmpeg and models (scripts/fetch_models.sh)");
        return;
    }
    let port = 39000 + (std::process::id() % 1000) as u16;
    let state = std::env::temp_dir().join(format!("vg360-live-test-{}", std::process::id()));
    let _srv = Server(Command::new(env!("CARGO_BIN_EXE_vg360-server"))
        .env("VG360_DEMO", "1").env("PORT", port.to_string()).env("VG360_STATE_DIR", &state).env("VG360_ROOT", root)
        .env_remove("VG360_VESDB_URL").stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap());

    // the demo camera starts producing frames
    let t0 = Instant::now();
    let live = loop {
        assert!(t0.elapsed() < Duration::from_secs(120), "no live frames from the demo camera");
        std::thread::sleep(Duration::from_millis(500));
        if TcpStream::connect(("127.0.0.1", port)).is_err() { continue; }
        let (code, _, b) = http(port, "GET", "/api/live", None, Duration::from_secs(5));
        if code != 200 { continue; }
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        if v["cameras"][0]["frames"].as_u64().unwrap_or(0) >= 5 { break v; }
    };
    let cam = &live["cameras"][0];
    assert_eq!(cam["id"], "DEMO");
    assert!(cam["error"].is_null(), "{cam}");
    assert!(cam["latest"]["people"].as_array().unwrap().len() >= 1, "the demo clip has workers on deck: {cam}");

    // latest frame as JPEG
    let (code, head, jpg) = http(port, "GET", "/api/live/DEMO/frame.jpg", None, Duration::from_secs(5));
    assert_eq!(code, 200);
    assert!(head.to_lowercase().contains("image/jpeg"));
    assert_eq!(&jpg[..2], &[0xff, 0xd8]);
    assert_eq!(http(port, "GET", "/api/live/NOPE/frame.jpg", None, Duration::from_secs(5)).0, 404);

    // SSE: hello + frame events
    let (code, head, sse) = http(port, "GET", "/api/live/events", None, Duration::from_secs(4));
    assert_eq!(code, 200);
    assert!(head.to_lowercase().contains("text/event-stream"));
    let sse = String::from_utf8_lossy(&sse);
    assert!(sse.contains("event: hello") && sse.contains("event: frame"), "{}", &sse[..sse.len().min(400)]);

    // site signals
    let (code, _, b) = http(port, "POST", "/api/signals", Some(r#"{"suspended_load":true,"active":["pipe-handling"]}"#), Duration::from_secs(5));
    assert_eq!(code, 200);
    let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(v["suspended_load"], true);
    assert_eq!(v["active"][0], "pipe-handling");

    // streaming chat completions: OpenAI chunks then [DONE]
    let req = r#"{"model":"vg360-ops","stream":true,"messages":[{"role":"user","content":"what is live now?"}]}"#;
    let (code, head, b) = http(port, "POST", "/v1/chat/completions", Some(req), Duration::from_secs(5));
    assert_eq!(code, 200);
    assert!(head.to_lowercase().contains("text/event-stream"));
    let body = String::from_utf8_lossy(&b);
    let datas: Vec<&str> = body.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
    assert_eq!(*datas.last().unwrap(), "[DONE]");
    let mut text = String::new();
    for d in &datas[..datas.len() - 1] {
        let c: serde_json::Value = serde_json::from_str(d).unwrap();
        assert_eq!(c["object"], "chat.completion.chunk");
        text += c["choices"][0]["delta"]["content"].as_str().unwrap_or("");
    }
    assert!(text.contains("LIVE DEMO"), "{text}");
    let last: serde_json::Value = serde_json::from_str(datas[datas.len() - 2]).unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");

    // non-streaming still returns one chat.completion
    let (code, _, b) = http(port, "POST", "/v1/chat/completions", Some(r#"{"messages":[{"role":"user","content":"live"}]}"#), Duration::from_secs(5));
    assert_eq!(code, 200);
    let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(v["object"], "chat.completion");

    // every alert snapshot on disk has the SHA-256 the alert (and its vesdb incident) carries
    for a in live["alerts"].as_array().unwrap() {
        if let (Some(id), Some(sha)) = (a["incident"].as_str(), a["snapshot_sha256"].as_str()) {
            let (code, _, jpg) = http(port, "GET", &format!("/api/evidence/{id}/snapshot.jpg"), None, Duration::from_secs(5));
            assert_eq!(code, 200);
            assert_eq!(vg360_evidence::sha256_hex(&jpg), sha);
        }
    }
    let _ = std::fs::remove_dir_all(&state);
}
