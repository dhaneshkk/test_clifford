//! POST /api/calibrate: deck edges and standing people of a known camera → fov, pitch, roll, height.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use vg360_geom::camera::Camera;

struct Server(Child);
impl Drop for Server { fn drop(&mut self) { let _ = self.0.kill(); } }

fn post(port: u16, path: &str, body: &str) -> serde_json::Value {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    write!(s, "POST {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    let mut buf = String::new();
    s.read_to_string(&mut buf).unwrap();
    serde_json::from_str(buf.split("\r\n\r\n").nth(1).unwrap()).unwrap()
}

#[test]
fn calibrate_from_deck_edges_and_people() {
    let root = env!("CARGO_MANIFEST_DIR");
    let port = 38000 + (std::process::id() % 1000) as u16;
    let state = std::env::temp_dir().join(format!("vg360-cal-{}", std::process::id()));
    let _srv = Server(Command::new(env!("CARGO_BIN_EXE_vg360-server")).env("PORT", port.to_string()).env("VG360_STATE_DIR", &state)
        .env("VG360_ROOT", root).env_remove("VG360_DEMO").stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap());
    let t0 = Instant::now();
    while TcpStream::connect(("127.0.0.1", port)).is_err() { assert!(t0.elapsed() < Duration::from_secs(60)); std::thread::sleep(Duration::from_millis(200)); }

    // a 70° camera, 12 m up, 32° down, looking along a bearing of 20°
    let cam = Camera { f_px: 640.0 / 35f64.to_radians().tan(), cx: 640.0, cy: 360.0, height_m: 12.0, pitch_deg: 32.0, yaw_deg: 20.0, roll_deg: 0.0, x0: 0.0, y0: 0.0 };
    let family = |bearing: f64| -> Vec<[f64; 4]> {
        let (d, n) = { let b = bearing.to_radians(); ([b.sin(), b.cos()], [b.cos(), -b.sin()]) };
        (0..5).map(|k| {
            let o = -8.0 + 4.0 * k as f64;
            let c = [n[0] * o + 22.0 * 20f64.to_radians().sin(), n[1] * o + 22.0 * 20f64.to_radians().cos()];
            let a = cam.project3_raw([c[0] - 5.0 * d[0], c[1] - 5.0 * d[1], 0.0]);
            let b = cam.project3_raw([c[0] + 5.0 * d[0], c[1] + 5.0 * d[1], 0.0]);
            [a[0], a[1], b[0], b[1]]
        }).collect()
    };
    let people: Vec<[f64; 4]> = [(-3.0, 18.0), (2.0, 24.0), (6.0, 15.0), (0.0, 30.0)].iter().map(|&(x, y)| {
        let f = cam.project3_raw([x, y, 0.0]); let h = cam.project3_raw([x, y, 1.70]);
        [f[0] - 12.0, h[1], f[0] + 12.0, f[1]]
    }).collect();
    let body = serde_json::json!({ "width": 1280, "height": 720, "lines_a": family(65.0), "lines_b": family(-25.0), "people": people }).to_string();
    let r = post(port, "/api/calibrate", &body);
    eprintln!("{r}");
    assert_eq!(r["ok"], true, "{r}");
    assert!((r["hfov_deg"].as_f64().unwrap() - 70.0).abs() < 0.5, "{r}");
    assert!((r["pitch_deg"].as_f64().unwrap() - 32.0).abs() < 0.5, "{r}");
    assert!(r["roll_deg"].as_f64().unwrap().abs() < 0.5, "{r}");
    assert!((r["height_m"].as_f64().unwrap() - 12.0).abs() < 0.3, "{r}");
    // parallel image lines (no vanishing point) are refused with a reason
    let bad = serde_json::json!({ "width": 1280, "height": 720, "lines_a": [[0, 0, 100, 0], [0, 50, 100, 50]], "lines_b": family(-25.0) }).to_string();
    assert_eq!(post(port, "/api/calibrate", &bad)["ok"], false);
    let _ = std::fs::remove_dir_all(&state);
}
