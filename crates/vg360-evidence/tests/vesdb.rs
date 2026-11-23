//! End to end against a real vesdb log + one independent witness (binaries from dhaneshkk/test1_vesdb).
//! Set VESDB_BIN_DIR to the directory holding `vesdb`, `vesdb-witness` and `vesdb-verify`; skipped otherwise.
//!
//! Writes the provenance of a VG360 alert (ruleset, model, calibration, camera guard, incident),
//! checks the incident cannot be written twice, survives an outage through the spool, then exports an
//! evidence bundle and verifies it offline with `vesdb-verify` — including the SHA-256 of the frame.
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::json;
use vg360_evidence::{Outcome, Vesdb, sha256_hex};

struct Kill(Vec<Child>);
impl Drop for Kill { fn drop(&mut self) { for c in &mut self.0 { let _ = c.kill(); let _ = c.wait(); } } }

fn bin(dir: &Path, n: &str) -> PathBuf { dir.join(n) }
fn free_port() -> u16 { std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port() }
fn wait_http(url: &str) {
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(20) {
        if ureq::get(url).call().is_ok() { return; }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("{url} did not come up");
}
fn token() -> String { sha256_hex(format!("{:?}{}", Instant::now(), std::process::id()).as_bytes())[..48].to_string() }

#[test]
fn alert_provenance_end_to_end() {
    let Ok(dir) = std::env::var("VESDB_BIN_DIR") else { eprintln!("VESDB_BIN_DIR not set — skipped"); return; };
    let dir = PathBuf::from(dir);
    let work = std::env::temp_dir().join(format!("vg360-vesdb-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let w = |p: &str| work.join(p);
    let run = |args: &[&str]| assert!(Command::new(bin(&dir, "vesdb")).args(args).stdout(Stdio::null()).status().unwrap().success());
    run(&["keygen", "--name", "rig-07.vg360.example/log", "--out", w("log.key").to_str().unwrap()]);
    run(&["keygen", "--name", "hse.operator.example", "--out", w("hse.key").to_str().unwrap()]);
    let (wp, sp) = (free_port(), free_port());
    let witness = Command::new(bin(&dir, "vesdb-witness"))
        .args(["run", "--key-file", w("hse.key").to_str().unwrap(), "--log-key", w("log.key.pub").to_str().unwrap(),
               "--state-file", w("hse.state").to_str().unwrap(), "--listen", &format!("127.0.0.1:{wp}")])
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
    let (wt, at) = (token(), token());
    let start_server = || Command::new(bin(&dir, "vesdb"))
        .args(["run", "--data-directory", w("data").to_str().unwrap(), "--signing-key-file", w("log.key").to_str().unwrap(),
               "--token", &format!("{wt}=read,write"), "--token", &format!("{at}=read,audit"),
               "--witness", &format!("http://127.0.0.1:{wp}={}", w("hse.key.pub").display()),
               "--listen", &format!("127.0.0.1:{sp}"), "--checkpoint-interval-ms", "200"])
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
    let mut procs = Kill(vec![witness, start_server()]);
    let base = format!("http://127.0.0.1:{sp}");
    wait_http(&format!("{base}/api/v2/system/ping"));

    let v = Vesdb::new(&base, "rig-07", &wt, Some(&at), Some(w("spool.ndjson")));
    let ids = |o: Outcome| match o { Outcome::Written(e) => e.into_iter().map(|e| e.id).collect::<Vec<_>>(), Outcome::Spooled => panic!("spooled") };

    // provenance of the decision chain
    let cfg = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/vg360_rules.draft.json")).unwrap();
    let mut evidence = ids(v.ruleset_deployed(&sha256_hex(&cfg), &["LIFT".into(), "RED".into(), "EDGE".into(), "FALL".into(), "PPE".into()], None).unwrap());
    evidence.extend(ids(v.model_deployed("yolox_nano", &"ab".repeat(32), json!({"engine": "onnxruntime"})).unwrap()));
    evidence.extend(ids(v.camera_event("CAM-05", "calibration-set", json!({"f_px": 914.0, "height_m": 10.0, "pitch_deg": 25.0})).unwrap()));
    evidence.extend(ids(v.camera_event("CAM-05", "camera-guard-changed", json!({"status": "Moved", "analytics": "Compensated"})).unwrap()));

    // the incident: atomically hazard + alert, frame hash included
    let frame = b"jpeg bytes of the frame the alert was raised on";
    let frame_sha = sha256_hex(frame);
    let inc = ids(v.incident("CAM-05-EDGE-7-24200",
        json!({"camera": "CAM-05", "rule": "EDGE", "subject": "7", "frame_sha256": frame_sha}),
        json!({"rule": "EDGE", "alert": "working-at-height-alert", "lsr": 9, "severity": "critical", "armed": true,
               "proof": ["the worker is near Deck-Edge (within 1 m)"], "ruleset_sha256": sha256_hex(&cfg)})).unwrap());
    assert_eq!(inc.len(), 2);
    evidence.extend(inc.clone());

    // the same incident id can never be written again (isSubjectPristine)
    let again = v.incident("CAM-05-EDGE-7-24200", json!({}), json!({}));
    assert!(again.is_err(), "a second write of the same incident must be rejected");

    // outage: the write is spooled, then replayed exactly once
    let server = procs.0.remove(1);
    let mut server = server; let _ = server.kill(); let _ = server.wait();
    let spooled = v.incident("CAM-05-FALL-7-31000", json!({"camera": "CAM-05", "rule": "FALL"}), json!({"rule": "FALL", "armed": true})).unwrap();
    assert!(matches!(spooled, Outcome::Spooled));
    procs.0.push(start_server());
    wait_http(&format!("{base}/api/v2/system/ping"));
    assert_eq!(v.flush_spool().unwrap(), 1);
    assert_eq!(v.flush_spool().unwrap(), 0, "spool is empty after a successful replay");

    // receipt for the alert, then a witness-cosigned checkpoint covering everything
    let r = v.receipt(&inc[1]).unwrap();
    assert!(r.to_string().len() > 20);
    let t = Instant::now();
    let cp = loop {
        if let Ok(cp) = v.checkpoint() { if cp["size"].as_u64().unwrap_or(0) as usize > inc[1].parse::<usize>().unwrap() { break cp; } }
        assert!(t.elapsed() < Duration::from_secs(20), "no cosigned checkpoint");
        std::thread::sleep(Duration::from_millis(200));
    };
    let bundle = v.bundle(&evidence).unwrap();
    std::fs::write(w("bundle.json"), bundle.to_string()).unwrap();
    std::fs::write(w("frame.jpg"), frame).unwrap();
    let out = Command::new(bin(&dir, "vesdb-verify"))
        .args(["bundle", w("bundle.json").to_str().unwrap(), "--log-key", w("log.key.pub").to_str().unwrap(),
               "--witness-key", w("hse.key.pub").to_str().unwrap(), "--file-hash", &format!("{}={}:frame_sha256", w("frame.jpg").display(), inc[0])])
        .output().unwrap();
    let report = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    eprintln!("checkpoint size {} · {} events in the bundle · vesdb-verify:\n{report}", cp["size"], evidence.len());
    assert!(out.status.success(), "offline verification failed: {report}");

    // tamper: change the frame → the file-hash check must fail
    std::fs::write(w("frame.jpg"), b"a different frame").unwrap();
    let bad = Command::new(bin(&dir, "vesdb-verify"))
        .args(["bundle", w("bundle.json").to_str().unwrap(), "--log-key", w("log.key.pub").to_str().unwrap(),
               "--witness-key", w("hse.key.pub").to_str().unwrap(), "--file-hash", &format!("{}={}:frame_sha256", w("frame.jpg").display(), inc[0])])
        .output().unwrap();
    assert!(!bad.status.success(), "a swapped frame must fail verification");
}
