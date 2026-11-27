//! Phase 3 acceptance test on the rig demo clip (testdata/clip, 10 fps).
//! A camera is knocked mid-clip by a known small amount: analytics must carry on through the shift
//! compensation and put people where the un-knocked camera puts them; a large knock must pause
//! analytics; a covered lens must pause them after the guard's hold time.
use vg360_geom::camera::Camera;
use vg360_infer::yolox::Yolox;
use vg360_pipeline::{CameraWorker, Mode, WorkerCfg};
use std::sync::Arc;
use vg360_pipeline::RuleSet;
use vg360_rules::{Deployed, Signals, SiteMap};
use vg360_scene::{scene::Scene, zones::default_zones};
use vg360_vision::img::Bgr;

fn root() -> String { format!("{}/../..", env!("CARGO_MANIFEST_DIR")) }
fn frame(i: usize) -> Bgr { Bgr::load(format!("{}/testdata/clip/f{i:02}.jpg", root())).unwrap() }
fn camera() -> Camera {
    // rig demo preset: 10 m mast, 25° down, 70° horizontal field of view, 1280×720
    Camera { f_px: 640.0 / 35f64.to_radians().tan(), cx: 640.0, cy: 360.0, height_m: 10.0, pitch_deg: 25.0, yaw_deg: 0.0, roll_deg: 0.0, x0: 0.0, y0: 0.0 }
}
fn rules() -> Arc<RuleSet> {
    let dep = Deployed::load(&std::fs::read_to_string(format!("{}/config/vg360_rules.draft.json", root())).unwrap()).unwrap();
    Arc::new(RuleSet::new(dep, SiteMap::rig_default(), None))
}
fn worker(tiles: usize) -> CameraWorker {
    let det = Yolox::load_tract(format!("{}/models/yolox_nano.onnx", root()), 416).expect("run scripts/fetch_models.sh");
    let mut w = CameraWorker::new("CAM-T", det, None, Scene::new(camera(), default_zones()), rules(),
                                  WorkerCfg { guard_every: 1, tiles, ..Default::default() });
    let n = w.set_reference(&frame(0), &[frame(1), frame(2)]);
    assert!(n >= 80, "{n} reference points");
    w
}
/// Knock: rotate `deg` about the image centre (counter-clockwise on screen), then shift (dx, dy) px.
fn knock(f: &Bgr, deg: f64, dx: f64, dy: f64) -> Bgr {
    let (c, s) = (deg.to_radians().cos(), deg.to_radians().sin());
    let (cx, cy) = (640.0, 360.0);
    let h = [[c, s, cx - c * cx - s * cy + dx], [-s, c, cy + s * cx - c * cy + dy], [0.0, 0.0, 1.0]];
    f.warp_perspective(&h, f.w, f.h, [30, 30, 30])
}
fn sig() -> Signals { Signals::default() }

#[test]
fn small_knock_is_compensated_and_keeps_people_on_the_same_deck_spots() {
    let mut a = worker(3);
    let mut b = worker(3);
    let (mut n, mut sum_comp, mut sum_naive, mut matched) = (0usize, 0.0, 0.0, 0usize);
    let mut ids_before = std::collections::BTreeSet::new();
    let mut ids_after = std::collections::BTreeSet::new();
    for i in 3..16 {
        let t = i as f64 * 0.1;
        let f = frame(i);
        let ra = a.process(&f, t, 0.1, &sig()).unwrap();
        let knocked = i >= 8;
        let fb = if knocked { knock(&f, 1.5, 32.0, -14.0) } else { f };
        let fb_h = fb.h as f64;
        let rb = b.process(&fb, t, 0.1, &sig()).unwrap();
        for r in rb.recs.iter().flatten() { if knocked { ids_after.insert(r.track); } else { ids_before.insert(r.track); } }
        if !knocked { assert_eq!(rb.mode, Mode::Aligned, "frame {i}"); continue; }
        assert_eq!(rb.mode, Mode::Compensated, "frame {i}: pose {:?}", rb.pose);
        let p = rb.pose.unwrap();
        assert!((p.angle - -1.5).abs() < 0.3 && (p.dx - 32.0 / 1280.0).abs() < 0.004, "frame {i}: {p:?}");
        // where the un-knocked camera puts each person vs. the knocked camera with and without compensation
        let feet_a: Vec<[f64; 2]> = ra.recs.iter().flatten().map(|r| r.foot).collect();
        for (r, d) in rb.recs.iter().zip(&rb.boxes) {
            let Some(r) = r else { continue };
            // a box cut by the bottom of the image has no visible feet: no deck position to compare
            if d.y2 >= fb_h - 4.0 { continue; }
            let near = |q: [f64; 2]| feet_a.iter().map(|p| (p[0] - q[0]).hypot(p[1] - q[1])).fold(f64::INFINITY, f64::min);
            let comp = near(r.foot);
            let Some((naive_foot, ..)) = b.scene.box_geometry([d.x1, d.y1, d.x2, d.y2]) else { continue };
            n += 1;
            if comp < 0.5 { matched += 1; }
            sum_comp += comp;
            sum_naive += near(naive_foot);
        }
    }
    let (mc, mn) = (sum_comp / n as f64, sum_naive / n as f64);
    eprintln!("knocked frames: {n} people · mean deck error compensated {mc:.2} m vs uncompensated {mn:.2} m · {matched}/{n} within 0.5 m");
    eprintln!("track ids before knock {ids_before:?}, after {ids_after:?}");
    assert!(n >= 10);
    assert!(mc < 0.3, "compensated error {mc}");
    assert!(mn > 2.0 * mc, "compensation should clearly help: {mn} vs {mc}");
    assert!(matched as f64 >= 0.9 * n as f64);
    assert!(ids_after.iter().filter(|id| !ids_before.contains(id)).count() <= 1, "tracks should survive the knock");
}

#[test]
fn large_knock_pauses_analytics() {
    let mut w = worker(1);
    let r = w.process(&knock(&frame(5), 9.0, 180.0, 60.0), 0.5, 0.1, &sig()).unwrap();
    eprintln!("large knock: mode {:?}, pose {:?}", r.mode, r.pose);
    assert_eq!(r.mode, Mode::Paused);
    assert!(r.recs.is_empty() && r.alerts.is_empty());
}

#[test]
fn covered_lens_suspends_analytics_at_once_and_raises_blocked_after_the_hold_time() {
    let mut w = worker(1);
    // a real frame first, so analytics are known to run before the lens is covered
    let live = w.process(&frame(5), 0.5, 0.1, &sig()).unwrap();
    assert!(live.analytics_suspended.is_none() && !live.boxes.is_empty(), "analytics run on the clear view");
    let black = Bgr::new(1280, 720);
    let (mut first_blocked, mut first_paused) = (None, None);
    for k in 0..70 {
        let t = 1.0 + k as f64 * 0.1;
        let r = w.process(&black, t, 0.1, &sig()).unwrap();
        // from the very first dark frame: no detections, no scene, no rule evaluated, reason given
        assert_eq!(r.analytics_suspended.as_deref(), Some("lens blocked"), "frame {k}");
        assert!(r.boxes.is_empty() && r.recs.is_empty() && r.alerts.is_empty(), "frame {k}");
        if k == 0 { assert_eq!(r.status, vg360_guard::policy::Status::Ok); assert!(r.guard_pending.is_some()); }
        if r.status == vg360_guard::policy::Status::Blocked && first_blocked.is_none() { first_blocked = Some(t - 1.0); }
        if r.mode == Mode::Paused && first_paused.is_none() { first_paused = Some(t - 1.0); }
    }
    let (b, p) = (first_blocked.expect("never Blocked"), first_paused.expect("never paused"));
    eprintln!("covered lens: analytics suspended from the first dark frame; status Blocked after {b:.1} s, mode Paused after {p:.1} s");
    let hold = vg360_guard::config::BLOCKED_HOLD;
    assert!((hold - 0.15..hold + 0.35).contains(&b));
    assert!((hold - 0.15..hold + 0.35).contains(&p));
}

#[test]
fn a_failed_tick_keeps_the_last_good_compensation() {
    let mut w = worker(1);
    assert_eq!(w.cur_to_ref(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let mut good = None;
    for i in 3..9 {
        let r = w.process(&knock(&frame(i), 1.5, 32.0, -14.0), i as f64 * 0.1, 0.1, &sig()).unwrap();
        if r.mode == Mode::Compensated { good = Some(w.cur_to_ref()); }
    }
    let good = good.expect("never compensated");
    assert_ne!(good[0][2], 0.0, "a real compensation matrix");
    // lens goes dark: the guard tick fails, analytics suspend, the matrix is NOT reset to identity
    let r = w.process(&Bgr::new(1280, 720), 1.0, 0.1, &sig()).unwrap();
    assert_eq!(r.analytics_suspended.as_deref(), Some("lens blocked"));
    assert_eq!(r.mode, Mode::Compensated, "still inside the hold time");
    assert_eq!(w.cur_to_ref(), good);
    // set_reference resets everything tied to the old reference
    w.set_reference(&frame(0), &[frame(1), frame(2)]);
    assert_eq!(w.cur_to_ref()[0][2], 0.0);
    assert_eq!(w.mode(), Mode::Aligned);
}

#[test]
fn compensated_while_status_moved_still_runs_analytics() {
    // a small knock held past MOVED_HOLD: the alarm says CAMERA MOVED but analytics keep running (the guide's promise)
    let mut w = worker(1);
    let mut last = None;
    for k in 0..45 {
        last = Some(w.process(&knock(&frame(5 + k % 4), 1.5, 32.0, -14.0), k as f64 * 0.1, 0.1, &sig()).unwrap());
    }
    let r = last.unwrap();
    eprintln!("after 4.5 s knocked: status {:?} mode {:?}", r.status, r.mode);
    assert_eq!(r.status, vg360_guard::policy::Status::Moved);
    assert_eq!(r.mode, Mode::Compensated);
    assert!(r.analytics_suspended.is_none() && !r.boxes.is_empty());
}

#[test]
fn rotated_box_is_not_inflated_when_mapped_to_reference() {
    let mut w = worker(1);
    for i in 3..9 { w.process(&knock(&frame(i), 3.0, 0.0, 0.0), i as f64 * 0.1, 0.1, &sig()).unwrap(); }
    assert_eq!(w.mode(), Mode::Compensated);
    let d = vg360_infer::yolox::Det { x1: 500.0, y1: 200.0, x2: 580.0, y2: 400.0, score: 0.9, class: 0 };
    let b = w.to_reference(&d);
    let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
    eprintln!("rotated 3 deg: 80x200 box maps to {bw:.1}x{bh:.1}");
    assert!((bw - 80.0).abs() < 1e-6, "width kept");
    assert!((bh - 200.0).abs() <= 4.0, "height inflated by more than 2 %: {bh}");
}

#[test]
fn provenance_follows_the_reference_and_the_camera() {
    let mut w = worker(1);
    assert_eq!(w.evidence_errors(), 0);
    let hash = |c: &Camera| vg360_evidence::sha256_hex(serde_json::to_string(c).unwrap().as_bytes());
    assert_eq!(w.provenance().unwrap().calibration_sha256, hash(&w.scene.cam));
    w.scene.cam.height_m = 7.0;                 // camera changed behind the worker's back
    w.set_reference(&frame(0), &[frame(1), frame(2)]);
    assert_eq!(w.provenance().unwrap().calibration_sha256, hash(&w.scene.cam));
}

#[test]
fn reference_capture_validates_and_persists() {
    let mut w = worker(1);
    let dark = vec![Bgr::new(1280, 720); 3];
    let why = w.capture_reference(&dark).unwrap_err();
    eprintln!("dark reference refused: {why}");
    assert!(why.contains("dark"), "{why}");
    let frames: Vec<Bgr> = (0..4).map(frame).collect();
    let (bi, info) = w.capture_reference(&frames).expect("clip reference is usable");
    eprintln!("best reference frame {bi}, {} points", info.points);
    assert!(info.points >= 80);
    let dir = std::env::temp_dir().join(format!("vg360-ref-{}", std::process::id()));
    let sha = w.save_reference(&dir, &frames[bi]).unwrap();
    // restart: a fresh worker loads the stored reference; the unmoved view is Aligned, a moved one is seen as moved
    let mut w2 = worker(1);
    let loaded = w2.load_reference(&dir).unwrap().expect("stored reference");
    assert_eq!(loaded.sha256, sha);
    eprintln!("stored reference reloaded: {} points (captured {})", loaded.points, info.points);
    let r = w2.process(&frame(6), 1.0, 0.1, &sig()).unwrap();
    assert_eq!(r.mode, Mode::Aligned, "{:?}", r.pose);
    let r = w2.process(&knock(&frame(6), 1.5, 32.0, -14.0), 1.1, 0.1, &sig()).unwrap();
    assert_eq!(r.mode, Mode::Compensated, "a camera moved while the server was down is seen as moved: {:?}", r.pose);
    // a tampered image is refused
    std::fs::write(dir.join("CAM-T.png"), b"not the reference").unwrap();
    assert!(worker(1).load_reference(&dir).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stream_loop_reports_a_stall_and_carries_on() {
    use vg360_pipeline::{Src, drive};
    let mut w = worker(1);
    let mut script = vec![Src::Frame(frame(3), Some(0.01)), Src::Stall, Src::Stall, Src::Frame(frame(4), Some(0.02)), Src::End].into_iter();
    let (tx, rx) = std::sync::mpsc::sync_channel(16);
    let mut t = 0.0;
    drive(&mut w, &mut || script.next().unwrap_or(Src::End), &mut || { t += 0.1; t }, 0.1, &|_| sig(), &tx);
    drop(tx);
    let got: Vec<_> = rx.iter().collect();
    assert_eq!(got.len(), 4);
    assert_eq!(got[0].as_ref().unwrap().frame_age_s, Some(0.01));
    assert_eq!(got[1].as_ref().unwrap_err(), "stream stalled: no frames for 30 s");
    assert!(got[2].is_err());
    assert_eq!(got[3].as_ref().unwrap().frame_age_s, Some(0.02), "frames resume after the stall");
}

#[test]
fn stream_worker_runs_on_a_file_replayed_as_a_camera() {
    if std::process::Command::new("ffmpeg").arg("-version").output().is_err() { eprintln!("ffmpeg not installed — skipped"); return; }
    let det = Yolox::load_tract(format!("{}/models/yolox_nano.onnx", root()), 416).unwrap();
    let w = CameraWorker::new("CAM-S", det, None, Scene::new(camera(), default_zones()), rules(),
                              WorkerCfg { tiles: 1, ..Default::default() });
    let rx = vg360_pipeline::spawn_stream(w, format!("{}/testdata/clip/clip.mp4", root()), 1280, 720, 10.0, 3, |_| sig());
    let got: Vec<_> = rx.iter().take(5).map(|r| r.unwrap()).collect();
    eprintln!("stream: {} results, modes {:?}, people {:?}", got.len(), got.iter().map(|r| r.mode).collect::<Vec<_>>(),
              got.iter().map(|r| r.recs.iter().flatten().count()).collect::<Vec<_>>());
    assert_eq!(got.len(), 5);
    assert!(got.iter().all(|r| r.mode == Mode::Aligned));
}
