//! Phase 3 acceptance test on the rig demo clip (testdata/clip, 10 fps).
//! A camera is knocked mid-clip by a known small amount: analytics must carry on through the shift
//! compensation and put people where the un-knocked camera puts them; a large knock must pause
//! analytics; a covered lens must pause them after the guard's hold time.
use vg360_geom::camera::Camera;
use vg360_infer::yolox::Yolox;
use vg360_pipeline::{CameraWorker, Mode, WorkerCfg};
use vg360_scene::{rules::{Ctx, RuleEngine, default_rules}, scene::Scene, zones::default_zones};
use vg360_vision::img::Bgr;

fn root() -> String { format!("{}/../..", env!("CARGO_MANIFEST_DIR")) }
fn frame(i: usize) -> Bgr { Bgr::load(format!("{}/testdata/clip/f{i:02}.jpg", root())).unwrap() }
fn camera() -> Camera {
    // rig demo preset: 10 m mast, 25° down, 70° horizontal field of view, 1280×720
    Camera { f_px: 640.0 / 35f64.to_radians().tan(), cx: 640.0, cy: 360.0, height_m: 10.0, pitch_deg: 25.0, yaw_deg: 0.0, roll_deg: 0.0, x0: 0.0, y0: 0.0 }
}
fn worker(tiles: usize) -> CameraWorker {
    let det = Yolox::load_tract(format!("{}/models/yolox_nano.onnx", root()), 416).expect("run scripts/fetch_models.sh");
    let mut w = CameraWorker::new("CAM-T", det, None, Scene::new(camera(), default_zones()), RuleEngine::new(default_rules(10.0), 0.5),
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
const CTX: Ctx = Ctx { suspended: false, pipe_handling: false };

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
        let ra = a.process(&f, t, 0.1, &CTX).unwrap();
        let knocked = i >= 8;
        let fb = if knocked { knock(&f, 1.5, 32.0, -14.0) } else { f };
        let fb_h = fb.h as f64;
        let rb = b.process(&fb, t, 0.1, &CTX).unwrap();
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
    let r = w.process(&knock(&frame(5), 9.0, 180.0, 60.0), 0.5, 0.1, &CTX).unwrap();
    eprintln!("large knock: mode {:?}, pose {:?}", r.mode, r.pose);
    assert_eq!(r.mode, Mode::Paused);
    assert!(r.recs.is_empty() && r.alerts.is_empty());
}

#[test]
fn covered_lens_pauses_after_the_hold_time() {
    let mut w = worker(1);
    let black = Bgr::new(1280, 720);
    let mut first_pause = None;
    for k in 0..70 {
        let t = 1.0 + k as f64 * 0.1;
        let r = w.process(&black, t, 0.1, &CTX).unwrap();
        if r.mode == Mode::Paused && first_pause.is_none() { first_pause = Some(t - 1.0); }
    }
    let p = first_pause.expect("never paused");
    eprintln!("covered lens: analytics paused after {p:.1} s");
    assert!((vg360_guard::config::BLOCKED_HOLD - 0.15..vg360_guard::config::BLOCKED_HOLD + 0.35).contains(&p));
}

#[test]
fn stream_worker_runs_on_a_file_replayed_as_a_camera() {
    if std::process::Command::new("ffmpeg").arg("-version").output().is_err() { eprintln!("ffmpeg not installed — skipped"); return; }
    let det = Yolox::load_tract(format!("{}/models/yolox_nano.onnx", root()), 416).unwrap();
    let w = CameraWorker::new("CAM-S", det, None, Scene::new(camera(), default_zones()), RuleEngine::new(default_rules(10.0), 0.5),
                              WorkerCfg { tiles: 1, ..Default::default() });
    let rx = vg360_pipeline::spawn_stream(w, format!("{}/testdata/clip/clip.mp4", root()), 1280, 720, 10.0, 3, |_| CTX);
    let got: Vec<_> = rx.iter().take(5).map(|r| r.unwrap()).collect();
    eprintln!("stream: {} results, modes {:?}, people {:?}", got.len(), got.iter().map(|r| r.mode).collect::<Vec<_>>(),
              got.iter().map(|r| r.recs.iter().flatten().count()).collect::<Vec<_>>());
    assert_eq!(got.len(), 5);
    assert!(got.iter().all(|r| r.mode == Mode::Aligned));
}
