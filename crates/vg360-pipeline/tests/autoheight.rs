//! Auto-height on the rig demo clip: a camera configured with the wrong mount height (from vanishing points
//! the height is unknown) fixes it from the first standing workers before any rule runs.
//! The clip has only 3-4 people (one sample per track), so the test lowers the distinct-track bar from 5 to 3.
use std::sync::Arc;

use vg360_geom::camera::Camera;
use vg360_infer::yolox::Yolox;
use vg360_pipeline::{CameraWorker, RuleSet, WorkerCfg};
use vg360_rules::{Deployed, Signals, SiteMap};
use vg360_scene::scene::Scene;
use vg360_scene::zones::default_zones;
use vg360_vision::img::Bgr;

fn root() -> String { format!("{}/../..", env!("CARGO_MANIFEST_DIR")) }
fn frame(i: usize) -> Bgr { Bgr::load(format!("{}/testdata/clip/f{i:02}.jpg", root())).unwrap() }

#[test]
fn wrong_mount_height_is_fixed_from_standing_workers_before_rules_run() {
    // the clip was rendered from 10 m; configure 6 m
    let cam = Camera { f_px: 640.0 / 35f64.to_radians().tan(), cx: 640.0, cy: 360.0, height_m: 6.0, pitch_deg: 25.0, yaw_deg: 0.0, roll_deg: 0.0, x0: 0.0, y0: 0.0 };
    let det = Yolox::load_tract(format!("{}/models/yolox_nano.onnx", root()), 416).expect("run scripts/fetch_models.sh");
    let dep = Deployed::load(&std::fs::read_to_string(format!("{}/config/vg360_rules.draft.json", root())).unwrap()).unwrap();
    let mut w = CameraWorker::new("AH", det, None, Scene::new(cam, default_zones()), Arc::new(RuleSet::new(dep, SiteMap::rig_default(), None)),
                                  WorkerCfg { tiles: 2, auto_height_boxes: 9, auto_height_min_tracks: 3, ..Default::default() });
    w.set_reference(&frame(0), &[frame(1)]);
    let mut done_at = None;
    let mut ah = None;
    for i in 0..16 {
        let r = w.process(&frame(i), i as f64 / 10.0, 0.1, &Signals::default()).unwrap();
        if r.calibrating { assert!(r.alerts.is_empty(), "no rule may run while calibrating"); }
        else if done_at.is_none() { done_at = Some((i, r.height_m)); ah = r.auto_height.clone(); }
    }
    let (i, h) = done_at.expect("height estimated within the clip");
    eprintln!("mount height 6.0 m → {h:.2} m (rendered from 10 m, workers 1.75 m tall) after frame {i}");
    assert!((h - 10.0).abs() < 1.0, "{h}");
    // the frame that reports the new height also says it is set, and the provenance hash follows the camera in use
    let ah = ah.expect("auto_height reported");
    assert_eq!(ah.state, "set");
    assert_eq!(ah.value, Some(h));
    let hash = |c: &Camera| vg360_evidence::sha256_hex(serde_json::to_string(c).unwrap().as_bytes());
    assert_eq!(w.provenance().unwrap().calibration_sha256, hash(&w.scene.cam));
    assert_ne!(w.provenance().unwrap().calibration_sha256, hash(&Camera { height_m: 6.0, ..w.scene.cam.clone() }));
}
