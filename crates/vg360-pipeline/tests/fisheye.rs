//! Fisheye cameras end to end (notebook §9b). The rig demo clip is resampled into what a 185° fisheye at the
//! same mast position would see; the worker analyses it through three upright virtual views and one tracker.
//! The people it puts on the deck must be where the pinhole worker puts them on the original frames.
use std::sync::Arc;

use vg360_geom::camera::Camera;
use vg360_infer::fisheye::FisheyeCam;
use vg360_infer::yolox::Yolox;
use vg360_pipeline::{CameraWorker, FisheyeRig, Mode, RuleSet, WorkerCfg};
use vg360_rules::{Deployed, Signals, SiteMap};
use vg360_scene::scene::Scene;
use vg360_scene::zones::default_zones;
use vg360_vision::img::Bgr;
use vg360_vision::lens::Fisheye;

fn root() -> String { format!("{}/../..", env!("CARGO_MANIFEST_DIR")) }
fn frame(i: usize) -> Bgr { Bgr::load(format!("{}/testdata/clip/f{i:02}.jpg", root())).unwrap() }
fn camera() -> Camera {
    Camera { f_px: 640.0 / 35f64.to_radians().tan(), cx: 640.0, cy: 360.0, height_m: 10.0, pitch_deg: 25.0, yaw_deg: 0.0, roll_deg: 0.0, x0: 0.0, y0: 0.0 }
}
/// the notebook's demo fisheye: Kannala–Brandt, 2880² (an 8 MP fisheye: about the pinhole's resolution on axis)
pub fn demo_fisheye() -> FisheyeCam {
    FisheyeCam::from_pose(Fisheye { fx: 892.0, fy: 892.0, cx: 1440.0, cy: 1440.0, k: [0.02, -0.01, 0.002, -0.0005] }, &camera())
}
fn worker(fish: bool) -> CameraWorker {
    let det = Yolox::load_tract(format!("{}/models/yolox_nano.onnx", root()), 416).expect("run scripts/fetch_models.sh");
    let dep = Deployed::load(&std::fs::read_to_string(format!("{}/config/vg360_rules.draft.json", root())).unwrap()).unwrap();
    let w = CameraWorker::new(if fish { "FE" } else { "PIN" }, det, None, Scene::new(camera(), default_zones()),
                              Arc::new(RuleSet::new(dep, SiteMap::rig_default(), None)), WorkerCfg { tiles: 2, ..Default::default() });
    if fish { w.with_fisheye(FisheyeRig::new(demo_fisheye(), 0.0, 25.0, Some(vec![(-30.0, 25.0), (0.0, 25.0), (30.0, 25.0)]), (1280, 720))) } else { w }
}

#[test]
fn fisheye_worker_puts_people_where_the_pinhole_worker_does() {
    let fc = demo_fisheye();
    let fish: Vec<Bgr> = (0..16).map(|i| fc.render_from_pinhole(&camera(), &frame(i), 2880, 2880, [25, 25, 25])).collect();
    let (mut pin, mut fe) = (worker(false), worker(true));
    pin.set_reference(&frame(0), &[frame(1)]);
    fe.set_reference(&fish[0], &[fish[1].clone()]);
    let (mut errs, mut n_pin, mut n_fe) = (Vec::new(), 0, 0);
    for i in 0..16 {
        let t = i as f64 / 10.0;
        let a = pin.process(&frame(i), t, 0.1, &Signals::default()).unwrap();
        let b = fe.process(&fish[i], t, 0.1, &Signals::default()).unwrap();
        assert_eq!(b.mode, Mode::Aligned);
        let pa: Vec<[f64; 2]> = a.recs.iter().flatten().map(|r| r.foot).collect();
        let pb: Vec<[f64; 2]> = b.recs.iter().flatten().map(|r| r.foot).collect();
        n_pin += pa.len(); n_fe += pb.len();
        for q in &pb {
            if let Some(d) = pa.iter().map(|p| (p[0] - q[0]).hypot(p[1] - q[1])).min_by(f64::total_cmp) { errs.push(d); }
        }
        // boxes are reported in fisheye pixels
        for d in &b.boxes { assert!(d.x1 >= 0.0 && d.y2 <= 2880.0 && d.x2 > d.x1, "{d:?}"); }
    }
    errs.sort_by(f64::total_cmp);
    let med = errs[errs.len() / 2];
    eprintln!("pinhole {n_pin} people, fisheye {n_fe} people over 16 frames; deck position vs pinhole: median {med:.2} m, max {:.2} m",
              errs.last().unwrap());
    assert!(n_fe as f64 >= 0.9 * n_pin as f64, "fisheye finds {n_fe} vs pinhole {n_pin}");
    assert!(med < 0.3, "median {med} m");
}
