//! End-to-end check on real frames: reference vs. a copy bumped by a known shift + rotation.
use std::time::Instant;
use vg360_vision::{features::{Detector, ratio_match}, img::Bgr, robust::{self, Model}};

fn frame(name: &str) -> Bgr { Bgr::load(format!("{}/../../testdata/frames/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap() }

#[test]
fn recovers_known_camera_bump() {
    let (a, b) = (frame("ref.png"), frame("moved.jpg"));
    let (w, h) = (960, 540);
    let ga = a.resize_area(w, h).to_gray().clahe(2.5, 8);
    let gb = b.resize_area(w, h).to_gray().clahe(2.5, 8);
    let det = Detector::default();
    let t = Instant::now();
    let (ka, da) = det.detect(&ga, |_, _| true);
    let (kb, db) = det.detect(&gb, |_, _| true);
    let t_det = t.elapsed();
    let m = ratio_match(&da, &db, 0.8);
    let src: Vec<_> = m.iter().map(|m| [ka[m.query].x as f64, ka[m.query].y as f64]).collect();
    let dst: Vec<_> = m.iter().map(|m| [kb[m.train].x as f64, kb[m.train].y as f64]).collect();
    let f = robust::fit(Model::Homography, &src, &dst, 8.0, 10_000, 0.999).unwrap();
    eprintln!("keys {}/{} matches {} inliers {} detect {:?}", ka.len(), kb.len(), m.len(), f.n_inliers, t_det);
    // truth (full-res): rotate 3.2° CCW about (640,360) then shift (+70, −25) → at 960 px scale by 0.75
    let s = 0.75;
    let (c, sn) = (3.2f64.to_radians().cos(), 3.2f64.to_radians().sin());
    for p in [[100.0, 100.0], [860.0, 100.0], [480.0, 270.0], [860.0, 440.0]] {
        let (x, y) = (p[0] / s - 640.0, p[1] / s - 360.0);
        let truth = [(c * x + sn * y + 640.0 + 70.0) * s, (-sn * x + c * y + 360.0 - 25.0) * s];
        let est = robust::project(&f.model, p);
        assert!((est[0] - truth[0]).abs() < 2.0 && (est[1] - truth[1]).abs() < 2.0, "{p:?}: {est:?} vs {truth:?}");
    }
}
