//! Deck calibration from clicked point pairs, replacing the notebook's
//! `cv2.findHomography` + focal search + `cv2.solvePnP(SOLVEPNP_IPPE)`.
//!
//! 1. Normalised DLT homography deck(X, Y) → image(u, v).
//! 2. For each candidate focal length, decompose H = λK[r₁ r₂ t] → camera centre and look
//!    direction → initial (height, pitch, yaw, x₀, y₀), roll 0.
//! 3. Levenberg–Marquardt on reprojection error over all 7 parameters
//!    (f, height, pitch, yaw, roll, x₀, y₀), starting from the best candidate.
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use vg360_geom::camera::Camera;

use crate::robust::homography_dlt;

#[derive(Clone, Debug)]
pub struct Calibration { pub camera: Camera, pub rms_px: f64, pub rms_m: f64 }

fn to_vec(c: &Camera) -> [f64; 7] { [c.f_px, c.height_m, c.pitch_deg, c.yaw_deg, c.roll_deg, c.x0, c.y0] }
fn from_vec(v: &[f64], cx: f64, cy: f64) -> Camera {
    Camera { f_px: v[0], cx, cy, height_m: v[1], pitch_deg: v[2], yaw_deg: v[3], roll_deg: v[4], x0: v[5], y0: v[6] }
}

fn residuals(c: &Camera, world: &[[f64; 2]], img: &[[f64; 2]]) -> Option<Vec<f64>> {
    let mut r = Vec::with_capacity(2 * world.len());
    for (w, p) in world.iter().zip(img) {
        let q = c.project(w[0], w[1])?;
        r.push(q[0] - p[0]);
        r.push(q[1] - p[1]);
    }
    Some(r)
}

fn rms(r: &[f64]) -> f64 { (r.iter().map(|v| v * v).sum::<f64>() / (r.len() / 2).max(1) as f64).sqrt() }

/// Initial pose from the homography for a given focal length.
fn decompose(h: &[[f64; 3]; 3], f: f64, cx: f64, cy: f64) -> Option<Camera> {
    let k_inv = Matrix3::new(1.0 / f, 0.0, -cx / f, 0.0, 1.0 / f, -cy / f, 0.0, 0.0, 1.0);
    let m = k_inv * Matrix3::from_fn(|r, c| h[r][c]);
    let (m1, m2, m3) = (m.column(0).into_owned(), m.column(1).into_owned(), m.column(2).into_owned());
    let mut lam = 2.0 / (m1.norm() + m2.norm());
    if m3[2] * lam < 0.0 { lam = -lam; } // deck in front of the camera
    let (r1, r2, t): (Vector3<f64>, Vector3<f64>, Vector3<f64>) = (m1 * lam, m2 * lam, m3 * lam);
    let r3 = r1.cross(&r2);
    // nearest rotation (world→camera)
    let svd = Matrix3::from_columns(&[r1, r2, r3]).svd(true, true);
    let r = svd.u? * svd.v_t?;
    let c = -(r.transpose() * t); // camera centre in world coordinates
    let fwd = r.row(2).transpose(); // camera z axis in world coordinates
    let height = c[2].abs();
    let pitch = (-fwd[2]).clamp(-1.0, 1.0).asin().to_degrees();
    let yaw = fwd[0].atan2(fwd[1]).to_degrees(); // compass bearing, as Camera::rotation
    Some(Camera { f_px: f, cx, cy, height_m: height, pitch_deg: pitch, yaw_deg: yaw, roll_deg: 0.0, x0: c[0], y0: c[1] })
}

fn levenberg_marquardt(start: &Camera, world: &[[f64; 2]], img: &[[f64; 2]], fix_f: bool) -> Option<(Camera, f64)> {
    let (cx, cy) = (start.cx, start.cy);
    let mut p = to_vec(start);
    let mut r = residuals(start, world, img)?;
    let mut cost: f64 = r.iter().map(|v| v * v).sum();
    let mut mu = 1e-3;
    let free: Vec<usize> = (if fix_f { 1 } else { 0 }..7).collect();
    for _ in 0..500 {
        let n = r.len();
        let mut j = DMatrix::<f64>::zeros(n, free.len());
        for (col, &k) in free.iter().enumerate() {
            let step = 1e-6 * p[k].abs().max(1e-3);
            let mut q = p; q[k] += step;
            let rq = residuals(&from_vec(&q, cx, cy), world, img)?;
            for i in 0..n { j[(i, col)] = (rq[i] - r[i]) / step; }
        }
        let jt = j.transpose();
        let g = &jt * DVector::from_vec(r.clone());
        let a = &jt * &j;
        let mut improved = false;
        for _ in 0..16 {
            let mut lhs = a.clone();
            for i in 0..free.len() { lhs[(i, i)] += mu * a[(i, i)].max(1e-9); }
            let Some(dx) = lhs.lu().solve(&(-&g)) else { mu *= 10.0; continue };
            let mut q = p;
            for (col, &k) in free.iter().enumerate() { q[k] += dx[col]; }
            if let Some(rq) = residuals(&from_vec(&q, cx, cy), world, img) {
                let c2: f64 = rq.iter().map(|v| v * v).sum();
                if c2 < cost {
                    p = q; r = rq; cost = c2; mu = (mu / 3.0).max(1e-12); improved = true;
                    break;
                }
            }
            mu *= 10.0;
        }
        if !improved { break; }
    }
    Some((from_vec(&p, cx, cy), rms(&r)))
}

/// Calibrate from ≥ 4 deck points (metres) and their pixels. `(cx, cy)`: principal point
/// (image centre unless known). Focal length is searched over 0.3 … 3 × image width.
pub fn calibrate_from_points(world: &[[f64; 2]], img: &[[f64; 2]], cx: f64, cy: f64) -> Option<Calibration> {
    let h = homography_dlt(world, img)?;
    let width = 2.0 * cx;
    let mut best: Option<(Camera, f64)> = None;
    for i in 0..28 {
        let f = width * 0.3 * (10f64).powf(i as f64 / 27.0);
        let Some(c0) = decompose(&h, f, cx, cy) else { continue };
        if let Some((c, e)) = levenberg_marquardt(&c0, world, img, true) {
            if best.as_ref().is_none_or(|b| e < b.1) { best = Some((c, e)); }
        }
    }
    let (c, _) = best?;
    let (cam, rms_px) = levenberg_marquardt(&c, world, img, world.len() < 5)?;
    let rms_m = (world.iter().zip(img).map(|(w, p)| {
        let g = cam.ground(p[0], p[1]).unwrap_or([f64::NAN; 2]);
        (g[0] - w[0]).powi(2) + (g[1] - w[1]).powi(2)
    }).sum::<f64>() / world.len() as f64).sqrt();
    Some(Calibration { camera: cam, rms_px, rms_m })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_notebook_selftest_camera() {
        // same camera as the notebook's point-calibration self-test
        let truth = Camera { f_px: 1100.0, cx: 960.0, cy: 540.0, height_m: 9.0, pitch_deg: 30.0, yaw_deg: 20.0, roll_deg: 3.0, x0: 0.0, y0: 0.0 };
        let mut world = vec![];
        let mut img = vec![];
        let mut s: u64 = 7;
        let mut noise = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s % 1000) as f64 / 1000.0 - 0.5) * 0.6 };
        for &(x, y) in &[(-6.0, 8.0), (2.0, 9.0), (9.0, 12.0), (-4.0, 16.0), (5.0, 18.0), (12.0, 22.0), (-2.0, 25.0), (8.0, 30.0)] {
            let p = truth.project(x, y).unwrap();
            world.push([x, y]);
            img.push([p[0] + noise(), p[1] + noise()]);
        }
        let c = calibrate_from_points(&world, &img, 960.0, 540.0).unwrap();
        let k = &c.camera;
        eprintln!("{k:?} rms {:.3}px {:.3}m", c.rms_px, c.rms_m);
        assert!(c.rms_px < 0.5, "{}", c.rms_px);
        assert!((k.f_px - 1100.0).abs() < 30.0 && (k.height_m - 9.0).abs() < 0.3);
        assert!((k.pitch_deg - 30.0).abs() < 0.5 && (k.yaw_deg - 20.0).abs() < 0.5 && (k.roll_deg - 3.0).abs() < 0.5);
        assert!(c.rms_m < 0.05);
    }
}

