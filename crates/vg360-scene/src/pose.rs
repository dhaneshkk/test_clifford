//! Single-view 3-D body pose by analysis-by-synthesis (notebook `fit_pose`): render 19 body
//! cuboids (3 standing heights, 2 tilts × 8 headings) through the camera, slide each on the deck
//! until its box bottom lands on the observed foot point, keep the best box IoU. The body tilt is
//! the angle of the Cl(3,0) rotor taking e₃ to the body axis (Lean: angle of R = (1 + b a)/|1 + b a|).
use std::sync::OnceLock;

use vg360_geom::camera::Camera;

pub const BODY_LAT: f64 = 0.25;
pub const BODY_SAG: f64 = 0.17;

/// (tilt rad, heading rad, height m)
pub fn hypotheses() -> &'static [(f64, f64, f64)] {
    static H: OnceLock<Vec<(f64, f64, f64)>> = OnceLock::new();
    H.get_or_init(|| {
        let mut v: Vec<(f64, f64, f64)> = [1.6, 1.75, 1.9].iter().map(|&h| (0.0, 0.0, h)).collect();
        for t in [45.0f64, 90.0] { for y in (0..360).step_by(45) { v.push((t.to_radians(), (y as f64).to_radians(), 1.75)); } }
        v
    })
}

/// Feet at `origin`; body axis tilted by `tilt` toward heading `yaw`. Corner order (sx, sy, z) as the notebook.
pub fn body_cuboid(origin: [f64; 2], h: f64, tilt: f64, yaw: f64) -> [[f64; 3]; 8] {
    let (ct, st, cy, sy) = (tilt.cos(), tilt.sin(), yaw.cos(), yaw.sin());
    let mut out = [[0.0; 3]; 8];
    let mut i = 0;
    for sx in [-1.0, 1.0] { for sgy in [-1.0, 1.0] { for z in [0.0, h] {
        let (x, y) = (sx * BODY_LAT, sgy * BODY_SAG);
        let (y1, z1) = (ct * y + st * z, -st * y + ct * z);
        let (x2, y2) = (cy * x - sy * y1, sy * x + cy * y1);
        out[i] = [x2 + origin[0], y2 + origin[1], z1.max(0.0)];
        i += 1;
    } } }
    out
}

fn local_cuboids() -> &'static [[[f64; 3]; 8]] {
    static L: OnceLock<Vec<[[f64; 3]; 8]>> = OnceLock::new();
    L.get_or_init(|| hypotheses().iter().map(|&(t, y, h)| body_cuboid([0.0, 0.0], h, t, y)).collect())
}

/// Rotor angle between two directions (degrees): 2·acos(⟨R⟩₀/|R|), R = 1 + b·a.
pub fn rotor_angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    let c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
    let nr = (2.0 + 2.0 * c).max(0.0).sqrt();
    if nr > 1e-9 { (2.0 * ((1.0 + c) / nr).min(1.0).acos()).to_degrees() } else { 180.0 }
}

#[derive(Clone, Debug)]
pub struct Pose { pub iou: f64, pub origin: [f64; 2], pub tilt_deg: f64, pub top_z: f64, pub height: f64 }

pub fn fit_pose(cam: &Camera, bx: [f64; 4], foot: [f64; 2]) -> Pose {
    let local = local_cuboids();
    let k = local.len();
    let r = cam.rotation();
    let c = cam.centre();
    let render = |o: &[[f64; 2]]| -> Vec<[f64; 4]> {
        (0..k).map(|i| {
            let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
            for p in &local[i] {
                let uv = cam.project3_raw([p[0] + o[i][0], p[1] + o[i][1], p[2]]);
                b[0] = b[0].min(uv[0]); b[1] = b[1].min(uv[1]); b[2] = b[2].max(uv[0]); b[3] = b[3].max(uv[1]);
            }
            b
        }).collect()
    };
    let mut o = vec![foot; k];
    for _ in 0..3 {
        let bb = render(&o);
        for i in 0..k {
            let (u, v) = ((bb[i][0] + bb[i][2]) / 2.0, bb[i][3]);
            let q = [(u - cam.cx) / cam.f_px, (v - cam.cy) / cam.f_px, 1.0];
            let d: [f64; 3] = std::array::from_fn(|j| r[j][0] * q[0] + r[j][1] * q[1] + r[j][2] * q[2]);
            if d[2] < -1e-9 {
                let s = -c[2] / d[2];
                let g = [c[0] + s * d[0], c[1] + s * d[1]];
                o[i] = [o[i][0] + foot[0] - g[0], o[i][1] + foot[1] - g[1]];
            }
        }
    }
    let bb = render(&o);
    let (mut best, mut kbest) = (f64::NEG_INFINITY, 0);
    for i in 0..k {
        let ix = (bb[i][2].min(bx[2]) - bb[i][0].max(bx[0])).max(0.0);
        let iy = (bb[i][3].min(bx[3]) - bb[i][1].max(bx[1])).max(0.0);
        let inter = ix * iy;
        let iou = inter / ((bb[i][2] - bb[i][0]) * (bb[i][3] - bb[i][1]) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - inter + 1e-9);
        if iou > best { best = iou; kbest = i; }
    }
    let cub: Vec<[f64; 3]> = local[kbest].iter().map(|p| [p[0] + o[kbest][0], p[1] + o[kbest][1], p[2]]).collect();
    let mean = |idx: [usize; 4]| -> [f64; 3] { std::array::from_fn(|j| idx.iter().map(|&i| cub[i][j]).sum::<f64>() / 4.0) };
    let (head, feet) = (mean([1, 3, 5, 7]), mean([0, 2, 4, 6]));
    let axis = [head[0] - feet[0], head[1] - feet[1], head[2] - feet[2]];
    Pose {
        iou: best,
        origin: o[kbest],
        tilt_deg: rotor_angle_deg([0.0, 0.0, 1.0], axis),
        top_z: cub.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max),
        height: hypotheses()[kbest].2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nineteen_hypotheses() { assert_eq!(hypotheses().len(), 19); }
    #[test]
    fn rotor_angle_matches_vector_angle() {
        assert!((rotor_angle_deg([0.0, 0.0, 1.0], [1.0, 0.0, 1.0]) - 45.0).abs() < 1e-9);
        assert!((rotor_angle_deg([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]) - 180.0).abs() < 1e-9);
    }
    #[test]
    fn standing_box_fits_standing_body() {
        let cam = Camera { f_px: 900.0, cx: 640.0, cy: 360.0, height_m: 6.0, pitch_deg: 35.0, yaw_deg: 0.0, roll_deg: 0.0, x0: 0.0, y0: 0.0 };
        let cub = body_cuboid([1.0, 10.0], 1.75, 0.0, 0.0);
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for p in cub { let uv = cam.project3_raw(p); b = [b[0].min(uv[0]), b[1].min(uv[1]), b[2].max(uv[0]), b[3].max(uv[1])]; }
        let foot = cam.ground((b[0] + b[2]) / 2.0, b[3]).unwrap(); // as the pipeline does: box bottom → deck
        let p = fit_pose(&cam, b, foot);
        assert!(p.tilt_deg < 1.0 && p.iou > 0.99 && (p.height - 1.75).abs() < 1e-9, "{p:?}");
    }
}
