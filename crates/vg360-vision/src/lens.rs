//! Lens models, replacing `cv2.undistort` / `initUndistortRectifyMap` and the `cv2.fisheye`
//! functions the notebook used.
//!
//! * [`Pinhole`] with Brown–Conrady distortion (k1, k2, p1, p2, k3) — OpenCV's default model.
//! * [`Fisheye`] — Kannala–Brandt θ_d = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸), identical to `cv2.fisheye`.
//! * [`virtual_view`] — an upright pinhole view rendered from a fisheye frame (detector input).
use crate::img::Bgr;

#[derive(Clone, Copy, Debug)]
pub struct Pinhole { pub fx: f64, pub fy: f64, pub cx: f64, pub cy: f64, pub k1: f64, pub k2: f64, pub p1: f64, pub p2: f64, pub k3: f64 }

impl Pinhole {
    /// Normalised undistorted (x, y) → distorted pixel.
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2;
        let xd = x * rad + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let yd = y * rad + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (self.fx * xd + self.cx, self.fy * yd + self.cy)
    }

    /// Distorted pixel → normalised undistorted (x, y) (fixed-point iteration, as `cv::undistortPoints`).
    pub fn undistort_point(&self, u: f64, v: f64) -> (f64, f64) {
        let (xd, yd) = ((u - self.cx) / self.fx, (v - self.cy) / self.fy);
        let (mut x, mut y) = (xd, yd);
        for _ in 0..20 {
            let r2 = x * x + y * y;
            let rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2;
            let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
            let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
            x = (xd - dx) / rad;
            y = (yd - dy) / rad;
        }
        (x, y)
    }

    /// Undistorted image with the same intrinsics (`cv2.undistort(img, K, dist)`).
    pub fn undistort(&self, img: &Bgr) -> Bgr {
        img.remap(img.w, img.h, [0; 3], |u, v| Some(self.distort((u - self.cx) / self.fx, (v - self.cy) / self.fy)))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Fisheye { pub fx: f64, pub fy: f64, pub cx: f64, pub cy: f64, pub k: [f64; 4] }

impl Fisheye {
    fn theta_d(&self, t: f64) -> f64 {
        let t2 = t * t;
        t * (1.0 + t2 * (self.k[0] + t2 * (self.k[1] + t2 * (self.k[2] + t2 * self.k[3]))))
    }

    /// Camera-frame ray (x right, y down, z forward) → pixel. Works beyond 90° off-axis.
    pub fn project(&self, d: [f64; 3]) -> Option<(f64, f64)> {
        let r = d[0].hypot(d[1]);
        let theta = r.atan2(d[2]);
        if r < 1e-12 { return Some((self.cx, self.cy)); }
        let td = self.theta_d(theta);
        Some((self.fx * td * d[0] / r + self.cx, self.fy * td * d[1] / r + self.cy))
    }

    /// Pixel → unit camera-frame ray (Newton on θ_d(θ) = r_d, as `cv2.fisheye.undistortPoints`).
    pub fn unproject(&self, u: f64, v: f64) -> [f64; 3] {
        let (xd, yd) = ((u - self.cx) / self.fx, (v - self.cy) / self.fy);
        let rd = xd.hypot(yd);
        if rd < 1e-12 { return [0.0, 0.0, 1.0]; }
        let mut t = rd;
        for _ in 0..20 {
            let t2 = t * t;
            let f = self.theta_d(t) - rd;
            let df = 1.0 + t2 * (3.0 * self.k[0] + t2 * (5.0 * self.k[1] + t2 * (7.0 * self.k[2] + t2 * 9.0 * self.k[3])));
            let step = f / df;
            t -= step;
            if step.abs() < 1e-12 { break; }
        }
        let s = t.sin();
        [s * xd / rd, s * yd / rd, t.cos()]
    }
}

/// Upright virtual pinhole view from a fisheye frame: look direction `yaw_deg` (about the
/// fisheye's y axis) and `pitch_deg` (about the view's x axis), horizontal FOV `hfov_deg`.
pub fn virtual_view(fe: &Fisheye, img: &Bgr, yaw_deg: f64, pitch_deg: f64, hfov_deg: f64, w: usize, h: usize) -> Bgr {
    let f = w as f64 / 2.0 / (hfov_deg.to_radians() / 2.0).tan();
    let (cy_, sy) = (yaw_deg.to_radians().cos(), yaw_deg.to_radians().sin());
    let (cp, sp) = (pitch_deg.to_radians().cos(), pitch_deg.to_radians().sin());
    img.remap(w, h, [0; 3], |u, v| {
        let d = [(u - w as f64 / 2.0) / f, (v - h as f64 / 2.0) / f, 1.0];
        let d = [d[0], cp * d[1] - sp * d[2], sp * d[1] + cp * d[2]]; // pitch
        let d = [cy_ * d[0] + sy * d[2], d[1], -sy * d[0] + cy_ * d[2]]; // yaw
        fe.project(d)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn brown_conrady_round_trip() {
        let p = Pinhole { fx: 1000.0, fy: 1000.0, cx: 960.0, cy: 540.0, k1: -0.28, k2: 0.08, p1: 0.001, p2: -0.0005, k3: 0.0 };
        for &(u, v) in &[(100.0, 80.0), (960.0, 540.0), (1800.0, 1000.0)] {
            let (x, y) = p.undistort_point(u, v);
            let (u2, v2) = p.distort(x, y);
            assert!((u - u2).abs() < 1e-3 && (v - v2).abs() < 1e-3, "{u},{v} -> {u2},{v2}");
        }
    }
    #[test]
    fn kannala_brandt_round_trip_to_95_degrees() {
        let fe = Fisheye { fx: 320.0, fy: 320.0, cx: 640.0, cy: 640.0, k: [0.02, -0.01, 0.002, -0.0003] };
        for deg in [0.0f64, 10.0, 45.0, 80.0, 95.0] {
            let t = deg.to_radians();
            let d = [t.sin() * 0.6, t.sin() * 0.8, t.cos()];
            let (u, v) = fe.project(d).unwrap();
            let r = fe.unproject(u, v);
            for i in 0..3 { assert!((r[i] - d[i]).abs() < 1e-9, "{deg}°: {r:?} vs {d:?}"); }
        }
    }
}
