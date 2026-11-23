//! Pinhole camera over a flat deck (PGA ray–plane meet in closed form).
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Camera {
    pub f_px: f64,
    pub cx: f64,
    pub cy: f64,
    pub height_m: f64,
    pub pitch_deg: f64,
    pub yaw_deg: f64,
    pub roll_deg: f64,
    pub x0: f64,
    pub y0: f64,
}

impl Camera {
    /// Rotation camera→world, columns = camera right, down, forward in world coordinates
    /// (world: x east, y north, z up). Same construction as the notebook's `Camera`:
    /// yaw is a compass bearing (clockwise from +y), pitch is the downward tilt, roll turns the image plane.
    pub fn rotation(&self) -> [[f64; 3]; 3] {
        let (th, ps, rl) = (self.pitch_deg.to_radians(), self.yaw_deg.to_radians(), self.roll_deg.to_radians());
        let fwd = [ps.sin() * th.cos(), ps.cos() * th.cos(), -th.sin()];
        let n = fwd[1].hypot(fwd[0]);
        let right = [fwd[1] / n, -fwd[0] / n, 0.0];
        let down = [fwd[1] * right[2] - fwd[2] * right[1], fwd[2] * right[0] - fwd[0] * right[2], fwd[0] * right[1] - fwd[1] * right[0]];
        let (c, s) = (rl.cos(), rl.sin());
        let r2: [f64; 3] = std::array::from_fn(|i| c * right[i] + s * down[i]);
        let d2: [f64; 3] = std::array::from_fn(|i| -s * right[i] + c * down[i]);
        std::array::from_fn(|i| [r2[i], d2[i], fwd[i]])
    }
    fn r(&self) -> [[f64; 3]; 3] { self.rotation() }

    pub fn centre(&self) -> [f64; 3] { [self.x0, self.y0, self.height_m] }

    /// Deck/world point → pixel without the in-front check (the notebook's `project`).
    pub fn project3_raw(&self, p: [f64; 3]) -> [f64; 2] {
        let r = self.r();
        let d = [p[0] - self.x0, p[1] - self.y0, p[2] - self.height_m];
        let c: [f64; 3] = std::array::from_fn(|i| r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2]);
        [self.f_px * c[0] / c[2] + self.cx, self.f_px * c[1] / c[2] + self.cy]
    }

    /// Pixel → world ray direction (not normalised).
    pub fn ray(&self, u: f64, v: f64) -> [f64; 3] {
        let d = [(u - self.cx) / self.f_px, (v - self.cy) / self.f_px, 1.0];
        let r = self.r();
        [0, 1, 2].map(|i| r[i][0] * d[0] + r[i][1] * d[1] + r[i][2] * d[2])
    }

    /// Pixel → deck point (metres). `None` if the ray points at or above the horizon.
    pub fn ground(&self, u: f64, v: f64) -> Option<[f64; 2]> {
        let d = [(u - self.cx) / self.f_px, (v - self.cy) / self.f_px, 1.0];
        let r = self.r();
        let w = [0, 1, 2].map(|i| r[i][0] * d[0] + r[i][1] * d[1] + r[i][2] * d[2]);
        if w[2] >= -1e-9 { return None; }
        let t = self.height_m / -w[2];
        Some([self.x0 + t * w[0], self.y0 + t * w[1]])
    }

    /// Deck point → pixel.
    pub fn project(&self, x: f64, y: f64) -> Option<[f64; 2]> {
        let r = self.r();
        let p = [x - self.x0, y - self.y0, -self.height_m];
        let c = [0, 1, 2].map(|i| r[0][i] * p[0] + r[1][i] * p[1] + r[2][i] * p[2]);
        if c[2] <= 1e-9 { return None; }
        Some([self.cx + self.f_px * c[0] / c[2], self.cy + self.f_px * c[1] / c[2]])
    }

    /// Compose a measured image shift (camera-guard homography, reference→current pixels)
    /// with the ground mapping: a current pixel is first mapped back into the reference frame.
    pub fn ground_through(&self, h_cur_to_ref: &[[f64; 3]; 3], u: f64, v: f64) -> Option<[f64; 2]> {
        let h = h_cur_to_ref;
        let w = h[2][0] * u + h[2][1] * v + h[2][2];
        self.ground((h[0][0] * u + h[0][1] * v + h[0][2]) / w, (h[1][0] * u + h[1][1] * v + h[1][2]) / w)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    fn cam() -> Camera {
        Camera { f_px: 1098.0, cx: 960.0, cy: 540.0, height_m: 9.0, pitch_deg: 30.0, yaw_deg: 20.0, roll_deg: 3.0, x0: 0.0, y0: 0.0 }
    }
    #[test]
    fn round_trip() {
        let c = cam();
        for &(x, y) in &[(2.0, 10.0), (-5.0, 25.0), (8.0, 14.0)] {
            let p = c.project(x, y).unwrap();
            let g = c.ground(p[0], p[1]).unwrap();
            assert!((g[0] - x).abs() < 1e-9 && (g[1] - y).abs() < 1e-9, "{g:?}");
        }
    }
    #[test]
    fn principal_ray_hits_deck_at_h_over_tan_pitch() {
        let c = Camera { yaw_deg: 0.0, roll_deg: 0.0, ..cam() };
        let g = c.ground(c.cx, c.cy).unwrap();
        assert!((g[1] - 9.0 / 30f64.to_radians().tan()).abs() < 1e-9 && g[0].abs() < 1e-9);
    }
    #[test]
    fn above_horizon_is_none() {
        assert!(cam().ground(960.0, -2000.0).is_none());
    }
}
