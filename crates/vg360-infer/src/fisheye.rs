//! Fisheye cameras: render upright virtual pinhole views (zero roll, so people stand upright),
//! detect in each view, map each person's foot point to the deck through that view's ordinary
//! camera model, and merge duplicates seen by two views in metres. Mirrors notebook §9b.
use vg360_geom::camera::Camera;
use vg360_vision::{img::Gray, lens::Fisheye};

use crate::engine::Result;
use crate::yolox::{Det, Yolox};

/// A fisheye camera in the world: lens model, camera→world rotation (columns = camera x right,
/// y down, z forward, in world coordinates) and centre (metres, z up).
#[derive(Clone, Debug)]
pub struct FisheyeCam { pub lens: Fisheye, pub r_cw: [[f64; 3]; 3], pub c: [f64; 3] }

pub struct View { pub cam: Camera, pub yaw: f64, pub pitch: f64, /// per view pixel: fisheye (u, v)
    pub map: Vec<(f32, f32)>, pub w: usize, pub h: usize }

#[derive(Clone, Debug)]
pub struct DeckDet { pub det: Det, pub view: usize, pub deck: [f64; 2] }

impl FisheyeCam {
    /// Fisheye with the pose of a pinhole camera (its rotation and centre). A camera looking straight down
    /// (pitch 90°) is built at 89.9° so that its heading stays defined.
    pub fn from_pose(lens: Fisheye, pose: &Camera) -> Self {
        let p = Camera { pitch_deg: pose.pitch_deg.min(89.9), ..pose.clone() };
        Self { lens, r_cw: p.rotation(), c: p.centre() }
    }

    /// Fisheye pixel → world direction.
    pub fn px_to_world_dir(&self, u: f64, v: f64) -> [f64; 3] {
        let d = self.lens.unproject(u, v);
        let r = &self.r_cw;
        [0, 1, 2].map(|i| r[i][0] * d[0] + r[i][1] * d[1] + r[i][2] * d[2])
    }

    /// What this fisheye would see of a scene filmed by the pinhole camera `pin` from the same centre
    /// (resample a pinhole clip into a fisheye one; the notebook's `fisheye_from_pinhole_maps`).
    pub fn render_from_pinhole(&self, pin: &Camera, src: &vg360_vision::img::Bgr, w: usize, h: usize, fill: [u8; 3]) -> vg360_vision::img::Bgr {
        let r = pin.rotation();
        src.remap(w, h, fill, |u, v| {
            let d = self.px_to_world_dir(u, v);
            let c = [0, 1, 2].map(|k| r[0][k] * d[0] + r[1][k] * d[1] + r[2][k] * d[2]);
            (c[2] > 1e-3).then(|| (pin.f_px * c[0] / c[2] + pin.cx, pin.f_px * c[1] / c[2] + pin.cy))
        })
    }

    /// World direction → fisheye pixel.
    pub fn world_dir_to_px(&self, d: [f64; 3]) -> Option<(f64, f64)> {
        let r = &self.r_cw;
        let dc = [0, 1, 2].map(|k| r[0][k] * d[0] + r[1][k] * d[1] + r[2][k] * d[2]); // R_cwᵀ d
        self.lens.project(dc)
    }

    /// Upright virtual views, one per (yaw°, pitch°) in world terms.
    pub fn views(&self, specs: &[(f64, f64)], w: usize, h: usize, hfov_deg: f64) -> Vec<View> {
        let f = w as f64 / 2.0 / (hfov_deg.to_radians() / 2.0).tan();
        specs.iter().map(|&(yaw, pitch)| {
            let cam = Camera { f_px: f, cx: w as f64 / 2.0, cy: h as f64 / 2.0, height_m: self.c[2], pitch_deg: pitch,
                               yaw_deg: yaw, roll_deg: 0.0, x0: self.c[0], y0: self.c[1] };
            let mut map = Vec::with_capacity(w * h);
            for v in 0..h { for u in 0..w {
                map.push(self.world_dir_to_px(cam.ray(u as f64, v as f64)).map_or((-1.0, -1.0), |(a, b)| (a as f32, b as f32)));
            } }
            View { cam, yaw, pitch, map, w, h }
        }).collect()
    }
}

impl View {
    /// Bilinear remap of the fisheye frame into this view (black outside the image circle).
    pub fn render(&self, fish: &Gray) -> Gray {
        let mut out = Gray::new(self.w, self.h);
        for (i, &(x, y)) in self.map.iter().enumerate() {
            if !(x >= 0.0 && y >= 0.0 && x <= (fish.w - 1) as f32 && y <= (fish.h - 1) as f32) { continue; }
            let (x0, y0) = (x.floor() as usize, y.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(fish.w - 1), (y0 + 1).min(fish.h - 1));
            let (ax, ay) = (x - x0 as f32, y - y0 as f32);
            let p = |a: usize, b: usize| fish.at(a, b) as f32;
            let v = (p(x0, y0) * (1.0 - ax) + p(x1, y0) * ax) * (1.0 - ay) + (p(x0, y1) * (1.0 - ax) + p(x1, y1) * ax) * ay;
            out.data[i] = v.round() as u8;
        }
        out
    }
}

/// Detect people in every view, place their feet on the deck, merge cross-view duplicates
/// closer than `merge_m` metres (keep the higher score).
pub fn detect_fisheye(det: &Yolox, fish: &Gray, views: &[View], tiles: usize, merge_m: f64) -> Result<Vec<DeckDet>> {
    let mut all = Vec::new();
    for (vi, v) in views.iter().enumerate() {
        let img = v.render(fish);
        for d in det.detect_tiled(&img, tiles, 0.25)? {
            if let Some(g) = v.cam.ground((d.x1 + d.x2) / 2.0, d.y2) { all.push(DeckDet { det: d, view: vi, deck: g }); }
        }
    }
    all.sort_by(|a, b| b.det.score.total_cmp(&a.det.score));
    let mut out: Vec<DeckDet> = Vec::new();
    for d in all {
        if !out.iter().any(|k| (k.deck[0] - d.deck[0]).hypot(k.deck[1] - d.deck[1]) < merge_m) { out.push(d); }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_pixel_and_fisheye_pixel_see_the_same_deck_point() {
        // fisheye looking straight down from 8 m: camera z = world −z, camera x = world x, camera y = world −y
        let fc = FisheyeCam {
            lens: Fisheye { fx: 300.0, fy: 300.0, cx: 640.0, cy: 640.0, k: [0.01, -0.005, 0.001, 0.0] },
            r_cw: [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            c: [0.0, 0.0, 8.0],
        };
        let views = fc.views(&[(0.0, 35.0), (120.0, 35.0), (240.0, 35.0)], 320, 180, 60.0);
        for v in &views {
            for &(u, w) in &[(160.0, 120.0), (40.0, 170.0), (300.0, 100.0)] {
                let g = v.cam.ground(u, w).unwrap();
                let d = [g[0] - fc.c[0], g[1] - fc.c[1], -fc.c[2]];
                let (fu, fv) = fc.world_dir_to_px(d).unwrap();
                let (mu, mv) = v.map[w as usize * v.w + u as usize];
                assert!((fu - mu as f64).abs() < 1e-2 && (fv - mv as f64).abs() < 1e-2, "{fu},{fv} vs {mu},{mv}");
            }
        }
    }
}
