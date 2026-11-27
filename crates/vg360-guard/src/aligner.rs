//! Position measurement, ported line-for-line in logic from camera_guard's `aligner.rs`.
//! OpenCV calls are replaced by `vg360-vision`: SIFT+FLANN → ORB-style features + exact
//! Hamming ratio matching; MAGSAC++ → MSAC with local optimisation; CLAHE / INTER_AREA as before.
use vg360_vision::features::{Desc, Detector, Key, ratio_match};
use vg360_vision::img::{Bgr, Gray};
use vg360_vision::robust::{self, Model};

use crate::config::*;
use crate::policy::Pose;

pub type M3 = [[f64; 3]; 3];
pub type Quad = [[f64; 2]; 4];

#[derive(Clone, Debug)]
pub struct Measure {
    pub ref_gen: u64,
    pub blocked: bool,
    pub ok: bool,
    pub inliers: i32,
    pub ref_points: i32,
    /// reference frame corners in current full-res pixels
    pub boxq: Quad,
    /// reference → current, analysis-resolution pixels
    pub h: M3,
    pub pose: Pose,
}

impl Measure {
    fn empty(ref_gen: u64, blocked: bool) -> Self {
        Self { ref_gen, blocked, ok: false, inliers: 0, ref_points: 0, boxq: [[0.0; 2]; 4], h: IDENTITY,
               pose: Pose { dx: 0.0, dy: 0.0, angle: 0.0, scale: 1.0, corner: 0.0 } }
    }
}

/// The still-point override only applies when still points are at least this share of ALL ratio
/// matches: if the background really moved, most matches are not still (fixed overlay text, housing
/// edge or dust give a few still points but cannot be the majority).
const STILL_FRAC: f64 = 0.5;
/// ...and a still-point set that is only a sliver of the reference (overlay text, dust) is not a
/// still background either: need at least this share of the reference points.
const STILL_REF_FRAC: f64 = 0.15;

/// What the reference looked like when it was captured (for an operator "is this a usable reference?" check).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefQuality {
    /// background keypoints kept
    pub points: usize,
    /// mean / std of the reference gray image (0..255, analysis resolution)
    pub mean: f64,
    pub std: f64,
    /// analysis-resolution size (w, h) the reference was captured at; `measure` needs the same size
    pub size: (usize, usize),
}

type Fit = (M3, i32, Quad, f64);

/// Model selection: keep the fit with more inliers; the homography wins ties.
pub fn pick(h: Option<Fit>, s: Option<Fit>) -> Option<Fit> {
    match (h, s) {
        (Some(h), Some(s)) => Some(if s.1 > h.1 { s } else { h }),
        (h, None) => h,
        (None, s) => s,
    }
}

const IDENTITY: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn project(h: &M3, x: f64, y: f64) -> [f64; 2] { robust::project(h, [x, y]) }

pub fn quad_area(q: &Quad) -> f64 {
    let mut a = 0.0;
    for i in 0..4 {
        let j = (i + 1) % 4;
        a += q[i][0] * q[j][1] - q[j][0] * q[i][1];
    }
    a.abs() / 2.0
}

fn quad_convex(q: &Quad) -> bool {
    let mut sign = 0.0;
    for i in 0..4 {
        let (a, b, c) = (q[i], q[(i + 1) % 4], q[(i + 2) % 4]);
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cross.abs() < 1e-9 { return false; }
        if sign == 0.0 { sign = cross.signum(); } else if cross.signum() != sign { return false; }
    }
    true
}

pub struct Aligner {
    det: Detector,
    kp: Vec<Key>,
    des: Vec<Desc>,
    s: f64,
    size: (f64, f64),
    pub ref_gen: u64,
    quality: RefQuality,
}

impl Default for Aligner { fn default() -> Self { Self::new() } }

impl Aligner {
    pub fn new() -> Self {
        Self { det: Detector::default(), kp: vec![], des: vec![], s: 1.0, size: (0.0, 0.0), ref_gen: 0,
               quality: RefQuality { points: 0, mean: 0.0, std: 0.0, size: (0, 0) } }
    }

    /// Returns (raw gray, CLAHE gray, scale).
    fn prep(&self, img: &Bgr) -> (Gray, Gray, f64) {
        let s = ANALYSIS_W as f64 / img.w as f64;
        let h = (img.h as f64 * s).round() as usize;
        let gray = img.to_gray().resize_area(ANALYSIS_W as usize, h);
        let eq = gray.clahe(2.5, 8);
        (gray, eq, s)
    }

    /// Corner areas where cameras burn in timestamp / logo: never use them — they don't move
    /// with the camera, so they would falsely say "camera still in place".
    fn outside_overlay(w: f64, h: f64) -> impl Fn(f32, f32) -> bool {
        let (ow, oh) = (w * OVERLAY_W, h * OVERLAY_H);
        move |x, y| {
            let (x, y) = (x as f64, y as f64);
            !((x < ow || x > w - ow) && (y < oh || y > h - oh))
        }
    }

    /// Keep only background points: keypoints that stay put across `extra` frames
    /// (people / moving objects are dropped). `static_mask(x, y)` restores a saved selection.
    pub fn set_reference(&mut self, img: &Bgr, extra: &[Bgr], static_mask: Option<&dyn Fn(f32, f32) -> bool>) -> usize {
        let (raw, g, s) = self.prep(img);
        let (ref_mean, ref_std) = raw.mean_std();
        let ov = Self::outside_overlay(g.w as f64, g.h as f64);
        let (mut kp, mut des) = self.det.detect(&g, |x, y| ov(x, y) && static_mask.is_none_or(|m| m(x / s as f32, y / s as f32)));
        if !extra.is_empty() && !des.is_empty() {
            let mut hits = vec![0usize; kp.len()];
            for f in extra {
                let (_, gf, _) = self.prep(f);
                let (kf, df) = self.det.detect(&gf, &ov);
                for m in ratio_match(&des, &df, RATIO) {
                    let (a, b) = (kp[m.query], kf[m.train]);
                    if (a.x - b.x).abs() < 2.5 && (a.y - b.y).abs() < 2.5 { hits[m.query] += 1; }
                }
            }
            let need = ((0.6 * extra.len() as f64) as usize).max(1);
            let keep: Vec<usize> = (0..kp.len()).filter(|&i| hits[i] >= need).collect();
            if keep.len() >= 80 {
                kp = keep.iter().map(|&i| kp[i]).collect();
                des = keep.iter().map(|&i| des[i]).collect();
            }
        }
        self.size = (g.w as f64, g.h as f64);
        self.s = s;
        self.kp = kp;
        self.des = des;
        self.ref_gen += 1;
        self.quality = RefQuality { points: self.kp.len(), mean: ref_mean, std: ref_std, size: (g.w, g.h) };
        self.kp.len()
    }

    /// Background points in full-resolution pixels (saved with the reference, replaces the mask PNG).
    pub fn static_points(&self) -> Vec<[f32; 2]> {
        self.kp.iter().map(|k| [k.x / self.s as f32, k.y / self.s as f32]).collect()
    }

    /// Quality of the current reference (points kept, brightness, contrast, analysis size).
    pub fn reference_quality(&self) -> RefQuality { self.quality }

    /// Measure the camera position against the reference. Returns `ok == false` (and `blocked == false`)
    /// when the live frame's analysis size differs from the reference's (a different stream
    /// resolution cannot be compared point by point), and never returns `ok` with non-finite values.
    pub fn measure(&self, img: &Bgr) -> Measure {
        let (raw, g, s) = self.prep(img);
        if self.quality.size != (0, 0) && (g.w, g.h) != self.quality.size { return Measure::empty(self.ref_gen, false); }
        let (mean, std) = raw.mean_std();
        let mut res = Measure::empty(self.ref_gen, std < 10.0 || mean < 15.0);
        if self.des.is_empty() { return res; }
        let ov = Self::outside_overlay(g.w as f64, g.h as f64);
        let (kp, des) = self.det.detect(&g, &ov);
        let good = ratio_match(&self.des, &des, RATIO);
        let mut src = Vec::with_capacity(good.len());
        let mut dst = Vec::with_capacity(good.len());
        let mut still = 0i32; // background points still at exactly the same pixel
        for m in &good {
            let (a, b) = (self.kp[m.query], kp[m.train]);
            if (a.x - b.x).abs() < STILL_TOL && (a.y - b.y).abs() < STILL_TOL { still += 1; }
            src.push([a.x as f64, a.y as f64]);
            dst.push([b.x as f64, b.y as f64]);
        }
        let (w, h) = self.size;
        let ref_points = self.kp.len() as i32;
        let identity = |res: &mut Measure, n: i32| {
            res.ok = true;
            res.inliers = n;
            res.ref_points = ref_points;
            res.boxq = [[0.0, 0.0], [w / s, 0.0], [w / s, h / s], [0.0, h / s]];
            res.h = IDENTITY;
            res.pose = Pose { dx: 0.0, dy: 0.0, angle: 0.0, scale: 1.0, corner: 0.0 };
        };
        if (good.len() as i32) < MIN_INLIERS {
            if still >= STILL_MIN { identity(&mut res, still); }
            return res;
        }
        let corners: Quad = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
        let valid = |hm: &M3, inl: i32| -> Option<(Quad, f64)> {
            if inl < MIN_INLIERS { return None; }
            let wc: Quad = std::array::from_fn(|i| project(hm, corners[i][0], corners[i][1]));
            if !wc.iter().flatten().all(|v| v.is_finite()) || !quad_convex(&wc) { return None; } // degenerate fit
            let sc = (quad_area(&wc) / (w * h)).sqrt();
            if !(0.4..2.5).contains(&sc) || hm[2][0].abs() * w > 0.6 || hm[2][1].abs() * h > 0.6 {
                return None; // absurd zoom / perspective
            }
            Some((wc, sc))
        };

        // 1) homography ref → current; people / moving objects become outliers
        let mut hfit: Option<Fit> = None;
        if let Some(f) = robust::fit(Model::Homography, &src, &dst, H_THRESH, 10_000, 0.999) {
            let inl = f.n_inliers as i32;
            if let Some((wc, sc)) = valid(&f.model, inl) { hfit = Some((f.model, inl, wc, sc)); }
        }
        // 2) few points → the 8-unknown homography is jumpy; also try shift + rotation + zoom (4 unknowns)
        //    and keep whichever model has more inliers (homography on ties)
        let mut sfit: Option<Fit> = None;
        if hfit.as_ref().is_none_or(|b| b.1 < CONFIRM_INLIERS) {
            if let Some(f) = robust::fit(Model::Similarity, &src, &dst, 2.0 * H_THRESH, 5_000, 0.999) {
                let inl = f.n_inliers as i32;
                if let Some((wc, sc)) = valid(&f.model, inl) { sfit = Some((f.model, inl, wc, sc)); }
            }
        }
        let best = pick(hfit, sfit);
        // People moving never shift the background: if enough background points are still exactly
        // where they were, the camera has NOT moved, whatever the fit says.
        let fit_inl = best.as_ref().map_or(0, |b| b.1);
        if still >= STILL_MIN && still as f64 >= STILL_SHARE * fit_inl as f64 && still as f64 >= STILL_FRAC * good.len() as f64
            && still as f64 >= STILL_REF_FRAC * ref_points as f64 {
            identity(&mut res, still.max(fit_inl));
            return res;
        }
        let Some((hm, inl, wc, scale)) = best else { return res };

        let (cx, cy) = (w / 2.0, h / 2.0);
        let p = project(&hm, cx, cy);
        let q = project(&hm, cx + 100.0, cy);
        // mean corner displacement as a fraction of width (same as camera_guard)
        let corner = (0..4)
            .map(|i| ((wc[i][0] - corners[i][0]).powi(2) + (wc[i][1] - corners[i][1]).powi(2)).sqrt())
            .sum::<f64>() / 4.0 / w;
        let pose = Pose {
            dx: (p[0] - cx) / w,
            dy: (p[1] - cy) / w,
            angle: (q[1] - p[1]).atan2(q[0] - p[0]).to_degrees(),
            scale,
            corner,
        };
        let boxq = wc.map(|c| [c[0] / s, c[1] / s]);
        let finite = [pose.dx, pose.dy, pose.angle, pose.scale, pose.corner].iter().all(|v| v.is_finite())
            && boxq.iter().flatten().all(|v| v.is_finite())
            && hm.iter().flatten().all(|v| v.is_finite());
        if !finite { return res; } // never report a position fix made of NaN / inf
        res.ok = true;
        res.inliers = inl;
        res.ref_points = ref_points;
        res.boxq = boxq;
        res.h = hm;
        res.pose = pose;
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fit(inl: i32, tag: f64) -> Fit { (IDENTITY, inl, [[0.0; 2]; 4], tag) }

    #[test]
    fn pick_prefers_more_inliers_and_homography_on_ties() {
        assert!(pick(None, None).is_none());
        assert_eq!(pick(Some(fit(10, 1.0)), None).unwrap().3, 1.0);
        assert_eq!(pick(None, Some(fit(10, 2.0))).unwrap().3, 2.0);
        assert_eq!(pick(Some(fit(30, 1.0)), Some(fit(40, 2.0))).unwrap().3, 2.0); // similarity has more
        assert_eq!(pick(Some(fit(50, 1.0)), Some(fit(40, 2.0))).unwrap().3, 1.0); // homography has more
        assert_eq!(pick(Some(fit(40, 1.0)), Some(fit(40, 2.0))).unwrap().3, 1.0); // tie -> homography
    }

    fn hash(a: i64, b: i64, seed: u64) -> u8 {
        let mut x = (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ seed;
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 32;
        (x & 0xFF) as u8
    }

    /// Deterministic block texture; `shift` moves the whole scene right; `overlay` pastes a fixed
    /// high-texture rectangle (burned-in text / housing edge) at the same place in every frame.
    fn scene(w: usize, h: usize, shift: i64, overlay: bool) -> Bgr {
        let mut im = Bgr::new(w, h);
        let (ox0, oy0, ox1, oy1) = (w * 4 / 10, h * 35 / 100, w * 6 / 10, h * 65 / 100);
        for y in 0..h {
            for x in 0..w {
                let v = if overlay && (ox0..ox1).contains(&x) && (oy0..oy1).contains(&y) {
                    hash((x / 5) as i64, (y / 5) as i64, 7)
                } else {
                    hash((x as i64 - shift).div_euclid(6), (y / 6) as i64, 1)
                };
                let i = (y * w + x) * 3;
                im.data[i] = v;
                im.data[i + 1] = v;
                im.data[i + 2] = v;
            }
        }
        im
    }

    #[test]
    fn fixed_overlay_does_not_hide_a_large_knock() {
        let mut a = Aligner::new();
        let n = a.set_reference(&scene(1280, 720, 0, true), &[], None);
        assert!(n >= 80, "{n} reference points");
        let m = a.measure(&scene(1280, 720, 102, true)); // 8 % of the width
        eprintln!("ok {} inl {} pose {:?}", m.ok, m.inliers, m.pose);
        let identity = m.ok && m.h == IDENTITY && m.pose.dx == 0.0 && m.pose.dy == 0.0;
        assert!(!identity, "a knock hidden by static overlay points");
        let m0 = a.measure(&scene(1280, 720, 0, true));
        assert!(m0.ok && m0.pose.dx.abs() < 1e-6, "{:?}", m0.pose);
    }

    /// Background replaced by an unrelated texture (camera knocked far away): only the fixed overlay
    /// still matches in place. The old rule (still >= 20 and no fit) declared "position correct".
    #[test]
    fn overlay_alone_does_not_hide_a_lost_background() {
        let mut a = Aligner::new();
        a.set_reference(&scene(1280, 720, 0, true), &[], None);
        let mut live = scene(1280, 720, 0, true);
        let (w, h) = (1280usize, 720usize);
        let (ox0, oy0, ox1, oy1) = (w * 4 / 10, h * 35 / 100, w * 6 / 10, h * 65 / 100);
        for y in 0..h {
            for x in 0..w {
                if (ox0..ox1).contains(&x) && (oy0..oy1).contains(&y) { continue; }
                let v = hash((x / 6) as i64, (y / 6) as i64, 99);
                let i = (y * w + x) * 3;
                live.data[i] = v; live.data[i + 1] = v; live.data[i + 2] = v;
            }
        }
        let m = a.measure(&live);
        eprintln!("ok {} inl {} pose {:?}", m.ok, m.inliers, m.pose);
        assert!(!(m.ok && m.h == IDENTITY), "lost background reported as position correct");
    }

    #[test]
    fn size_mismatch_is_not_ok_and_not_blocked() {
        let mut a = Aligner::new();
        a.set_reference(&scene(1280, 720, 0, false), &[], None);
        let q = a.reference_quality();
        assert_eq!(q.size, (960, 540));
        assert!(q.points >= 80 && q.std > 10.0 && q.mean > 15.0, "{q:?}");
        let m = a.measure(&scene(640, 480, 0, false));
        assert!(!m.ok && !m.blocked);
    }
}
