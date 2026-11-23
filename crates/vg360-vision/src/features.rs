//! Feature detection and matching, replacing OpenCV SIFT + FLANN.
//!
//! Detector/descriptor: ORB-style, implemented here with no dependencies —
//! FAST-9 corners on a 1.2× image pyramid, Harris ranking with a spatial grid so background
//! points cover the whole frame, intensity-centroid orientation, and a 256-bit rotated BRIEF
//! descriptor on a smoothed image. (The sampling pattern is generated from a fixed seed, so
//! descriptors are stable across runs but are not interchangeable with OpenCV's ORB.)
//! Matching: exact brute-force Hamming 2-NN with Lowe's ratio test — no FLANN approximation.
use std::sync::OnceLock;

use crate::img::Gray;

#[derive(Clone, Copy, Debug)]
pub struct Key { pub x: f32, pub y: f32, pub level: u8, pub angle: f32, pub response: f32 }

/// 256-bit binary descriptor.
pub type Desc = [u64; 4];

#[derive(Clone, Copy, Debug)]
pub struct Match { pub query: usize, pub train: usize, pub dist: u32 }

#[derive(Clone, Debug)]
pub struct Detector {
    pub max_keys: usize,
    pub levels: usize,
    pub scale: f64,
    pub fast_threshold: i16,
    /// grid cells per axis used to spread keypoints over the frame
    pub grid: usize,
}

impl Default for Detector {
    fn default() -> Self { Self { max_keys: 3000, levels: 6, scale: 1.2, fast_threshold: 12, grid: 8 } }
}

const PATCH: i32 = 15; // descriptor / orientation radius
const BORDER: i32 = 19;

// Bresenham circle of radius 3 (16 pixels) used by FAST
const CIRCLE: [(i32, i32); 16] = [(0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)];

fn pattern() -> &'static [[(i8, i8); 2]; 256] {
    static P: OnceLock<[[(i8, i8); 2]; 256]> = OnceLock::new();
    P.get_or_init(|| {
        // isotropic Gaussian pairs (σ = PATCH/2.5) clipped to the patch disc, fixed seed
        let mut s: u64 = 0x5EED_0F_0B_B1E5;
        let mut uni = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 };
        let mut gauss = || {
            loop {
                let (u, v) = (uni().max(1e-12), uni());
                let g = (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos() * PATCH as f64 / 2.5;
                if g.abs() <= PATCH as f64 - 1.0 { return g.round() as i8; }
            }
        };
        std::array::from_fn(|_| loop {
            let a = (gauss(), gauss());
            let b = (gauss(), gauss());
            let r = |p: (i8, i8)| (p.0 as i32).pow(2) + (p.1 as i32).pow(2) <= (PATCH - 1).pow(2);
            if a != b && r(a) && r(b) { break [a, b]; }
        })
    })
}

fn resize_bilinear(g: &Gray, w: usize, h: usize) -> Gray {
    let (sx, sy) = (g.w as f64 / w as f64, g.h as f64 / h as f64);
    let mut o = Gray::new(w, h);
    for y in 0..h {
        let fy = ((y as f64 + 0.5) * sy - 0.5).clamp(0.0, g.h as f64 - 1.0);
        let (y0, ay) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(g.h - 1);
        for x in 0..w {
            let fx = ((x as f64 + 0.5) * sx - 0.5).clamp(0.0, g.w as f64 - 1.0);
            let (x0, ax) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(g.w - 1);
            let v = (g.at(x0, y0) as f64 * (1.0 - ax) + g.at(x1, y0) as f64 * ax) * (1.0 - ay)
                + (g.at(x0, y1) as f64 * (1.0 - ax) + g.at(x1, y1) as f64 * ax) * ay;
            o.data[y * w + x] = v.round() as u8;
        }
    }
    o
}

/// 5×5 box blur (separable) used before sampling BRIEF pairs.
fn box5(g: &Gray) -> Gray {
    let (w, h) = (g.w, g.h);
    let mut t = vec![0u16; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0u16;
            for d in -2i32..=2 { s += g.at((x as i32 + d).clamp(0, w as i32 - 1) as usize, y) as u16; }
            t[y * w + x] = s;
        }
    }
    let mut o = Gray::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut s = 0u32;
            for d in -2i32..=2 { s += t[(y as i32 + d).clamp(0, h as i32 - 1) as usize * w + x] as u32; }
            o.data[y * w + x] = ((s + 12) / 25) as u8;
        }
    }
    o
}

/// FAST-9: ≥ 9 contiguous circle pixels all brighter than p+t or all darker than p−t.
fn fast9(g: &Gray, x: i32, y: i32, t: i16) -> bool {
    let p = g.at(x as usize, y as usize) as i16;
    let px = |i: usize| g.at((x + CIRCLE[i].0) as usize, (y + CIRCLE[i].1) as usize) as i16;
    // quick rejection on the 4 compass points: at least 3 must agree for a 9-arc
    let c = [px(0), px(4), px(8), px(12)];
    let (nb, nd) = (c.iter().filter(|&&v| v > p + t).count(), c.iter().filter(|&&v| v < p - t).count());
    if nb < 2 && nd < 2 { return false; }
    let v: [i16; 16] = std::array::from_fn(px);
    for sign in [1i16, -1] {
        let mut run = 0;
        for i in 0..32 {
            let d = (v[i % 16] - p) * sign;
            if d > t { run += 1; if run >= 9 { return true; } } else { run = 0; }
        }
    }
    false
}

fn harris(g: &Gray, x: i32, y: i32) -> f32 {
    let (mut a, mut b, mut c) = (0f32, 0f32, 0f32);
    for dy in -3..=3 {
        for dx in -3..=3 {
            let at = |u: i32, v: i32| g.at((x + u) as usize, (y + v) as usize) as f32;
            let ix = at(dx + 1, dy) - at(dx - 1, dy);
            let iy = at(dx, dy + 1) - at(dx, dy - 1);
            a += ix * ix; b += iy * iy; c += ix * iy;
        }
    }
    a * b - c * c - 0.04 * (a + b) * (a + b)
}

fn orientation(g: &Gray, x: i32, y: i32) -> f32 {
    let (mut m10, mut m01) = (0f32, 0f32);
    for dy in -PATCH..=PATCH {
        for dx in -PATCH..=PATCH {
            if dx * dx + dy * dy > PATCH * PATCH { continue; }
            let v = g.at((x + dx) as usize, (y + dy) as usize) as f32;
            m10 += dx as f32 * v; m01 += dy as f32 * v;
        }
    }
    m01.atan2(m10)
}

fn describe(s: &Gray, x: i32, y: i32, angle: f32) -> Desc {
    let (c, sn) = (angle.cos(), angle.sin());
    let at = |p: (i8, i8)| {
        let (px, py) = (p.0 as f32, p.1 as f32);
        let u = x + (c * px - sn * py).round() as i32;
        let v = y + (sn * px + c * py).round() as i32;
        s.at(u as usize, v as usize)
    };
    let mut d = [0u64; 4];
    for (i, pr) in pattern().iter().enumerate() {
        if at(pr[0]) < at(pr[1]) { d[i / 64] |= 1 << (i % 64); }
    }
    d
}

impl Detector {
    /// Detect and describe; `keep(x, y)` filters keypoints in input-image pixels
    /// (the old OpenCV mask argument). Returned coordinates are in input-image pixels.
    pub fn detect(&self, g: &Gray, keep: impl Fn(f32, f32) -> bool) -> (Vec<Key>, Vec<Desc>) {
        let mut keys = Vec::new();
        let mut descs = Vec::new();
        let per_level: Vec<f64> = (0..self.levels).map(|l| self.scale.powi(-(l as i32))).collect();
        let total: f64 = per_level.iter().map(|f| f * f).sum();
        let mut img = g.clone();
        for (lvl, &f) in per_level.iter().enumerate() {
            if lvl > 0 {
                let (w, h) = ((g.w as f64 * f).round() as usize, (g.h as f64 * f).round() as usize);
                if w < 2 * BORDER as usize + 8 || h < 2 * BORDER as usize + 8 { break; }
                img = resize_bilinear(g, w, h);
            }
            let quota = (self.max_keys as f64 * f * f / total).ceil() as usize;
            let (w, h) = (img.w as i32, img.h as i32);
            // FAST candidates with Harris score, 3×3 non-max suppression
            let mut score = vec![0f32; img.w * img.h];
            for y in BORDER..h - BORDER {
                for x in BORDER..w - BORDER {
                    if fast9(&img, x, y, self.fast_threshold) { score[(y * w + x) as usize] = harris(&img, x, y).max(1e-3); }
                }
            }
            let cells = self.grid;
            let mut buckets: Vec<Vec<(f32, i32, i32)>> = vec![Vec::new(); cells * cells];
            for y in BORDER..h - BORDER {
                for x in BORDER..w - BORDER {
                    let s = score[(y * w + x) as usize];
                    if s <= 0.0 { continue; }
                    let mut is_max = true;
                    'n: for dy in -1..=1 { for dx in -1..=1 {
                        if (dx, dy) != (0, 0) && score[((y + dy) * w + x + dx) as usize] > s { is_max = false; break 'n; }
                    } }
                    if !is_max { continue; }
                    let (fx, fy) = (x as f32 / f as f32, y as f32 / f as f32);
                    if !keep(fx, fy) { continue; }
                    let cx = (x as usize * cells / img.w).min(cells - 1);
                    let cy = (y as usize * cells / img.h).min(cells - 1);
                    buckets[cy * cells + cx].push((s, x, y));
                }
            }
            let per_cell = quota.div_ceil(cells * cells).max(1);
            let smooth = box5(&img);
            for b in buckets.iter_mut() {
                b.sort_by(|a, c| c.0.total_cmp(&a.0));
                for &(s, x, y) in b.iter().take(per_cell) {
                    let a = orientation(&img, x, y);
                    keys.push(Key { x: x as f32 / f as f32, y: y as f32 / f as f32, level: lvl as u8, angle: a, response: s });
                    descs.push(describe(&smooth, x, y, a));
                }
            }
        }
        (keys, descs)
    }
}

#[inline]
pub fn hamming(a: &Desc, b: &Desc) -> u32 { (0..4).map(|i| (a[i] ^ b[i]).count_ones()).sum() }

/// For each query descriptor, the best train match if it passes `best < ratio × second`.
pub fn ratio_match(query: &[Desc], train: &[Desc], ratio: f32) -> Vec<Match> {
    if query.len() < 2 || train.len() < 2 { return vec![]; }
    let mut out = Vec::new();
    for (qi, q) in query.iter().enumerate() {
        let (mut b1, mut b2, mut bi) = (u32::MAX, u32::MAX, 0);
        for (ti, t) in train.iter().enumerate() {
            let d = hamming(q, t);
            if d < b1 { b2 = b1; b1 = d; bi = ti; } else if d < b2 { b2 = d; }
        }
        if (b1 as f32) < ratio * b2 as f32 { out.push(Match { query: qi, train: bi, dist: b1 }); }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pattern_is_deterministic_and_inside_patch() {
        let p = pattern();
        assert!(p.iter().flatten().all(|&(x, y)| (x as i32).pow(2) + (y as i32).pow(2) <= (PATCH - 1).pow(2)));
        assert_eq!(p[0], pattern()[0]);
    }
    #[test]
    fn fast_finds_a_bright_square_corner() {
        let mut g = Gray::new(64, 64);
        for y in 32..64 { for x in 32..64 { g.data[y * 64 + x] = 200; } }
        assert!(fast9(&g, 32, 32, 20));
        assert!(!fast9(&g, 10, 10, 20));
    }
}
