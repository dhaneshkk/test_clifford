//! Robust 2-D model fitting, replacing `findHomography(USAC_MAGSAC)` and
//! `estimateAffinePartial2D(RANSAC)`.
//!
//! Estimator: MSAC scoring (truncated quadratic) with adaptive iteration count, followed by
//! local optimisation (iterated least-squares refit on the inlier set). This plays the same
//! role as MAGSAC++ in camera_guard — reject people / moving objects as outliers — but it is
//! not a line-by-line MAGSAC++ port (no σ-marginalisation). Deterministic: fixed-seed PRNG.
use nalgebra::{DMatrix, Matrix3};

use crate::M3;

pub type Pt = [f64; 2];

#[derive(Clone, Debug)]
pub struct Fit { pub model: M3, pub inliers: Vec<bool>, pub n_inliers: usize }

pub fn project(h: &M3, p: Pt) -> Pt {
    let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
    [(h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / w, (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / w]
}

pub fn invert(h: &M3) -> Option<M3> {
    let m = Matrix3::from_fn(|r, c| h[r][c]).try_inverse()?;
    Some(std::array::from_fn(|r| std::array::from_fn(|c| m[(r, c)])))
}

pub fn mul(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

/// Hartley normalisation: centroid to origin, mean distance √2.
fn normaliser(pts: &[Pt]) -> M3 {
    let n = pts.len() as f64;
    let (cx, cy) = (pts.iter().map(|p| p[0]).sum::<f64>() / n, pts.iter().map(|p| p[1]).sum::<f64>() / n);
    let d = pts.iter().map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt()).sum::<f64>() / n;
    let s = if d > 1e-12 { std::f64::consts::SQRT_2 / d } else { 1.0 };
    [[s, 0.0, -s * cx], [0.0, s, -s * cy], [0.0, 0.0, 1.0]]
}

/// Normalised DLT homography (≥ 4 correspondences), `dst ≈ H src`.
pub fn homography_dlt(src: &[Pt], dst: &[Pt]) -> Option<M3> {
    let n = src.len();
    if n < 4 || dst.len() != n { return None; }
    let (ts, td) = (normaliser(src), normaliser(dst));
    let mut a = DMatrix::<f64>::zeros(2 * n.max(5), 9); // pad to ≥ 9 rows so SVD yields V fully
    for i in 0..n {
        let s = project(&ts, src[i]);
        let d = project(&td, dst[i]);
        let (x, y, u, v) = (s[0], s[1], d[0], d[1]);
        let r1 = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u];
        let r2 = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v];
        for j in 0..9 { a[(2 * i, j)] = r1[j]; a[(2 * i + 1, j)] = r2[j]; }
    }
    let svd = (a.transpose() * &a).symmetric_eigen();
    let (mut k, mut best) = (0, f64::INFINITY);
    for (i, &e) in svd.eigenvalues.iter().enumerate() { if e < best { best = e; k = i; } }
    let h = svd.eigenvectors.column(k);
    let hn: M3 = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]];
    let h = mul(&invert(&td)?, &mul(&hn, &ts));
    let s = h[2][2];
    if !s.is_finite() || s.abs() < 1e-12 { return None; }
    let h = h.map(|r| r.map(|v| v / s));
    h.iter().flatten().all(|v| v.is_finite()).then_some(h)
}

/// Least-squares similarity (rotation + uniform scale + translation), as a 3×3 matrix.
pub fn similarity_lsq(src: &[Pt], dst: &[Pt]) -> Option<M3> {
    let n = src.len();
    if n < 2 { return None; }
    let nf = n as f64;
    let (sx, sy) = (src.iter().map(|p| p[0]).sum::<f64>() / nf, src.iter().map(|p| p[1]).sum::<f64>() / nf);
    let (dx, dy) = (dst.iter().map(|p| p[0]).sum::<f64>() / nf, dst.iter().map(|p| p[1]).sum::<f64>() / nf);
    let (mut a, mut b, mut ss) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (src[i][0] - sx, src[i][1] - sy);
        let (u, v) = (dst[i][0] - dx, dst[i][1] - dy);
        a += x * u + y * v;
        b += x * v - y * u;
        ss += x * x + y * y;
    }
    if ss < 1e-12 { return None; }
    let (c, s) = (a / ss, b / ss); // c = k cosθ, s = k sinθ
    Some([[c, -s, dx - c * sx + s * sy], [s, c, dy - s * sx - c * sy], [0.0, 0.0, 1.0]])
}

/// Reprojection error² in the destination image.
fn err2(h: &M3, s: Pt, d: Pt) -> f64 {
    let p = project(h, s);
    if !p[0].is_finite() { return f64::INFINITY; }
    (p[0] - d[0]).powi(2) + (p[1] - d[1]).powi(2)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn sample(&mut self, n: usize, k: usize, out: &mut Vec<usize>) {
        out.clear();
        while out.len() < k { let i = (self.next() % n as u64) as usize; if !out.contains(&i) { out.push(i); } }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Model { Homography, Similarity }

/// MSAC + local optimisation. `thresh` in pixels (same meaning as OpenCV's ransacReprojThreshold).
pub fn fit(model: Model, src: &[Pt], dst: &[Pt], thresh: f64, max_iters: usize, confidence: f64) -> Option<Fit> {
    let n = src.len();
    let k = match model { Model::Homography => 4, Model::Similarity => 2 };
    if n < k { return None; }
    let estimate = |s: &[Pt], d: &[Pt]| match model { Model::Homography => homography_dlt(s, d), Model::Similarity => similarity_lsq(s, d) };
    let t2 = thresh * thresh;
    let cost = |h: &M3| -> f64 { (0..n).map(|i| err2(h, src[i], dst[i]).min(t2)).sum() };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut best, mut best_cost) = (None::<M3>, f64::INFINITY);
    let (mut iters, mut i) = (max_iters, 0);
    let mut idx = Vec::with_capacity(k);
    while i < iters {
        i += 1;
        rng.sample(n, k, &mut idx);
        let (s, d): (Vec<Pt>, Vec<Pt>) = idx.iter().map(|&j| (src[j], dst[j])).unzip();
        let Some(h) = estimate(&s, &d) else { continue };
        let c = cost(&h);
        if c < best_cost {
            best_cost = c;
            best = Some(h);
            let inl = (0..n).filter(|&j| err2(&h, src[j], dst[j]) < t2).count() as f64 / n as f64;
            let p_good = inl.powi(k as i32).clamp(1e-12, 1.0 - 1e-12);
            let need = ((1.0 - confidence).ln() / (1.0 - p_good).ln()).ceil();
            if need.is_finite() { iters = iters.min((need as usize).max(i + 1)); }
        }
    }
    // local optimisation: refit on inliers until the inlier set stops growing
    let mut h = best?;
    let mut last = 0;
    for _ in 0..10 {
        let inl: Vec<usize> = (0..n).filter(|&j| err2(&h, src[j], dst[j]) < t2).collect();
        if inl.len() < k || inl.len() <= last { break; }
        last = inl.len();
        let (s, d): (Vec<Pt>, Vec<Pt>) = inl.iter().map(|&j| (src[j], dst[j])).unzip();
        match estimate(&s, &d) { Some(h2) if cost(&h2) <= cost(&h) + 1e-9 => h = h2, _ => break }
    }
    let inliers: Vec<bool> = (0..n).map(|j| err2(&h, src[j], dst[j]) < t2).collect();
    let n_inliers = inliers.iter().filter(|&&b| b).count();
    Some(Fit { model: h, inliers, n_inliers })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn truth() -> M3 { [[0.98, -0.06, 40.0], [0.05, 1.01, -18.0], [2e-5, -1e-5, 1.0]] }

    fn data(outlier_every: usize) -> (Vec<Pt>, Vec<Pt>) {
        let h = truth();
        let mut src = vec![]; let mut dst = vec![];
        let mut r = Rng(42);
        for i in 0..300 {
            let p = [(r.next() % 960) as f64, (r.next() % 540) as f64];
            let mut q = project(&h, p);
            q[0] += ((r.next() % 1000) as f64 / 1000.0 - 0.5) * 1.0;
            q[1] += ((r.next() % 1000) as f64 / 1000.0 - 0.5) * 1.0;
            if outlier_every > 0 && i % outlier_every == 0 { q = [(r.next() % 960) as f64, (r.next() % 540) as f64]; }
            src.push(p); dst.push(q);
        }
        (src, dst)
    }

    #[test]
    fn dlt_exact() {
        let (s, d) = data(0);
        let h = homography_dlt(&s[..10], &d[..10]).unwrap();
        let t = truth();
        let p = project(&h, [500.0, 300.0]); let q = project(&t, [500.0, 300.0]);
        assert!((p[0] - q[0]).abs() < 1.0 && (p[1] - q[1]).abs() < 1.0);
    }

    #[test]
    fn msac_rejects_40pct_outliers() {
        let (s, d) = data(2); // every 2nd point is garbage → 50 % outliers
        let f = fit(Model::Homography, &s, &d, 3.0, 10_000, 0.999).unwrap();
        assert!(f.n_inliers >= 140, "{}", f.n_inliers);
        for p in [[0.0, 0.0], [960.0, 540.0], [480.0, 270.0]] {
            let (a, b) = (project(&f.model, p), project(&truth(), p));
            assert!((a[0] - b[0]).abs() < 1.5 && (a[1] - b[1]).abs() < 1.5, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn similarity_recovers_rotation_scale_shift() {
        let (c, s) = (1.02 * 0.1f64.cos(), 1.02 * 0.1f64.sin());
        let t: M3 = [[c, -s, 30.0], [s, c, -7.0], [0.0, 0.0, 1.0]];
        let src: Vec<Pt> = (0..50).map(|i| [(i * 17 % 900) as f64, (i * 31 % 500) as f64]).collect();
        let dst: Vec<Pt> = src.iter().map(|&p| project(&t, p)).collect();
        let f = fit(Model::Similarity, &src, &dst, 2.0, 1000, 0.999).unwrap();
        for r in 0..2 { for k in 0..3 { assert!((f.model[r][k] - t[r][k]).abs() < 1e-6); } }
    }
}
