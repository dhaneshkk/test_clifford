//! Camera auto-calibration without surveyed deck points (notebook §3, "Lehar: parallel lines meet at infinity").
//!
//! * [`vanishing_point`] — least-squares meeting point of image segments of parallel world lines
//!   (2-D PGA: each segment is the join of its end points; the vanishing point is the null vector of the
//!   stacked, normalised lines).
//! * [`from_vanishing_points`] — two vanishing points of orthogonal horizontal directions (deck plate edges,
//!   rails, pipe racks) → focal length, pitch and roll. `f² = −(v₁ − p)·(v₂ − p)` is proved in
//!   `lean/VG360/Calib.lean` (`focal_from_orthogonal_vps`).
//! * [`height_from_people`] — every metric estimate scales linearly with the mount height
//!   (`Calib.head_height_scales`), so the median head height of standing workers seen with a 1 m probe camera
//!   fixes it: `h = 1.70 m / median`.
use crate::camera::Camera;

/// Intersection of the image lines of parallel world edges, each given by two pixels. `None` if the lines are
/// parallel in the image too (vanishing point at infinity) or fewer than two segments are usable.
///
/// Hartley-normalised (points translated to their centroid and scaled to mean distance sqrt 2 before the eigen
/// solve) and weighted by segment length, so long, well-conditioned edges dominate short noisy ones.
pub fn vanishing_point(segments: &[([f64; 2], [f64; 2])]) -> Option<[f64; 2]> {
    let usable: Vec<_> = segments.iter().filter(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]) > 1e-9).collect();
    if usable.len() < 2 { return None; }
    let k = 2.0 * usable.len() as f64;
    let (mx, my) = usable.iter().fold((0.0, 0.0), |s, (a, b)| (s.0 + a[0] + b[0], s.1 + a[1] + b[1]));
    let (mx, my) = (mx / k, my / k);
    let md = usable.iter().map(|(a, b)| (a[0] - mx).hypot(a[1] - my) + (b[0] - mx).hypot(b[1] - my)).sum::<f64>() / k;
    if md < 1e-12 { return None; }
    let sc = std::f64::consts::SQRT_2 / md;
    let mut m = [[0f64; 3]; 3];
    for (a, b) in usable {
        let (a, b) = ([(a[0] - mx) * sc, (a[1] - my) * sc], [(b[0] - mx) * sc, (b[1] - my) * sc]);
        // join of the homogeneous points (a, 1) and (b, 1); unit line (distance residual) weighted by segment length
        let l = [a[1] - b[1], b[0] - a[0], a[0] * b[1] - a[1] * b[0]];
        let len = l[0].hypot(l[1]);
        let l = l.map(|x| x / len);
        for i in 0..3 { for j in 0..3 { m[i][j] += len * l[i] * l[j]; } }
    }
    let x = smallest_eigenvector(m);
    if x[2].abs() < 1e-12 { return None; }
    // undo the normalisation
    Some([x[0] / x[2] / sc + mx, x[1] / x[2] / sc + my])
}

/// Focal length, pitch and roll from two vanishing points of orthogonal horizontal world directions.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct VpCalib { pub f_px: f64, pub pitch_deg: f64, pub roll_deg: f64, pub hfov_deg: f64 }

/// Why [`vp_calibrate`] refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VpError { TooFewLines, Degenerate, NearInfinity, NotOrthogonal, FocalOutOfRange }

impl std::fmt::Display for VpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            VpError::TooFewLines => "each direction needs at least two usable line segments",
            VpError::Degenerate => "the lines of one direction are parallel or identical in the image, so they have no vanishing point",
            VpError::NearInfinity => "a vanishing point lies more than 50 image sizes from the centre; pick lines with more perspective",
            VpError::NotOrthogonal => "the two directions are not orthogonal as seen by a real camera (f^2 <= 0); check the line groups",
            VpError::FocalOutOfRange => "the recovered focal length is outside 0.3 to 10 image widths; check the lines or enter the lens by hand",
        })
    }
}
impl std::error::Error for VpError {}

/// Calibrate from two groups of image segments of orthogonal horizontal world directions (deck edges, rails).
/// Principal point = image centre. Validates every step instead of returning a bare `None`.
pub fn vp_calibrate(lines_a: &[([f64; 2], [f64; 2])], lines_b: &[([f64; 2], [f64; 2])], w: f64, h: f64) -> Result<VpCalib, VpError> {
    let usable = |l: &[([f64; 2], [f64; 2])]| l.iter().filter(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]) > 1e-9).count();
    if usable(lines_a) < 2 || usable(lines_b) < 2 { return Err(VpError::TooFewLines); }
    // sine of the largest angle between two segment directions; identical or parallel lines give ~0
    let spread = |l: &[([f64; 2], [f64; 2])]| {
        let dirs: Vec<[f64; 2]> = l.iter().filter_map(|(a, b)| {
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let n = dx.hypot(dy);
            (n > 1e-9).then(|| [dx / n, dy / n])
        }).collect();
        dirs.iter().flat_map(|p| dirs.iter().map(move |q| (p[0] * q[1] - p[1] * q[0]).abs())).fold(0.0, f64::max)
    };
    let vp = |l: &[([f64; 2], [f64; 2])]| if spread(l) < 1e-6 { None } else { vanishing_point(l) };
    let (v1, v2) = match (vp(lines_a), vp(lines_b)) { (Some(a), Some(b)) => (a, b), _ => return Err(VpError::Degenerate) };
    let (cx, cy) = (w / 2.0, h / 2.0);
    let far = 50.0 * w.max(h);
    if [v1, v2].iter().any(|v| !v[0].is_finite() || !v[1].is_finite() || (v[0] - cx).hypot(v[1] - cy) > far) { return Err(VpError::NearInfinity); }
    let f2 = -((v1[0] - cx) * (v2[0] - cx) + (v1[1] - cy) * (v2[1] - cy));
    if f2 <= 0.0 { return Err(VpError::NotOrthogonal); }
    let f = f2.sqrt();
    if f < 0.3 * w || f > 10.0 * w { return Err(VpError::FocalOutOfRange); }
    from_vanishing_points(v1, v2, cx, cy, w).ok_or(VpError::Degenerate)
}

pub fn from_vanishing_points(v1: [f64; 2], v2: [f64; 2], cx: f64, cy: f64, width: f64) -> Option<VpCalib> {
    let f2 = -((v1[0] - cx) * (v2[0] - cx) + (v1[1] - cy) * (v2[1] - cy));
    if f2 <= 0.0 { return None; }               // not orthogonal directions seen by a real camera
    let f = f2.sqrt();
    // horizon = join of the two vanishing points; its distance from the principal point is f·tan(pitch)
    let (dx, dy) = (v2[0] - v1[0], v2[1] - v1[1]);
    let len = dx.hypot(dy);
    if len < 1e-9 { return None; }
    let (dx, dy) = if dx < 0.0 { (-dx, -dy) } else { (dx, dy) };
    // signed distance of p below the horizon (image y grows downwards)
    let dist = ((cx - v1[0]) * (-dy) + (cy - v1[1]) * dx) / len;
    let pitch = dist.atan2(f).to_degrees();
    let roll = (-dy).atan2(dx).to_degrees();
    Some(VpCalib { f_px: f, pitch_deg: pitch, roll_deg: roll, hfov_deg: 2.0 * (width / 2.0 / f).atan().to_degrees() })
}

/// Deck point under pixel (u, v) with the ray pointing forward (s > 0).
pub fn feet_on_deck(cam: &Camera, u: f64, v: f64) -> Option<[f64; 3]> {
    let c = cam.centre();
    let d = cam.ray(u, v);
    if d[2].abs() < 1e-12 { return None; }
    let s = -c[2] / d[2];
    if !(s > 0.0) || !s.is_finite() { return None; }
    Some([c[0] + s * d[0], c[1] + s * d[1], 0.0])
}

/// Feet on the deck, head height and foot column of a standing person whose box is `b` = [x1, y1, x2, y2].
///
/// The box is the hull of the body, so its centre column is the mean of the foot and head columns (verticals
/// converge in the image). Iterate: cast the feet ray through column `uf`, the head ray through `uh`, take the
/// point of the head ray closest to the vertical line through the feet, re-project it to get the true head column
/// and re-centre `uf`/`uh` symmetrically around the box centre. Returns `(feet, head_z, u_foot)`.
pub fn person_vertical(cam: &Camera, b: [f64; 4]) -> Option<([f64; 3], f64, f64)> {
    let c = cam.centre();
    let uc = (b[0] + b[2]) / 2.0;
    let (mut uf, mut uh) = (uc, uc);
    let mut out = None;
    for _ in 0..6 {
        let ft = feet_on_deck(cam, uf, b[3])?;
        let d = cam.ray(uh, b[1]);
        let dxy2 = d[0] * d[0] + d[1] * d[1];
        if dxy2 < 1e-18 { return None; }
        let s = (d[0] * (ft[0] - c[0]) + d[1] * (ft[1] - c[1])) / dxy2;
        if !(s > 0.0) { return None; }
        let z = c[2] + s * d[2];
        out = Some((ft, z, uf));
        let uh_true = cam.project3_raw([ft[0], ft[1], z])[0];
        let half = (uh_true - uf) / 2.0;
        if !half.is_finite() { return None; }
        uf = uc - half;
        uh = uc + half;
    }
    let (ft, z, _) = out?;
    // final foot and head with the last columns
    let ft = feet_on_deck(cam, uf, b[3]).unwrap_or(ft);
    let d = cam.ray(uh, b[1]);
    let dxy2 = d[0] * d[0] + d[1] * d[1];
    if dxy2 < 1e-18 { return None; }
    let s = (d[0] * (ft[0] - c[0]) + d[1] * (ft[1] - c[1])) / dxy2;
    if !(s > 0.0) { return None; }
    let z = if (c[2] + s * d[2]).is_finite() { c[2] + s * d[2] } else { z };
    (z > 0.0 && z.is_finite()).then_some((ft, z, uf))
}

/// Head height (m) of a box [x1, y1, x2, y2] for this camera (see [`person_vertical`]).
pub fn head_height(cam: &Camera, b: [f64; 4]) -> Option<f64> {
    person_vertical(cam, b).map(|(_, z, _)| z)
}

/// Mount height from boxes of standing people (assumed `person_m` tall, 1.70 m in the notebook), with the
/// rest of the camera (focal length, pitch, roll) known. Median over boxes, so a few crouching or partly hidden
/// workers do not move it much. `None` without usable boxes.
pub fn height_from_people(cam: &Camera, boxes: &[[f64; 4]], person_m: f64) -> Option<f64> {
    let probe = Camera { height_m: 1.0, ..cam.clone() };
    let mut hs: Vec<f64> = boxes.iter().filter_map(|b| head_height(&probe, *b)).filter(|h| *h > 0.0).collect();
    if hs.is_empty() { return None; }
    hs.sort_by(f64::total_cmp);
    let n = hs.len();
    let med = if n % 2 == 1 { hs[n / 2] } else { (hs[n / 2 - 1] + hs[n / 2]) / 2.0 };
    Some(person_m / med)
}

/// Eigenvector of the smallest eigenvalue of a symmetric 3×3 matrix (cyclic Jacobi rotations).
fn smallest_eigenvector(mut a: [[f64; 3]; 3]) -> [f64; 3] {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        let off = a[0][1].powi(2) + a[0][2].powi(2) + a[1][2].powi(2);
        if off < 1e-30 { break; }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 { continue; }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = (if theta >= 0.0 { 1.0 } else { -1.0 }) / (theta.abs() + (theta * theta + 1.0).sqrt());
            let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let (vkp, vkq) = (v[k][p], v[k][q]);
                v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let i = (0..3).min_by(|&x, &y| a[x][x].total_cmp(&a[y][y])).unwrap();
    [v[0][i], v[1][i], v[2][i]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(pitch: f64, roll: f64, yaw: f64) -> Camera {
        Camera { f_px: 900.0, cx: 640.0, cy: 360.0, height_m: 12.0, pitch_deg: pitch, yaw_deg: yaw, roll_deg: roll, x0: 0.0, y0: 0.0 }
    }

    /// image segments of deck lines along a horizontal direction (dx, dy), offset across the deck
    fn family(c: &Camera, dir: [f64; 2], noise: f64) -> Vec<([f64; 2], [f64; 2])> {
        let n = [-dir[1], dir[0]];
        (0..6).map(|k| {
            let o = -10.0 + 4.0 * k as f64;
            let base = [n[0] * o + 25.0 * c.yaw_deg.to_radians().sin(), n[1] * o + 25.0 * c.yaw_deg.to_radians().cos()];
            let p = |s: f64| c.project3_raw([base[0] + s * dir[0], base[1] + s * dir[1], 0.0]);
            let (mut a, mut b) = (p(-6.0), p(6.0));
            a[0] += noise * ((k * 7 % 5) as f64 - 2.0); b[1] -= noise * ((k * 3 % 5) as f64 - 2.0);
            (a, b)
        }).collect()
    }

    #[test]
    fn recovers_focal_pitch_and_roll_from_two_orthogonal_line_families() {
        for (pitch, roll, yaw) in [(25.0, 0.0, 0.0), (35.0, 4.0, 30.0), (50.0, -6.0, -70.0)] {
            let c = cam(pitch, roll, yaw);
            // deck edges at ±45° to the viewing direction (compass bearing β → [sin β, cos β])
            let (b1, b2) = ((yaw + 45.0f64).to_radians(), (yaw - 45.0f64).to_radians());
            let v1 = vanishing_point(&family(&c, [b1.sin(), b1.cos()], 0.3)).unwrap();
            let v2 = vanishing_point(&family(&c, [b2.sin(), b2.cos()], 0.3)).unwrap();
            let k = from_vanishing_points(v1, v2, c.cx, c.cy, 1280.0).unwrap();
            eprintln!("pitch {pitch} roll {roll} yaw {yaw} → {k:?}");
            assert!((k.f_px - 900.0).abs() < 20.0, "{k:?}");
            assert!((k.pitch_deg - pitch).abs() < 1.0, "{k:?}");
            assert!((k.roll_deg - roll).abs() < 1.0, "{k:?}");
        }
    }

    #[test]
    fn mount_height_from_standing_people() {
        let c = cam(35.0, 0.0, 0.0);
        // 1.70 m people standing at a few deck spots, plus one crouching (1.0 m)
        let mut boxes = Vec::new();
        for (x, y, h) in [(-3.0, 15.0, 1.7), (2.0, 20.0, 1.7), (5.0, 12.0, 1.7), (0.0, 25.0, 1.7), (-6.0, 18.0, 1.0)] {
            let f = c.project3_raw([x, y, 0.0]);
            let t = c.project3_raw([x, y, h]);
            boxes.push([f[0] - 15.0, t[1], f[0] + 15.0, f[1]]);
        }
        let h = height_from_people(&c, &boxes, 1.70).unwrap();
        assert!((h - 12.0).abs() < 0.5, "{h}");
    }

    /// 1.70 m person, 0.5 x 0.3 m footprint centred on (x, y); box = hull of the projected 8 corners,
    /// top row at (x, y, 1.70), bottom row at (x, y, 0)
    fn person_box(c: &Camera, x: f64, y: f64) -> Option<[f64; 4]> {
        let r = c.rotation();
        let mut us = Vec::new();
        for dx in [-0.25, 0.25] { for dy in [-0.15, 0.15] { for z in [0.0, 1.7] {
            let d = [x + dx - c.x0, y + dy - c.y0, z - c.height_m];
            let cz = r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2];
            if cz <= 0.1 { return None; }
            us.push(c.project3_raw([x + dx, y + dy, z])[0]);
        }}}
        let (u1, u2) = (us.iter().cloned().fold(f64::MAX, f64::min), us.iter().cloned().fold(f64::MIN, f64::max));
        let (t, f) = (c.project3_raw([x, y, 1.7]), c.project3_raw([x, y, 0.0]));
        Some([u1, t[1], u2, f[1]])
    }

    #[test]
    fn head_height_unbiased_offaxis() {
        let mut n = 0;
        for pitch in [35.0, 50.0] { for roll in [0.0, 5.0] {
            let c = cam(pitch, roll, 0.0);
            for (x, y) in [(0.0, 12.0), (4.0, 12.0), (8.0, 12.0), (8.0, 14.0), (-6.0, 20.0), (10.0, 8.0)] {
                let Some(b) = person_box(&c, x, y) else { eprintln!("skip pitch {pitch} ({x},{y}): behind camera"); continue };
                let h = height_from_people(&c, &[b], 1.70).unwrap_or(f64::NAN);
                eprintln!("pitch {pitch} roll {roll} ({x},{y}) -> {h:.4}");
                assert!((h - 12.0).abs() < 0.06, "pitch {pitch} roll {roll} at ({x},{y}): {h}");
                n += 1;
            }
        }}
        assert!(n >= 20, "only {n} boxes were usable");
    }

    fn seg(c: &Camera, dir: [f64; 2], o: f64) -> ([f64; 2], [f64; 2]) {
        let n = [-dir[1], dir[0]];
        let base = [n[0] * o, n[1] * o + 25.0];
        let p = |s: f64| c.project3_raw([base[0] + s * dir[0], base[1] + s * dir[1], 0.0]);
        (p(-6.0), p(6.0))
    }

    #[test]
    fn vp_calibrate_ok_and_errors() {
        let c = cam(35.0, 0.0, 0.0);
        let q = std::f64::consts::FRAC_1_SQRT_2;
        let (d1, d2) = ([q, q], [q, -q]);
        let g1: Vec<_> = (0..5).map(|k| seg(&c, d1, -8.0 + 4.0 * k as f64)).collect();
        let g2: Vec<_> = (0..5).map(|k| seg(&c, d2, -8.0 + 4.0 * k as f64)).collect();
        let k = vp_calibrate(&g1, &g2, 1280.0, 720.0).unwrap();
        assert!((k.f_px - 900.0).abs() < 5.0 && (k.pitch_deg - 35.0).abs() < 0.5, "{k:?}");
        assert_eq!(vp_calibrate(&g1[..1], &g2, 1280.0, 720.0), Err(VpError::TooFewLines));
        let par: Vec<_> = (0..3).map(|k| ([100.0, 100.0 + 50.0 * k as f64], [500.0, 100.0 + 50.0 * k as f64])).collect();
        assert_eq!(vp_calibrate(&par, &g2, 1280.0, 720.0), Err(VpError::Degenerate));
        assert_eq!(vp_calibrate(&[g1[0], g1[0]], &g2, 1280.0, 720.0), Err(VpError::Degenerate));
        // nearly parallel lines: the vanishing point is millions of pixels away
        let near: Vec<_> = (0..3).map(|k| ([100.0, 100.0 + 50.0 * k as f64], [500.0, 100.0 + 50.0 * k as f64 + 0.01 * k as f64])).collect();
        assert_eq!(vp_calibrate(&near, &g2, 1280.0, 720.0), Err(VpError::NearInfinity));
        // both vanishing points on the same side of the centre -> f^2 <= 0
        assert_eq!(vp_calibrate(&g1, &g1, 1280.0, 720.0), Err(VpError::NotOrthogonal));
        // orthogonal-looking but implausible focal length: vanishing points 20 px either side of the centre -> f = 20
        let toward = |vx: f64, vy: f64| -> Vec<([f64; 2], [f64; 2])> {
            [(0.3, 1.0), (-0.3, 1.0), (0.0, 1.0)].iter().map(|(dx, dy)| ([vx + 100.0 * dx, vy + 100.0 * dy], [vx + 300.0 * dx, vy + 300.0 * dy])).collect()
        };
        assert_eq!(vp_calibrate(&toward(660.0, 360.0), &toward(620.0, 360.0), 1280.0, 720.0), Err(VpError::FocalOutOfRange));
        assert!(!VpError::NotOrthogonal.to_string().is_empty());
    }
}
