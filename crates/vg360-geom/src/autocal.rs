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
pub fn vanishing_point(segments: &[([f64; 2], [f64; 2])]) -> Option<[f64; 2]> {
    let mut m = [[0f64; 3]; 3];
    let mut n = 0;
    for (a, b) in segments {
        // join of the homogeneous points (a, 1) and (b, 1)
        let l = [a[1] - b[1], b[0] - a[0], a[0] * b[1] - a[1] * b[0]];
        let s = l[0].hypot(l[1]);
        if s < 1e-9 { continue; }
        let l = l.map(|x| x / s);
        for i in 0..3 { for j in 0..3 { m[i][j] += l[i] * l[j]; } }
        n += 1;
    }
    if n < 2 { return None; }
    let x = smallest_eigenvector(m);
    (x[2].abs() > 1e-12).then(|| [x[0] / x[2], x[1] / x[2]])
}

/// Focal length, pitch and roll from two vanishing points of orthogonal horizontal world directions.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct VpCalib { pub f_px: f64, pub pitch_deg: f64, pub roll_deg: f64, pub hfov_deg: f64 }

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

fn ray_plane(c: [f64; 3], d: [f64; 3], n: [f64; 3], delta: f64) -> Option<[f64; 3]> {
    let nd = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
    if nd.abs() < 1e-12 { return None; }
    let s = -((n[0] * c[0] + n[1] * c[1] + n[2] * c[2]) + delta) / nd;
    Some([c[0] + s * d[0], c[1] + s * d[1], c[2] + s * d[2]])
}

/// Head height (m) of a box [x1, y1, x2, y2] for this camera: feet ray ∧ deck, head ray ∧ the vertical plane
/// through the feet facing the camera (the notebook's `head_height`).
pub fn head_height(cam: &Camera, b: [f64; 4]) -> Option<f64> {
    let c = cam.centre();
    let uc = (b[0] + b[2]) / 2.0;
    let ft = ray_plane(c, cam.ray(uc, b[3]), [0.0, 0.0, 1.0], 0.0)?;
    let mut n = [ft[0] - c[0], ft[1] - c[1], 0.0];
    let l = n[0].hypot(n[1]);
    if l < 1e-9 { return None; }
    n = [n[0] / l, n[1] / l, 0.0];
    ray_plane(c, cam.ray(uc, b[1]), n, -(n[0] * ft[0] + n[1] * ft[1])).map(|x| x[2])
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
}
