//! 3-D hazard zones (notebook §4). Spheres and half-spaces are tested with the conformal (CGA)
//! inner product on 5-vectors (x, a, b) = x + a·e∞ + b·e₀:  P·S = ½(r² − d²) for a sphere,
//! n·x − δ for a plane (Lean: `cga_sphere`, `cga_plane`). Floor polygons use 2-D PGA side tests.

/// Conformal embedding of a point: (x, y, z, ½|x|², 1).
pub fn up5(x: [f64; 3]) -> [f64; 5] { [x[0], x[1], x[2], 0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]), 1.0] }
pub fn sphere5(c: [f64; 3], r: f64) -> [f64; 5] { let mut s = up5(c); s[3] -= 0.5 * r * r; s }
pub fn plane5(n: [f64; 3], d: f64) -> [f64; 5] {
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    [n[0] / l, n[1] / l, n[2] / l, d, 0.0]
}
/// Conformal inner product (metric e₀·e∞ = −1).
pub fn cdot5(p: &[f64; 5], s: &[f64; 5]) -> f64 { p[0] * s[0] + p[1] * s[1] + p[2] * s[2] - (p[3] * s[4] + p[4] * s[3]) }

pub enum Centre { Fixed([f64; 3]), Moving(Box<dyn Fn(f64) -> [f64; 3] + Send + Sync>) }

pub enum Shape {
    /// convex floor polygon, counter-clockwise
    Polygon(Vec<[f64; 2]>),
    Sphere { centre: Centre, r: f64 },
    /// points with n·x > d are past the plane
    HalfSpace { n: [f64; 3], d: f64 },
}

pub struct Zone { pub name: String, pub shape: Shape }

pub fn polygon_inside(poly: &[[f64; 2]], p: [f64; 2]) -> bool {
    (0..poly.len()).all(|i| {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0
    })
}

/// Clearance in metres for one person's body points (feet first); < 0 means inside.
pub fn zone_hits(zones: &[Zone], body: &[[f64; 3]], t: f64) -> Vec<(String, f64)> {
    zones.iter().map(|z| {
        let c = match &z.shape {
            Shape::Polygon(poly) => if polygon_inside(poly, [body[0][0], body[0][1]]) { -0.01 } else { 1.0 },
            Shape::Sphere { centre, r } => {
                let c = match centre { Centre::Fixed(c) => *c, Centre::Moving(f) => f(t) };
                let s = sphere5(c, *r);
                let v = body.iter().map(|p| cdot5(&up5(*p), &s)).fold(f64::NEG_INFINITY, f64::max);
                (r * r - 2.0 * v).max(0.0).sqrt() - r
            }
            Shape::HalfSpace { n, d } => {
                let s = plane5(*n, *d);
                -body.iter().map(|p| cdot5(&up5(*p), &s)).fold(f64::NEG_INFINITY, f64::max)
            }
        };
        (z.name.clone(), c)
    }).collect()
}

/// The notebook's simulated-site zones (`default_zones`).
pub fn default_zones() -> Vec<Zone> {
    vec![
        Zone { name: "pump_skid_keepout".into(), shape: Shape::Polygon(vec![[-6.0, 8.0], [-2.0, 8.0], [-2.0, 11.0], [-6.0, 11.0]]) },
        Zone { name: "crane_hook".into(), shape: Shape::Sphere { centre: Centre::Fixed([3.0, 12.0, 1.0]), r: 2.0 } },
        Zone { name: "suspended_load".into(), shape: Shape::Sphere {
            centre: Centre::Moving(Box::new(|t| [-4.0 + 8.0 * (0.5 + 0.5 * (0.1 * t).sin()), 14.0, 1.2])), r: 1.8 } },
        Zone { name: "rig_edge".into(), shape: Shape::HalfSpace { n: [0.0, 1.0, 0.0], d: 16.5 } },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sphere_clearance_is_distance_minus_radius() {
        let z = [Zone { name: "s".into(), shape: Shape::Sphere { centre: Centre::Fixed([0.0, 0.0, 1.0]), r: 2.0 } }];
        let c = zone_hits(&z, &[[3.0, 4.0, 1.0]], 0.0)[0].1;
        assert!((c - 3.0).abs() < 1e-12);
    }
    #[test]
    fn halfspace_clearance() {
        let z = [Zone { name: "e".into(), shape: Shape::HalfSpace { n: [0.0, 2.0, 0.0], d: 16.5 } }];
        assert!((zone_hits(&z, &[[0.0, 15.0, 0.1], [0.0, 15.2, 1.7]], 0.0)[0].1 - 1.3).abs() < 1e-12);
    }
}
