//! Zone tests. CGA: for a point P and sphere S, P·S = ½(r² − d²), so the sign says inside/outside.
//! 2-D PGA: the join of two vertices is a line; its meet-sign with a point is the side test.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Zone {
    Circle { c: [f64; 2], r: f64 },
    Polygon { pts: Vec<[f64; 2]> },
}

/// ½(r² − d²): positive inside, zero on the boundary, negative outside.
pub fn cga_sphere(p: [f64; 2], c: [f64; 2], r: f64) -> f64 {
    let d2 = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2);
    0.5 * (r * r - d2)
}

/// Signed area: > 0 when p is left of a→b.
pub fn side_of_line(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

impl Zone {
    pub fn contains(&self, p: [f64; 2]) -> bool {
        match self {
            Zone::Circle { c, r } => cga_sphere(p, *c, *r) >= 0.0,
            Zone::Polygon { pts } => {
                // winding number via side tests (works for non-convex polygons)
                let mut wn = 0i32;
                for i in 0..pts.len() {
                    let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
                    if a[1] <= p[1] {
                        if b[1] > p[1] && side_of_line(a, b, p) > 0.0 { wn += 1; }
                    } else if b[1] <= p[1] && side_of_line(a, b, p) < 0.0 { wn -= 1; }
                }
                wn != 0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sphere_sign() {
        assert!(cga_sphere([1.0, 0.0], [0.0, 0.0], 2.0) > 0.0);
        assert_eq!(cga_sphere([2.0, 0.0], [0.0, 0.0], 2.0), 0.0);
        assert!(cga_sphere([3.0, 0.0], [0.0, 0.0], 2.0) < 0.0);
    }
    #[test]
    fn l_shaped_polygon() {
        let z = Zone::Polygon { pts: vec![[0., 0.], [4., 0.], [4., 1.], [1., 1.], [1., 4.], [0., 4.]] };
        assert!(z.contains([0.5, 3.0]) && z.contains([3.0, 0.5]) && !z.contains([3.0, 3.0]));
    }
}
