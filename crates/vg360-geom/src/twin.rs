//! Rig Twin → camera models. The Rig Twin editor (console `/twin`) exports `vg360_twin: 1` JSON:
//! components, cameras (position, mount height, heading, pitch, FOV, resolution, deck surface) and
//! zones with the rules switched on. This turns each twin camera into the runtime [`Camera`].
//!
//! Conventions: the twin's heading ψ is a math angle (0° = east, counter-clockwise); the runtime yaw is a
//! compass bearing (0° = north, clockwise), so yaw = 90° − ψ. A camera looking at a raised deck
//! (`surf` metres) is modelled against that deck: height above it = h − surf.
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::camera::Camera;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwinCamera {
    pub id: String,
    pub camera: Camera,
    pub width: u32,
    pub height: u32,
    /// deck surface the camera looks at (m)
    pub surf: f64,
    /// calibration points: name, deck world (x, y), pixel the twin predicts
    pub points: Vec<(String, [f64; 2], [f64; 2])>,
}

impl TwinCamera {
    /// RMS pixel distance between the runtime camera's projection of the twin's calibration points
    /// and the pixels the twin itself predicts (checks the conventions agree).
    pub fn rms_px(&self) -> Option<f64> {
        if self.points.is_empty() { return None; }
        let s: f64 = self.points.iter().map(|(_, w, p)| {
            let q = self.camera.project(w[0], w[1]).unwrap_or([f64::INFINITY; 2]);
            (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)
        }).sum();
        Some((s / self.points.len() as f64).sqrt())
    }
}

pub fn wrap180(a: f64) -> f64 { let x = (a + 180.0).rem_euclid(360.0) - 180.0; if x == -180.0 { 180.0 } else { x } }

/// Parse a twin export. Errors name the first problem.
pub fn cameras(twin: &Value) -> Result<Vec<TwinCamera>, String> {
    if twin["vg360_twin"].as_i64() != Some(1) { return Err("not a VG360 twin file (vg360_twin != 1)".into()); }
    let cams = twin["cameras"].as_array().ok_or("twin has no cameras")?;
    cams.iter().map(|c| {
        let id = c["id"].as_str().ok_or("camera without id")?.to_string();
        let num = |k: &str| c[k].as_f64().ok_or(format!("camera {id}: missing {k}"));
        let res = c["res"].as_str().ok_or(format!("camera {id}: missing res (export from the current Rig Twin)"))?;
        let (w, h) = res.split_once('x').and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?))).ok_or(format!("camera {id}: bad res {res}"))?;
        let (fov, surf) = (num("fov")?, c["surf"].as_f64().unwrap_or(0.0));
        let f = (w as f64 / 2.0) / (fov.to_radians() / 2.0).tan();
        let camera = Camera { f_px: f, cx: w as f64 / 2.0, cy: h as f64 / 2.0, height_m: num("h")? - surf, pitch_deg: num("pitch")?,
                              yaw_deg: wrap180(90.0 - num("yaw")?), roll_deg: 0.0, x0: num("x")?, y0: num("y")? };
        if camera.height_m <= 0.0 { return Err(format!("camera {id}: mounted below its deck")); }
        let points = c["calibration_points"].as_array().into_iter().flatten().filter_map(|p| Some((
            p["name"].as_str().unwrap_or("").to_string(),
            [p["world"][0].as_f64()?, p["world"][1].as_f64()?],
            [p["predicted_px"][0].as_f64()?, p["predicted_px"][1].as_f64()?]))).collect();
        Ok(TwinCamera { id, camera, width: w, height: h, surf, points })
    }).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn rig_twin_cameras_reproduce_the_twins_own_projection() {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/twin/rig_twin.json")).unwrap()).unwrap();
        let cams = super::cameras(&v).unwrap();
        assert_eq!(cams.len(), v["cameras"].as_array().unwrap().len());
        let mut checked = 0;
        for c in &cams {
            if let Some(rms) = c.rms_px() {
                // the twin rounds its predicted pixels to whole numbers
                assert!(rms < 1.0, "{}: {rms:.2} px between runtime camera and twin", c.id);
                checked += 1;
            }
        }
        assert!(checked >= 5, "{checked} cameras with calibration points");
    }
}

/// A twin zone on the deck, as a counter-clockwise floor polygon (metres).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwinZone { pub id: String, pub name: String, pub polygon: Vec<[f64; 2]>, pub surface_m: f64, pub rules: Vec<String>, pub cameras: Vec<String> }

/// Zones of the twin: circles become 24-gons, rects `[x, y, w, h]` become 4-gons; only rules switched on.
pub fn zones(twin: &Value) -> Vec<TwinZone> {
    twin["zones"].as_array().into_iter().flatten().filter_map(|z| {
        let polygon = match z["shape"].as_str()? {
            "circle" => {
                let (c, r) = ([z["center"][0].as_f64()?, z["center"][1].as_f64()?], z["r"].as_f64()?);
                (0..24).map(|i| { let a = i as f64 * std::f64::consts::TAU / 24.0; [c[0] + r * a.cos(), c[1] + r * a.sin()] }).collect()
            }
            "rect" => {
                let v: Vec<f64> = (0..4).map(|i| z["rect"][i].as_f64()).collect::<Option<_>>()?;
                vec![[v[0], v[1]], [v[0] + v[2], v[1]], [v[0] + v[2], v[1] + v[3]], [v[0], v[1] + v[3]]]
            }
            _ => return None,
        };
        let strs = |k: &str, f: &dyn Fn(&Value) -> Option<String>| z[k].as_array().into_iter().flatten().filter_map(f).collect::<Vec<_>>();
        Some(TwinZone {
            id: z["id"].as_str()?.into(), name: z["name"].as_str().unwrap_or("").into(), polygon,
            surface_m: z["surface_m"].as_f64().unwrap_or(0.0),
            rules: strs("rules", &|r| (r["on"] != false).then(|| r["id"].as_str().map(String::from)).flatten()),
            cameras: strs("cameras", &|c| c.as_str().map(String::from)),
        })
    }).collect()
}

#[cfg(test)]
mod zone_tests {
    #[test]
    fn twin_zones_are_ccw_polygons() {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/twin/rig_twin.json")).unwrap()).unwrap();
        let zs = super::zones(&v);
        assert_eq!(zs.len(), v["zones"].as_array().unwrap().len());
        for z in &zs {
            let a: f64 = (0..z.polygon.len()).map(|i| { let (p, q) = (z.polygon[i], z.polygon[(i + 1) % z.polygon.len()]); p[0] * q[1] - q[0] * p[1] }).sum();
            assert!(a > 0.0, "{} is not counter-clockwise", z.id);
        }
        assert!(zs.iter().any(|z| z.rules.contains(&"lift".to_string())));
    }
}
