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
    /// pinhole model; for a fisheye camera, its pose (centre, heading, tilt) — the lens is in `lens`
    pub camera: Camera,
    /// fisheye lens (Kannala–Brandt), when the twin camera has `"lens": {"model": "kannala-brandt", …}`
    #[serde(default)]
    pub lens: Option<LensSpec>,
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
    /// Not computed (None) for fisheye cameras: `camera` is only their pose, not their lens.
    pub fn rms_px(&self) -> Option<f64> {
        if self.points.is_empty() || self.lens.is_some() { return None; }
        let s: f64 = self.points.iter().map(|(_, w, p)| {
            let q = self.camera.project(w[0], w[1]).unwrap_or([f64::INFINITY; 2]);
            (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)
        }).sum();
        Some((s / self.points.len() as f64).sqrt())
    }
}

pub fn wrap180(a: f64) -> f64 { let x = (a + 180.0).rem_euclid(360.0) - 180.0; if x == -180.0 { 180.0 } else { x } }

/// Twin angle (the export's `yaw`, and each fisheye `lens.views` bearing) → runtime yaw (compass, 0° = north,
/// clockwise): `wrap180(90 − b)`. The ONE conversion used for camera heading and virtual views alike.
pub fn bearing_to_yaw(b: f64) -> f64 { wrap180(90.0 - b) }

/// Kannala–Brandt fisheye intrinsics in the camera's own pixels, plus optional virtual views.
/// In the twin JSON `views` are `[[bearing°, pitch°], …]` in the same angle convention as the camera's `yaw`;
/// parsing converts them with [`bearing_to_yaw`], so `views` here hold runtime scene yaw (compass) and pitch,
/// ready for `vg360_pipeline::FisheyeRig::new` (which also picks the default when `None`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LensSpec { pub fx: f64, pub fy: f64, pub cx: f64, pub cy: f64, pub k: [f64; 4], pub views: Option<Vec<[f64; 2]>> }

fn lens(c: &Value, id: &str) -> Result<Option<LensSpec>, String> {
    let l = &c["lens"];
    if l.is_null() { return Ok(None); }
    let m = l["model"].as_str().unwrap_or("");
    if !matches!(m, "kannala-brandt" | "kb" | "fisheye") { return Err(format!("camera {id}: lens model {m:?} (supported: kannala-brandt)")); }
    let n = |k: &str| l[k].as_f64().ok_or(format!("camera {id}: lens.{k} missing"));
    let k = l["k"].as_array().ok_or(format!("camera {id}: lens.k missing"))?;
    if k.len() != 4 { return Err(format!("camera {id}: lens.k needs 4 coefficients")); }
    let mut kk = [0.0; 4];
    for (i, v) in k.iter().enumerate() { kk[i] = v.as_f64().ok_or(format!("camera {id}: lens.k[{i}] is not a number"))?; }
    let fx = n("fx")?;
    let fy = if l["fy"].is_null() { fx } else { l["fy"].as_f64().ok_or(format!("camera {id}: lens.fy is not a number"))? };
    let views = if l["views"].is_null() { None } else {
        let v = l["views"].as_array().ok_or(format!("camera {id}: lens.views is not an array"))?;
        if v.is_empty() { return Err(format!("camera {id}: lens.views is empty; omit it for defaults")); }
        Some(v.iter().enumerate().map(|(i, p)| {
            let bad = || format!("camera {id}: lens.views[{i}] must be [bearing, pitch] (two numbers)");
            let a = p.as_array().filter(|a| a.len() == 2).ok_or_else(bad)?;
            Ok([bearing_to_yaw(a[0].as_f64().ok_or_else(bad)?), a[1].as_f64().ok_or_else(bad)?])
        }).collect::<Result<Vec<_>, String>>()?)
    };
    Ok(Some(LensSpec { fx, fy, cx: n("cx")?, cy: n("cy")?, k: kk, views }))
}

fn pair(v: &Value) -> Option<[f64; 2]> { let a = v.as_array().filter(|a| a.len() == 2)?; Some([a[0].as_f64()?, a[1].as_f64()?]) }

/// Parse a twin export. Errors name the first problem.
pub fn cameras(twin: &Value) -> Result<Vec<TwinCamera>, String> {
    if twin["vg360_twin"].as_i64() != Some(1) { return Err("not a VG360 twin file (vg360_twin != 1)".into()); }
    let cams = twin["cameras"].as_array().ok_or("twin has no cameras")?;
    cams.iter().map(|c| {
        let id = c["id"].as_str().ok_or("camera without id")?.to_string();
        let num = |k: &str| c[k].as_f64().ok_or(format!("camera {id}: missing {k}"));
        let res = c["res"].as_str().ok_or(format!("camera {id}: missing res (export from the current Rig Twin)"))?;
        let (w, h) = res.split_once('x').and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?))).ok_or(format!("camera {id}: bad res {res}"))?;
        if w == 0 || h == 0 { return Err(format!("camera {id}: res {res} has a zero side")); }
        let lens = lens(c, &id)?;
        let (fov, surf) = (if lens.is_some() { c["fov"].as_f64().unwrap_or(90.0) } else { num("fov")? }, c["surf"].as_f64().unwrap_or(0.0));
        if lens.is_none() && !(10.0..=170.0).contains(&fov) { return Err(format!("camera {id}: fov {fov} outside 10..170 degrees")); }
        let pitch = num("pitch")?;
        if !(-90.0..=90.0).contains(&pitch) { return Err(format!("camera {id}: pitch {pitch} outside -90..90 degrees")); }
        let f = (w as f64 / 2.0) / (fov.to_radians() / 2.0).tan();
        // roll is always 0: the twin has no roll field (a fisheye is levelled by its upright virtual views)
        let camera = Camera { f_px: f, cx: w as f64 / 2.0, cy: h as f64 / 2.0, height_m: num("h")? - surf, pitch_deg: pitch,
                              yaw_deg: bearing_to_yaw(num("yaw")?), roll_deg: 0.0, x0: num("x")?, y0: num("y")? };
        if camera.height_m <= 0.0 { return Err(format!("camera {id}: mounted below its deck")); }
        let points = match &c["calibration_points"] {
            Value::Null => vec![],
            Value::Array(a) => a.iter().enumerate().map(|(i, p)| {
                let bad = |what: &str| format!("camera {id}: calibration_points[{i}]: {what} must be two numbers");
                Ok((p["name"].as_str().unwrap_or("").to_string(), pair(&p["world"]).ok_or_else(|| bad("world"))?,
                    pair(&p["predicted_px"]).ok_or_else(|| bad("predicted_px"))?))
            }).collect::<Result<Vec<_>, String>>()?,
            _ => return Err(format!("camera {id}: calibration_points is not an array")),
        };
        Ok(TwinCamera { id, camera, lens, width: w, height: h, surf, points })
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

    #[test]
    fn fisheye_lens_in_a_twin_camera() {
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/twin/rig_twin.json")).unwrap()).unwrap();
        v["cameras"][0]["lens"] = serde_json::json!({ "model": "kannala-brandt", "fx": 892, "cx": 1440, "cy": 1440, "k": [0.02, -0.01, 0.002, -0.0005],
                                                      "views": [[0, 35], [120, 35], [240, 35]] });
        let cams = super::cameras(&v).unwrap();
        let l = cams[0].lens.as_ref().unwrap();
        assert_eq!((l.fx, l.fy, l.k[3]), (892.0, 892.0, -0.0005));
        assert_eq!(l.views.as_ref().unwrap().len(), 3);
        assert!(cams[1].lens.is_none());
        v["cameras"][0]["lens"]["model"] = "mei".into();
        assert!(super::cameras(&v).is_err());
    }

    fn sample() -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/twin/rig_twin.json")).unwrap()).unwrap()
    }
    fn fish() -> serde_json::Value {
        let mut v = sample();
        v["cameras"][0]["lens"] = serde_json::json!({ "model": "kannala-brandt", "fx": 892, "cx": 1440, "cy": 1440, "k": [0.02, -0.01, 0.002, -0.0005] });
        v
    }
    fn err(v: &serde_json::Value) -> String { super::cameras(v).unwrap_err() }

    #[test]
    fn fisheye_rms_is_not_computed() {
        let c = super::cameras(&fish()).unwrap();
        assert!(c[0].lens.is_some() && !c[0].points.is_empty());
        assert!(c[0].rms_px().is_none());
        assert!(c[1].rms_px().is_some());
    }

    #[test]
    fn view_bearings_use_the_camera_yaw_conversion() {
        let mut v = fish();
        v["cameras"][0]["lens"]["views"] = serde_json::json!([[0, 35], [90, 35], [250, 20]]);
        let c = &super::cameras(&v).unwrap()[0];
        for (b, view) in [0.0, 90.0, 250.0].iter().zip(c.lens.as_ref().unwrap().views.as_ref().unwrap()) {
            assert_eq!(view[0], super::bearing_to_yaw(*b));
            // the deck point straight ahead of the virtual view lies along the converted compass direction
            let cam = crate::camera::Camera { yaw_deg: view[0], pitch_deg: view[1], x0: 0.0, y0: 0.0, ..c.camera };
            let g = cam.ground(cam.cx, cam.cy).unwrap();
            let compass = g[0].atan2(g[1]).to_degrees();
            assert!(super::wrap180(compass - super::wrap180(90.0 - b)).abs() < 1e-6, "{b}: {compass}");
        }
        // a view with the camera's own heading looks where the camera looks
        let psi = v["cameras"][0]["yaw"].as_f64().unwrap();
        v["cameras"][0]["lens"]["views"] = serde_json::json!([[psi, 35]]);
        let c = &super::cameras(&v).unwrap()[0];
        assert_eq!(c.lens.as_ref().unwrap().views.as_ref().unwrap()[0][0], c.camera.yaw_deg);
    }

    #[test]
    fn strict_lens_parsing() {
        let mut v = fish();
        v["cameras"][0]["lens"]["views"] = serde_json::json!([]);
        assert!(err(&v).contains("lens.views is empty; omit it for defaults"));
        v["cameras"][0]["lens"]["views"] = serde_json::json!([[0, 35], [120]]);
        assert!(err(&v).contains("views[1]"));
        v["cameras"][0]["lens"]["views"] = serde_json::json!([["a", 35]]);
        assert!(err(&v).contains("views[0]"));
        v["cameras"][0]["lens"]["views"] = serde_json::json!([[0, 35, 1]]);
        assert!(err(&v).contains("views[0]"));
        v["cameras"][0]["lens"]["views"] = serde_json::Value::Null;
        assert!(super::cameras(&v).is_ok());
        v["cameras"][0]["lens"]["k"] = serde_json::json!([0.02, "x", 0.002, 0.0]);
        assert!(err(&v).contains("lens.k[1]"));
        let mut v = fish();
        v["cameras"][0]["lens"]["fy"] = "big".into();
        assert!(err(&v).contains("lens.fy"));
        v["cameras"][0]["lens"]["fy"] = 900.into();
        assert_eq!(super::cameras(&v).unwrap()[0].lens.as_ref().unwrap().fy, 900.0);
    }

    #[test]
    fn strict_camera_parsing() {
        let mut v = sample();
        v["cameras"][1]["calibration_points"][1] = serde_json::json!({ "name": "p", "world": [1, "x"], "predicted_px": [1, 2] });
        assert!(err(&v).contains("calibration_points[1]"));
        let mut v = sample();
        v["cameras"][1]["calibration_points"][0]["predicted_px"] = serde_json::json!([1]);
        assert!(err(&v).contains("calibration_points[0]"));
        for (k, bad, needle) in [("fov", 5, "fov"), ("fov", 175, "fov"), ("pitch", 95, "pitch"), ("pitch", -91, "pitch")] {
            let mut v = sample();
            v["cameras"][1][k] = bad.into();
            assert!(err(&v).contains(needle), "{k}={bad}");
        }
        for res in ["0x720", "1280x0"] {
            let mut v = sample();
            v["cameras"][1]["res"] = res.into();
            assert!(err(&v).contains("zero side"), "{res}");
        }
        // a fisheye camera's fov is unused, so it is not range-checked
        let mut v = fish();
        v["cameras"][0]["fov"] = 185.into();
        assert!(super::cameras(&v).is_ok());
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
