//! POST /api/calibrate: validation and the solve behind it. The HTTP handler in `api.rs` is a thin call to [`run`].
//! Input checks run first (a 400 with a field-specific message); the solve itself is [`solve`], the one step to
//! swap when a different solver (e.g. `vg360_geom::autocal::vp_calibrate`) replaces the vanishing-point one.
use serde::Deserialize;
use serde_json::{Value, json};
use topcoat::router::StatusCode;

use vg360_geom::autocal::{self, VpCalib};
use vg360_geom::camera::Camera;

#[derive(Deserialize)]
pub struct CalibReq {
    pub width: f64,
    pub height: f64,
    /// image segments [x1, y1, x2, y2] of one family of parallel horizontal deck edges …
    pub lines_a: Vec<[f64; 4]>,
    /// … and of a second family at right angles to the first
    pub lines_b: Vec<[f64; 4]>,
    /// optional boxes [x1, y1, x2, y2] of people standing upright → mount height
    #[serde(default)] pub people: Vec<[f64; 4]>,
    #[serde(default)] pub person_m: Option<f64>,
}

/// Input checks: sizes, person height, finite and non-degenerate segments, well-formed boxes inside the frame.
pub fn validate(r: &CalibReq) -> Result<(), String> {
    let dim = |name: &str, v: f64| {
        if v.is_finite() && (100.0..=16384.0).contains(&v) { Ok(()) } else { Err(format!("{name} must be a number in 100..=16384 (got {v})")) }
    };
    dim("width", r.width)?;
    dim("height", r.height)?;
    if let Some(p) = r.person_m {
        if !(p.is_finite() && (1.0..=2.5).contains(&p)) { return Err(format!("person_m must be a height in metres, 1.0..=2.5 (got {p})")); }
    }
    for (name, fam) in [("lines_a", &r.lines_a), ("lines_b", &r.lines_b)] {
        for (i, s) in fam.iter().enumerate() {
            if !s.iter().all(|v| v.is_finite()) { return Err(format!("{name}[{i}] has a non-finite coordinate")); }
            if (s[2] - s[0]).hypot(s[3] - s[1]) < 1.0 { return Err(format!("{name}[{i}] is shorter than 1 px")); }
        }
    }
    for (i, b) in r.people.iter().enumerate() {
        if !b.iter().all(|v| v.is_finite()) { return Err(format!("people[{i}] has a non-finite coordinate")); }
        if b[2] <= b[0] || b[3] <= b[1] { return Err(format!("people[{i}] box is inverted: need x2 > x1 and y2 > y1")); }
        let (w, h) = (r.width, r.height);
        if b[0] < -0.05 * w || b[2] > 1.05 * w || b[1] < -0.05 * h || b[3] > 1.05 * h {
            return Err(format!("people[{i}] box lies outside the frame"));
        }
    }
    Ok(())
}

/// Plausibility of the solved camera: a sane field of view and a downward pitch.
pub fn check_range(hfov_deg: f64, pitch_deg: f64) -> Result<(), String> {
    if (10.0..=170.0).contains(&hfov_deg) && (-5.0..=89.0).contains(&pitch_deg) { return Ok(()); }
    Err(format!("result outside plausible range (hfov {hfov_deg:.1}°, pitch {pitch_deg:.1}°): check that the two families are at right angles on the deck"))
}

/// Result of the solve step.
pub struct Solved {
    pub vps: [[f64; 2]; 2],
    pub k: VpCalib,
}

/// The solve: two vanishing points from the two line families, then focal length, pitch and roll.
/// Errors are the full `ok:false` response body.
pub fn solve(r: &CalibReq) -> Result<Solved, Value> {
    let seg = |l: &[[f64; 4]]| l.iter().map(|s| ([s[0], s[1]], [s[2], s[3]])).collect::<Vec<_>>();
    let vp = |name: &str, fam: &[[f64; 4]]| {
        autocal::vanishing_point(&seg(fam)).ok_or_else(|| {
            let msg = if fam.len() < 2 {
                format!("{name} needs ≥ 2 segments to find a vanishing point")
            } else {
                "lines within a family must converge (not be parallel in the image); pick edges that recede into the distance".to_string()
            };
            json!({ "ok": false, "error": msg })
        })
    };
    let v1 = vp("lines_a", &r.lines_a)?;
    let v2 = vp("lines_b", &r.lines_b)?;
    // checked solve: rejects near-infinite vanishing points, f² ≤ 0 and implausible focal lengths
    let k = autocal::vp_calibrate(&seg(&r.lines_a), &seg(&r.lines_b), r.width, r.height)
        .map_err(|e| json!({ "ok": false, "vanishing_points": [v1, v2], "error": e.to_string() }))?;
    Ok(Solved { vps: [v1, v2], k })
}

/// Validate, solve, range-check and build the response body.
pub fn compute(r: &CalibReq) -> Result<Value, Value> {
    validate(r).map_err(|e| json!({ "ok": false, "error": e }))?;
    let Solved { vps, k } = solve(r)?;
    check_range(k.hfov_deg, k.pitch_deg).map_err(|e| json!({ "ok": false, "vanishing_points": vps, "error": e }))?;
    let cam = Camera { f_px: k.f_px, cx: r.width / 2.0, cy: r.height / 2.0, height_m: 1.0, pitch_deg: k.pitch_deg,
                       yaw_deg: 0.0, roll_deg: k.roll_deg, x0: 0.0, y0: 0.0 };
    let h = (!r.people.is_empty()).then(|| autocal::height_from_people(&cam, &r.people, r.person_m.unwrap_or(1.70))).flatten();

    let mut warnings: Vec<String> = Vec::new();
    if k.roll_deg.abs() > 2.0 { warnings.push(format!("roll above 2° ({:.1}°): level the camera or calibrate from deck points", k.roll_deg)); }
    if !r.people.is_empty() && r.people.len() < 3 { warnings.push("fewer than 3 people: height is approximate".into()); }
    if !r.people.is_empty() && h.is_none() { warnings.push("no height could be estimated from the people boxes".into()); }

    Ok(json!({ "ok": true, "vanishing_points": vps, "f_px": k.f_px, "hfov_deg": k.hfov_deg, "pitch_deg": k.pitch_deg,
        "roll_deg": k.roll_deg, "height_m": h,
        "twin": { "fov": (k.hfov_deg * 10.0).round() / 10.0, "pitch": (k.pitch_deg * 10.0).round() / 10.0, "h_above_deck": h.map(|h| (h * 100.0).round() / 100.0) },
        "notes": [ "pitch/fov/height assume square pixels and a centred principal point",
                   if k.roll_deg.abs() > 2.0 { "roll above 2°: level the camera or calibrate from deck points (the twin assumes zero roll)" } else { "roll ≈ 0" },
                   if h.is_none() { "no height: add boxes of standing people, or set VG360_CAM_<ID>_AUTO_HEIGHT=1 to estimate it live" } else { "height from the median head height of the people boxes (1.70 m assumed)" } ],
        "warnings": warnings }))
}

/// HTTP status and JSON body for a calibration request: 200 with the result, or 400 with `ok:false` and a reason.
pub fn run(r: &CalibReq) -> (StatusCode, Value) {
    match compute(r) {
        Ok(v) => (StatusCode::OK, v),
        Err(v) => (StatusCode::BAD_REQUEST, v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> CalibReq {
        CalibReq { width: 1280.0, height: 720.0, lines_a: vec![], lines_b: vec![], people: vec![], person_m: None }
    }

    #[test]
    fn accepts_plain_input() {
        assert!(validate(&req()).is_ok());
    }

    #[test]
    fn rejects_zero_width() {
        let mut r = req();
        r.width = 0.0;
        assert!(validate(&r).unwrap_err().contains("width"));
    }

    #[test]
    fn rejects_nan_height() {
        let mut r = req();
        r.height = f64::NAN;
        assert!(validate(&r).unwrap_err().contains("height"));
    }

    #[test]
    fn rejects_out_of_range_person_height() {
        let mut r = req();
        r.person_m = Some(0.0);
        assert!(validate(&r).unwrap_err().contains("person_m"));
        r.person_m = Some(f64::INFINITY);
        assert!(validate(&r).is_err());
        r.person_m = Some(1.75);
        assert!(validate(&r).is_ok());
    }

    #[test]
    fn rejects_inverted_person_box() {
        let mut r = req();
        r.people = vec![[100.0, 200.0, 90.0, 300.0]];
        assert!(validate(&r).unwrap_err().contains("inverted"));
        r.people = vec![[100.0, 300.0, 120.0, 200.0]];
        assert!(validate(&r).unwrap_err().contains("inverted"));
    }

    #[test]
    fn rejects_box_outside_frame() {
        let mut r = req();
        r.people = vec![[100.0, 100.0, 1400.0, 300.0]];
        assert!(validate(&r).unwrap_err().contains("outside the frame"));
        r.people = vec![[100.0, 100.0, 140.0, 300.0]];
        assert!(validate(&r).is_ok());
    }

    #[test]
    fn rejects_zero_length_segment() {
        let mut r = req();
        r.lines_a = vec![[10.0, 10.0, 10.0, 10.0]];
        assert!(validate(&r).unwrap_err().contains("shorter than 1 px"));
    }

    #[test]
    fn rejects_non_finite_segment() {
        let mut r = req();
        r.lines_b = vec![[0.0, f64::NAN, 50.0, 50.0]];
        assert!(validate(&r).unwrap_err().contains("non-finite"));
    }

    #[test]
    fn range_check_bounds() {
        assert!(check_range(70.0, 30.0).is_ok());
        assert!(check_range(9.0, 30.0).is_err());
        assert!(check_range(171.0, 30.0).is_err());
        assert!(check_range(70.0, -6.0).is_err());
        assert!(check_range(70.0, 90.0).is_err());
        assert!(check_range(70.0, -5.0).is_ok());
    }

    #[test]
    fn parallel_lines_get_convergence_message() {
        let mut r = req();
        r.lines_a = vec![[0.0, 0.0, 100.0, 0.0], [0.0, 50.0, 100.0, 50.0]];
        r.lines_b = vec![[0.0, 0.0, 0.0, 100.0], [50.0, 0.0, 50.0, 100.0]];
        let e = compute(&r).unwrap_err();
        assert_eq!(e["ok"], false);
        assert!(e["error"].as_str().unwrap().contains("must converge"), "{e}");
        let (status, _) = run(&r);
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn too_few_segments_gets_count_message() {
        let mut r = req();
        r.lines_a = vec![[0.0, 0.0, 100.0, 0.0]];
        let e = compute(&r).unwrap_err();
        assert!(e["error"].as_str().unwrap().contains("≥ 2 segments"), "{e}");
    }
}
