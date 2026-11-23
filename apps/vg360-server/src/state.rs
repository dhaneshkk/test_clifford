//! Site state. In this prototype it is loaded from files the notebook wrote
//! (`testdata/rule_studio_export.json`, `data/notebook_report.json`) plus a sample camera
//! list; in production the inference workers update it live.
use std::collections::BTreeMap;

use serde::Serialize;
use vg360_geom::ledger::{Entry, Ledger, Verify};
use vg360_geom::rules::RuleRecord;
use vg360_guard::aligner::Aligner;
use vg360_guard::policy::{self, Guard, Observation, Pose, Status};
use vg360_vision::img::Bgr;

#[derive(Clone, Copy, Serialize)]
pub enum Lens { Pinhole, Fisheye }

#[derive(Clone, Serialize)]
pub struct Cam {
    pub id: &'static str,
    pub name: &'static str,
    pub area: &'static str,
    pub frame: &'static str,
    pub reference: Option<&'static str>,
    pub lens: Lens,
    pub height_m: f64,
    pub pitch_deg: f64,
    pub hfov_deg: f64,
    pub calib_rms_m: f64,
    pub calib_method: &'static str,
    pub guard: Status,
    pub pose: Pose,
    /// reference-frame corners in the live image (camera-guard "correct view is here" box)
    pub ref_quad: Option<[[f64; 2]; 4]>,
    pub inliers: i32,
    pub fps: f64,
    pub infer_ms: f64,
    pub rules: &'static [&'static str],
    /// world position on the site plan (m) and look direction (deg, 0 = north, clockwise)
    pub plan_xy: [f64; 2],
    pub plan_dir: f64,
}

impl Cam {
    pub fn guidance(&self) -> Vec<String> { policy::guidance(&self.pose) }
    pub fn env_var(&self) -> String { format!("VG360_CAM_{}_URL", self.id.replace('-', "_")) }
}

#[derive(Serialize)]
pub struct Site {
    pub name: &'static str,
    pub cams: Vec<Cam>,
    pub rules: BTreeMap<String, RuleRecord>,
    pub ledger: Ledger,
    pub ledger_check: Verify,
    pub ledger_source: String,
    pub report: serde_json::Value,
}

#[derive(serde::Deserialize)]
struct Export { rules: BTreeMap<String, RuleRecord>, ledger: Vec<Entry>, source: String }

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

impl Site {
    pub fn load(root: &str) -> Self {
        let x: Export = serde_json::from_str(&read(&format!("{root}/../../testdata/rule_studio_export.json"))).unwrap();
        let report: serde_json::Value = serde_json::from_str(&read(&format!("{root}/data/notebook_report.json"))).unwrap();
        // camera guard, measured for real at start-up (pure Rust, no OpenCV): reference frame vs the bumped frame
        let mut aligner = Aligner::new();
        let reference = Bgr::load(format!("{root}/../../testdata/frames/ref.png")).expect("reference frame");
        let ref_pts = aligner.set_reference(&reference, &[], None);
        let t = std::time::Instant::now();
        let m = aligner.measure(&Bgr::load(format!("{root}/public/samples/cam5_moved.jpg")).expect("live frame"));
        eprintln!("camera guard CAM-05: {ref_pts} reference points, {} inliers, {:?} in {:?}", m.inliers, m.pose, t.elapsed());
        let mut guard = Guard::default();
        let status = (0..20).map(|i| guard.step(i as f64 * 0.25, if m.ok { Observation::Fix(m.pose) } else if m.blocked { Observation::Blocked } else { Observation::NoFix })).last().unwrap();
        let ledger = Ledger { entries: x.ledger };
        let ledger_check = ledger.verify();
        let home = Pose { dx: 0.0, dy: 0.0, angle: 0.0, scale: 1.0, corner: 0.0 };
        let cam = |id, name, area, frame, h, p, f, rules, xy, dir| Cam {
            id, name, area, frame, reference: None, lens: Lens::Pinhole, height_m: h, pitch_deg: p, hfov_deg: f,
            calib_rms_m: 0.007, calib_method: "8 deck points · solvePnP", guard: Status::Ok, pose: home,
            ref_quad: None, inliers: 212, fps: 5.0, infer_ms: 41.0, rules, plan_xy: xy, plan_dir: dir,
        };
        let cams = vec![
            cam("CAM-01", "Drill floor · NW mast", "Drill floor", "/samples/cam1.jpg", 12.0, 35.0, 70.0,
                &["red", "edge", "fall", "ppe"], [-18.0, 22.0], 135.0),
            cam("CAM-02", "Pipe deck · catwalk", "Pipe deck", "/samples/cam2.jpg", 10.0, 30.0, 70.0,
                &["lift", "ppe", "fall"], [24.0, 18.0], 225.0),
            cam("CAM-03", "Pump skids · south", "Mud pumps", "/samples/cam3.jpg", 9.0, 32.0, 80.0,
                &["red", "ppe"], [-38.0, -26.0], 50.0),
            Cam { infer_ms: 44.0, ..cam("CAM-04", "Crane 1 · lift zone", "Crane", "/samples/cam4.jpg", 14.0, 40.0, 60.0,
                &["lift", "ppe"], [40.0, -22.0], 300.0) },
            Cam {
                reference: Some("/samples/cam5_reference.jpg"), guard: status, pose: m.pose, ref_quad: m.ok.then_some(m.boxq),
                inliers: m.inliers, calib_method: "suspended — camera moved",
                ..cam("CAM-05", "Mud tanks · east", "Mud tanks", "/samples/cam5_moved.jpg", 10.0, 25.0, 70.0,
                    &["edge", "red"], [42.0, 4.0], 265.0)
            },
            Cam {
                lens: Lens::Fisheye, hfov_deg: 190.0, pitch_deg: 90.0, calib_rms_m: 0.021, infer_ms: 63.0,
                calib_method: "Kannala–Brandt k1..k4 · 3 virtual views",
                ..cam("CAM-06", "Wellhead · fisheye", "Cellar / wellhead", "/samples/cam6_fisheye.jpg", 8.0, 90.0, 190.0,
                    &["red", "fall"], [0.0, 0.0], 0.0)
            },
        ];
        Site { name: "Desert Rig 7 (sample data)", cams, rules: x.rules, ledger, ledger_check, ledger_source: x.source, report }
    }

    pub fn cam(&self, id: &str) -> Option<&Cam> { self.cams.iter().find(|c| c.id.eq_ignore_ascii_case(id)) }

    pub fn rule_name(&self, id: &str) -> String {
        self.rules.get(id).map_or_else(|| id.to_string(), |r| r.name.clone())
    }
}
