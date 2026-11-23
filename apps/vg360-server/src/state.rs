//! Site state. In this prototype it is loaded from files the notebook wrote
//! (`testdata/rule_studio_export.json`, `data/notebook_report.json`) plus a sample camera
//! list; in production the inference workers update it live.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::RwLock;

use serde_json::Value;
use vg360_evidence::Vesdb;
use vg360_geom::twin::{self, TwinCamera};
use vg360_rules::Deployed;

use serde::Serialize;
use vg360_geom::ledger::{Entry, Ledger, Verify};
use vg360_geom::rules::RuleRecord;
use vg360_guard::aligner::Aligner;
use vg360_pipeline::{CompLimits, Mode, compensation_mode};
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
    /// analytics mode decided from the guard measurement (aligned / compensated / paused)
    pub mode: Mode,
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
    /// IOGP rules compiled (and, in production, approved) in iogp-editor
    #[serde(skip)]
    pub iogp: IogpRules,
    /// the Rig Twin applied to this site
    #[serde(skip)]
    pub twin: RwLock<Option<TwinState>>,
    #[serde(skip)]
    pub state_dir: PathBuf,
    /// tamper-evident evidence log (VG360_VESDB_URL / tokens from the environment)
    #[serde(skip)]
    pub vesdb: Option<Vesdb>,
}

#[derive(Clone, Debug, Serialize)]
pub struct IogpRule {
    pub id: String,
    pub text: String,
    pub readback: Vec<String>,
    pub fol: String,
    pub alert: String,
    pub lsr: u8,
    pub severity: Option<String>,
    pub tier: String,
    pub floor: f64,
    pub cm: Option<[u64; 4]>,
    pub mcc: Option<f64>,
    pub validated_on: String,
    pub approved_by: Option<String>,
    /// approved and MCC ≥ floor: may sound
    pub armed: bool,
}

pub struct IogpRules { pub config_path: String, pub sha256: String, pub rules: Vec<IogpRule>, pub draft: bool }

impl IogpRules {
    pub fn get(&self, id: &str) -> Option<&IogpRule> { self.rules.iter().find(|r| r.id.eq_ignore_ascii_case(id)) }
    pub fn armed(&self) -> usize { self.rules.iter().filter(|r| r.armed).count() }
}

#[derive(Clone)]
pub struct TwinState { pub json: Value, pub applied_at: String, pub source: String, pub cameras: Vec<TwinCamera>, pub zones: usize, pub rules_on: usize }

impl TwinState {
    pub fn from_json(json: Value, applied_at: String, source: String) -> Result<Self, String> {
        let cameras = twin::cameras(&json)?;
        let zones = json["zones"].as_array().map_or(0, |z| z.len());
        let rules_on = json["zones"].as_array().into_iter().flatten()
            .flat_map(|z| z["rules"].as_array().into_iter().flatten()).filter(|r| r["on"] != false).count();
        Ok(Self { json, applied_at, source, cameras, zones, rules_on })
    }
}

fn now_rfc3339() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let (d, r) = (s / 86400, s % 86400);
    // civil date from days since 1970-01-01 (Howard Hinnant)
    let z = d as i64 + 719468; let era = z.div_euclid(146097); let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; let y = yoe + era * 400; let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153; let dd = doy - (153 * mp + 2) / 5 + 1; let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y + (m <= 2) as i64, m, dd, r / 3600, r % 3600 / 60, r % 60)
}

impl Site {
    /// Apply a twin from the Rig Twin editor: validate, persist, and record each camera model as evidence.
    pub fn apply_twin(&self, json: Value) -> Result<Value, String> {
        let t = TwinState::from_json(json, now_rfc3339(), "applied from /twin".into())?;
        std::fs::create_dir_all(&self.state_dir).map_err(|e| e.to_string())?;
        std::fs::write(self.state_dir.join("twin.json"), serde_json::to_string_pretty(&t.json).unwrap()).map_err(|e| e.to_string())?;
        let mut evidence = 0;
        if let Some(v) = &self.vesdb {
            for c in &t.cameras {
                let data = serde_json::json!({ "source": "rig-twin", "camera": c.camera, "resolution": [c.width, c.height], "deck_surface_m": c.surf,
                    "calibration_points": c.points.len(), "twin_rms_px": c.rms_px(), "twin_sha256": vg360_evidence::sha256_hex(t.json.to_string().as_bytes()) });
                if v.camera_event(&c.id, "calibration-set", data).is_ok() { evidence += 1; }
            }
        }
        let out = serde_json::json!({ "ok": true, "cameras": t.cameras.len(), "zones": t.zones, "rules_on": t.rules_on, "evidence_events": evidence,
            "applied_at": t.applied_at });
        *self.twin.write().unwrap() = Some(t);
        Ok(out)
    }
}

#[derive(serde::Deserialize)]
struct Export { rules: BTreeMap<String, RuleRecord>, ledger: Vec<Entry>, source: String }

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Rules from iogp-editor's export, their MCC on the simulated site and their approval state.
fn load_iogp(root: &str, legacy: &BTreeMap<String, RuleRecord>) -> IogpRules {
    let path = std::env::var("VG360_RULES_CONFIG").unwrap_or_else(|_| format!("{root}/../../config/vg360_rules.draft.json"));
    let json = read(&path);
    let dep = Deployed::load(&json).unwrap_or_else(|e| panic!("{path}: {e}"));
    let val: Value = serde_json::from_str(&read(&format!("{root}/../../config/vg360_rules.validation.json"))).unwrap_or_default();
    let raw: Value = serde_json::from_str(&json).unwrap();
    let rules = raw["rules"].as_array().unwrap().iter().map(|r| {
        let id = r["id"].as_str().unwrap_or("").to_string();
        let sev = r["alert"]["severity"].as_str().map(String::from);
        let (tier, floor) = vg360_rules::tier(sev.as_deref());
        let v = &val["rules"][&id];
        // PPE needs imagery: it is scored on the PPE crop test set (notebook §7)
        let (cm, validated_on) = match v["cm"].as_array() {
            Some(a) => (Some([0, 1, 2, 3].map(|i| a[i].as_u64().unwrap_or(0))), val["validated_on"].as_str().unwrap_or("").to_string()),
            None => (legacy.get(&id.to_lowercase()).and_then(|l| l.cm), legacy.get(&id.to_lowercase()).map_or(String::new(), |l| l.validated_on.clone())),
        };
        let mcc = cm.and_then(vg360_rules::validate::mcc);
        let approved_by = r["version"]["approved_by"].as_str().map(String::from);
        let armed = approved_by.is_some() && mcc.is_some_and(|m| m >= floor);
        let alert = dep.rule(&id).map(|x| x.alert.rule.clone()).unwrap_or_default();
        IogpRule { lsr: dep.rule(&id).map_or(0, |x| x.alert.lsr), id, text: r["text"].as_str().unwrap_or("").into(),
                   readback: serde_json::from_value(r["readback"].clone()).unwrap_or_default(), fol: r["fol"].as_str().unwrap_or("").into(),
                   alert, severity: sev, tier: tier.into(), floor, cm, mcc, validated_on, approved_by, armed }
    }).collect::<Vec<_>>();
    let draft = rules.iter().all(|r| r.approved_by.is_none());
    IogpRules { config_path: path.rsplit('/').next().unwrap_or("").into(), sha256: dep.sha256, rules, draft }
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
            calib_rms_m: 0.007, calib_method: "8 deck points · DLT + LM", guard: Status::Ok, mode: Mode::Aligned, pose: home,
            ref_quad: None, inliers: 212, fps: 5.0, infer_ms: 41.0, rules, plan_xy: xy, plan_dir: dir,
        };
        let cams = vec![
            cam("CAM-01", "Drill floor · NW mast", "Drill floor", "/samples/cam1.jpg", 12.0, 35.0, 70.0,
                &["RED", "EDGE", "FALL", "PPE"], [-18.0, 22.0], 135.0),
            cam("CAM-02", "Pipe deck · catwalk", "Pipe deck", "/samples/cam2.jpg", 10.0, 30.0, 70.0,
                &["LIFT", "PPE", "FALL"], [24.0, 18.0], 225.0),
            cam("CAM-03", "Pump skids · south", "Mud pumps", "/samples/cam3.jpg", 9.0, 32.0, 80.0,
                &["RED", "PPE"], [-38.0, -26.0], 50.0),
            Cam { infer_ms: 44.0, ..cam("CAM-04", "Crane 1 · lift zone", "Crane", "/samples/cam4.jpg", 14.0, 40.0, 60.0,
                &["LIFT", "PPE"], [40.0, -22.0], 300.0) },
            Cam {
                reference: Some("/samples/cam5_reference.jpg"), guard: status, pose: m.pose, ref_quad: m.ok.then_some(m.boxq),
                inliers: m.inliers, mode: compensation_mode(&m, &CompLimits::default()),
                calib_method: "8 deck points · DLT + LM (through measured shift)",
                ..cam("CAM-05", "Mud tanks · east", "Mud tanks", "/samples/cam5_moved.jpg", 10.0, 25.0, 70.0,
                    &["EDGE", "RED"], [42.0, 4.0], 265.0)
            },
            Cam {
                lens: Lens::Fisheye, hfov_deg: 190.0, pitch_deg: 90.0, calib_rms_m: 0.021, infer_ms: 63.0,
                calib_method: "Kannala–Brandt k1..k4 · 3 virtual views",
                ..cam("CAM-06", "Wellhead · fisheye", "Cellar / wellhead", "/samples/cam6_fisheye.jpg", 8.0, 90.0, 190.0,
                    &["RED", "FALL"], [0.0, 0.0], 0.0)
            },
        ];
        let iogp = load_iogp(root, &x.rules);
        let state_dir = std::env::var("VG360_STATE_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(format!("{root}/state")));
        let twin = std::fs::read_to_string(state_dir.join("twin.json")).ok().and_then(|t| serde_json::from_str(&t).ok())
            .and_then(|j| TwinState::from_json(j, "saved".into(), "applied from /twin".into()).ok())
            .or_else(|| serde_json::from_str(&read(&format!("{root}/../../testdata/twin/rig_twin.json"))).ok()
                .and_then(|j| TwinState::from_json(j, "sample".into(), "sample rig twin (testdata)".into()).ok()));
        Site { name: "Desert Rig 7 (sample data)", cams, rules: x.rules, ledger, ledger_check, ledger_source: x.source, report,
               iogp, twin: RwLock::new(twin), state_dir, vesdb: Vesdb::from_env() }
    }

    pub fn cam(&self, id: &str) -> Option<&Cam> { self.cams.iter().find(|c| c.id.eq_ignore_ascii_case(id)) }

    pub fn rule_name(&self, id: &str) -> String {
        self.iogp.get(id).map(|r| r.alert.clone()).or_else(|| self.rules.get(id).map(|r| r.name.clone())).unwrap_or_else(|| id.to_string())
    }
}
