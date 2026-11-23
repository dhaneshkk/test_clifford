//! VisionGuard360 alert rules on **iogp-runtime**.
//!
//! Rules are written in controlled English and approved in iogp-editor (`dhaneshkk/test1_logpeditor`),
//! which exports `config.json`. At inference time each camera frame's scene state — tracks, zone
//! clearances, fall state, PPE, site signals — is turned into IOGP objects and facts:
//!
//! | scene state | IOGP |
//! |---|---|
//! | track `7` on the deck | object `7` class `person`; fact `in(7, Deck)` for the site walkway |
//! | clearance < 0 to a mapped zone | fact `in(7, Red-Zone)` |
//! | clearance < `near_m` to a mapped zone | fact `near(7, Deck-Edge)` |
//! | crane load suspended (load cell) | object `Load-1` class `suspended_load` |
//! | inside the load's drop volume | fact `under(7, Load-1)` |
//! | pipe handling active (drilling control system) | signal `Pipe-Handling`; `have(Red-Zone, Pipe-Handling)` |
//! | fall pose held | object `7` also class `fall_pose` |
//! | hard hat seen (P ≥ 0.5) or not classifiable | object `7/hardhat`, fact `wear(7, 7/hardhat)` |
//!
//! PPE follows the previous engine: only a confident "no hard hat" (P < 0.5) counts as not wearing;
//! a head too small to classify is not an alarm. The runtime then applies evidence smoothing (grace),
//! exact durations and the raise-once / cool-down alert machine.
use std::collections::BTreeSet;

use iogp_runtime::fast::Kind;
pub use iogp_runtime::{AlertEvent, Engine, RuleSet, load_config};
pub mod validate;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// How one scene zone appears in the rules.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZoneMap {
    /// zone name in the scene (vg360-scene)
    pub scene_zone: String,
    /// named zone instance in the catalog (e.g. "Red-Zone")
    pub instance: String,
    /// zone type noun in the catalog (e.g. "exclusion-zone")
    pub zone_type: String,
    /// emit `near` when the clearance is below this many metres
    #[serde(default)] pub near_m: Option<f64>,
    /// site signals this zone carries while they are active (e.g. "pipe-handling")
    #[serde(default)] pub signals: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SiteMap {
    pub zones: Vec<ZoneMap>,
    /// scene zone that is the drop volume of the suspended load (gives `under` facts)
    #[serde(default)] pub load_zone: Option<String>,
    /// named walkway instance covering the whole deck (e.g. "Deck")
    #[serde(default)] pub walkway: Option<String>,
    /// `near` distance for zones without their own `near_m`: taken from the compiled rules
    /// (the catalog's `near` metres, shown in each rule's readback)
    #[serde(default = "three")] pub near_default_m: f64,
}

fn three() -> f64 { 3.0 }

impl SiteMap {
    /// The rig demo / simulated site.
    pub fn rig_default() -> Self {
        SiteMap {
            zones: vec![
                ZoneMap { scene_zone: "pump_skid_keepout".into(), instance: "Red-Zone".into(), zone_type: "exclusion-zone".into(), near_m: None, signals: vec!["pipe-handling".into()] },
                ZoneMap { scene_zone: "rig_edge".into(), instance: "Deck-Edge".into(), zone_type: "height-zone".into(), near_m: None, signals: vec![] },
            ],
            load_zone: Some("suspended_load".into()),
            walkway: Some("Deck".into()),
            near_default_m: 3.0,
        }
    }
}

/// Live site signals (crane load cell, drilling control system, permit-to-work …).
#[derive(Clone, Debug, Default)]
pub struct Signals { pub suspended_load: bool, pub active: Vec<String> }

/// What the rules see of one person in one frame.
#[derive(Clone, Debug)]
pub struct Person<'a> { pub id: String, pub clear: &'a [(String, f64)], pub down: bool, pub p_helmet: Option<f64> }

/// Feed one frame to the engine and return the raised / cleared events.
pub fn step(eng: &mut Engine, site: &SiteMap, t: f64, camera: Option<&str>, sig: &Signals, people: &[Person]) -> Vec<AlertEvent> {
    eng.begin(t, camera);
    for z in &site.zones { eng.object(&z.instance, Kind::Zone, &z.zone_type, None); }
    if let Some(w) = &site.walkway { eng.object(w, Kind::Zone, "walkway", None); }
    let load = "Load-1";
    if sig.suspended_load { eng.object(load, Kind::Class, "suspended_load", None); }
    for s in &sig.active {
        let sid = signal_id(s);
        eng.object(&sid, Kind::Signal, s, None);
        for z in site.zones.iter().filter(|z| z.signals.contains(s)) { eng.fact("have", &z.instance, &sid); }
    }
    for p in people {
        eng.object(&p.id, Kind::Class, "person", None);
        if let Some(w) = &site.walkway { eng.fact("in", &p.id, w); }
        // the fall pose is a second type of the same track: iogp-runtime keeps every type believed over
        // missed frames (dhaneshkk/test1_logpeditor#1), so the alert's subject is the track id itself
        if p.down { eng.object(&p.id, Kind::Class, "fall_pose", None); }
        let clear = |name: &str| p.clear.iter().find(|(z, _)| z == name).map(|c| c.1);
        for z in &site.zones {
            let Some(c) = clear(&z.scene_zone) else { continue };
            if c < 0.0 { eng.fact("in", &p.id, &z.instance); }
            if c < z.near_m.unwrap_or(site.near_default_m) { eng.fact("near", &p.id, &z.instance); }
        }
        if sig.suspended_load && site.load_zone.as_deref().and_then(clear).is_some_and(|c| c < 0.0) {
            eng.fact("under", &p.id, load);
        }
        if p.p_helmet.is_none_or(|h| h >= 0.5) {
            let hid = format!("{}/hardhat", p.id);
            eng.object(&hid, Kind::Class, "hardhat", None);
            eng.fact("wear", &p.id, &hid);
        }
    }
    eng.end_events()
}

/// Track id behind an alert subject (subjects are track ids; anything after '/' is ignored).
pub fn track_of(subject: &str) -> &str { subject.split('/').next().unwrap_or(subject) }

fn signal_id(s: &str) -> String {
    s.split('-').map(|w| { let mut c = w.chars(); c.next().map_or(String::new(), |f| f.to_uppercase().collect::<String>() + c.as_str()) }).collect::<Vec<_>>().join("-")
}

/// Priority tier and MCC floor from an IOGP severity (critical → P1 0.85, high → P2 0.75, medium/low → P3 0.60).
pub fn tier(severity: Option<&str>) -> (&'static str, f64) {
    match severity { Some("critical") => ("P1", 0.85), Some("high") => ("P2", 0.75), _ => ("P3", 0.60) }
}

/// One rule's approval record from iogp-editor's export (`version`); `None` for a draft.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Approval { pub v: u32, pub author: String, pub approved_by: Option<String>, pub approved_at: Option<String>, pub text_sha256: Option<String> }

/// The ruleset as deployed: the exported config, its SHA-256 (what the evidence log records) and the
/// approval of each rule. Only rules approved in iogp-editor (four-eyes sign-off) may raise alarms; a
/// draft rule is evaluated in shadow mode.
pub struct Deployed { pub json: String, pub sha256: String, pub rules: RuleSet, pub approvals: Vec<Option<Approval>>, pub near_m: Option<f64> }

impl Deployed {
    pub fn load(json: &str) -> Result<Self, String> {
        let rules = load_config(json).map_err(|e| e.0)?;
        let sha256 = Sha256::digest(json.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let approvals = v["rules"].as_array().map(|rs| rs.iter().map(|r| serde_json::from_value::<Approval>(r["version"].clone()).ok()).collect()).unwrap_or_default();
        let near_m = v["rules"].as_array().into_iter().flatten().flat_map(|r| r["conditions"].as_array().into_iter().flatten())
            .filter(|c| c["relation"] == "near").filter_map(|c| c["metres"].as_f64()).reduce(f64::max);
        Ok(Self { json: json.into(), sha256, rules, approvals, near_m })
    }
    /// Approved in iogp-editor (has an approver)?
    pub fn approved(&self, id: &str) -> bool {
        self.rules.rules.iter().position(|r| r.id == id)
            .and_then(|i| self.approvals.get(i).cloned().flatten()).is_some_and(|a| a.approved_by.is_some())
    }
    pub fn rule(&self, id: &str) -> Option<&iogp_runtime::Rule> { self.rules.rules.iter().find(|r| r.id == id) }
    pub fn engine(&self) -> Engine { Engine::new(self.rules.rules.clone(), self.rules.grace_seconds, self.rules.cooldown_seconds) }
    /// Strict engine (no evidence smoothing) for ground truth.
    pub fn strict_engine(&self) -> Engine { Engine::new(self.rules.rules.clone(), 0.0, self.rules.cooldown_seconds) }
    pub fn ids(&self) -> Vec<String> { self.rules.rules.iter().map(|r| r.id.clone()).collect() }
}

/// Which (rule, subject) alerts are currently active.
pub fn active(eng: &Engine) -> BTreeSet<(String, String)> { eng.active() }
