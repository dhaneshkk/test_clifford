//! Rule Studio engine (notebook §7b): per (subject, rule) run length of the rule's condition,
//! an alert on the rising edge when it reaches the rule's duration, short drop-outs (≤ gap) do not
//! reset the timer, P2/P3 alerts are rate-limited by a cooldown, P1 alerts never are
//! (Lean: `debounce_keeps_p1`), and every alert is appended to the SHA-256 ledger (`alert_has_evidence`).
use std::collections::HashMap;

use vg360_geom::ledger::Ledger;

/// Live site context (crane load cell, drilling control system, …).
#[derive(Clone, Copy, Debug, Default)]
pub struct Ctx { pub suspended: bool, pub pipe_handling: bool }

/// What a rule looks at for one person in one frame.
#[derive(Clone, Debug, Default)]
pub struct State { pub clear: Vec<(String, f64)>, pub down: bool, pub p_helmet: Option<f64> }

impl State {
    fn clear(&self, zone: &str, default: f64) -> f64 { self.clear.iter().find(|(z, _)| z == zone).map_or(default, |c| c.1) }
}

#[derive(Clone, Debug)]
pub enum Cond {
    /// inside `zone` while the crane load is suspended
    InZoneWhileSuspended { zone: String },
    /// inside `zone` while pipe handling is active
    InZoneWhilePipeHandling { zone: String },
    /// closer than `m` metres to the plane / zone boundary
    ClearanceBelow { zone: String, m: f64 },
    Down,
    NoHelmet,
}

#[derive(Clone, Debug)]
pub struct Rule { pub id: String, pub hz: String, pub pri: Option<String>, pub name: String, pub cond: Cond, pub secs: f64 }

impl Rule {
    fn holds(&self, s: &State, c: &Ctx) -> bool {
        match &self.cond {
            Cond::InZoneWhileSuspended { zone } => c.suspended && s.clear(zone, 1.0) < 0.0,
            Cond::InZoneWhilePipeHandling { zone } => c.pipe_handling && s.clear(zone, 1.0) < 0.0,
            Cond::ClearanceBelow { zone, m } => s.clear(zone, 99.0) < *m,
            Cond::Down => s.down,
            Cond::NoHelmet => s.p_helmet.is_some_and(|p| p < 0.5),
        }
    }
    pub fn cooldown_s(&self) -> f64 {
        match self.pri.as_deref() { Some("P1") => 0.0, Some("P2") => 30.0, Some("P3") => 300.0, _ => 30.0 }
    }
}

/// The five Rule Studio rules of the notebook. `fps` converts the PPE rule's frame count to seconds.
pub fn default_rules(fps: f64) -> Vec<Rule> {
    let r = |id: &str, hz: &str, pri: Option<&str>, name: &str, cond, secs| Rule {
        id: id.into(), hz: hz.into(), pri: pri.map(Into::into), name: name.into(), cond, secs };
    vec![
        r("lift", "H01", Some("P1"), "Person under a suspended load", Cond::InZoneWhileSuspended { zone: "suspended_load".into() }, 3.0),
        r("red", "H02", Some("P1"), "Red zone during pipe handling", Cond::InZoneWhilePipeHandling { zone: "pump_skid_keepout".into() }, 0.0),
        r("edge", "", None, "Unguarded deck edge", Cond::ClearanceBelow { zone: "rig_edge".into(), m: 1.0 }, 0.0),
        r("fall", "H06", Some("P2"), "Slip, trip or person down", Cond::Down, 2.0),
        r("ppe", "H03", Some("P1"), "Missing PPE", Cond::NoHelmet, 15.0 / fps),
    ]
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Alert { pub t: f64, pub rule: String, pub hz: String, pub pri: Option<String>, pub name: String, pub subject: String, pub hash: String }

pub struct RuleEngine {
    pub rules: Vec<Rule>,
    pub ledger: Ledger,
    pub gap_s: f64,
    run: HashMap<(String, usize), f64>,
    off: HashMap<(String, usize), f64>,
    last: HashMap<(String, usize), f64>,
    active: HashMap<(String, usize), bool>,
}

impl RuleEngine {
    pub fn new(rules: Vec<Rule>, gap_s: f64) -> Self {
        Self { rules, ledger: Ledger::default(), gap_s, run: HashMap::new(), off: HashMap::new(), last: HashMap::new(), active: HashMap::new() }
    }

    fn held(&self, k: &(String, usize)) -> bool {
        self.run.get(k).copied().unwrap_or(0.0) >= self.rules[k.1].secs.max(1e-9)
    }

    /// One frame for one subject (track id). Returns the alerts raised, already in the ledger.
    pub fn step(&mut self, subject: &str, s: &State, c: &Ctx, t: f64, dt: f64, clip: &str) -> Vec<Alert> {
        let mut fired = Vec::new();
        for i in 0..self.rules.len() {
            let k = (subject.to_string(), i);
            if self.rules[i].holds(s, c) {
                self.off.insert(k.clone(), 0.0);
                *self.run.entry(k.clone()).or_insert(0.0) += dt;
            } else {
                let off = self.off.entry(k.clone()).or_insert(0.0);
                *off += dt;
                if *off > self.gap_s { self.run.insert(k.clone(), 0.0); }
            }
            let held = self.held(&k);
            let alert = held && !self.active.get(&k).copied().unwrap_or(false); // rising edge: one alert per episode
            self.active.insert(k.clone(), held);
            let r = &self.rules[i];
            let recent = t - self.last.get(&k).copied().unwrap_or(-1e9) < r.cooldown_s();
            if alert && (r.pri.as_deref() == Some("P1") || !recent) {
                self.last.insert(k, t);
                let hash = self.ledger.append(t, &r.id, &format!("Track {subject}"), clip).hash.clone();
                fired.push(Alert { t, rule: r.id.clone(), hz: r.hz.clone(), pri: r.pri.clone(), name: r.name.clone(), subject: subject.into(), hash });
            }
        }
        fired
    }
}
