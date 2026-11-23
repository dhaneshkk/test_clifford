//! Rule Studio rules: duration rules with gap tolerance, and per-priority MCC floors.
use serde::{Deserialize, Serialize};

/// Minimum validated MCC before a rule may raise alerts (notebook §7b).
pub fn floor_for(priority: Option<&str>) -> f64 {
    match priority.unwrap_or("") { "P1" => 0.85, "P2" => 0.75, "P3" => 0.60, _ => 0.80 }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuleRecord {
    pub hz: String,
    /// `None` = untiered (floor 0.80); the notebook export currently leaves `edge` untiered.
    pub pri: Option<String>,
    pub name: String,
    pub en: String,
    pub fol: String,
    pub floor: f64,
    /// [tp, fp, fn, tn]
    pub cm: Option<[u64; 4]>,
    pub validated_on: String,
    pub mcc: Option<f64>,
}

impl RuleRecord {
    /// A rule without a validation result never passes.
    pub fn passes_floor(&self) -> bool { self.mcc.is_some_and(|m| m >= self.floor) }
    pub fn mcc_from_cm(&self) -> Option<f64> {
        let [tp, fp, fn_, tn] = self.cm?.map(|x| x as f64);
        Some(crate::metrics::mcc(tp, fp, fn_, tn))
    }
}

/// "inside for more than `hold_s`", tolerating detector drop-outs shorter than `gap_s`.
#[derive(Clone, Debug)]
pub struct Debounce {
    pub hold_s: f64,
    pub gap_s: f64,
    start: Option<f64>,
    last_seen: Option<f64>,
    fired: bool,
}

impl Debounce {
    pub fn new(hold_s: f64, gap_s: f64) -> Self { Self { hold_s, gap_s, start: None, last_seen: None, fired: false } }
    /// Feed one observation; returns true exactly once per continuous episode, when it fires.
    pub fn step(&mut self, t: f64, inside: bool) -> bool {
        if inside {
            if self.last_seen.is_none_or(|l| t - l > self.gap_s) { self.start = Some(t); self.fired = false; }
            self.last_seen = Some(t);
            if !self.fired && t - self.start.unwrap() >= self.hold_s { self.fired = true; return true; }
        } else if self.last_seen.is_some_and(|l| t - l > self.gap_s) {
            self.start = None; self.last_seen = None; self.fired = false;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn short_dropout_does_not_reset() {
        let mut d = Debounce::new(3.0, 0.5);
        let mut fired = vec![];
        for i in 0..50 {
            let t = i as f64 * 0.1;
            let inside = !(15..=18).contains(&i); // 0.4 s drop-out
            if d.step(t, inside) { fired.push(t); }
        }
        assert_eq!(fired.len(), 1);
        assert!((fired[0] - 3.0).abs() < 1e-9);
    }
    #[test]
    fn floors() { assert_eq!(floor_for(Some("P1")), 0.85); assert_eq!(floor_for(None), 0.80); }
}
