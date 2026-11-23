//! Per-rule MCC of a compiled ruleset on the notebook's simulated site (testdata/golden/scene.json):
//! predicted = the rule's alert is active for the track after the frame (rules fed from the Rust scene
//! on noisy boxes); truth = the same rule, strict (no smoothing), fed from the true 3-D state of the person.
use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

use crate::{Deployed, Person, Signals, SiteMap, step, tier};

#[derive(Clone, Debug, Serialize)]
pub struct RuleScore { pub id: String, pub severity: Option<String>, pub tier: String, pub floor: f64, pub cm: [u64; 4], pub mcc: Option<f64>, pub approved: bool }

pub fn mcc(cm: [u64; 4]) -> Option<f64> {
    let [tp, fp, fn_, tn] = cm.map(|x| x as f64);
    let d = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    (d > 0.0).then(|| (tp * tn - fp * fn_) / d)
}

/// `frames`: per frame `(t, ctx_suspended, ctx_pipe, predicted people, true people)`; people as
/// (id, clearances, down). Returns scores for rules that can be evaluated without imagery.
pub fn score(dep: &Deployed, site: &SiteMap, frames: &[(f64, bool, bool, Vec<(String, Vec<(String, f64)>, bool)>, Vec<(String, Vec<(String, f64)>, bool)>)],
             skip: &[&str]) -> Vec<RuleScore> {
    // Detection quality, not alert lifecycle: with cooldown 0 an alert is active exactly while the rule's
    // antecedent holds (durations and evidence smoothing included), as in the notebook's "held" metric.
    let rs = &dep.rules;
    let mut pe = crate::Engine::new(rs.rules.clone(), rs.grace_seconds, 0.0);
    // Truth is only known for people the simulated detector reported (it drops 5 % of boxes), so the truth
    // engine bridges up to two missing frames (0.25 s); otherwise every miss would reset a true duration.
    let mut ge = crate::Engine::new(rs.rules.clone(), 0.25, 0.0);
    let mut cms: BTreeMap<String, [u64; 4]> = dep.ids().into_iter().filter(|i| !skip.contains(&i.as_str())).map(|i| (i, [0; 4])).collect();
    for (t, susp, pipe, pred, truth) in frames {
        let sig = Signals { suspended_load: *susp, active: if *pipe { vec!["pipe-handling".into()] } else { vec![] } };
        let pp: Vec<Person> = pred.iter().map(|(id, c, d)| Person { id: id.clone(), clear: c, down: *d, p_helmet: None }).collect();
        let gp: Vec<Person> = truth.iter().map(|(id, c, d)| Person { id: id.clone(), clear: c, down: *d, p_helmet: None }).collect();
        step(&mut pe, site, *t, None, &sig, &pp);
        step(&mut ge, site, *t, None, &sig, &gp);
        let (pa, ga) = (pe.active(), ge.active());
        for ((p, _, _), (g, _, _)) in pred.iter().zip(truth) {
            for (rid, cm) in cms.iter_mut() {
                let a = pa.iter().any(|(r, s)| r == rid && crate::track_of(s) == p);
                let b = ga.iter().any(|(r, s)| r == rid && crate::track_of(s) == g);
                let k = match (a, b) { (true, true) => 0, (true, false) => 1, (false, true) => 2, (false, false) => 3 };
                cm[k] += 1;
            }
        }
    }
    cms.into_iter().map(|(id, cm)| {
        let sev = dep.rule(&id).and_then(|r| r.alert.severity.clone());
        let (tr, floor) = tier(sev.as_deref());
        RuleScore { approved: dep.approved(&id), id, severity: sev, tier: tr.into(), floor, cm, mcc: mcc(cm) }
    }).collect()
}

/// Parse testdata/golden/scene.json's ground-truth and context fields for [`score`] (predicted people come
/// from the caller, normally a Rust `Scene` replay).
pub fn golden_truth(g: &Value) -> Vec<(f64, bool, bool, Vec<(String, Vec<(String, f64)>, bool)>)> {
    let dt = g["dt"].as_f64().unwrap_or(0.1);
    g["frames"].as_array().into_iter().flatten().enumerate().map(|(k, f)| {
        let t = k as f64 * dt;
        let (s, p) = (f["ctx"]["suspended"].as_bool().unwrap_or(false), f["ctx"]["pipe_handling"].as_bool().unwrap_or(false));
        let truth = f["gt"].as_array().into_iter().flatten().map(|x| {
            let clear = x["clear"].as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(1.0))).collect()).unwrap_or_default();
            (format!("gt{}", x["pid"]), clear, x["down"].as_bool().unwrap_or(false))
        }).collect();
        (t, s, p, truth)
    }).collect()
}
