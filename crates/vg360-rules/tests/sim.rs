//! The five VG360 rules, written in controlled English and compiled by iogp-editor
//! (config/vg360_rules.draft.json), scored by MCC on the notebook's simulated site.
use vg360_geom::camera::Camera;
use vg360_rules::{Deployed, SiteMap, validate};
use vg360_scene::{scene::Scene, zones::default_zones};

fn root() -> String { format!("{}/../..", env!("CARGO_MANIFEST_DIR")) }

pub fn run() -> Vec<validate::RuleScore> {
    let dep = Deployed::load(&std::fs::read_to_string(format!("{}/config/vg360_rules.draft.json", root())).unwrap()).unwrap();
    let g: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{}/testdata/golden/scene.json", root())).unwrap()).unwrap();
    let c = &g["camera"];
    let cam = Camera { f_px: c["f"].as_f64().unwrap(), cx: c["cx"].as_f64().unwrap(), cy: c["cy"].as_f64().unwrap(), height_m: c["h"].as_f64().unwrap(),
                       pitch_deg: c["pitch"].as_f64().unwrap(), yaw_deg: c["yaw"].as_f64().unwrap(), roll_deg: 0.0, x0: 0.0, y0: 0.0 };
    let mut scene = Scene::new(cam, default_zones());
    let truth = validate::golden_truth(&g);
    let mut frames = Vec::new();
    for (f, (t, s, p, gt)) in g["frames"].as_array().unwrap().iter().zip(truth) {
        let boxes: Vec<[f64; 4]> = serde_json::from_value(f["boxes"].clone()).unwrap();
        let recs = scene.analyse(&boxes, t);
        let pred: Vec<_> = recs.iter().flatten().map(|r| (r.track.to_string(), r.zones.clone(), r.state.down)).collect();
        assert_eq!(pred.len(), gt.len());
        frames.push((t, s, p, pred, gt));
    }
    let mut site = SiteMap::rig_default();
    site.near_default_m = dep.near_m.unwrap_or(3.0);
    validate::score(&dep, &site, &frames, &["PPE"])
}

#[test]
fn iogp_rules_on_the_simulated_site() {
    let scores = run();
    for s in &scores {
        eprintln!("{:5} {:8} {} floor {:.2}  MCC {:.3}  [tp fp fn tn] {:?}  approved: {}", s.id, s.severity.as_deref().unwrap_or("-"), s.tier, s.floor,
                  s.mcc.unwrap_or(f64::NAN), s.cm, s.approved);
    }
    assert_eq!(scores.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), ["EDGE", "FALL", "LIFT", "RED"]);
    assert!(scores.iter().all(|s| s.mcc.is_some_and(|m| m > 0.5)), "every rule must be informative on the simulated site");
    assert!(scores.iter().all(|s| !s.approved), "the committed config is a draft: approval happens in iogp-editor");
    // the committed validation file must be what this run measures
    let committed: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{}/config/vg360_rules.validation.json", root())).unwrap()).unwrap();
    for s in &scores {
        let c = &committed["rules"][&s.id];
        assert_eq!(c["cm"], serde_json::json!(s.cm), "{}: config/vg360_rules.validation.json is stale — run `cargo run -p vg360-rules --example validate`", s.id);
    }
}
