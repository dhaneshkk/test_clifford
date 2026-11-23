//! cargo run --release -p vg360-rules --example validate → config/vg360_rules.validation.json
#[path = "../tests/sim.rs"]
#[allow(dead_code)]
mod sim;

fn main() {
    let scores = sim::run();
    let rules: serde_json::Map<String, serde_json::Value> = scores.iter().map(|s| (s.id.clone(), serde_json::to_value(s).unwrap())).collect();
    let out = serde_json::json!({
        "validated_on": "notebook simulated site, 60 s, 600 frames, 8 018 person records (scripts/export_golden_scene.py); PPE is scored on the PPE crop test set",
        "config": "config/vg360_rules.draft.json",
        "rules": rules,
    });
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/vg360_rules.validation.json");
    std::fs::write(p, serde_json::to_string_pretty(&out).unwrap() + "\n").unwrap();
    for s in &scores { println!("{} MCC {:.3} floor {:.2}", s.id, s.mcc.unwrap_or(f64::NAN), s.floor); }
}
