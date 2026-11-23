//! Golden test: the ledger and rule MCCs written by the Python notebook verify in Rust.
use std::collections::BTreeMap;
use vg360_geom::{ledger::{Entry, Ledger, Verify}, rules::RuleRecord};

#[derive(serde::Deserialize)]
struct Export { rules: BTreeMap<String, RuleRecord>, ledger: Vec<Entry>, ledger_verified: bool }

fn load() -> Export {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/rule_studio_export.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn notebook_ledger_verifies_in_rust() {
    let x = load();
    assert!(x.ledger_verified);
    let l = Ledger { entries: x.ledger };
    assert!(matches!(l.verify(), Verify::Ok { .. }), "{:?}", l.verify());
}

#[test]
fn notebook_mcc_matches_confusion_matrix() {
    for (id, r) in load().rules {
        let (a, b) = (r.mcc.unwrap(), r.mcc_from_cm().unwrap());
        assert!((a - b).abs() < 1e-9, "{id}: sklearn {a} vs rust {b}");
        assert_eq!(r.floor, vg360_geom::rules::floor_for(r.pri.as_deref()), "{id}");
    }
}
