//! Append-only evidence ledger. hash = SHA-256(compact JSON [time, rule, who, clip, prev]).
//! Byte-compatible with the notebook's `Ledger` (json.dumps(separators=(",",":"), ensure_ascii=False)).
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    pub time: String,
    pub rule: String,
    pub who: String,
    pub clip: String,
    pub prev: String,
    pub hash: String,
}

pub fn digest(time: &str, rule: &str, who: &str, clip: &str, prev: &str) -> String {
    let canon = serde_json::to_string(&[time, rule, who, clip, prev]).expect("strings serialise");
    let h = Sha256::digest(canon.as_bytes());
    h.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Ledger {
    pub entries: Vec<Entry>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Verify {
    Ok { entries: usize, head: String },
    Broken { index: usize, reason: &'static str },
}

impl Ledger {
    pub fn append(&mut self, time_s: f64, rule: &str, who: &str, clip: &str) -> &Entry {
        let prev = self.entries.last().map_or(GENESIS.to_string(), |e| e.hash.clone());
        let time = format!("{time_s:.1}");
        let hash = digest(&time, rule, who, clip, &prev);
        self.entries.push(Entry { time, rule: rule.into(), who: who.into(), clip: clip.into(), prev, hash });
        self.entries.last().unwrap()
    }

    pub fn verify(&self) -> Verify {
        let mut prev = GENESIS.to_string();
        for (i, e) in self.entries.iter().enumerate() {
            if e.prev != prev { return Verify::Broken { index: i, reason: "prev link does not match" }; }
            if digest(&e.time, &e.rule, &e.who, &e.clip, &e.prev) != e.hash {
                return Verify::Broken { index: i, reason: "hash does not match content" };
            }
            prev = e.hash.clone();
        }
        Verify::Ok { entries: self.entries.len(), head: prev }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn append_then_verify_then_tamper() {
        let mut l = Ledger::default();
        l.append(1.0, "edge", "Track 1", "sim-0");
        l.append(2.5, "lift", "Track 2", "sim-0");
        assert!(matches!(l.verify(), Verify::Ok { entries: 2, .. }));
        l.entries[0].who = "Track 9".into();
        assert_eq!(l.verify(), Verify::Broken { index: 0, reason: "hash does not match content" });
    }
}
