//! Model cards written by the training notebook next to each exported model
//! (`models/ppe_clifford_folded.onnx` → `models/ppe_clifford_folded.card.json`).
//!
//! The card records the ONNX file's SHA-256, its test MCC (with bootstrap CI), the data it was trained
//! on, the SAM settings and whether it clears its rule's MCC floor. [`verify`] refuses a model whose bytes
//! are not the ones the card describes, so the metrics shown on site belong to the model that runs.
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
pub struct Card { pub path: PathBuf, pub json: Value }

impl Card {
    pub fn sha256(&self) -> &str { self.json["sha256"].as_str().unwrap_or("") }
    pub fn test_mcc(&self) -> Option<f64> { self.json["metrics"]["test_mcc"].as_f64() }
    pub fn mcc_ci95(&self) -> Option<(f64, f64)> {
        let c = self.json["metrics"]["test_mcc_ci95"].as_array()?;
        Some((c.first()?.as_f64()?, c.get(1)?.as_f64()?))
    }
    pub fn data_source(&self) -> &str { self.json["data"]["source"].as_str().unwrap_or("") }
    /// trained on real site crops (a synthetic smoke-test model must never arm a rule)
    pub fn real_data(&self) -> bool { self.data_source().starts_with("real:") }
    pub fn meets_floor(&self) -> bool { self.json["gate"]["meets_floor"].as_bool().unwrap_or(false) }
}

pub fn card_path(model: &Path) -> PathBuf { model.with_extension("card.json") }

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Sha256::digest(&b).iter().map(|x| format!("{x:02x}")).collect())
}

/// Load the card of `model` and check that it describes these exact bytes.
pub fn verify(model: impl AsRef<Path>) -> Result<Card, String> {
    let model = model.as_ref();
    let path = card_path(model);
    let json: Value = serde_json::from_str(&std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let card = Card { path, json };
    let sha = sha256_file(model)?;
    if card.sha256() != sha {
        return Err(format!("{} is not the model its card describes: sha256 {sha} vs card {}", model.display(), card.sha256()));
    }
    Ok(card)
}
