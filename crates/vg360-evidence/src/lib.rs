//! Tamper-evident provenance for VisionGuard360 on **vesdb** (`dhaneshkk/test1_vesdb`).
//!
//! Everything that decides or explains an alert is written as an event over vesdb's
//! EventSourcingDB-compatible API, so it lands in an RFC 9162 Merkle log with a signed write receipt,
//! witness-cosigned checkpoints and offline evidence bundles (`vesdb-verify`):
//!
//! | subject | type (`ai.innoura.vg360.*`) | data |
//! |---|---|---|
//! | `/rigs/<rig>/rulesets/iogp` | `ruleset-deployed` | config SHA-256, rule ids, editor revision |
//! | `/rigs/<rig>/models/<name>` | `model-deployed` | model SHA-256, MCC, training (SAM ρ) |
//! | `/rigs/<rig>/cameras/<cam>` | `calibration-set`, `camera-guard-changed` | camera model / guard status, analytics mode, measured shift |
//! | `/rigs/<rig>/incidents/<id>` | `hazard-detected`, `alert-raised` | rule, LSR, severity, proof lines, frame SHA-256, ruleset and model hashes |
//!
//! The incident subject is opened with an `isSubjectPristine` precondition, so an incident id can never
//! be written twice. Every request carries an `Idempotency-Key` (SHA-256 of the body): when vesdb is
//! unreachable the request is spooled to disk and replayed later exactly once, in order.
use std::fs::OpenOptions;
use std::io::{BufRead, Write};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub type Error = Box<dyn std::error::Error + Send + Sync>;

pub fn sha256_hex(b: &[u8]) -> String { Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect() }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate { pub source: String, pub subject: String, #[serde(rename = "type")] pub ty: String, pub data: Value }

/// An event as stored by vesdb (the fields VG360 uses).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stored {
    pub id: String,
    pub subject: String,
    #[serde(rename = "type")] pub ty: String,
    pub time: String,
    pub hash: String,
    /// RFC 9162 leaf hash (hex)
    #[serde(default)] pub vesdbleaf: Option<String>,
    #[serde(default)] pub data: Value,
}

#[derive(Debug)]
pub enum Outcome { Written(Vec<Stored>), Spooled }

pub struct Vesdb {
    pub base: String,
    pub rig: String,
    pub source: String,
    write_token: String,
    audit_token: Option<String>,
    agent: ureq::Agent,
    spool: Option<PathBuf>,
}

impl Vesdb {
    /// `base` e.g. `http://127.0.0.1:3000`; tokens from the environment, never from the repository.
    pub fn new(base: &str, rig: &str, write_token: &str, audit_token: Option<&str>, spool: Option<PathBuf>) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .allow_non_standard_methods(true)
            .timeout_global(Some(std::time::Duration::from_secs(10)))
            .build()
            .into();
        Self { base: base.trim_end_matches('/').into(), rig: rig.into(), source: format!("https://{rig}.vg360.innoura.ai"),
               write_token: write_token.into(), audit_token: audit_token.map(Into::into), agent, spool }
    }

    /// From `VG360_VESDB_URL`, `VG360_VESDB_WRITE_TOKEN`, `VG360_VESDB_AUDIT_TOKEN`, `VG360_RIG`, `VG360_VESDB_SPOOL`.
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("VG360_VESDB_URL").ok()?;
        let w = std::env::var("VG360_VESDB_WRITE_TOKEN").ok()?;
        Some(Self::new(&url, &std::env::var("VG360_RIG").unwrap_or("rig".into()), &w,
                       std::env::var("VG360_VESDB_AUDIT_TOKEN").ok().as_deref(), std::env::var("VG360_VESDB_SPOOL").ok().map(PathBuf::from)))
    }

    fn request(&self, method: &str, path: &str, token: &str, body: Option<&str>, idem: Option<&str>) -> Result<(u16, String), Error> {
        let mut b = ureq::http::Request::builder().method(method).uri(format!("{}{path}", self.base))
            .header("authorization", format!("Bearer {token}"));
        if body.is_some() { b = b.header("content-type", "application/json"); }
        if let Some(k) = idem { b = b.header("idempotency-key", k); }
        let mut r = self.agent.run(b.body(body.unwrap_or("").to_string())?)?;
        let status = r.status().as_u16();
        Ok((status, r.body_mut().read_to_string()?))
    }

    fn audit(&self) -> Result<&str, Error> { self.audit_token.as_deref().ok_or_else(|| "no audit token (VG360_VESDB_AUDIT_TOKEN)".into()) }

    pub fn candidate(&self, subject: &str, ty: &str, data: Value) -> Candidate {
        Candidate { source: self.source.clone(), subject: subject.into(), ty: format!("ai.innoura.vg360.{ty}"), data }
    }

    /// Write a batch atomically. Network failures are spooled (if a spool file is set); a rejected
    /// request (e.g. a violated precondition) is returned as an error and never spooled.
    pub fn write(&self, events: &[Candidate], preconditions: &[Value]) -> Result<Outcome, Error> {
        let body = json!({ "events": events, "preconditions": preconditions }).to_string();
        let key = sha256_hex(body.as_bytes());
        match self.send_write(&body, &key) {
            Ok(v) => Ok(Outcome::Written(v)),
            Err(e) if e.to_string().starts_with("vesdb rejected") => Err(e),
            Err(e) => match &self.spool {
                Some(p) => {
                    let mut f = OpenOptions::new().create(true).append(true).open(p)?;
                    writeln!(f, "{}", json!({ "key": key, "body": body }))?;
                    f.sync_all()?;
                    Ok(Outcome::Spooled)
                }
                None => Err(e),
            },
        }
    }

    fn send_write(&self, body: &str, key: &str) -> Result<Vec<Stored>, Error> {
        let (status, text) = self.request("POST", "/api/v2/data/write-events", &self.write_token, Some(body), Some(key))?;
        if status >= 500 || status == 429 { return Err(format!("vesdb unavailable: HTTP {status}").into()); }
        let mut out = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line)?;
            match v["type"].as_str() {
                Some("event") => out.push(serde_json::from_value(v["data"].clone())?),
                Some("error") => return Err(format!("vesdb rejected the write: {}", v["data"]).into()),
                _ => {}
            }
        }
        if status >= 400 { return Err(format!("vesdb rejected the write: HTTP {status} {text}").into()); }
        Ok(out)
    }

    /// Replay spooled writes in order (each exactly once, thanks to its Idempotency-Key).
    pub fn flush_spool(&self) -> Result<usize, Error> {
        let Some(p) = &self.spool else { return Ok(0) };
        let Ok(f) = std::fs::File::open(p) else { return Ok(0) };
        let lines: Vec<String> = std::io::BufReader::new(f).lines().collect::<Result<_, _>>()?;
        let mut done = 0;
        for l in &lines {
            let v: Value = serde_json::from_str(l)?;
            match self.send_write(v["body"].as_str().unwrap_or(""), v["key"].as_str().unwrap_or("")) {
                Ok(_) => done += 1,
                Err(e) if e.to_string().starts_with("vesdb rejected") => done += 1, // already decided; keep the order going
                Err(_) => break,
            }
        }
        let rest = &lines[done..];
        std::fs::write(p, rest.iter().map(|l| format!("{l}\n")).collect::<String>())?;
        Ok(done)
    }

    pub fn checkpoint(&self) -> Result<Value, Error> {
        let (s, t) = self.request("GET", "/tlog/checkpoint", &self.write_token, None, None)?;
        if s != 200 { return Err(format!("no cosigned checkpoint yet (HTTP {s})").into()); }
        Ok(serde_json::from_str(&t)?)
    }
    pub fn receipt(&self, id: &str) -> Result<Value, Error> {
        let (s, t) = self.request("QUERY", "/api/v2/tlog/receipt", self.audit()?, Some(&json!({"id": id}).to_string()), None)?;
        if s != 200 { return Err(format!("receipt: HTTP {s} {t}").into()); }
        Ok(serde_json::from_str(&t)?)
    }
    /// Offline evidence bundle for these event ids (verify with `vesdb-verify bundle`).
    pub fn bundle(&self, ids: &[String]) -> Result<Value, Error> {
        let (s, t) = self.request("QUERY", "/api/v2/tlog/bundle", self.audit()?, Some(&json!({"ids": ids}).to_string()), None)?;
        if s != 200 { return Err(format!("bundle: HTTP {s} {t}").into()); }
        Ok(serde_json::from_str(&t)?)
    }
    /// Time-series points (e.g. camera-guard shift), Gorilla-compressed and sealed into the log by vesdb.
    pub fn ts_write(&self, series: &str, times_ns: &[i64], values: &[f64]) -> Result<(), Error> {
        let body = json!({ "series": [{ "name": series, "times": times_ns, "values": values }] }).to_string();
        let (s, t) = self.request("POST", "/api/v2/ts/write", &self.write_token, Some(&body), None)?;
        if s >= 300 { return Err(format!("ts write: HTTP {s} {t}").into()); }
        Ok(())
    }

    // ---------------------------------------------------------------- VG360 vocabulary
    pub fn ruleset_deployed(&self, config_sha256: &str, rule_ids: &[String], revision: Option<&str>) -> Result<Outcome, Error> {
        let c = self.candidate(&format!("/rigs/{}/rulesets/iogp", self.rig), "ruleset-deployed",
            json!({ "config_sha256": config_sha256, "rules": rule_ids, "revision": revision, "editor": "iogp-editor" }));
        self.write(&[c], &[])
    }
    pub fn model_deployed(&self, name: &str, model_sha256: &str, extra: Value) -> Result<Outcome, Error> {
        let mut d = json!({ "model_sha256": model_sha256 });
        if let (Some(o), Some(e)) = (d.as_object_mut(), extra.as_object()) { o.extend(e.clone()); }
        self.write(&[self.candidate(&format!("/rigs/{}/models/{name}", self.rig), "model-deployed", d)], &[])
    }
    pub fn camera_event(&self, camera: &str, ty: &str, data: Value) -> Result<Outcome, Error> {
        self.write(&[self.candidate(&format!("/rigs/{}/cameras/{camera}", self.rig), ty, data)], &[])
    }
    /// Open an incident: `hazard-detected` + `alert-raised` in one atomic, never-repeated write.
    pub fn incident(&self, id: &str, hazard: Value, alert: Value) -> Result<Outcome, Error> {
        let subject = format!("/rigs/{}/incidents/{id}", self.rig);
        let evs = [self.candidate(&subject, "hazard-detected", hazard), self.candidate(&subject, "alert-raised", alert)];
        self.write(&evs, &[json!({ "type": "isSubjectPristine", "payload": { "subject": subject } })])
    }
}
