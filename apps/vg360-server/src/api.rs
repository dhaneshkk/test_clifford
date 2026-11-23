//! OpenAI-compatible endpoints. The assistant is deterministic: it answers from site state
//! (camera guard, rules, ledger) without an LLM, so every number it gives is auditable.
//! Point any OpenAI client at `http://<host>:3000/v1` with model `vg360-ops`.
use serde::Deserialize;
use serde_json::{Value, json};
use topcoat::{
    Result,
    context::{Cx, context},
    router::{content::Json, route},
};
use vg360_guard::policy::Status;

use crate::state::Site;

#[route(GET "/v1/models")]
pub async fn models(cx: &Cx) -> Result<Json<Value>> {
    Ok(Json(json!({"object": "list", "data": [
        {"id": "vg360-ops", "object": "model", "owned_by": "visionguard360"}
    ]})))
}

#[derive(Deserialize)]
pub struct Msg { pub role: String, pub content: String }

#[derive(Deserialize)]
pub struct ChatReq { pub model: Option<String>, pub messages: Vec<Msg> }

fn now() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) }

pub fn answer(site: &Site, q: &str) -> String {
    let q = q.to_lowercase();
    let mut out = Vec::new();
    if q.contains("camera") || q.contains("guard") || q.contains("moved") || q.contains("status") {
        for c in &site.cams {
            let mut line = format!("{} {} — {}", c.id, c.name, c.guard.label());
            if c.guard != Status::Ok {
                let mode = match c.mode { vg360_pipeline::Mode::Compensated => "analytics continue, shift-compensated", vg360_pipeline::Mode::Paused => "analytics paused", vg360_pipeline::Mode::Aligned => "analytics running" };
                line += &format!(" ({mode}; {})", c.guidance().join(", "));
            }
            out.push(line);
        }
    }
    if q.contains("rule") || q.contains("mcc") || q.contains("armed") {
        for (id, r) in &site.rules {
            out.push(format!("{id}: {} — MCC {:.3} vs floor {:.2} → {}", r.name, r.mcc.unwrap_or(0.0), r.floor,
                if r.passes_floor() { "armed" } else { "shadow" }));
        }
    }
    if q.contains("alert") || q.contains("ledger") || q.contains("evidence") {
        out.push(format!("{} alerts in the ledger; verification: {:?}", site.ledger.entries.len(), site.ledger_check));
        for e in site.ledger.entries.iter().rev().take(5) {
            out.push(format!("t={}s {} {} hash {}…", e.time, e.rule, e.who, &e.hash[..12]));
        }
    }
    if out.is_empty() {
        out.push("Ask about camera status, rules / MCC, or alerts / ledger.".into());
    }
    out.join("\n")
}

#[route(POST "/v1/chat/completions")]
pub async fn chat(cx: &Cx, Json(req): Json<ChatReq>) -> Result<Json<Value>> {
    let site = context::<Site>(cx);
    let q = req.messages.iter().rev().find(|m| m.role == "user").map_or("", |m| m.content.as_str());
    let text = answer(site, q);
    Ok(Json(json!({
        "id": format!("chatcmpl-vg360-{}", now()),
        "object": "chat.completion",
        "created": now(),
        "model": req.model.unwrap_or_else(|| "vg360-ops".into()),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })))
}

#[route(GET "/v1/rules")]
pub async fn rules(cx: &Cx) -> Result<Json<Value>> {
    let site = context::<Site>(cx);
    let rules: Vec<Value> = site.rules.iter().map(|(id, r)| json!({
        "id": id, "name": r.name, "priority": r.pri, "floor": r.floor, "mcc": r.mcc, "armed": r.passes_floor(), "fol": r.fol
    })).collect();
    Ok(Json(json!({"rules": rules})))
}

#[route(GET "/v1/ledger")]
pub async fn ledger(cx: &Cx) -> Result<Json<Value>> {
    let site = context::<Site>(cx);
    Ok(Json(json!({"verify": site.ledger_check, "entries": site.ledger.entries})))
}

#[route(GET "/v1/cameras")]
pub async fn cameras(cx: &Cx) -> Result<Json<Value>> {
    let site = context::<Site>(cx);
    Ok(Json(json!({"cameras": site.cams.iter().map(|c| json!({
        "id": c.id, "name": c.name, "guard": c.guard, "analytics": c.mode,
        "pose": c.pose, "guidance": c.guidance(), "rules": c.rules, "url_env": c.env_var()
    })).collect::<Vec<_>>()})))
}
