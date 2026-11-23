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
        if site.iogp.draft { out.push("Rules config is a draft (not approved in iogp-editor): all rules run in shadow mode.".into()); }
        for r in &site.iogp.rules {
            out.push(format!("{}: {} — MCC {:.3} vs floor {:.2}, {} → {}", r.id, r.text, r.mcc.unwrap_or(0.0), r.floor,
                if r.approved_by.is_some() { "approved" } else { "draft" }, if r.armed { "armed" } else { "shadow" }));
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
    Ok(Json(json!({"config_sha256": site.iogp.sha256, "draft": site.iogp.draft, "rules": site.iogp.rules})))
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

// ───────────────────────────── Rig Twin and Rule Studio data

#[route(GET "/api/twin")]
pub async fn twin_get(cx: &Cx) -> Result<Json<Value>> {
    let site = context::<Site>(cx);
    let t = site.twin.read().unwrap();
    Ok(Json(match t.as_ref() {
        Some(t) => json!({ "twin": t.json, "applied_at": t.applied_at, "source": t.source }),
        None => json!({ "twin": null }),
    }))
}

#[route(POST "/api/twin")]
pub async fn twin_apply(cx: &Cx, Json(v): Json<Value>) -> Result<Json<Value>> {
    let site = context::<Site>(cx);
    Ok(Json(site.apply_twin(v).unwrap_or_else(|e| json!({ "ok": false, "error": e }))))
}

/// Rule Studio's `vg360_notebook_export` format, filled from the live IOGP rules and their validation.
#[route(GET "/api/rule-studio")]
pub async fn rule_studio(cx: &Cx) -> Result<Json<Value>> {
    let site = context::<Site>(cx);
    let hz = |id: &str| match id { "LIFT" => "H01", "RED" => "H02", "PPE" => "H03", "FALL" => "H06", _ => "" };
    let by_id: serde_json::Map<String, Value> = site.iogp.rules.iter().map(|r| (r.id.to_lowercase(), json!({
        "hz": hz(&r.id), "pri": r.tier, "name": r.alert, "en": r.text, "fol": r.fol, "floor": r.floor, "cm": r.cm,
        "validated_on": r.validated_on, "mcc": r.mcc, "approved_by": r.approved_by, "armed": r.armed
    }))).collect();
    Ok(Json(json!({
        "vg360_notebook_export": 1, "generated": "live", "source": format!("site console · iogp-editor config {}", site.iogp.config_path),
        "rules": by_id, "ledger": site.ledger.entries, "ledger_verified": matches!(site.ledger_check, vg360_geom::ledger::Verify::Ok { .. })
    })))
}

#[route(GET "/api/evidence")]
pub async fn evidence(cx: &Cx) -> Result<Json<Value>> {
    let site = context::<Site>(cx);
    Ok(Json(match &site.vesdb {
        None => json!({ "connected": false }),
        Some(v) => json!({ "connected": true, "url": v.base, "rig": v.rig, "checkpoint": v.checkpoint().ok() }),
    }))
}
