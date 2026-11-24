//! OpenAI-compatible endpoints. The assistant is deterministic: it answers from site state
//! (camera guard, rules, ledger) without an LLM, so every number it gives is auditable.
//! Point any OpenAI client at `http://<host>:3000/v1` with model `vg360-ops`.
use serde::Deserialize;
use serde_json::{Value, json};
use topcoat::{
    Result,
    context::{Cx, context},
    router::{
        content::{Json, sse::{Event, KeepAlive, Sse}},
        error::not_found,
        path_param,
        response::{IntoResponse, Response},
        route,
    },
};
use futures_util::{Stream, StreamExt};
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
pub struct ChatReq { pub model: Option<String>, pub messages: Vec<Msg>, #[serde(default)] pub stream: bool }

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
    if q.contains("live") || q.contains("now") || q.contains("people") || q.contains("alert") || q.contains("incident") {
        let l = site.live.snapshot();
        let cams = l["cameras"].as_array().cloned().unwrap_or_default();
        if cams.is_empty() { out.push("No live streams (set VG360_CAM_<ID>_URL for twin cameras, or VG360_DEMO=1).".into()); }
        for c in &cams {
            let lt = &c["latest"];
            out.push(format!("LIVE {} — {:.1} fps, {} frames, {} people in view, analytics {}{}", c["id"].as_str().unwrap_or(""),
                c["fps"].as_f64().unwrap_or(0.0), c["frames"], lt["people"].as_array().map_or(0, |p| p.len()),
                lt["mode"].as_str().unwrap_or("starting"), c["error"].as_str().map(|e| format!(" (error: {e})")).unwrap_or_default()));
        }
        for a in l["alerts"].as_array().into_iter().flatten().take(5) {
            out.push(format!("{} t={:.1}s {} {} {} [{}]{}", a["camera"].as_str().unwrap_or(""), a["t"].as_f64().unwrap_or(0.0),
                a["kind"].as_str().unwrap_or(""), a["alert"].as_str().unwrap_or(""), a["track"].as_str().unwrap_or(""),
                if a["armed"] == true { "armed" } else { "shadow" }, a["incident"].as_str().map(|i| format!(" incident {i}")).unwrap_or_default()));
        }
    }
    if out.is_empty() {
        out.push("Ask about camera status, rules / MCC, alerts / ledger, or what is live now.".into());
    }
    out.join("\n")
}

/// OpenAI `chat.completion.chunk` stream: role, one chunk per answer line, finish, then `[DONE]`.
fn chunks(id: String, model: String, text: String) -> impl Stream<Item = Result<Event>> + Send + 'static {
    let created = now();
    let chunk = move |delta: Value, finish: Value| json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] }).to_string();
    let lines: Vec<String> = text.split_inclusive('\n').map(String::from).collect();
    let mut evs = vec![chunk(json!({"role": "assistant", "content": ""}), Value::Null)];
    evs.extend(lines.into_iter().map(|l| chunk(json!({"content": l}), Value::Null)));
    evs.push(chunk(json!({}), json!("stop")));
    evs.push("[DONE]".into());
    futures_util::stream::iter(evs.into_iter().map(|d| Ok(Event::new().data(d))))
}

#[route(POST "/v1/chat/completions")]
pub async fn chat(cx: &Cx, Json(req): Json<ChatReq>) -> Result<Response> {
    let site = context::<Site>(cx);
    let q = req.messages.iter().rev().find(|m| m.role == "user").map_or("", |m| m.content.as_str());
    let text = answer(site, q);
    let (id, model) = (format!("chatcmpl-vg360-{}", now()), req.model.unwrap_or_else(|| "vg360-ops".into()));
    if req.stream { return Sse::new(chunks(id, model, text)).into_response(cx); }
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": now(),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })).into_response(cx)
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

// ───────────────────────────── live console

#[route(GET "/api/live")]
pub async fn live_state(cx: &Cx) -> Result<Json<Value>> { Ok(Json(context::<Site>(cx).live.snapshot())) }

/// Server-sent events: `frame` (≤ 4 Hz per camera) and `alert` (immediately). Browsers: `new EventSource("/api/live/events")`.
#[route(GET "/api/live/events")]
pub async fn live_events(cx: &Cx) -> Result<Sse<impl Stream<Item = Result<Event>> + use<>>> {
    let rx = context::<Site>(cx).live.tx.subscribe();
    let hello = Event::new().event("hello").data(context::<Site>(cx).live.snapshot().to_string());
    let evs = tokio_stream::wrappers::BroadcastStream::new(rx)
        .filter_map(|m| async move { m.ok().map(|(name, data)| Ok(Event::new().event(name).data(data))) });
    Ok(Sse::new(futures_util::stream::once(async move { Ok(hello) }).chain(evs)).keep_alive(KeepAlive::new()))
}

fn jpeg(b: Vec<u8>) -> ([(&'static str, &'static str); 2], Vec<u8>) { ([("content-type", "image/jpeg"), ("cache-control", "no-store")], b) }

path_param!(cam);

/// Latest live frame of a camera (JPEG).
#[route(GET "/api/live/{cam}/frame.jpg")]
pub async fn live_jpeg(cx: &Cx) -> Result<Response> {
    let site = context::<Site>(cx);
    let b = site.live.cams.read().unwrap().get(cam(cx)).and_then(|c| c.jpeg.clone()).ok_or_else(not_found)?;
    jpeg(b.as_ref().clone()).into_response(cx)
}

path_param!(incident);

/// The frame an alert was raised on — its SHA-256 is in the incident's `hazard-detected` event.
#[route(GET "/api/evidence/{incident}/snapshot.jpg")]
pub async fn live_evidence_jpeg(cx: &Cx) -> Result<Response> {
    let site = context::<Site>(cx);
    let id = incident(cx);
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') { return Err(not_found().into()); }
    let b = std::fs::read(site.state_dir.join("evidence").join(format!("{id}.jpg"))).map_err(|_| not_found())?;
    jpeg(b).into_response(cx)
}

#[derive(Deserialize)]
pub struct SignalsReq { #[serde(default)] pub suspended_load: Option<bool>, #[serde(default)] pub active: Option<Vec<String>> }

/// Site signals from the rig's control system (crane load, pipe handling …): they gate the rules.
#[route(POST "/api/signals")]
pub async fn signals(cx: &Cx, Json(r): Json<SignalsReq>) -> Result<Json<Value>> {
    let site = context::<Site>(cx);
    {
        let mut s = site.live.signals.write().unwrap();
        if let Some(l) = r.suspended_load { s.suspended_load = l; }
        if let Some(a) = r.active { s.active = a; }
    }
    let v = site.live.snapshot()["signals"].clone();
    let _ = site.live.tx.send(("signals".into(), v.to_string()));
    Ok(Json(v))
}
