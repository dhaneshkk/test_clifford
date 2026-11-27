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

/// One camera for the assistant and `/v1/cameras`: a live stream (from the live snapshot) or a static sample.
#[derive(Debug, Clone, PartialEq)]
pub struct CamLine {
    pub source: &'static str,
    pub id: String,
    pub label: String,
    pub status: Value,
    pub status_label: String,
    pub mode: String,
    pub warning: Option<String>,
    pub guidance: Vec<String>,
    pub error: Option<String>,
    pub stale_s: Option<f64>,
    pub fps: f64,
}

fn str_of(v: &Value) -> Option<String> { v.as_str().map(str::to_owned).filter(|s| !s.is_empty()) }

/// Live cameras from a `Live::snapshot()` JSON value. Every field is optional: missing ones fall back to neutral defaults.
pub fn live_lines(snapshot: &Value) -> Vec<CamLine> {
    snapshot["cameras"].as_array().into_iter().flatten().map(|c| {
        let lt = &c["latest"];
        let status = lt.get("status").cloned().unwrap_or(Value::Null);
        let status_label = str_of(&lt["status_label"]).or_else(|| str_of(&status)).unwrap_or_else(|| "UNKNOWN".into());
        let warning = str_of(&lt["warning"]).or_else(|| str_of(&lt["warning"]["text"]));
        CamLine {
            source: "live",
            id: str_of(&c["id"]).unwrap_or_default(),
            label: str_of(&c["label"]).unwrap_or_default(),
            status,
            status_label,
            mode: str_of(&lt["mode"]).unwrap_or_else(|| "starting".into()),
            warning,
            guidance: lt["guidance"].as_array().into_iter().flatten().filter_map(str_of).collect(),
            error: str_of(&c["error"]),
            stale_s: c["stale_s"].as_f64(),
            fps: c["fps"].as_f64().unwrap_or(0.0),
        }
    }).collect()
}

/// Live cameras first (from the live snapshot), then the static sample cameras.
pub fn camera_lines(site: &Site) -> Vec<CamLine> {
    let mut v = live_lines(&site.live.snapshot());
    v.extend(site.cams.iter().map(|c| CamLine {
        source: "sample",
        id: c.id.to_string(),
        label: c.name.to_string(),
        status: json!(c.guard),
        status_label: c.guard.label().to_string(),
        mode: format!("{:?}", c.mode).to_lowercase(),
        warning: None,
        guidance: if c.guard == Status::Ok { vec![] } else { c.guidance() },
        error: None,
        stale_s: None,
        fps: c.fps,
    }));
    v
}

fn capitalize(s: &str) -> String {
    let mut ch = s.chars();
    ch.next().map_or(String::new(), |f| f.to_uppercase().chain(ch).collect())
}

/// One human line: `LIVE DEMO — CAMERA MOVED · Compensated — <warning> (<first guidance>)`.
/// Samples read `ID name (sample) — …` when `tag_samples` is set (live cameras exist).
pub fn render_cam(c: &CamLine, tag_samples: bool) -> String {
    let head = if c.source == "live" { format!("LIVE {}", c.id) } else if tag_samples { format!("{} {} (sample)", c.id, c.label) } else { format!("{} {}", c.id, c.label) };
    let mut s = format!("{head} — {} · {}", c.status_label, capitalize(&c.mode));
    if let Some(w) = &c.warning { s += &format!(" — {w}"); }
    if let Some(g) = c.guidance.first() { s += &format!(" ({g})"); }
    s
}

pub fn answer(site: &Site, q: &str) -> String {
    let q = q.to_lowercase();
    let mut out = Vec::new();
    if q.contains("camera") || q.contains("guard") || q.contains("moved") || q.contains("status") {
        let lines = camera_lines(site);
        let tag = lines.iter().any(|c| c.source == "live");
        out.extend(lines.iter().map(|c| render_cam(c, tag)));
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
        for (c, lc) in cams.iter().zip(live_lines(&l)) {
            let lt = &c["latest"];
            out.push(format!("LIVE {} — {} · {:.1} fps, {} frames, {} people in view, analytics {}{}", lc.id, lc.status_label,
                c["fps"].as_f64().unwrap_or(0.0), c["frames"], lt["people"].as_array().map_or(0, |p| p.len()),
                lc.mode, lc.error.as_ref().map(|e| format!(" (error: {e})")).unwrap_or_default()));
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
    let live = live_lines(&site.live.snapshot()).into_iter().map(|c| json!({
        "id": c.id, "label": c.label, "source": "live", "status": c.status, "status_label": c.status_label, "mode": c.mode,
        "warning": c.warning, "guidance": c.guidance, "error": c.error, "stale_s": c.stale_s, "fps": c.fps
    }));
    let samples = site.cams.iter().map(|c| json!({
        "id": c.id, "name": c.name, "source": "sample", "guard": c.guard, "analytics": c.mode,
        "pose": c.pose, "guidance": c.guidance(), "rules": c.rules, "url_env": c.env_var()
    }));
    Ok(Json(json!({"cameras": live.chain(samples).collect::<Vec<_>>()})))
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

// ───────────────────────────── auto-calibration (no surveyed deck points): see calib_api.rs

/// Vanishing points → focal length, field of view, pitch, roll; standing people → mount height.
/// The result's `twin` block is what to enter for the camera in the Rig Twin (fov, pitch, h above the deck).
#[route(POST "/api/calibrate")]
pub async fn calibrate(cx: &Cx, Json(r): Json<crate::calib_api::CalibReq>) -> Result<Response> {
    let (status, body) = crate::calib_api::run(&r);
    (status, Json(body)).into_response(cx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> Value {
        json!({ "cameras": [{
            "id": "DEMO", "label": "LIVE DEMO", "source": "demo", "frames": 120, "fps": 9.5, "error": null, "stale_s": 0.4,
            "latest": { "status": "moved", "status_label": "CAMERA MOVED", "mode": "compensated",
                "guidance": ["Move view RIGHT 5.0%", "Re-aim the camera"],
                "warning": { "level": "amber", "text": "Camera moved — analytics continue through shift compensation; please re-aim the camera" },
                "people": [1, 2] }
        }, {
            "id": "CAM2", "fps": 0.0, "frames": 0, "error": "stream closed", "latest": {}
        }], "alerts": [], "signals": {} })
    }

    #[test]
    fn live_lines_read_status_warning_and_guidance() {
        let lines = live_lines(&snapshot());
        assert_eq!(lines.len(), 2);
        let demo = &lines[0];
        assert_eq!(demo.source, "live");
        assert_eq!(demo.id, "DEMO");
        assert_eq!(demo.status_label, "CAMERA MOVED");
        assert_eq!(demo.mode, "compensated");
        assert_eq!(demo.guidance, vec!["Move view RIGHT 5.0%", "Re-aim the camera"]);
        assert!(demo.warning.as_deref().unwrap().starts_with("Camera moved"));
        assert_eq!(demo.stale_s, Some(0.4));
        assert_eq!(demo.error, None);
    }

    #[test]
    fn live_lines_default_when_fields_missing() {
        let lines = live_lines(&snapshot());
        let bare = &lines[1];
        assert_eq!(bare.status_label, "UNKNOWN");
        assert_eq!(bare.mode, "starting");
        assert!(bare.warning.is_none() && bare.guidance.is_empty());
        assert_eq!(bare.error.as_deref(), Some("stream closed"));
        assert_eq!(live_lines(&json!({})).len(), 0);
    }

    #[test]
    fn render_live_camera_line() {
        let lines = live_lines(&snapshot());
        assert_eq!(
            render_cam(&lines[0], true),
            "LIVE DEMO — CAMERA MOVED · Compensated — Camera moved — analytics continue through shift compensation; please re-aim the camera (Move view RIGHT 5.0%)"
        );
    }

    #[test]
    fn render_sample_line_tagged_only_when_live_exists() {
        let sample = CamLine {
            source: "sample", id: "C1".into(), label: "Gate".into(), status: json!("moved"), status_label: "CAMERA MOVED".into(),
            mode: "paused".into(), warning: None, guidance: vec!["Re-aim".into()], error: None, stale_s: None, fps: 0.0,
        };
        assert_eq!(render_cam(&sample, true), "C1 Gate (sample) — CAMERA MOVED · Paused (Re-aim)");
        assert_eq!(render_cam(&sample, false), "C1 Gate — CAMERA MOVED · Paused (Re-aim)");
    }
}
