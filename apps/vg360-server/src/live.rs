//! Live analytics: one camera worker thread per stream, results fanned out to the console.
//!
//! Streams come from the environment, never from the repository:
//! * `VG360_CAM_<ID>_URL` for every camera of the applied Rig Twin (e.g. `VG360_CAM_C5_URL`) — the worker
//!   uses that camera's twin calibration and the twin zones it watches;
//! * `VG360_DEMO=1` adds camera `DEMO`: the simulated rig clip (`testdata/clip/clip.mp4`) replayed in real
//!   time as if it were a camera, with its own (known) camera model;
//! * `RTSP_FEED` adds camera `LIVE` (replacing DEMO): a real stream with a generic pinhole from
//!   `VG360_LIVE_SIZE` / `_HFOV` / `_PITCH` / `_HEIGHT`, no zones, mount height estimated if `_HEIGHT` is unset.
//!
//! Each worker's results are summarised (≤ 4 per second per camera) and pushed on an in-process broadcast
//! channel that `/api/live/events` streams as server-sent events; alerts are pushed immediately. Alert
//! snapshots are kept under `state/evidence/<incident>.jpg` next to their vesdb incident.
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde_json::{Value, json};
use tokio::sync::broadcast;
use vg360_evidence::Vesdb;
use vg360_geom::camera::Camera;
use vg360_geom::twin::{LensSpec, TwinCamera, TwinZone};
use vg360_infer::{ppe::Ppe, yolox::Yolox};
use vg360_guard::config::REF_CAPTURE_S;
use vg360_guard::policy::{Status, guidance};
use vg360_pipeline::{CameraWorker, FrameResult, Mode, Provenance, RefPolicy, RuleSet, WorkerCfg};
use vg360_rules::{Deployed, Signals, SiteMap, ZoneMap};
use vg360_scene::scene::Scene;
use vg360_scene::zones::{Shape, Zone, default_zones};

pub struct CamLive {
    pub label: String,
    pub source: String,
    pub summary: Option<Value>,
    pub jpeg: Option<Arc<Vec<u8>>>,
    pub frames: u64,
    pub fps: f64,
    pub error: Option<String>,
    /// frame size (px) and lens, for the overlay
    pub size: (usize, usize),
    pub fisheye: bool,
    last_push: Option<Instant>,
    pub last_frame_at: Option<Instant>,
    /// last (status, mode, warning level, warning text) broadcast as a `guard` event
    guard_key: Option<(String, String, String, String)>,
    started: Instant,
}

pub struct Live {
    pub cams: RwLock<BTreeMap<String, CamLive>>,
    pub alerts: RwLock<VecDeque<Value>>,
    pub signals: Arc<RwLock<Signals>>,
    pub tx: broadcast::Sender<(String, String)>,
}

impl Live {
    pub fn is_running(&self) -> bool { !self.cams.read().unwrap().is_empty() }
    pub fn snapshot(&self) -> Value {
        let cams = self.cams.read().unwrap();
        json!({
            "cameras": cams.iter().map(|(id, c)| {
                let stale = c.last_frame_at.map(|t| t.elapsed().as_secs_f64());
                // never leave a camera without a warning: no frame yet, a stream/config error, or a stall all override
                let mut latest = c.summary.clone().unwrap_or_else(|| json!({ "status_label": if c.error.is_some() { "NO SIGNAL" } else { "STARTING" },
                                                                              "mode": "Paused", "guidance": [] }));
                let over = match (&c.error, stale) {
                    (Some(e), _) => Some(warning(Status::Ok, Mode::Paused, None, None, None, Some(e), None)),
                    (None, Some(s)) if s > STALE_S => Some(stall_warning(s)),
                    (None, None) => Some(("warn", "Starting — waiting for the first frame".to_string())),
                    _ => None,
                };
                if let (Some((l, t)), Value::Object(m)) = (over, &mut latest) { m.insert("warning".into(), json!({ "level": l, "text": t })); }
                json!({ "id": id, "label": c.label, "source": c.source, "frames": c.frames, "w": c.size.0, "h": c.size.1, "fisheye": c.fisheye,
                        "fps": c.fps, "error": c.error, "stale_s": stale, "latest": latest })
            }).collect::<Vec<_>>(),
            "alerts": self.alerts.read().unwrap().iter().cloned().collect::<Vec<_>>(),
            "signals": { "suspended_load": self.signals.read().unwrap().suspended_load, "active": self.signals.read().unwrap().active },
        })
    }
}

/// a frame older than this (s) at the console means the stream has stalled
const STALE_S: f64 = 5.0;

fn stall_warning(stale_s: f64) -> (&'static str, String) {
    ("alarm", format!("Stream stalled — no frame for {stale_s:.0} s; analytics are not running"))
}

/// The one operator-facing warning for a camera: (level "ok"|"warn"|"alarm", text).
pub fn warning(status: Status, mode: Mode, pending: Option<(&str, f64)>, analytics_suspended: Option<&str>,
               auto_height_state: Option<&str>, error: Option<&str>, stale_s: Option<f64>) -> (&'static str, String) {
    if let Some(e) = error { return ("alarm", format!("No signal: {e}")); }
    if let Some(s) = stale_s.filter(|s| *s > STALE_S) { return stall_warning(s); }
    match status {
        Status::Blocked => return ("alarm", "Lens blocked — analytics paused (clean / uncover the camera)".into()),
        Status::ViewLost => return ("alarm", "View not recognised — analytics paused (ground calibration no longer valid)".into()),
        Status::Moved if mode == Mode::Paused => return ("alarm", "Camera moved — analytics paused (ground calibration no longer valid); re-aim the camera".into()),
        Status::Moved => return ("warn", "Camera moved — analytics continue through shift compensation; please re-aim the camera".into()),
        Status::Realigning => return ("warn", "Realigning — hold the camera steady".into()),
        Status::Ok => {}
    }
    // status still Ok: say what is actually happening while the guard waits out its hold time
    let confirming = |el: Option<f64>| el.map(|e| format!(" — confirming ({e:.0} s)")).unwrap_or_default();
    let pend_s = pending.map(|(_, e)| e);
    if let Some(why) = analytics_suspended { return ("warn", format!("Analytics suspended: {why}{}", confirming(pend_s))); }
    if let Some((what, el)) = pending {
        let what = match what { "Blocked" => "Possible lens blockage", "ViewLost" => "View not recognised", _ => "Possible camera movement" };
        let tail = if mode == Mode::Paused { "; analytics paused meanwhile" } else { "" };
        return ("warn", format!("{what} — confirming ({el:.0} s){tail}"));
    }
    match mode {
        Mode::Paused => return ("warn", "Camera position changed — analytics paused while the guard confirms".into()),
        Mode::Compensated => return ("warn", "Small camera shift — analytics continue through shift compensation; check the camera".into()),
        Mode::Aligned => {}
    }
    match auto_height_state {
        Some("estimating") => ("warn", "Estimating mount height — rules off".into()),
        Some("failed") => ("alarm", "Mount height could not be estimated — rules off; set the height in the twin / VG360_LIVE_HEIGHT".into()),
        _ => ("ok", "Position correct".into()),
    }
}

/// Compact, UI-ready view of one frame result.
pub fn summarise(r: &FrameResult) -> Value {
    let people: Vec<Value> = r.boxes.iter().enumerate().map(|(i, d)| {
        let rec = r.recs.get(i).and_then(|x| x.as_ref());
        json!({ "box": [d.x1, d.y1, d.x2, d.y2], "score": d.score, "helmet": r.helmet.get(i).copied().flatten(),
                "track": rec.map(|x| x.track), "foot": rec.map(|x| x.foot), "head_z": rec.map(|x| x.head_z), "tilt": rec.map(|x| x.tilt),
                "down": rec.map(|x| x.state.down), "inside": rec.map(|x| x.zones.iter().filter(|(_, c)| *c < 0.0).map(|(z, _)| z.clone()).collect::<Vec<_>>()) })
    }).collect();
    let guide = if r.status == Status::Ok && r.mode == Mode::Aligned { vec![] }
                else { r.smoothed_pose.as_ref().or(r.pose.as_ref()).map(guidance).unwrap_or_default() };
    let ah = r.auto_height.as_ref();
    let (level, text) = warning(r.status, r.mode, r.guard_pending.as_ref().map(|(s, e)| (s.as_str(), *e)), r.analytics_suspended.as_deref(),
                                ah.map(|a| a.state), None, None);
    json!({ "camera": r.camera, "t": r.t, "status": r.status, "mode": r.mode, "pose": r.pose, "inliers": r.inliers,
            "ref_box": r.ref_box, "people": people, "ms": r.ms, "height_m": r.height_m, "calibrating": r.calibrating,
            "status_label": r.status.label(), "guidance": guide,
            "pending": r.guard_pending.as_ref().map(|(s, e)| json!({ "status": s, "elapsed_s": e })),
            "analytics_suspended": r.analytics_suspended, "ref_points": r.ref_points, "smoothed_pose": r.smoothed_pose,
            "frame_age_s": r.frame_age_s,
            "auto_height": ah.map(|a| json!({ "state": a.state, "value": a.value, "samples": a.samples })),
            "warning": { "level": level, "text": text } })
}

fn alert_json(r: &FrameResult, a: &vg360_pipeline::Alarm) -> Value {
    json!({ "camera": r.camera, "t": a.event.t, "kind": a.event.kind, "rule": a.event.rule, "alert": a.event.alert, "lsr": a.event.lsr,
            "severity": a.event.severity, "track": a.track, "armed": a.armed, "mode": a.mode, "proof": a.event.proof,
            "incident": a.incident, "evidence": a.evidence, "snapshot_sha256": a.snapshot_sha256 })
}

/// A `guard` event when (status, mode, warning level, warning text) differs from the previous summary.
fn guard_event(cam: &str, sum: &Value, prev: &mut Option<(String, String, String, String)>) -> Option<Value> {
    let k = |v: &Value| v.as_str().unwrap_or_default().to_string();
    let key = (k(&sum["status"]), k(&sum["mode"]), k(&sum["warning"]["level"]), k(&sum["warning"]["text"]));
    if prev.as_ref() == Some(&key) { return None; }
    *prev = Some(key);
    Some(json!({ "camera": cam, "status": sum["status"], "status_label": sum["status_label"], "mode": sum["mode"],
                 "warning": sum["warning"], "guidance": sum["guidance"] }))
}

struct Spec { id: String, label: String, url: String, camera: Camera, size: (usize, usize), zones: Vec<Zone>, site: SiteMap,
              lens: Option<LensSpec>, auto_height: bool,
              /// configuration error: the camera is listed with this error and no worker is started
              config_error: Option<String> }

/// Scene zones and rule mapping for a twin camera: every zone the twin says this camera watches, on its deck.
fn twin_zones(cam: &TwinCamera, zones: &[TwinZone]) -> (Vec<Zone>, SiteMap) {
    let mut out = Vec::new();
    let mut site = SiteMap { zones: vec![], load_zone: None, walkway: Some("Deck".into()), near_default_m: 3.0 };
    for z in zones.iter().filter(|z| z.cameras.contains(&cam.id) && (z.surface_m - cam.surf).abs() < 0.5) {
        out.push(Zone { name: z.id.clone(), shape: Shape::Polygon(z.polygon.iter().map(|p| [p[0], p[1]]).collect()) });
        for r in &z.rules {
            match r.as_str() {
                "red" | "excl" => site.zones.push(ZoneMap { scene_zone: z.id.clone(), instance: "Red-Zone".into(), zone_type: "exclusion-zone".into(),
                                                            near_m: None, signals: vec!["pipe-handling".into()] }),
                "height" => site.zones.push(ZoneMap { scene_zone: z.id.clone(), instance: "Deck-Edge".into(), zone_type: "height-zone".into(),
                                                      near_m: Some(0.0), signals: vec![] }),
                "lift" if site.load_zone.is_none() => site.load_zone = Some(z.id.clone()),
                _ => {}
            }
        }
    }
    (out, site)
}

fn specs(root: &str, twin: Option<&crate::state::TwinState>) -> Vec<Spec> {
    let mut v = Vec::new();
    if let Some(t) = twin {
        let zones = vg360_geom::twin::zones(&t.json);
        for c in &t.cameras {
            let Ok(url) = std::env::var(format!("VG360_CAM_{}_URL", c.id.replace('-', "_"))) else { continue };
            let (z, site) = twin_zones(c, &zones);
            let env = |k: &str| std::env::var(format!("VG360_CAM_{}_{k}", c.id.replace('-', "_"))).is_ok_and(|v| v == "1");
            v.push(Spec { id: c.id.clone(), label: format!("{} (Rig Twin{})", c.id, if c.lens.is_some() { ", fisheye" } else { "" }), url,
                          camera: c.camera.clone(), size: (c.width as usize, c.height as usize), zones: z, site, lens: c.lens.clone(),
                          auto_height: env("AUTO_HEIGHT"), config_error: None });
        }
    }
    if let Ok(url) = std::env::var("RTSP_FEED") {
        // a standalone real camera: generic pinhole from VG360_LIVE_*, no zones; mount height estimated from
        // standing people unless VG360_LIVE_HEIGHT is given (rules stay off until it is known)
        // a malformed or out-of-range VG360_LIVE_* value refuses the LIVE camera (logged, not silently defaulted)
        let url2 = url.clone();
        let live = (move || -> Result<Spec, String> {
            let (w, h) = crate::env::env_size("VG360_LIVE_SIZE")?.unwrap_or((1280, 720));
            let hfov = crate::env::env_f64("VG360_LIVE_HFOV", 10.0, 170.0)?.unwrap_or(70.0);
            let pitch = crate::env::env_f64("VG360_LIVE_PITCH", -5.0, 89.0)?.unwrap_or(25.0);
            let height = crate::env::env_f64("VG360_LIVE_HEIGHT", 0.5, 100.0)?;
            let camera = Camera { f_px: (w as f64 / 2.0) / (hfov / 2.0).to_radians().tan(), cx: w as f64 / 2.0, cy: h as f64 / 2.0,
                                  height_m: height.unwrap_or(6.0), pitch_deg: pitch, yaw_deg: 0.0, roll_deg: 0.0, x0: 0.0, y0: 0.0 };
            Ok(Spec { id: "LIVE".into(), label: "Live RTSP camera (generic pinhole, no zones)".into(), url, camera, size: (w, h),
                      zones: vec![], site: SiteMap { zones: vec![], load_zone: None, walkway: Some("Deck".into()), near_default_m: 3.0 },
                      lens: None, auto_height: height.is_none(), config_error: None })
        })();
        match live {
            Ok(spec) => v.push(spec),
            Err(e) => {
                eprintln!("LIVE camera not started: {e}");
                v.push(Spec { id: "LIVE".into(), label: "Live RTSP camera (not started: bad configuration)".into(), url: url2,
                              camera: Camera { f_px: 1000.0, cx: 640.0, cy: 360.0, height_m: 6.0, pitch_deg: 25.0, yaw_deg: 0.0, roll_deg: 0.0, x0: 0.0, y0: 0.0 },
                              size: (1280, 720), zones: vec![], site: SiteMap { zones: vec![], load_zone: None, walkway: Some("Deck".into()), near_default_m: 3.0 },
                              lens: None, auto_height: false, config_error: Some(format!("configuration: {e}")) });
            }
        }
    } else if std::env::var("VG360_DEMO").is_ok_and(|d| d == "1") {
        let url = std::env::var("VG360_DEMO_CLIP").unwrap_or_else(|_| format!("{root}/../../testdata/clip/clip.mp4"));
        // the simulated clip's own camera: 10 m mast, 25° down, 70° HFOV, 1280×720
        let camera = Camera { f_px: 640.0 / 35f64.to_radians().tan(), cx: 640.0, cy: 360.0, height_m: 10.0, pitch_deg: 25.0, yaw_deg: 0.0, roll_deg: 0.0, x0: 0.0, y0: 0.0 };
        v.push(Spec { id: "DEMO".into(), label: "Rig demo clip (simulated render, replayed as a camera)".into(), url, camera: camera.clone(),
                      size: (1280, 720), zones: default_zones(), site: SiteMap::rig_default(), lens: None, auto_height: false, config_error: None });
        if std::env::var("VG360_DEMO_FISHEYE").is_ok_and(|d| d == "1") {
            // the same clip resampled into a 185° fisheye on the same mast (examples/fisheye_clip.rs)
            v.push(Spec { id: "DEMO-FE".into(), label: "Rig demo clip as a 185° fisheye (3 virtual views)".into(),
                          url: format!("{root}/../../testdata/clip/clip_fisheye.mp4"), camera, size: (2880, 2880), zones: default_zones(),
                          site: SiteMap::rig_default(), auto_height: false, config_error: None,
                          lens: Some(LensSpec { fx: 892.0, fy: 892.0, cx: 1440.0, cy: 1440.0, k: [0.02, -0.01, 0.002, -0.0005],
                                                views: Some(vec![[-30.0, 25.0], [0.0, 25.0], [30.0, 25.0]]) }) });
        }
    }
    v
}

/// The person detector to run: the stock COCO model, or a fine-tuned one from the training notebook
/// (`VG360_DETECTOR=models/yolox_nano_ft.onnx`), which must match its card and must have been trained on real
/// site frames (`VG360_ALLOW_SMOKE_MODELS=1` overrides that for testing a smoke-set model).
pub fn detector(root: &str) -> (String, Result<Option<vg360_infer::card::Card>, String>) {
    match std::env::var("VG360_DETECTOR") {
        Err(_) => (format!("{root}/../../models/yolox_nano.onnx"), Ok(None)),
        Ok(p) => {
            let c = vg360_infer::card::verify(&p).and_then(|c| {
                if c.real_data() || std::env::var("VG360_ALLOW_SMOKE_MODELS").is_ok_and(|v| v == "1") { Ok(Some(c)) }
                else { Err(format!("{p} was trained on {} — not served (VG360_ALLOW_SMOKE_MODELS=1 to test it)", c.data_source())) }
            });
            (p, c)
        }
    }
}

fn load_detector(root: &str) -> Result<(Yolox, Option<Ppe>, Provenance), String> {
    let m = |f: &str| format!("{root}/../../models/{f}");
    let (det_path, det_card) = detector(root);
    det_card?;
    let sha = |p: &str| std::fs::read(p).map(|b| vg360_evidence::sha256_hex(&b)).unwrap_or_default();
    #[cfg(feature = "ort")]
    let (det, ppe) = {
        let n = std::thread::available_parallelism().map_or(1, |n| n.get());
        (Yolox::load_ort(&det_path, 416, n).map_err(|e| e.to_string())?, Ppe::load_ort(m("ppe_clifford_folded.onnx"), 48).ok())
    };
    #[cfg(not(feature = "ort"))]
    let (det, ppe) = (Yolox::load_tract(&det_path, 416).map_err(|e| format!("{e} (run scripts/fetch_models.sh)"))?,
                      Ppe::load_tract(m("ppe_clifford_folded.onnx"), 48).ok());
    // the PPE model runs only if it is the exact model its training-notebook card describes
    let ppe = match vg360_infer::card::verify(m("ppe_clifford_folded.onnx")) {
        Ok(_) => ppe,
        Err(e) => { eprintln!("PPE classifier disabled: {e}"); None }
    };
    let prov = Provenance { model_sha256: sha(&det_path), ppe_model_sha256: Some(sha(&m("ppe_clifford_folded.onnx"))), calibration_sha256: String::new() };
    Ok((det, ppe, prov))
}

/// Start a worker for every configured stream. Returns immediately; workers run on their own threads.
pub fn start(root: &str, deployed_json: &str, validation: Option<&Value>, twin: Option<&crate::state::TwinState>, state_dir: PathBuf) -> Arc<Live> {
    let (tx, _) = broadcast::channel(256);
    let live = Arc::new(Live { cams: RwLock::new(BTreeMap::new()), alerts: RwLock::new(VecDeque::new()),
                               signals: Arc::new(RwLock::new(Signals::default())), tx });
    let tiles: usize = std::env::var("VG360_TILES").ok().and_then(|t| t.parse().ok()).unwrap_or(1);
    let fps: f64 = std::env::var("VG360_FPS").ok().and_then(|t| t.parse().ok()).unwrap_or(10.0);
    let specs = specs(root, twin);
    if !specs.is_empty() {
        // record which models this site runs, with the notebook's model card (MCC, data, SAM settings)
        if let (Some(v), Ok(card)) = (Vesdb::from_env(), vg360_infer::card::verify(format!("{root}/../../models/ppe_clifford_folded.onnx"))) {
            let _ = v.model_deployed("ppe_clifford_folded", card.sha256(), json!({ "card": card.json }));
        }
        if let (Some(v), (_, Ok(Some(card)))) = (Vesdb::from_env(), detector(root)) {
            let _ = v.model_deployed("yolox_nano_ft", card.sha256(), json!({ "card": card.json }));
        }
    }
    for s in specs {
        live.cams.write().unwrap().insert(s.id.clone(), CamLive { label: s.label.clone(), source: if s.url.starts_with("rtsp") { "rtsp".into() } else { "file".into() },
            summary: None, jpeg: None, frames: 0, fps: 0.0, error: s.config_error.clone(), size: s.size, fisheye: s.lens.is_some(), last_push: None, last_frame_at: None, guard_key: None, started: Instant::now() });
        if s.config_error.is_some() { continue; }
        let (live2, root2, cfg_json, val, sd) = (live.clone(), root.to_string(), deployed_json.to_string(), validation.cloned(), state_dir.clone());
        std::thread::spawn(move || {
            let fail = |e: String| {
                eprintln!("camera {}: {e}", s.id);
                if let Some(c) = live2.cams.write().unwrap().get_mut(&s.id) { c.error = Some(e.clone()); }
                let _ = live2.tx.send(("camera-error".into(), json!({ "camera": s.id, "error": e }).to_string()));
            };
            let (det, ppe, mut prov) = match load_detector(&root2) { Ok(x) => x, Err(e) => return fail(e) };
            let dep = match Deployed::load(&cfg_json) { Ok(d) => d, Err(e) => return fail(e) };
            prov.calibration_sha256 = vg360_evidence::sha256_hex(serde_json::to_string(&s.camera).unwrap_or_default().as_bytes());
            let rules = Arc::new(RuleSet::new(dep, s.site, val.as_ref()));
            let mut w = CameraWorker::new(&s.id, det, ppe, Scene::new(s.camera.clone(), s.zones), rules,
                                          WorkerCfg { tiles, snapshot_every: 2, auto_height_boxes: if s.auto_height { 12 } else { 0 }, reference_dir: Some(sd.join("reference")), ..Default::default() });
            if let Some(l) = &s.lens {
                let fc = vg360_infer::fisheye::FisheyeCam::from_pose(vg360_vision::lens::Fisheye { fx: l.fx, fy: l.fy, cx: l.cx, cy: l.cy, k: l.k }, &s.camera);
                let views = l.views.as_ref().map(|v| v.iter().map(|p| (p[0], p[1])).collect());
                w = w.with_fisheye(vg360_pipeline::FisheyeRig::new(fc, s.camera.yaw_deg, s.camera.pitch_deg, views, (960, 540)));
            }
            if let Some(v) = Vesdb::from_env() { w = w.with_evidence(Arc::new(v), prov); }
            let sig = live2.signals.clone();
            let rereference = std::env::var(format!("VG360_CAM_{}_REREFERENCE", s.id.replace('-', "_"))).is_ok_and(|v| v == "1");
            let policy = RefPolicy { capture_s: REF_CAPTURE_S, dir: Some(sd.join("reference")), rereference, ..Default::default() };
            let rx = vg360_pipeline::spawn_stream_ref(w, s.url, s.size.0, s.size.1, fps, policy, move |_| sig.read().unwrap().clone());
            for r in rx {
                let r = match r {
                    Ok(r) => r,
                    // non-fatal (stalled stream, unusable reference): show it and keep going
                    Err(e) => { fail(e); continue; }
                };
                for a in &r.alerts {
                    if let (Some(id), Some(jpg)) = (&a.incident, &a.snapshot) {
                        let dir = sd.join("evidence");
                        let _ = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(dir.join(format!("{id}.jpg")), jpg));
                    }
                    let j = alert_json(&r, a);
                    { let mut al = live2.alerts.write().unwrap(); al.push_front(j.clone()); al.truncate(200); }
                    let _ = live2.tx.send(("alert".into(), j.to_string()));
                }
                let push = {
                    let mut cams = live2.cams.write().unwrap();
                    let Some(c) = cams.get_mut(&s.id) else { return };
                    c.frames += 1;
                    c.fps = c.frames as f64 / c.started.elapsed().as_secs_f64().max(1e-3);
                    c.last_frame_at = Some(Instant::now());
                    let recovered = c.error.take().is_some();
                    let sum = summarise(&r);
                    if let Some(g) = guard_event(&s.id, &sum, &mut c.guard_key) { let _ = live2.tx.send(("guard".into(), g.to_string())); }
                    if recovered { let _ = live2.tx.send(("camera-ok".into(), json!({ "camera": s.id }).to_string())); }
                    c.summary = Some(sum);
                    if let Some(j) = r.jpeg.clone() { c.jpeg = Some(Arc::new(j)); }
                    let due = c.last_push.is_none_or(|t| t.elapsed().as_millis() >= 250);
                    if due { c.last_push = Some(Instant::now()); }
                    due.then(|| json!({ "camera": s.id, "fps": c.fps, "frames": c.frames, "error": c.error, "latest": c.summary }).to_string())
                };
                if let Some(p) = push { let _ = live2.tx.send(("frame".into(), p)); }
            }
        });
    }
    live
}

#[cfg(test)]
mod tests {
    use super::*;
    fn w(st: Status, m: Mode, p: Option<(&str, f64)>, sus: Option<&str>, ah: Option<&str>, e: Option<&str>, stale: Option<f64>) -> (&'static str, String) {
        warning(st, m, p, sus, ah, e, stale)
    }
    #[test]
    fn warning_while_the_guard_is_still_confirming() {
        // a covered lens: suspended analytics win over the generic "camera movement" pending text
        let (l, t) = w(Status::Ok, Mode::Aligned, Some(("Blocked", 2.0)), Some("lens blocked"), None, None, None);
        assert_eq!(l, "warn"); assert!(t.starts_with("Analytics suspended: lens blocked — confirming (2 s)"), "{t}");
        let (_, t) = w(Status::Ok, Mode::Aligned, Some(("ViewLost", 4.0)), None, None, None, None);
        assert!(t.starts_with("View not recognised — confirming"), "{t}");
        let (_, t) = w(Status::Ok, Mode::Paused, Some(("Moved", 1.0)), None, None, None, None);
        assert!(t.starts_with("Possible camera movement") && t.ends_with("analytics paused meanwhile"), "{t}");
        // a big knock before the hold timer starts must not read "Position correct"
        let (l, t) = w(Status::Ok, Mode::Paused, None, None, None, None, None);
        assert_eq!(l, "warn"); assert!(t.contains("analytics paused"), "{t}");
        let (l, t) = w(Status::Ok, Mode::Compensated, None, None, None, None, None);
        assert_eq!(l, "warn"); assert!(t.starts_with("Small camera shift"), "{t}");
    }

    #[test]
    fn warning_branches() {
        let a = Mode::Aligned;
        assert_eq!(w(Status::Ok, a, None, None, None, None, None), ("ok", "Position correct".into()));
        assert_eq!(w(Status::Ok, a, None, None, Some("set"), None, Some(1.0)).0, "ok");
        let (l, t) = w(Status::Ok, a, None, None, None, Some("end of stream"), None);
        assert_eq!(l, "alarm"); assert!(t.starts_with("No signal:"));
        let (l, t) = w(Status::Ok, a, None, None, None, None, Some(9.0));
        assert_eq!(l, "alarm"); assert!(t.starts_with("Stream stalled"));
        assert!(w(Status::Blocked, Mode::Paused, None, None, None, None, None).1.starts_with("Lens blocked"));
        assert_eq!(w(Status::ViewLost, Mode::Paused, None, None, None, None, None).0, "alarm");
        assert_eq!(w(Status::Moved, Mode::Compensated, None, None, None, None, None).0, "warn");
        let (l, t) = w(Status::Moved, Mode::Paused, None, None, None, None, None);
        assert_eq!(l, "alarm"); assert!(t.contains("analytics paused"));
        assert_eq!(w(Status::Realigning, a, None, None, None, None, None).1, "Realigning — hold the camera steady");
        assert_eq!(w(Status::Ok, a, Some(("moved", 2.4)), None, None, None, None).1, "Possible camera movement — confirming (2 s)");
        assert_eq!(w(Status::Ok, a, None, Some("lens blocked"), None, None, None).1, "Analytics suspended: lens blocked");
        assert_eq!(w(Status::Ok, a, None, None, Some("estimating"), None, None).1, "Estimating mount height — rules off");
        assert_eq!(w(Status::Ok, a, None, None, Some("failed"), None, None).0, "alarm");
    }
    #[test]
    fn guard_event_only_on_change() {
        let s = json!({ "status": "Ok", "mode": "Aligned", "status_label": "POSITION CORRECT", "guidance": [], "warning": { "level": "ok", "text": "Position correct" } });
        let mut prev = None;
        assert!(guard_event("C", &s, &mut prev).is_some());
        assert!(guard_event("C", &s, &mut prev).is_none());
        let s2 = json!({ "status": "Moved", "mode": "Paused", "warning": { "level": "alarm", "text": "x" } });
        assert!(guard_event("C", &s2, &mut prev).is_some());
    }
}
