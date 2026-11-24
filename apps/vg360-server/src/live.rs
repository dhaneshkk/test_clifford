//! Live analytics: one camera worker thread per stream, results fanned out to the console.
//!
//! Streams come from the environment, never from the repository:
//! * `VG360_CAM_<ID>_URL` for every camera of the applied Rig Twin (e.g. `VG360_CAM_C5_URL`) — the worker
//!   uses that camera's twin calibration and the twin zones it watches;
//! * `VG360_DEMO=1` adds camera `DEMO`: the simulated rig clip (`testdata/clip/clip.mp4`) replayed in real
//!   time as if it were a camera, with its own (known) camera model.
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
use vg360_geom::twin::{TwinCamera, TwinZone};
use vg360_infer::{ppe::Ppe, yolox::Yolox};
use vg360_pipeline::{CameraWorker, FrameResult, Provenance, RuleSet, WorkerCfg};
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
    last_push: Option<Instant>,
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
            "cameras": cams.iter().map(|(id, c)| json!({ "id": id, "label": c.label, "source": c.source, "frames": c.frames,
                "fps": c.fps, "error": c.error, "latest": c.summary })).collect::<Vec<_>>(),
            "alerts": self.alerts.read().unwrap().iter().cloned().collect::<Vec<_>>(),
            "signals": { "suspended_load": self.signals.read().unwrap().suspended_load, "active": self.signals.read().unwrap().active },
        })
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
    json!({ "camera": r.camera, "t": r.t, "status": r.status, "mode": r.mode, "pose": r.pose, "inliers": r.inliers,
            "ref_box": r.ref_box, "people": people, "ms": r.ms })
}

fn alert_json(r: &FrameResult, a: &vg360_pipeline::Alarm) -> Value {
    json!({ "camera": r.camera, "t": a.event.t, "kind": a.event.kind, "rule": a.event.rule, "alert": a.event.alert, "lsr": a.event.lsr,
            "severity": a.event.severity, "track": a.track, "armed": a.armed, "mode": a.mode, "proof": a.event.proof,
            "incident": a.incident, "evidence": a.evidence, "snapshot_sha256": a.snapshot_sha256 })
}

struct Spec { id: String, label: String, url: String, camera: Camera, size: (usize, usize), zones: Vec<Zone>, site: SiteMap }

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
            v.push(Spec { id: c.id.clone(), label: format!("{} (Rig Twin)", c.id), url, camera: c.camera.clone(),
                          size: (c.width as usize, c.height as usize), zones: z, site });
        }
    }
    if std::env::var("VG360_DEMO").is_ok_and(|d| d == "1") {
        let url = std::env::var("VG360_DEMO_CLIP").unwrap_or_else(|_| format!("{root}/../../testdata/clip/clip.mp4"));
        // the simulated clip's own camera: 10 m mast, 25° down, 70° HFOV, 1280×720
        let camera = Camera { f_px: 640.0 / 35f64.to_radians().tan(), cx: 640.0, cy: 360.0, height_m: 10.0, pitch_deg: 25.0, yaw_deg: 0.0, roll_deg: 0.0, x0: 0.0, y0: 0.0 };
        v.push(Spec { id: "DEMO".into(), label: "Rig demo clip (simulated render, replayed as a camera)".into(), url, camera,
                      size: (1280, 720), zones: default_zones(), site: SiteMap::rig_default() });
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
            summary: None, jpeg: None, frames: 0, fps: 0.0, error: None, last_push: None, started: Instant::now() });
        let (live2, root2, cfg_json, val, sd) = (live.clone(), root.to_string(), deployed_json.to_string(), validation.cloned(), state_dir.clone());
        std::thread::spawn(move || {
            let fail = |e: String| { if let Some(c) = live2.cams.write().unwrap().get_mut(&s.id) { c.error = Some(e); } };
            let (det, ppe, mut prov) = match load_detector(&root2) { Ok(x) => x, Err(e) => return fail(e) };
            let dep = match Deployed::load(&cfg_json) { Ok(d) => d, Err(e) => return fail(e) };
            prov.calibration_sha256 = vg360_evidence::sha256_hex(serde_json::to_string(&s.camera).unwrap_or_default().as_bytes());
            let rules = Arc::new(RuleSet::new(dep, s.site, val.as_ref()));
            let mut w = CameraWorker::new(&s.id, det, ppe, Scene::new(s.camera, s.zones), rules,
                                          WorkerCfg { tiles, snapshot_every: 2, ..Default::default() });
            if let Some(v) = Vesdb::from_env() { w = w.with_evidence(Arc::new(v), prov); }
            let sig = live2.signals.clone();
            let rx = vg360_pipeline::spawn_stream(w, s.url, s.size.0, s.size.1, fps, 3, move |_| sig.read().unwrap().clone());
            for r in rx {
                let r = match r { Ok(r) => r, Err(e) => return fail(e) };
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
                    c.summary = Some(summarise(&r));
                    if let Some(j) = r.jpeg.clone() { c.jpeg = Some(Arc::new(j)); }
                    let due = c.last_push.is_none_or(|t| t.elapsed().as_millis() >= 250);
                    if due { c.last_push = Some(Instant::now()); }
                    due.then(|| json!({ "fps": c.fps, "frames": c.frames, "latest": c.summary }).to_string())
                };
                if let Some(p) = push { let _ = live2.tx.send(("frame".into(), p)); }
            }
        });
    }
    live
}
