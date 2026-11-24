//! One camera, end to end: camera guard → (shift compensation) → person detection → scene
//! (deck geometry, tracks, zones, falls) → IOGP rules (iogp-runtime, rules approved in iogp-editor)
//! → tamper-evident evidence in vesdb.
//!
//! **Alarms.** A rule raises an alarm only when it is *armed*: approved in iogp-editor (four-eyes
//! sign-off) **and** its validated MCC clears the floor for its severity. Any other rule runs in shadow
//! mode — evaluated and recorded as evidence with `armed: false`, never sounded.
//!
//! **Shift compensation.** The guard measures the homography H from the reference view (the one the
//! deck calibration was made on) to the live view. When the camera has been knocked by a small amount,
//! detections are mapped back through H⁻¹ into reference pixels and the *original* calibration keeps
//! working — analytics continue (mode [`Mode::Compensated`]) while the technician alarm still asks for
//! the camera to be put back. Only when the move is too large to trust, or the view is blocked / lost,
//! do analytics pause ([`Mode::Paused`]). This is the composition "ground ∘ H⁻¹" of
//! `vg360_geom::camera::Camera::ground_through`, applied to whole boxes.
use std::collections::BTreeSet;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use vg360_guard::aligner::{Aligner, Measure};
use vg360_guard::config::{ANALYSIS_W, CONFIRM_INLIERS};
use vg360_guard::grabber::Grabber;
use vg360_guard::policy::{Guard, Observation, Pose, Status, is_aligned};
use vg360_infer::ppe::Ppe;
use vg360_infer::yolox::{Det, Yolox};
use serde_json::json;
use vg360_evidence::{Vesdb, sha256_hex};
use vg360_rules::{AlertEvent, Deployed, Engine, Person, Signals, SiteMap, track_of};
use vg360_scene::scene::{Rec, Scene};
use vg360_vision::img::Bgr;
use vg360_vision::robust::{self, invert, mul};

pub type M3 = [[f64; 3]; 3];
pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// How far a moved camera may be before its calibration is no longer trusted, even with compensation.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct CompLimits { pub max_shift: f64, pub max_rot_deg: f64, pub max_scale: f64, pub min_inliers: i32 }

impl Default for CompLimits {
    fn default() -> Self { Self { max_shift: 0.08, max_rot_deg: 5.0, max_scale: 0.06, min_inliers: CONFIRM_INLIERS } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Mode {
    /// view matches the reference; calibration used as is
    Aligned,
    /// small measured move; detections mapped back through H⁻¹ before the deck geometry
    Compensated,
    /// move too large, view blocked or lost: no analytics, no alerts
    Paused,
}

#[derive(Clone, Copy, Debug)]
pub struct WorkerCfg {
    /// run the guard on every n-th frame (camera_guard: 4 Hz)
    pub guard_every: usize,
    pub tiles: usize,
    pub limits: CompLimits,
    /// rule engine drop-out tolerance (s)
    pub rule_gap_s: f64,
    /// attach a JPEG of the live frame to every n-th result (0 = never), for live views
    pub snapshot_every: usize,
}

impl Default for WorkerCfg {
    fn default() -> Self { Self { guard_every: 3, tiles: 3, limits: CompLimits::default(), rule_gap_s: 0.5, snapshot_every: 0 } }
}

#[derive(Clone, Debug, Serialize)]
pub struct FrameResult {
    pub camera: String,
    pub t: f64,
    pub status: Status,
    pub mode: Mode,
    pub pose: Option<Pose>,
    pub inliers: i32,
    /// reference-view corners in live pixels (the "correct view is here" box)
    pub ref_box: Option<[[f64; 2]; 4]>,
    /// detections in live-image pixels
    pub boxes: Vec<Det>,
    pub recs: Vec<Option<Rec>>,
    pub helmet: Vec<Option<f64>>,
    pub alerts: Vec<Alarm>,
    pub ms: f64,
    /// live-view JPEG (see `WorkerCfg::snapshot_every`)
    #[serde(skip)]
    pub jpeg: Option<Vec<u8>>,
}

/// One raised or cleared rule alert, with whether it sounds and where its evidence went.
#[derive(Clone, Debug, Serialize)]
pub struct Alarm {
    pub event: AlertEvent,
    /// track id behind the alert subject
    pub track: String,
    /// approved in iogp-editor and MCC ≥ floor: the alarm sounds; otherwise shadow mode
    pub armed: bool,
    pub mode: Mode,
    /// vesdb event ids of the incident (hazard-detected, alert-raised), or "spooled"
    pub evidence: Vec<String>,
    /// incident id written to the evidence log
    pub incident: Option<String>,
    /// SHA-256 of `snapshot` (also in the incident's `hazard-detected` event)
    pub snapshot_sha256: Option<String>,
    /// the frame the alert was raised on, as JPEG: keep it next to the evidence bundle —
    /// `vesdb-verify --file-hash snapshot.jpg=<event id>:frame_sha256` proves it is the same frame
    #[serde(skip)]
    pub snapshot: Option<Vec<u8>>,
}

/// Rules and their provenance, shared by every camera worker on the site.
pub struct RuleSet {
    pub deployed: Deployed,
    pub site: SiteMap,
    /// rule ids that may sound (approved and validated); the rest run in shadow mode
    pub armed: BTreeSet<String>,
}

impl RuleSet {
    /// `validation`: config/vg360_rules.validation.json-style `{rules: {ID: {mcc, floor}}}`.
    pub fn new(deployed: Deployed, mut site: SiteMap, validation: Option<&serde_json::Value>) -> Self {
        if let Some(m) = deployed.near_m { site.near_default_m = m; }
        let armed = deployed.ids().into_iter().filter(|id| {
            let v = validation.map(|v| &v["rules"][id]);
            let ok = v.is_some_and(|v| v["mcc"].as_f64().zip(v["floor"].as_f64()).is_some_and(|(m, f)| m >= f));
            deployed.approved(id) && ok
        }).collect();
        Self { deployed, site, armed }
    }
}

/// Hashes of what produced a decision, written with every incident.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Provenance { pub model_sha256: String, pub ppe_model_sha256: Option<String>, pub calibration_sha256: String }

/// Analytics mode for one guard measurement: aligned, small enough to compensate, or paused.
pub fn compensation_mode(m: &Measure, l: &CompLimits) -> Mode {
    if !m.ok { return Mode::Paused; }
    let p = &m.pose;
    if is_aligned(p) { return Mode::Aligned; }
    let ok = p.dx.abs() <= l.max_shift && p.dy.abs() <= l.max_shift && p.angle.abs() <= l.max_rot_deg
        && (p.scale - 1.0).abs() <= l.max_scale && m.inliers >= l.min_inliers;
    if ok { Mode::Compensated } else { Mode::Paused }
}

pub struct CameraWorker {
    pub id: String,
    pub cfg: WorkerCfg,
    aligner: Aligner,
    guard: Guard,
    det: Yolox,
    ppe: Option<Ppe>,
    pub scene: Scene,
    pub rules: Arc<RuleSet>,
    engine: Engine,
    evidence: Option<Arc<Vesdb>>,
    pub provenance: Provenance,
    frame_no: usize,
    last_reported: Option<(Status, Mode)>,
    last: Option<Measure>,
    mode: Mode,
    /// live → reference, full-resolution pixels
    cur_to_ref: M3,
}

const I3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl CameraWorker {
    pub fn new(id: &str, det: Yolox, ppe: Option<Ppe>, scene: Scene, rules: Arc<RuleSet>, cfg: WorkerCfg) -> Self {
        let engine = rules.deployed.engine();
        Self { id: id.into(), cfg, aligner: Aligner::new(), guard: Guard::default(), det, ppe, scene, rules, engine,
               evidence: None, provenance: Provenance::default(), frame_no: 0, last_reported: None, last: None, mode: Mode::Aligned, cur_to_ref: I3 }
    }

    /// Write incidents and camera-guard changes to vesdb.
    pub fn with_evidence(mut self, v: Arc<Vesdb>, provenance: Provenance) -> Self { self.evidence = Some(v); self.provenance = provenance; self }

    fn record_guard(&mut self, t: f64) {
        let now = (self.guard.status, self.mode);
        if self.last_reported == Some(now) { return; }
        self.last_reported = Some(now);
        if let Some(v) = &self.evidence {
            let pose = self.last.as_ref().filter(|m| m.ok).map(|m| m.pose);
            let _ = v.camera_event(&self.id, "camera-guard-changed", json!({ "t": t, "status": now.0, "analytics": now.1, "pose": pose,
                "inliers": self.last.as_ref().map(|m| m.inliers) }));
        }
    }

    fn incident_id(&self, e: &AlertEvent) -> String {
        format!("{}-{}-{}-{}", self.id, e.rule, track_of(&e.subject).replace('/', "_"), (e.t * 1000.0).round() as i64)
    }

    fn record_incident(&self, e: &AlertEvent, rec: Option<&Rec>, armed: bool, frame_sha256: &str) -> Vec<String> {
        let Some(v) = &self.evidence else { return vec![] };
        let id = self.incident_id(e);
        let hazard = json!({ "camera": self.id, "rule": e.rule, "subject": e.subject, "t": e.t,
            "box": rec.map(|r| r.bbox), "foot_m": rec.map(|r| r.foot), "head_z_m": rec.map(|r| r.head_z), "tilt_deg": rec.map(|r| r.tilt),
            "analytics": self.mode, "guard": self.guard.status, "frame_sha256": frame_sha256 });
        let alert = json!({ "rule": e.rule, "alert": e.alert, "lsr": e.lsr, "severity": e.severity, "proof": e.proof, "armed": armed,
            "ruleset_sha256": self.rules.deployed.sha256, "model_sha256": self.provenance.model_sha256,
            "ppe_model_sha256": self.provenance.ppe_model_sha256, "calibration_sha256": self.provenance.calibration_sha256 });
        match v.incident(&id, hazard, alert) {
            Ok(vg360_evidence::Outcome::Written(evs)) => evs.into_iter().map(|e| e.id).collect(),
            Ok(vg360_evidence::Outcome::Spooled) => vec!["spooled".into()],
            Err(err) => vec![format!("error: {err}")],
        }
    }

    /// Capture the reference view (the view the deck calibration belongs to). `extra` frames from the
    /// next few seconds let the aligner keep only static background points.
    pub fn set_reference(&mut self, reference: &Bgr, extra: &[Bgr]) -> usize {
        self.mode = Mode::Aligned;
        self.cur_to_ref = I3;
        self.aligner.set_reference(reference, extra, None)
    }

    fn decide(&self, m: &Measure, img_w: usize) -> (Mode, M3) {
        let mode = compensation_mode(m, &self.cfg.limits);
        if mode != Mode::Compensated { return (mode, I3); }
        // aligner H is reference → live at analysis resolution; lift to full resolution and invert
        let s = ANALYSIS_W as f64 / img_w as f64;
        let sm: M3 = [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]];
        let si: M3 = [[1.0 / s, 0.0, 0.0], [0.0, 1.0 / s, 0.0], [0.0, 0.0, 1.0]];
        let ref_to_cur = mul(&si, &mul(&m.h, &sm));
        match invert(&ref_to_cur) { Some(h) => (Mode::Compensated, h), None => (Mode::Paused, I3) }
    }

    /// Map a live-image box into reference pixels (axis-aligned hull of the mapped corners).
    pub fn to_reference(&self, d: &Det) -> [f64; 4] {
        let cs = [[d.x1, d.y1], [d.x2, d.y1], [d.x2, d.y2], [d.x1, d.y2]].map(|p| robust::project(&self.cur_to_ref, p));
        [cs.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min), cs.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
         cs.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max), cs.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max)]
    }

    pub fn mode(&self) -> Mode { self.mode }

    /// Process one live frame at time `t` (seconds) with the site context (crane load, pipe handling …).
    pub fn process(&mut self, frame: &Bgr, t: f64, _dt: f64, sig: &Signals) -> Result<FrameResult, Error> {
        let t0 = Instant::now();
        if self.frame_no % self.cfg.guard_every.max(1) == 0 {
            let m = self.aligner.measure(frame);
            let obs = if m.blocked { Observation::Blocked } else if m.ok { Observation::Fix(m.pose) } else { Observation::NoFix };
            self.guard.step(t, obs);
            let (mode, h) = self.decide(&m, frame.w);
            // blocked / lost only pause once the guard has confirmed it (hold time), like the alarm
            self.mode = match (mode, self.guard.status) {
                (Mode::Paused, Status::Ok) if !m.ok => self.mode,
                _ => mode,
            };
            if self.mode == Mode::Compensated { self.cur_to_ref = h; } else if self.mode == Mode::Aligned { self.cur_to_ref = I3; }
            self.last = Some(m);
            self.record_guard(t);
        }
        self.frame_no += 1;
        let (pose, inliers, ref_box) = self.last.as_ref().map_or((None, 0, None), |m| (m.ok.then_some(m.pose), m.inliers, m.ok.then_some(m.boxq)));
        let mut res = FrameResult { camera: self.id.clone(), t, status: self.guard.status, mode: self.mode, pose, inliers, ref_box,
                                    boxes: vec![], recs: vec![], helmet: vec![], alerts: vec![], ms: 0.0, jpeg: None };
        if self.mode != Mode::Paused {
            let gray = frame.to_gray();
            let boxes = self.det.detect_tiled(&gray, self.cfg.tiles, 0.25)?;
            let helmet = match &self.ppe { Some(p) => p.classify(&gray, &boxes)?, None => vec![None; boxes.len()] };
            let mapped: Vec<[f64; 4]> = boxes.iter().map(|d| self.to_reference(d)).collect();
            let recs = self.scene.analyse(&mapped, t);
            let people: Vec<Person> = recs.iter().zip(&helmet).filter_map(|(r, h)| r.as_ref().map(|r|
                Person { id: r.track.to_string(), clear: &r.zones, down: r.state.down, p_helmet: *h })).collect();
            let events = vg360_rules::step(&mut self.engine, &self.rules.site, t, Some(&self.id), sig, &people);
            if !events.is_empty() {
                // the evidence frame is the JPEG that is kept, so its hash can be checked later
                let jpeg = frame.encode_jpeg(90);
                let frame_sha256 = sha256_hex(&jpeg);
                for e in events {
                    let track = track_of(&e.subject).to_string();
                    let armed = self.rules.armed.contains(&e.rule);
                    let rec = recs.iter().flatten().find(|r| r.track.to_string() == track);
                    let raised = e.kind == "raised";
                    let evidence = if raised { self.record_incident(&e, rec, armed, &frame_sha256) } else { vec![] };
                    if raised { self.scene.log_rule(t, &e.rule, &track); }
                    let incident = raised.then(|| self.incident_id(&e));
                    res.alerts.push(Alarm { event: e, track, armed, mode: self.mode, evidence, incident,
                        snapshot_sha256: raised.then(|| frame_sha256.clone()), snapshot: raised.then(|| jpeg.clone()) });
                }
            }
            res.boxes = boxes;
            res.recs = recs;
            res.helmet = helmet;
        }
        res.ms = t0.elapsed().as_secs_f64() * 1e3;
        let n = self.cfg.snapshot_every;
        if n > 0 && (self.frame_no - 1) % n == 0 { res.jpeg = Some(frame.encode_jpeg(75)); }
        Ok(res)
    }
}

/// Run a worker on a live stream (RTSP URL or a file replayed as a camera) in its own thread.
/// The first `ref_frames` frames become the reference view. Results arrive on the returned channel;
/// dropping the receiver stops the worker.
pub fn spawn_stream(mut w: CameraWorker, url: String, width: usize, height: usize, fps: f64, ref_frames: usize,
                    signals: impl Fn(f64) -> Signals + Send + 'static) -> mpsc::Receiver<Result<FrameResult, String>> {
    let (tx, rx) = mpsc::sync_channel(8);
    thread::spawn(move || {
        let g = Grabber::start(&url, width, height, false);
        let dt = 1.0 / fps;
        let next = |seen: &mut u64| -> Option<Bgr> {
            let t0 = Instant::now();
            loop {
                let n = g.frames();
                if n > *seen { *seen = n; return g.read(); }
                if t0.elapsed() > Duration::from_secs(30) { return None; }
                thread::sleep(Duration::from_millis(5));
            }
        };
        let mut seen = 0u64;
        let mut refs = Vec::new();
        while refs.len() < ref_frames.max(1) { match next(&mut seen) { Some(f) => refs.push(f), None => { let _ = tx.send(Err("no frames from stream".into())); g.stop(); return; } } }
        w.set_reference(&refs[0], &refs[1..]);
        let start = Instant::now();
        while let Some(f) = next(&mut seen) {
            let t = start.elapsed().as_secs_f64();
            if tx.send(w.process(&f, t, dt, &signals(t)).map_err(|e| e.to_string())).is_err() { break; }
        }
        g.stop();
    });
    rx
}
