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
//!
//! **Fisheye cameras** ([`CameraWorker::with_fisheye`]). Each frame is rendered into upright virtual
//! pinhole views; people are detected and placed on the deck through each view's own camera, the same
//! worker seen by two views is merged (feet closer than 0.8 m, larger box kept) and one tracker runs over
//! all of them (notebook §9b). Boxes are reported in fisheye pixels. A homography cannot describe a moved
//! fisheye, so a measured move pauses analytics instead of compensating.
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use vg360_guard::aligner::{Aligner, Measure};
use vg360_guard::config::{ANALYSIS_W, CONFIRM_INLIERS, REF_CAPTURE_S};
use vg360_guard::grabber::Grabber;
use vg360_guard::policy::{Guard, Observation, Pose, Status, is_aligned};
use vg360_infer::fisheye::{FisheyeCam, View};
use vg360_infer::ppe::Ppe;
use vg360_infer::yolox::{Det, Yolox};
use serde_json::json;
use vg360_evidence::{Vesdb, sha256_hex};
use vg360_rules::{AlertEvent, Deployed, Engine, Person, Signals, SiteMap, track_of};
use vg360_scene::scene::{Rec, Scene, box_geometry};
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

#[derive(Clone, Debug)]
pub struct WorkerCfg {
    /// run the guard on every n-th frame (camera_guard: 4 Hz)
    pub guard_every: usize,
    pub tiles: usize,
    pub limits: CompLimits,
    /// attach a JPEG of the live frame to every n-th result (0 = never), for live views
    pub snapshot_every: usize,
    /// estimate the mount height from the first `n` person boxes (assumed 1.70 m standing) before any rule is
    /// evaluated — for cameras calibrated from vanishing points, whose height is unknown (0 = off; pinhole only)
    pub auto_height_boxes: usize,
    /// distinct people (tracks, one sample each) auto-height needs, at least; the bar is `max(this, auto_height_boxes / 3)`
    pub auto_height_min_tracks: usize,
    /// a reference with fewer background points than this is refused (too dark, flat or featureless to guard)
    pub min_ref_points: usize,
    /// where `spawn_stream_ref` keeps `{camera}.png` + `{camera}.json` (None = reference is not persisted)
    pub reference_dir: Option<PathBuf>,
}

impl Default for WorkerCfg {
    fn default() -> Self { Self { guard_every: 3, tiles: 3, limits: CompLimits::default(), snapshot_every: 0, auto_height_boxes: 0, auto_height_min_tracks: 3, min_ref_points: 80, reference_dir: None } }
}

#[derive(Clone, Debug, Serialize)]
pub struct FrameResult {
    pub camera: String,
    pub t: f64,
    pub status: Status,
    pub mode: Mode,
    /// this tick's raw measured pose (None when the last guard tick had no fix)
    pub pose: Option<Pose>,
    /// median of the guard's last few fixes (what the status and the compensation decision follow)
    pub smoothed_pose: Option<Pose>,
    pub inliers: i32,
    /// background points in the reference view
    pub ref_points: i32,
    /// candidate guard status waiting out its hold time, with seconds elapsed so far
    pub guard_pending: Option<(String, f64)>,
    /// why no analytics ran on this frame ("lens blocked" / "view not recognised"): no detections, no scene,
    /// no rule evaluated, from the first bad guard tick on. The alarm `status` still follows the guard hold times.
    pub analytics_suspended: Option<String>,
    /// age of the frame at the grabber when it was handed over (spawn_stream only)
    pub frame_age_s: Option<f64>,
    /// auto-height progress (None when auto-height is off)
    pub auto_height: Option<AutoHeight>,
    /// reference-view corners in live pixels (the "correct view is here" box)
    pub ref_box: Option<[[f64; 2]; 4]>,
    /// detections in live-image pixels
    pub boxes: Vec<Det>,
    pub recs: Vec<Option<Rec>>,
    pub helmet: Vec<Option<f64>>,
    pub alerts: Vec<Alarm>,
    pub ms: f64,
    /// mount height in use (m); changes once when auto-height finishes
    pub height_m: f64,
    /// auto-height still collecting boxes: people are tracked but no rule is evaluated. After a failed
    /// estimate this is false but rules stay off (see `auto_height.state == "failed"`).
    pub calibrating: bool,
    /// live-view JPEG (see `WorkerCfg::snapshot_every`)
    #[serde(skip)]
    pub jpeg: Option<Vec<u8>>,
}

/// Mount-height estimation progress. `state`: "estimating" | "set" | "failed".
#[derive(Clone, Debug, Serialize)]
pub struct AutoHeight { pub state: &'static str, pub value: Option<f64>, pub samples: usize }

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AhState { Estimating, Set, Failed }

/// Camera events (guard changes, reference, height) are written to vesdb from a background thread so the
/// frame loop never waits on HTTP. Bounded: when vesdb is slower than the events arrive, they are dropped and counted.
struct EvidenceQueue { tx: mpsc::SyncSender<(String, serde_json::Value)>, errors: Arc<AtomicU64> }

impl EvidenceQueue {
    fn new(v: Arc<Vesdb>, camera: String) -> Self {
        let (tx, rx) = mpsc::sync_channel::<(String, serde_json::Value)>(64);
        let errors = Arc::new(AtomicU64::new(0));
        let e2 = errors.clone();
        thread::spawn(move || {
            for (ty, data) in rx {
                if let Err(e) = v.camera_event(&camera, &ty, data) {
                    e2.fetch_add(1, Ordering::Relaxed);
                    let msg: String = e.to_string().chars().take(160).collect();
                    eprintln!("vg360-pipeline: evidence event {ty} for {camera} failed: {msg}");
                }
            }
        });
        Self { tx, errors }
    }
    fn push(&self, ty: &str, data: serde_json::Value) {
        if self.tx.try_send((ty.to_string(), data)).is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
            eprintln!("vg360-pipeline: evidence queue full, dropped {ty}");
        }
    }
}

/// What a reference capture produced.
#[derive(Clone, Debug)]
pub struct RefInfo { pub points: usize, pub mean: f64, pub std: f64, pub sha256: String }

/// Why a candidate reference view is unusable (None = usable).
pub fn reference_problem(q: &vg360_guard::aligner::RefQuality, min_points: usize) -> Option<String> {
    if q.mean < 15.0 { return Some(format!("too dark (mean {:.0})", q.mean)); }
    if q.std < 10.0 { return Some(format!("flat, no contrast (std {:.1})", q.std)); }
    if q.points < min_points { return Some(format!("only {} background points, need {min_points}", q.points)); }
    None
}

fn lift(m: &Measure, img_w: usize) -> Option<M3> {
    // aligner H is reference → live at analysis resolution; lift to full resolution and invert
    let s = ANALYSIS_W as f64 / img_w as f64;
    let sm: M3 = [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]];
    let si: M3 = [[1.0 / s, 0.0, 0.0], [0.0, 1.0 / s, 0.0], [0.0, 0.0, 1.0]];
    let h = invert(&mul(&si, &mul(&m.h, &sm)))?;
    h.iter().flatten().all(|v| v.is_finite()).then_some(h)
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
    ev_queue: Option<EvidenceQueue>,
    pub provenance: Provenance,
    frame_no: usize,
    last_reported: Option<(Status, Mode)>,
    last_mode_event_t: f64,
    last: Option<Measure>,
    mode: Mode,
    suspended: Option<&'static str>,
    /// live → reference, full-resolution pixels
    cur_to_ref: M3,
    fisheye: Option<FisheyeRig>,
    ah_state: AhState,
    ah_tracks: BTreeMap<u64, Vec<(f64, [f64; 4])>>,
    ah_rejects: u32,
    ah_t0: Option<f64>,
    ah_value: Option<f64>,
}

/// A fisheye camera and the upright virtual views it is analysed through.
pub struct FisheyeRig { pub cam: FisheyeCam, pub views: Vec<View> }

impl FisheyeRig {
    /// Default views: looking down (pitch ≥ 60°) → three views 120° apart at 35° down; otherwise three views
    /// at −45°, 0°, +45° around the camera's heading at its pitch. 70° HFOV, `size` pixels each
    /// (960×540 is a good default: people stay as large as in a 1080p pinhole view of the same area).
    pub fn new(cam: FisheyeCam, yaw_deg: f64, pitch_deg: f64, views: Option<Vec<(f64, f64)>>, size: (usize, usize)) -> Self {
        let specs = views.unwrap_or_else(|| if pitch_deg >= 60.0 {
            vec![(yaw_deg, 35.0), (yaw_deg + 120.0, 35.0), (yaw_deg + 240.0, 35.0)]
        } else {
            vec![(yaw_deg - 45.0, pitch_deg), (yaw_deg, pitch_deg), (yaw_deg + 45.0, pitch_deg)]
        });
        let views = cam.views(&specs, size.0, size.1, 70.0);
        Self { cam, views }
    }

    /// View box → axis-aligned box in fisheye pixels (hull of the mapped border points).
    pub fn to_fisheye(&self, view: usize, d: &Det) -> Option<[f64; 4]> {
        let v = &self.views[view];
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for k in 0..=8 {
            let a = k as f64 / 8.0;
            for (x, y) in [(d.x1 + a * (d.x2 - d.x1), d.y1), (d.x1 + a * (d.x2 - d.x1), d.y2), (d.x1, d.y1 + a * (d.y2 - d.y1)), (d.x2, d.y1 + a * (d.y2 - d.y1))] {
                let (u, w) = ((x.round() as isize).clamp(0, v.w as isize - 1) as usize, (y.round() as isize).clamp(0, v.h as isize - 1) as usize);
                let (fx, fy) = v.map[w * v.w + u];
                if fx < 0.0 { continue; }
                b = [b[0].min(fx as f64), b[1].min(fy as f64), b[2].max(fx as f64), b[3].max(fy as f64)];
            }
        }
        b[0].is_finite().then_some(b)
    }
}

const I3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const AH_MAX_REJECTS: u32 = 5;
const AH_TIMEOUT_S: f64 = 300.0;
/// at most this many samples per tracked person, at least `AH_SPACING_S` apart
const AH_PER_TRACK: usize = 4;
const AH_SPACING_S: f64 = 0.5;

fn within(l: &CompLimits, p: &Pose) -> bool {
    p.dx.abs() <= l.max_shift && p.dy.abs() <= l.max_shift && p.angle.abs() <= l.max_rot_deg && (p.scale - 1.0).abs() <= l.max_scale
}

impl CameraWorker {
    pub fn new(id: &str, det: Yolox, ppe: Option<Ppe>, scene: Scene, rules: Arc<RuleSet>, cfg: WorkerCfg) -> Self {
        let engine = rules.deployed.engine();
        let mut w = Self { id: id.into(), cfg, aligner: Aligner::new(), guard: Guard::default(), det, ppe, scene, rules, engine,
               evidence: None, ev_queue: None, provenance: Provenance::default(), frame_no: 0, last_reported: None, last_mode_event_t: f64::NEG_INFINITY,
               last: None, mode: Mode::Aligned, suspended: None, cur_to_ref: I3, fisheye: None,
               ah_state: AhState::Estimating, ah_tracks: BTreeMap::new(), ah_rejects: 0, ah_t0: None, ah_value: None };
        w.refresh_calibration_hash();
        w
    }

    /// Analyse this camera as a fisheye through upright virtual views (see the module docs).
    pub fn with_fisheye(mut self, rig: FisheyeRig) -> Self { self.fisheye = Some(rig); self }

    pub fn is_fisheye(&self) -> bool { self.fisheye.is_some() }

    /// Write incidents and camera-guard changes to vesdb (camera events go through a background thread).
    pub fn with_evidence(mut self, v: Arc<Vesdb>, provenance: Provenance) -> Self {
        self.ev_queue = Some(EvidenceQueue::new(v.clone(), self.id.clone()));
        self.evidence = Some(v);
        self.provenance = provenance;
        self
    }

    /// Camera events that could not be written (vesdb errors) or were dropped (queue full) so far.
    pub fn evidence_errors(&self) -> u64 { self.ev_queue.as_ref().map_or(0, |q| q.errors.load(Ordering::Relaxed)) }

    /// Hashes of what is producing decisions right now (the calibration hash follows the camera in use).
    pub fn provenance(&self) -> Option<&Provenance> { Some(&self.provenance) }

    fn refresh_calibration_hash(&mut self) {
        self.provenance.calibration_sha256 = sha256_hex(serde_json::to_string(&self.scene.cam).unwrap_or_default().as_bytes());
    }

    fn camera_event(&self, ty: &str, data: serde_json::Value) {
        if let Some(q) = &self.ev_queue { q.push(ty, data); }
    }

    /// Debounced: every status change is recorded; a mode-only flip at most every 2 s.
    fn record_guard(&mut self, t: f64) {
        let now = (self.guard.status, self.mode);
        let Some(prev) = self.last_reported else { return self.emit_guard(t, now) };
        if prev == now { return; }
        if prev.0 != now.0 || t - self.last_mode_event_t >= 2.0 { self.emit_guard(t, now); }
    }

    fn emit_guard(&mut self, t: f64, now: (Status, Mode)) {
        self.last_reported = Some(now);
        self.last_mode_event_t = t;
        let pose = self.last.as_ref().filter(|m| m.ok).map(|m| m.pose);
        let inliers = self.last.as_ref().map(|m| m.inliers);
        self.camera_event("camera-guard-changed", json!({ "t": t, "status": now.0, "analytics": now.1, "pose": pose, "inliers": inliers,
            "suspended": self.suspended }));
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

    /// Forget everything tied to the previous reference: guard state, mode, compensation, height estimate.
    fn reset_for_reference(&mut self) {
        self.guard.reset();
        self.mode = Mode::Aligned;
        self.cur_to_ref = I3;
        self.suspended = None;
        self.last = None;
        self.last_reported = None;
        self.frame_no = 0;
        self.ah_state = AhState::Estimating;
        self.ah_tracks.clear();
        self.ah_rejects = 0;
        self.ah_t0 = None;
        self.ah_value = None;
        self.refresh_calibration_hash();
    }

    /// Capture the reference view (the view the deck calibration belongs to). `extra` frames from the
    /// next few seconds let the aligner keep only static background points. Resets the guard, mode,
    /// compensation, auto-height state and provenance.
    pub fn set_reference(&mut self, reference: &Bgr, extra: &[Bgr]) -> usize {
        self.reset_for_reference();
        self.aligner.set_reference(reference, extra, None)
    }

    /// Restore a saved reference: `static_points` (from [`CameraWorker::reference_points`]) select the background keypoints.
    pub fn set_reference_masked(&mut self, reference: &Bgr, static_points: &[[f32; 2]]) -> usize {
        self.reset_for_reference();
        let set: HashSet<(u32, u32)> = static_points.iter().map(|p| (p[0].to_bits(), p[1].to_bits())).collect();
        let mask = move |x: f32, y: f32| set.contains(&(x.to_bits(), y.to_bits()));
        self.aligner.set_reference(reference, &[], Some(&mask))
    }

    /// Background points of the current reference (full-resolution pixels).
    pub fn reference_points(&self) -> Vec<[f32; 2]> { self.aligner.static_points() }

    pub fn reference_quality(&self) -> vg360_guard::aligner::RefQuality { self.aligner.reference_quality() }

    /// Pick the best reference from frames captured over a few seconds: the candidate (first / middle / last)
    /// with the most background points, the other frames serving as the static-background filter. Refuses a dark,
    /// flat or featureless view (`Err(why)`; the aligner then holds that unusable reference, so do not run on it).
    pub fn capture_reference(&mut self, frames: &[Bgr]) -> Result<(usize, RefInfo), String> {
        if frames.is_empty() { return Err("no frames".into()); }
        let n = frames.len();
        let mut cands = vec![0, n / 2, n - 1];
        cands.dedup();
        let extras_for = |c: usize| -> Vec<Bgr> {
            let others: Vec<usize> = (0..n).filter(|&i| i != c).collect();
            let k = others.len().min(4);
            (0..k).map(|j| frames[others[j * others.len() / k]].clone()).collect()
        };
        let mut best: Option<(usize, usize)> = None;
        let mut last_tried = 0;
        for &c in &cands {
            let pts = self.set_reference(&frames[c], &extras_for(c));
            last_tried = c;
            if best.is_none_or(|b| pts > b.1) { best = Some((c, pts)); }
        }
        let (bi, _) = best.unwrap();
        if bi != last_tried { self.set_reference(&frames[bi], &extras_for(bi)); }
        let q = self.aligner.reference_quality();
        if let Some(why) = reference_problem(&q, self.cfg.min_ref_points) { return Err(why); }
        Ok((bi, RefInfo { points: q.points, mean: q.mean, std: q.std, sha256: sha256_hex(&frames[bi].data) }))
    }

    /// Write `{dir}/{id}.png` and `{id}.json` (points, sha256 of the png, time, background points).
    /// Returns the png's sha256.
    pub fn save_reference(&self, dir: &Path, img: &Bgr) -> Result<String, Error> {
        std::fs::create_dir_all(dir)?;
        let png = dir.join(format!("{}.png", self.id));
        img.save(&png)?;
        let sha = sha256_hex(&std::fs::read(&png)?);
        let q = self.aligner.reference_quality();
        let meta = json!({ "camera": self.id, "points": q.points, "sha256": sha, "width": img.w, "height": img.h,
            "saved_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()),
            "static_points": self.reference_points() });
        std::fs::write(dir.join(format!("{}.json", self.id)), serde_json::to_vec(&meta)?)?;
        Ok(sha)
    }

    /// Load a saved reference (verifying the png hash and that it is still usable). `Ok(None)` when none is stored.
    pub fn load_reference(&mut self, dir: &Path) -> Result<Option<RefInfo>, String> {
        let (png, js) = (dir.join(format!("{}.png", self.id)), dir.join(format!("{}.json", self.id)));
        if !png.exists() || !js.exists() { return Ok(None); }
        let bytes = std::fs::read(&png).map_err(|e| e.to_string())?;
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(&js).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let sha = sha256_hex(&bytes);
        if meta["sha256"].as_str() != Some(sha.as_str()) { return Err("stored reference image does not match its recorded hash".into()); }
        let pts: Vec<[f32; 2]> = serde_json::from_value(meta["static_points"].clone()).map_err(|e| e.to_string())?;
        let img = Bgr::decode(&bytes).map_err(|e| e.to_string())?;
        self.set_reference_masked(&img, &pts);
        let q = self.aligner.reference_quality();
        if let Some(why) = reference_problem(&q, self.cfg.min_ref_points) { return Err(format!("stored reference unusable: {why}")); }
        Ok(Some(RefInfo { points: q.points, mean: q.mean, std: q.std, sha256: sha }))
    }

    /// Record that a reference view was set (vesdb camera event `reference-set`).
    pub fn note_reference(&self, info: &RefInfo, source: &str) {
        self.camera_event("reference-set", json!({ "sha256": info.sha256, "points": info.points, "mean": info.mean, "std": info.std, "source": source }));
    }

    /// live → reference matrix (full-resolution pixels) in use; identity unless compensating.
    pub fn cur_to_ref(&self) -> M3 { self.cur_to_ref }

    /// Map a live-image box into reference pixels. x: the mapped feet (bottom centre) at the original width, so
    /// a rotated view does not widen the box. y: the vertical extent of the mapped corners (head row = higher top
    /// corner, feet row = lower bottom corner), which measured closest to the un-knocked deck positions
    /// (0.24 m vs 0.28 m mean with a head/feet-point rebuild); it exceeds the true height by only ~1 % per 1.5° of rotation.
    pub fn to_reference(&self, d: &Det) -> [f64; 4] {
        let xc = (d.x1 + d.x2) / 2.0;
        let w = d.x2 - d.x1;
        let c = [[d.x1, d.y1], [d.x2, d.y1], [d.x2, d.y2], [d.x1, d.y2]].map(|p| robust::project(&self.cur_to_ref, p));
        let foot = robust::project(&self.cur_to_ref, [xc, d.y2]);
        [foot[0] - w / 2.0, c[0][1].min(c[1][1]), foot[0] + w / 2.0, c[2][1].max(c[3][1])]
    }

    pub fn mode(&self) -> Mode { self.mode }

    /// One guard measurement: status, analytics suspension, mode and compensation matrix.
    fn guard_tick(&mut self, frame: &Bgr, t: f64) {
        let m = self.aligner.measure(frame);
        let obs = if m.blocked { Observation::Blocked } else if m.ok { Observation::Fix(m.pose, m.inliers) } else { Observation::NoFix };
        self.guard.step(t, obs);
        // the first bad tick stops analytics at once; the alarm status follows the hold times
        self.suspended = if m.blocked { Some("lens blocked") } else if !m.ok { Some("view not recognised") } else { None };
        if m.ok {
            let sp = self.guard.smoothed_pose().unwrap_or(m.pose);
            let l = self.cfg.limits;
            let (mut mode, mut h) = if is_aligned(&m.pose) && is_aligned(&sp) { (Mode::Aligned, I3) }
                else if within(&l, &m.pose) && within(&l, &sp) && m.inliers >= l.min_inliers {
                    match lift(&m, frame.w) { Some(h) => (Mode::Compensated, h), None => (Mode::Paused, self.cur_to_ref) }
                } else { (Mode::Paused, self.cur_to_ref) };
            if self.fisheye.is_some() && mode == Mode::Compensated { mode = Mode::Paused; h = self.cur_to_ref; }
            self.mode = mode;
            self.cur_to_ref = h;       // only ever replaced by a valid fit; a failed tick keeps the last good matrix
        } else if matches!(self.guard.status, Status::Blocked | Status::ViewLost) {
            self.mode = Mode::Paused;
        }
        self.last = Some(m);
        self.record_guard(t);
    }

    fn ah_active(&self) -> bool { self.cfg.auto_height_boxes > 0 && self.fisheye.is_none() }

    fn ah_fail(&mut self, t: f64, reason: &str) {
        self.ah_state = AhState::Failed;
        self.camera_event("height-estimate-failed", json!({ "t": t, "reason": reason, "rejected_estimates": self.ah_rejects,
            "samples": self.ah_samples() }));
    }

    fn ah_samples(&self) -> usize { self.ah_tracks.values().map(Vec::len).sum() }

    /// Collect standing-person boxes (up to `AH_PER_TRACK` per tracked person, `AH_SPACING_S` apart) and, with enough
    /// distinct people, fix the mount height from them (`vg360_geom::autocal::height_from_people`, proved in
    /// `lean/VG360/Calib.lean`) and record it. Samples: detector score >= 0.5, >= 40 px tall, upright (aspect > 1.2),
    /// not touching the image edge; only while the view is aligned. Needs `auto_height_min_tracks` (3) distinct people
    /// and `auto_height_boxes` samples; at `AH_TIMEOUT_S` it estimates from what it has if enough people were seen,
    /// otherwise (or after five estimates outside 1.5-60 m) it is marked failed. Returns true on the frame the height is set.
    fn auto_height(&mut self, live: &[Det], mapped: &[[f64; 4]], recs: &[Option<Rec>], (iw, ih): (usize, usize), t: f64) -> bool {
        let t0 = *self.ah_t0.get_or_insert(t);
        let need_tracks = self.cfg.auto_height_min_tracks.max(1);
        let timed_out = t - t0 > AH_TIMEOUT_S;
        if timed_out && self.ah_tracks.len() < need_tracks { self.ah_fail(t, "timeout: too few distinct people seen"); return false; }
        if !timed_out {
            if self.mode != Mode::Aligned { return false; }
            let (mx, my) = (iw as f64 * 0.01, ih as f64 * 0.01);
            for ((d, b), r) in live.iter().zip(mapped).zip(recs) {
                let Some(r) = r else { continue };
                let ok = d.score >= 0.5 && d.y2 - d.y1 >= 40.0 && (d.y2 - d.y1) > 1.2 * (d.x2 - d.x1)
                    && d.x1 > mx && d.y1 > my && d.x2 < iw as f64 - mx && d.y2 < ih as f64 - my;
                if !ok { continue; }
                let v = self.ah_tracks.entry(r.track).or_default();
                if v.len() < AH_PER_TRACK && v.last().is_none_or(|(ts, _)| t - ts >= AH_SPACING_S) { v.push((t, *b)); }
            }
            let need_samples = self.cfg.auto_height_boxes.max(need_tracks);
            if self.ah_tracks.len() < need_tracks || self.ah_samples() < need_samples { return false; }
        }
        let boxes: Vec<[f64; 4]> = self.ah_tracks.values().flatten().map(|(_, b)| *b).collect();
        let was = self.scene.cam.height_m;
        match vg360_geom::autocal::height_from_people(&self.scene.cam, &boxes, 1.70) {
            Some(h) if (1.5..=60.0).contains(&h) => {
                self.scene.cam.height_m = h;
                self.scene.tracker = vg360_scene::track::Tracker::new(vg360_scene::track::TrackerCfg::default());
                self.ah_state = AhState::Set;
                self.ah_value = Some(h);
                self.refresh_calibration_hash();
                self.camera_event("height-estimated", json!({ "t": t, "height_m": h, "previous_m": was,
                    "boxes": boxes.len(), "people": self.ah_tracks.len(), "assumed_person_m": 1.70,
                    "method": "median head height of standing people, up to 4 samples per tracked person",
                    "calibration_sha256": self.provenance.calibration_sha256 }));
                true
            }
            _ => {
                self.ah_rejects += 1;
                self.ah_tracks.clear();
                if timed_out || self.ah_rejects >= AH_MAX_REJECTS { self.ah_fail(t, "estimates outside 1.5-60 m"); }
                false
            }
        }
    }

    /// Process one live frame at time `t` (seconds) with the site context (crane load, pipe handling …).
    pub fn process(&mut self, frame: &Bgr, t: f64, _dt: f64, sig: &Signals) -> Result<FrameResult, Error> {
        let t0 = Instant::now();
        if self.frame_no % self.cfg.guard_every.max(1) == 0 { self.guard_tick(frame, t); }
        self.frame_no += 1;
        let (pose, inliers, ref_box) = self.last.as_ref().map_or((None, 0, None), |m| (m.ok.then_some(m.pose), m.inliers, m.ok.then_some(m.boxq)));
        let ah_active = self.ah_active();
        let mut res = FrameResult { camera: self.id.clone(), t, status: self.guard.status, mode: self.mode, pose,
                                    smoothed_pose: self.guard.smoothed_pose(), inliers, ref_points: self.aligner.reference_quality().points as i32,
                                    guard_pending: self.guard.pending().map(|(s, since)| (format!("{s:?}"), (t - since).max(0.0))),
                                    analytics_suspended: self.suspended.map(String::from), frame_age_s: None, auto_height: None,
                                    ref_box, boxes: vec![], recs: vec![], helmet: vec![], alerts: vec![], ms: 0.0,
                                    height_m: self.scene.cam.height_m, calibrating: ah_active && self.ah_state == AhState::Estimating, jpeg: None };
        let mut skip_rules = ah_active && self.ah_state != AhState::Set;
        if self.suspended.is_none() && self.mode != Mode::Paused {
            let gray = frame.to_gray();
            let (boxes, helmet, recs) = if let Some(rig) = &self.fisheye {
                // every view: detect, classify PPE, place on the deck through the view's camera
                let mut cand: Vec<(Det, Option<f64>, ([f64; 2], f64, f64, f64), f64)> = Vec::new();
                for (vi, v) in rig.views.iter().enumerate() {
                    let img = v.render(&gray);
                    let b = self.det.detect_tiled(&img, self.cfg.tiles.min(2), 0.25)?;
                    let h = match &self.ppe { Some(p) => p.classify(&img, &b)?, None => vec![None; b.len()] };
                    for (d, p) in b.iter().zip(h) {
                        let Some(geo) = box_geometry(&v.cam, [d.x1, d.y1, d.x2, d.y2]) else { continue };
                        let Some(fb) = rig.to_fisheye(vi, d) else { continue };
                        cand.push((Det { x1: fb[0], y1: fb[1], x2: fb[2], y2: fb[3], ..*d }, p, geo, d.y2 - d.y1));
                    }
                }
                // the same worker seen by two views → keep the larger box
                cand.sort_by(|a, b| b.3.total_cmp(&a.3));
                let mut kept: Vec<(Det, Option<f64>, ([f64; 2], f64, f64, f64), f64)> = Vec::new();
                for c in cand {
                    if kept.iter().all(|k| (k.2.0[0] - c.2.0[0]).hypot(k.2.0[1] - c.2.0[1]) > 0.8) { kept.push(c); }
                }
                let fb: Vec<[f64; 4]> = kept.iter().map(|k| [k.0.x1, k.0.y1, k.0.x2, k.0.y2]).collect();
                let geo: Vec<Option<([f64; 2], f64, f64, f64)>> = kept.iter().map(|k| Some(k.2)).collect();
                let recs = self.scene.analyse_geo(&fb, &geo, t);
                (kept.iter().map(|k| k.0).collect::<Vec<_>>(), kept.iter().map(|k| k.1).collect::<Vec<_>>(), recs)
            } else {
                let boxes = self.det.detect_tiled(&gray, self.cfg.tiles, 0.25)?;
                let helmet = match &self.ppe { Some(p) => p.classify(&gray, &boxes)?, None => vec![None; boxes.len()] };
                let mapped: Vec<[f64; 4]> = boxes.iter().map(|d| self.to_reference(d)).collect();
                let recs = self.scene.analyse(&mapped, t);
                if ah_active && self.ah_state == AhState::Estimating {
                    // the frame that sets the height still runs no rule (its records use the old height)
                    if self.auto_height(&boxes, &mapped, &recs, (frame.w, frame.h), t) { res.height_m = self.scene.cam.height_m; }
                    res.calibrating = self.ah_state == AhState::Estimating;
                    skip_rules = true;
                }
                (boxes, helmet, recs)
            };

            let people: Vec<Person> = recs.iter().zip(&helmet).filter_map(|(r, h)| r.as_ref().map(|r|
                Person { id: r.track.to_string(), clear: &r.zones, down: r.state.down, p_helmet: *h })).collect();
            // no rule runs on an uncalibrated camera (auto-height still collecting, or failed)
            let events = if skip_rules { vec![] } else { vg360_rules::step(&mut self.engine, &self.rules.site, t, Some(&self.id), sig, &people) };
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
        if ah_active {
            res.auto_height = Some(AutoHeight { state: match self.ah_state { AhState::Estimating => "estimating", AhState::Set => "set", AhState::Failed => "failed" },
                                                value: self.ah_value, samples: self.ah_samples() });
        }
        res.ms = t0.elapsed().as_secs_f64() * 1e3;
        let n = self.cfg.snapshot_every;
        if n > 0 && (self.frame_no - 1) % n == 0 { res.jpeg = Some(frame.encode_jpeg(75)); }
        Ok(res)
    }
}

/// How `spawn_stream_ref` obtains the reference view.
#[derive(Clone, Debug)]
pub struct RefPolicy {
    /// seconds of frames collected to pick the best reference and its static background
    pub capture_s: f64,
    /// where the reference is kept (None: `WorkerCfg::reference_dir`, else not persisted)
    pub dir: Option<PathBuf>,
    /// ignore a stored reference and capture a new one (the new one replaces it)
    pub rereference: bool,
    /// stop capturing after this many frames instead of `capture_s` (what the old `spawn_stream` does)
    pub ref_frames: Option<usize>,
}

impl Default for RefPolicy {
    fn default() -> Self { Self { capture_s: REF_CAPTURE_S, dir: None, rereference: false, ref_frames: None } }
}

/// One step of a frame source for [`drive`].
pub enum Src {
    /// a frame and its age at the grabber (s)
    Frame(Bgr, Option<f64>),
    /// 30 s without a new frame
    Stall,
    /// the source is finished (tests, files)
    End,
}

/// Consecutive `process` errors after which the worker gives up.
pub const MAX_CONSECUTIVE_ERRORS: usize = 20;

/// The frame loop of a stream worker: results (and errors) go to `tx`. A stall is reported and waited out;
/// a `process` error is reported and the loop continues, until `MAX_CONSECUTIVE_ERRORS` in a row.
pub fn drive(w: &mut CameraWorker, next: &mut dyn FnMut() -> Src, now: &mut dyn FnMut() -> f64, dt: f64,
             signals: &dyn Fn(f64) -> Signals, tx: &mpsc::SyncSender<Result<FrameResult, String>>) {
    let mut errs = 0usize;
    loop {
        match next() {
            Src::End => return,
            Src::Stall => { if tx.send(Err("stream stalled: no frames for 30 s".into())).is_err() { return; } }
            Src::Frame(f, age) => {
                let t = now();
                match w.process(&f, t, dt, &signals(t)) {
                    Ok(mut r) => { errs = 0; r.frame_age_s = age; if tx.send(Ok(r)).is_err() { return; } }
                    Err(e) => {
                        errs += 1;
                        if tx.send(Err(e.to_string())).is_err() { return; }
                        if errs >= MAX_CONSECUTIVE_ERRORS {
                            let _ = tx.send(Err(format!("worker stopped after {errs} consecutive errors")));
                            return;
                        }
                    }
                }
            }
        }
    }
}

/// Run a worker on a live stream (RTSP URL or a file replayed as a camera) in its own thread.
/// The first `ref_frames` frames become the reference view. Results arrive on the returned channel;
/// dropping the receiver stops the worker. See [`spawn_stream_ref`] for validated, persisted references.
pub fn spawn_stream(w: CameraWorker, url: String, width: usize, height: usize, fps: f64, ref_frames: usize,
                    signals: impl Fn(f64) -> Signals + Send + 'static) -> mpsc::Receiver<Result<FrameResult, String>> {
    spawn_stream_ref(w, url, width, height, fps, RefPolicy { ref_frames: Some(ref_frames.max(1)), ..Default::default() }, signals)
}

/// Like [`spawn_stream`], with the reference chosen by `policy`: a stored reference is loaded (so a camera moved
/// while the server was down reports Moved) unless `rereference`; otherwise frames are captured over `capture_s`,
/// the best view is picked, validated (not dark / flat, at least `min_ref_points` background points; retried for
/// 30 s, then `Err("reference unusable: …")` is sent on every failed attempt) and stored.
pub fn spawn_stream_ref(mut w: CameraWorker, url: String, width: usize, height: usize, fps: f64, policy: RefPolicy,
                        signals: impl Fn(f64) -> Signals + Send + 'static) -> mpsc::Receiver<Result<FrameResult, String>> {
    let (tx, rx) = mpsc::sync_channel(8);
    thread::spawn(move || {
        let g = Grabber::start(&url, width, height, false);
        let dt = 1.0 / fps;
        let mut seen = 0u64;
        let mut next = || -> Src {
            let t0 = Instant::now();
            loop {
                let n = g.frames();
                if n > seen { seen = n; if let Some(f) = g.read() { return Src::Frame(f, g.age_s()); } }
                if t0.elapsed() > Duration::from_secs(30) { return Src::Stall; }
                thread::sleep(Duration::from_millis(5));
            }
        };
        let dir = policy.dir.clone().or_else(|| w.cfg.reference_dir.clone());
        let mut have_ref = false;
        if let (Some(d), false) = (&dir, policy.rereference) {
            match w.load_reference(d) {
                Ok(Some(info)) => { w.note_reference(&info, "stored"); have_ref = true; }
                Ok(None) => {}
                Err(e) => eprintln!("vg360-pipeline: {}: {e}; capturing a new reference", w.id),
            }
        }
        let first_try = Instant::now();
        while !have_ref {
            let start = Instant::now();
            let mut frames: Vec<Bgr> = Vec::new();
            loop {
                match next() {
                    Src::Frame(f, _) => frames.push(f),
                    _ => { if tx.send(Err("stream stalled: no frames for 30 s".into())).is_err() { g.stop(); return; } }
                }
                let done = match policy.ref_frames { Some(n) => frames.len() >= n, None => start.elapsed().as_secs_f64() >= policy.capture_s && !frames.is_empty() };
                if done || frames.len() >= 60 { break; }
            }
            match w.capture_reference(&frames) {
                Ok((bi, info)) => {
                    let mut info = info;
                    if let Some(d) = &dir {
                        match w.save_reference(d, &frames[bi]) {
                            Ok(sha) => info.sha256 = sha,
                            Err(e) => eprintln!("vg360-pipeline: {}: reference not stored: {e}", w.id),
                        }
                    }
                    w.note_reference(&info, "captured");
                    have_ref = true;
                }
                Err(why) => {
                    if first_try.elapsed().as_secs_f64() >= 30.0 && tx.send(Err(format!("reference unusable: {why}"))).is_err() { g.stop(); return; }
                }
            }
        }
        let start = Instant::now();
        drive(&mut w, &mut next, &mut || start.elapsed().as_secs_f64(), dt, &signals, &tx);
        g.stop();
    });
    rx
}
