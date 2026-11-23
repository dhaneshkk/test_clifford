//! One camera, end to end: camera guard → (shift compensation) → person detection → scene
//! (deck geometry, tracks, zones, falls) → Rule Studio rules → evidence ledger.
//!
//! **Shift compensation.** The guard measures the homography H from the reference view (the one the
//! deck calibration was made on) to the live view. When the camera has been knocked by a small amount,
//! detections are mapped back through H⁻¹ into reference pixels and the *original* calibration keeps
//! working — analytics continue (mode [`Mode::Compensated`]) while the technician alarm still asks for
//! the camera to be put back. Only when the move is too large to trust, or the view is blocked / lost,
//! do analytics pause ([`Mode::Paused`]). This is the composition "ground ∘ H⁻¹" of
//! `vg360_geom::camera::Camera::ground_through`, applied to whole boxes.
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use vg360_guard::aligner::{Aligner, Measure};
use vg360_guard::config::{ANALYSIS_W, CONFIRM_INLIERS};
use vg360_guard::grabber::Grabber;
use vg360_guard::policy::{Guard, Observation, Pose, Status, is_aligned};
use vg360_infer::ppe::Ppe;
use vg360_infer::yolox::{Det, Yolox};
use vg360_scene::rules::{Alert, Ctx, RuleEngine, State};
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
}

impl Default for WorkerCfg {
    fn default() -> Self { Self { guard_every: 3, tiles: 3, limits: CompLimits::default(), rule_gap_s: 0.5 } }
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
    pub alerts: Vec<Alert>,
    pub ms: f64,
}

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
    pub rules: RuleEngine,
    frame_no: usize,
    last: Option<Measure>,
    mode: Mode,
    /// live → reference, full-resolution pixels
    cur_to_ref: M3,
}

const I3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl CameraWorker {
    pub fn new(id: &str, det: Yolox, ppe: Option<Ppe>, scene: Scene, rules: RuleEngine, cfg: WorkerCfg) -> Self {
        Self { id: id.into(), cfg, aligner: Aligner::new(), guard: Guard::default(), det, ppe, scene, rules,
               frame_no: 0, last: None, mode: Mode::Aligned, cur_to_ref: I3 }
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
    pub fn process(&mut self, frame: &Bgr, t: f64, dt: f64, ctx: &Ctx) -> Result<FrameResult, Error> {
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
        }
        self.frame_no += 1;
        let (pose, inliers, ref_box) = self.last.as_ref().map_or((None, 0, None), |m| (m.ok.then_some(m.pose), m.inliers, m.ok.then_some(m.boxq)));
        let mut res = FrameResult { camera: self.id.clone(), t, status: self.guard.status, mode: self.mode, pose, inliers, ref_box,
                                    boxes: vec![], recs: vec![], helmet: vec![], alerts: vec![], ms: 0.0 };
        if self.mode != Mode::Paused {
            let gray = frame.to_gray();
            let boxes = self.det.detect_tiled(&gray, self.cfg.tiles, 0.25)?;
            let helmet = match &self.ppe { Some(p) => p.classify(&gray, &boxes)?, None => vec![None; boxes.len()] };
            let mapped: Vec<[f64; 4]> = boxes.iter().map(|d| self.to_reference(d)).collect();
            let recs = self.scene.analyse(&mapped, t);
            let clip = format!("{}-{}", self.id, (t * 1000.0).round() as i64);
            for (r, h) in recs.iter().zip(&helmet) {
                let Some(r) = r else { continue };
                let s = State { clear: r.zones.clone(), down: r.state.down, p_helmet: *h };
                for a in self.rules.step(&r.track.to_string(), &s, ctx, t, dt, &clip) {
                    self.scene.log_rule(t, &a.rule, &a.subject);
                    res.alerts.push(a);
                }
            }
            res.boxes = boxes;
            res.recs = recs;
            res.helmet = helmet;
        }
        res.ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(res)
    }
}

/// Run a worker on a live stream (RTSP URL or a file replayed as a camera) in its own thread.
/// The first `ref_frames` frames become the reference view. Results arrive on the returned channel;
/// dropping the receiver stops the worker.
pub fn spawn_stream(mut w: CameraWorker, url: String, width: usize, height: usize, fps: f64, ref_frames: usize,
                    ctx: impl Fn(f64) -> Ctx + Send + 'static) -> mpsc::Receiver<Result<FrameResult, String>> {
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
            if tx.send(w.process(&f, t, dt, &ctx(t)).map_err(|e| e.to_string())).is_err() { break; }
        }
        g.stop();
    });
    rx
}
