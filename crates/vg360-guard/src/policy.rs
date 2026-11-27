use crate::config::*;
use serde::Serialize;

/// Smoothed position of the live view relative to the reference view.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Pose {
    /// shift, fractions of image width (+ = view must move right / down)
    pub dx: f64,
    pub dy: f64,
    /// degrees, + = clockwise
    pub angle: f64,
    /// 1.0 = same zoom
    pub scale: f64,
    /// residual perspective (tilt), fraction of width
    pub corner: f64,
}

impl Default for Pose {
    fn default() -> Self { Pose { dx: 0.0, dy: 0.0, angle: 0.0, scale: 1.0, corner: 0.0 } }
}

pub fn is_moved(p: &Pose) -> bool {
    p.dx.abs() > MOVE_SHIFT || p.dy.abs() > MOVE_SHIFT || p.angle.abs() > MOVE_ROT
        || (p.scale - 1.0).abs() > MOVE_SCALE || p.corner > MOVE_CORNER
}

pub fn is_aligned(p: &Pose) -> bool {
    p.dx.abs() < OK_SHIFT && p.dy.abs() < OK_SHIFT && p.angle.abs() < OK_ROT
        && (p.scale - 1.0).abs() < OK_SCALE && p.corner < OK_CORNER
}

/// "Follow the box" instructions for the technician (same wording as camera_guard).
pub fn guidance(p: &Pose) -> Vec<String> {
    let mut t = Vec::new();
    if p.dx.abs() >= OK_SHIFT { t.push(format!("Move view {}  {:.1}%", if p.dx > 0.0 { "RIGHT" } else { "LEFT" }, p.dx.abs() * 100.0)); }
    if p.dy.abs() >= OK_SHIFT { t.push(format!("Move view {}  {:.1}%", if p.dy > 0.0 { "DOWN" } else { "UP" }, p.dy.abs() * 100.0)); }
    if p.angle.abs() >= OK_ROT {
        t.push(format!("Rotate view {}  {:.1} deg", if p.angle > 0.0 { "CLOCKWISE" } else { "ANTI-CLOCKWISE" }, p.angle.abs()));
    }
    if (p.scale - 1.0).abs() >= OK_SCALE {
        t.push(format!("Zoom {}  {:.1}%", if p.scale > 1.0 { "OUT" } else { "IN" }, (p.scale - 1.0).abs() * 100.0));
    }
    if t.is_empty() && p.corner >= OK_CORNER { t.push("Camera tilt differs - match the blue outlines".into()); }
    t
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Status { Ok, Moved, Blocked, ViewLost, Realigning }

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Ok => "POSITION CORRECT",
            Status::Moved => "CAMERA MOVED",
            Status::Blocked => "CAMERA BLOCKED",
            Status::ViewLost => "VIEW LOST",
            Status::Realigning => "REALIGNING",
        }
    }
    /// Whether the ground calibration may be trusted (analytics run).
    pub fn analytics_enabled(self) -> bool { matches!(self, Status::Ok) }
}

/// One analysis tick from the aligner. `Fix(pose, inliers)`: a fix with fewer than
/// `CONFIRM_INLIERS` inliers may raise Moved but can never clear an alarm.
#[derive(Clone, Copy, Debug)]
pub enum Observation { Fix(Pose, i32), Blocked, NoFix }

fn finite(p: &Pose) -> bool {
    p.dx.is_finite() && p.dy.is_finite() && p.angle.is_finite() && p.scale.is_finite() && p.corner.is_finite()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Alarm state machine: a condition must persist for its hold time before it is raised,
/// and the view must stay aligned for OK_HOLD before the alarm clears.
#[derive(Clone, Debug)]
pub struct Guard {
    pub status: Status,
    since: Option<(Status, f64)>,
    aligned_since: Option<f64>,
    buf: Vec<Pose>,
}

impl Default for Guard {
    fn default() -> Self { Self { status: Status::Ok, since: None, aligned_since: None, buf: Vec::new() } }
}

impl Guard {
    /// Forget all state (status Ok, no pending timers, empty smoothing buffer).
    pub fn reset(&mut self) { *self = Self::default(); }

    /// The candidate status currently waiting out its hold time, with the time (same clock as
    /// `step`'s `t`) it first appeared. Callers compute elapsed = now - since_t.
    pub fn pending(&self) -> Option<(Status, f64)> { self.since }

    /// Per-field median of the last `SMOOTH_N` valid fix poses (None until a fix has arrived).
    pub fn smoothed_pose(&self) -> Option<Pose> {
        if self.buf.is_empty() { return None; }
        let f = |g: fn(&Pose) -> f64| median(self.buf.iter().map(g).collect());
        Some(Pose { dx: f(|p| p.dx), dy: f(|p| p.dy), angle: f(|p| p.angle), scale: f(|p| p.scale), corner: f(|p| p.corner) })
    }

    pub fn step(&mut self, t: f64, obs: Observation) -> Status {
        let obs = match obs { Observation::Fix(p, _) if !finite(&p) => Observation::NoFix, o => o };
        let candidate = match obs {
            Observation::Blocked => Status::Blocked,
            Observation::NoFix => Status::ViewLost,
            Observation::Fix(p, inliers) => {
                self.buf.push(p);
                if self.buf.len() > SMOOTH_N { self.buf.remove(0); }
                let sp = self.smoothed_pose().unwrap();
                if is_moved(&sp) {
                    Status::Moved
                } else if inliers < CONFIRM_INLIERS {
                    return self.status; // weak fix: cannot clear or advance, no timer changes
                } else if is_aligned(&sp) {
                    Status::Ok
                } else {
                    // dead band: no longer blocked / lost / moved, but not yet trusted
                    self.since = None;
                    self.aligned_since = None;
                    if self.status != Status::Ok { self.status = Status::Realigning; }
                    return self.status;
                }
            }
        };
        if candidate == Status::Ok {
            self.since = None;
            if self.status != Status::Ok {
                let s = *self.aligned_since.get_or_insert(t);
                self.status = if t - s >= OK_HOLD { self.aligned_since = None; Status::Ok } else { Status::Realigning };
            }
            return self.status;
        }
        self.aligned_since = None;
        if candidate == self.status { self.since = None; return self.status; }
        let hold = match candidate { Status::Moved => MOVED_HOLD, Status::Blocked => BLOCKED_HOLD, Status::ViewLost => LOST_HOLD, _ => 0.0 };
        match self.since {
            Some((c, s)) if c == candidate => {
                if t - s >= hold {
                    self.status = candidate;
                    self.since = None;
                    if matches!(candidate, Status::Blocked | Status::ViewLost) { self.buf.clear(); }
                }
            }
            _ => self.since = Some((candidate, t)),
        }
        self.status
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MOVED: Pose = Pose { dx: 0.055, dy: -0.02, angle: 3.2, scale: 1.0, corner: 0.0 };
    const HOME: Pose = Pose { dx: 0.0, dy: 0.0, angle: 0.0, scale: 1.0, corner: 0.0 };
    // between OK_* and MOVE_*: neither moved nor aligned
    const DEAD: Pose = Pose { dx: 0.012, dy: 0.0, angle: 0.0, scale: 1.0, corner: 0.0 };
    const STRONG: i32 = CONFIRM_INLIERS + 10;

    /// Feed `obs` every 0.25 s from `t` until `t_end`; returns the next t.
    fn run(g: &mut Guard, mut t: f64, t_end: f64, obs: Observation) -> f64 {
        while t < t_end { g.step(t, obs); t += 0.25; }
        t
    }
    fn into_moved(g: &mut Guard) -> f64 {
        let t = run(g, 0.0, MOVED_HOLD + 0.5, Observation::Fix(MOVED, STRONG));
        assert_eq!(g.status, Status::Moved);
        t
    }

    #[test]
    fn guidance_wording() {
        let g = guidance(&MOVED);
        assert_eq!(g[0], "Move view RIGHT  5.5%");
        assert_eq!(g[2], "Rotate view CLOCKWISE  3.2 deg");
    }

    #[test]
    fn moved_needs_hold_then_clears_after_ok_hold() {
        let mut g = Guard::default();
        let mut t = 0.0;
        while t < MOVED_HOLD - 0.5 { assert_eq!(g.step(t, Observation::Fix(MOVED, STRONG)), Status::Ok); t += 0.25; }
        while t < MOVED_HOLD + 0.5 { g.step(t, Observation::Fix(MOVED, STRONG)); t += 0.25; }
        assert_eq!(g.status, Status::Moved);
        assert!(!g.status.analytics_enabled());
        // the median needs a majority of HOME fixes before the move is considered over
        for _ in 0..SMOOTH_N / 2 { g.step(t, Observation::Fix(HOME, STRONG)); }
        assert_eq!(g.step(t, Observation::Fix(HOME, STRONG)), Status::Realigning);
        assert_eq!(g.step(t + OK_HOLD, Observation::Fix(HOME, STRONG)), Status::Ok);
    }

    #[test]
    fn brief_blockage_is_ignored() {
        let mut g = Guard::default();
        for i in 0..8 { g.step(i as f64 * 0.25, Observation::Blocked); } // 1.75 s < BLOCKED_HOLD
        assert_eq!(g.step(2.0, Observation::Fix(HOME, STRONG)), Status::Ok);
    }

    #[test]
    fn stale_since_does_not_shortcut_blocked_hold() {
        let mut g = Guard::default();
        let t0 = into_moved(&mut g);
        g.step(t0, Observation::Blocked); // pending (Blocked, t0)
        let t = run(&mut g, t0 + 0.25, t0 + BLOCKED_HOLD + 0.1, Observation::Fix(MOVED, STRONG));
        assert_eq!(g.pending(), None);
        g.step(t, Observation::Blocked);
        assert_eq!(g.status, Status::Moved, "stale timer must not flip to Blocked");
    }

    #[test]
    fn stale_since_does_not_shortcut_viewlost_hold() {
        let mut g = Guard::default();
        let t0 = into_moved(&mut g);
        g.step(t0, Observation::NoFix);
        let t = run(&mut g, t0 + 0.25, t0 + LOST_HOLD + 0.1, Observation::Fix(MOVED, STRONG));
        g.step(t, Observation::NoFix);
        assert_eq!(g.status, Status::Moved);
    }

    #[test]
    fn dead_band_fix_leaves_blocked_to_realigning() {
        let mut g = Guard::default();
        let t = run(&mut g, 0.0, BLOCKED_HOLD + 0.5, Observation::Blocked);
        assert_eq!(g.status, Status::Blocked);
        assert_eq!(g.step(t, Observation::Fix(DEAD, STRONG)), Status::Realigning);
        assert!(!g.status.analytics_enabled());
    }

    #[test]
    fn dead_band_from_moved_then_aligned_clears_after_ok_hold() {
        let mut g = Guard::default();
        let mut t = into_moved(&mut g);
        t = run(&mut g, t, t + 2.0, Observation::Fix(DEAD, STRONG));
        assert_eq!(g.status, Status::Realigning);
        // aligned fixes (enough of them for the median) must then last OK_HOLD from their start
        let ta = t;
        for _ in 0..SMOOTH_N { g.step(t, Observation::Fix(HOME, STRONG)); t += 0.05; }
        assert_eq!(g.status, Status::Realigning);
        assert_eq!(g.step(ta + OK_HOLD + 0.5, Observation::Fix(HOME, STRONG)), Status::Ok);
    }

    #[test]
    fn dead_band_while_ok_stays_ok() {
        let mut g = Guard::default();
        for i in 0..40 { assert_eq!(g.step(i as f64 * 0.25, Observation::Fix(DEAD, STRONG)), Status::Ok); }
    }

    #[test]
    fn weak_inliers_cannot_clear_moved() {
        let mut g = Guard::default();
        let mut t = into_moved(&mut g);
        t = run(&mut g, t, t + 10.0, Observation::Fix(HOME, CONFIRM_INLIERS - 1));
        assert_eq!(g.status, Status::Moved);
        run(&mut g, t, t + 10.0, Observation::Fix(HOME, CONFIRM_INLIERS));
        assert_eq!(g.status, Status::Ok);
    }

    #[test]
    fn weak_inliers_can_still_raise_moved() {
        let mut g = Guard::default();
        run(&mut g, 0.0, MOVED_HOLD + 0.5, Observation::Fix(MOVED, 1));
        assert_eq!(g.status, Status::Moved);
    }

    #[test]
    fn nan_pose_is_nofix() {
        let mut g = Guard::default();
        g.step(0.0, Observation::Fix(Pose { dx: f64::NAN, ..HOME }, STRONG));
        assert_eq!(g.pending().map(|p| p.0), Some(Status::ViewLost));
        assert!(g.smoothed_pose().is_none());
        g.step(LOST_HOLD, Observation::Fix(Pose { angle: f64::INFINITY, ..HOME }, STRONG));
        assert_eq!(g.status, Status::ViewLost);
    }

    #[test]
    fn smoothing_ignores_single_tick_spike() {
        let mut g = Guard::default();
        for i in 0..3 { g.step(i as f64 * 0.25, Observation::Fix(HOME, STRONG)); }
        g.step(1.0, Observation::Fix(MOVED, STRONG));
        assert_eq!(g.pending(), None, "one spike must not even start a Moved timer");
        let sp = g.smoothed_pose().unwrap();
        assert!(sp.dx.abs() < 1e-12 && (sp.scale - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pose_default_scale_one() {
        assert_eq!(Pose::default().scale, 1.0);
        assert!(is_aligned(&Pose::default()));
    }
}
