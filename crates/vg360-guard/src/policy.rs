use crate::config::*;
use serde::Serialize;

/// Smoothed position of the live view relative to the reference view.
#[derive(Clone, Copy, Debug, Default, Serialize)]
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

/// One analysis tick from the aligner.
#[derive(Clone, Copy, Debug)]
pub enum Observation { Fix(Pose), Blocked, NoFix }

/// Alarm state machine: a condition must persist for its hold time before it is raised,
/// and the view must stay aligned for OK_HOLD before the alarm clears.
#[derive(Clone, Debug)]
pub struct Guard { pub status: Status, since: Option<(Status, f64)>, aligned_since: Option<f64> }

impl Default for Guard { fn default() -> Self { Self { status: Status::Ok, since: None, aligned_since: None } } }

impl Guard {
    pub fn step(&mut self, t: f64, obs: Observation) -> Status {
        let candidate = match obs {
            Observation::Blocked => Status::Blocked,
            Observation::NoFix => Status::ViewLost,
            Observation::Fix(p) if is_moved(&p) => Status::Moved,
            Observation::Fix(p) if is_aligned(&p) => Status::Ok,
            Observation::Fix(_) => self.status, // in the dead band: keep the current state
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
        if candidate == self.status { return self.status; }
        let hold = match candidate { Status::Moved => MOVED_HOLD, Status::Blocked => BLOCKED_HOLD, Status::ViewLost => LOST_HOLD, _ => 0.0 };
        match self.since {
            Some((c, s)) if c == candidate => { if t - s >= hold { self.status = candidate; self.since = None; } }
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
        while t < MOVED_HOLD - 0.5 { assert_eq!(g.step(t, Observation::Fix(MOVED)), Status::Ok); t += 0.25; }
        while t < MOVED_HOLD + 0.5 { g.step(t, Observation::Fix(MOVED)); t += 0.25; }
        assert_eq!(g.status, Status::Moved);
        assert!(!g.status.analytics_enabled());
        assert_eq!(g.step(t, Observation::Fix(HOME)), Status::Realigning);
        assert_eq!(g.step(t + OK_HOLD, Observation::Fix(HOME)), Status::Ok);
    }

    #[test]
    fn brief_blockage_is_ignored() {
        let mut g = Guard::default();
        for i in 0..8 { g.step(i as f64 * 0.25, Observation::Blocked); } // 1.75 s < BLOCKED_HOLD
        assert_eq!(g.step(2.0, Observation::Fix(HOME)), Status::Ok);
    }
}
