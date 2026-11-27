//! Tuning constants from camera_guard/src/config.rs.
//! The RTSP URL is NOT a constant any more: it is read from `VG360_CAM_<ID>_URL` at runtime,
//! so no camera credentials live in the repository.


pub const ANALYSIS_W: i32 = 960; // frames are downscaled to this width for matching
pub const RATIO: f32 = 0.80; // Lowe ratio test
pub const H_THRESH: f64 = 8.0; // homography tolerance (px @ ANALYSIS_W); loose for wide-angle lens
pub const MIN_INLIERS: i32 = 15; // fewer agreeing points = no position fix this check
pub const CONFIRM_INLIERS: i32 = 60; // "POSITION CORRECT" needs at least this many

// "Camera still in place" test (makes people / groups moving harmless):
pub const STILL_TOL: f32 = 4.0; // px @ ANALYSIS_W: a background point counts as unmoved within this
pub const STILL_MIN: i32 = 20; // this many unmoved background points => camera has not moved
pub const STILL_SHARE: f64 = 0.5; // ...and at least half as many as the best "moved" fit
pub const OVERLAY_W: f64 = 0.38; // corner areas ignored (burned-in timestamp / logo)
pub const OVERLAY_H: f64 = 0.10;

// Alarm when ANY is exceeded (shift = fraction of image width)
pub const MOVE_SHIFT: f64 = 0.020;
pub const MOVE_ROT: f64 = 1.5;
pub const MOVE_SCALE: f64 = 0.04;
pub const MOVE_CORNER: f64 = 0.03;
// "Correct" when ALL are below (tighter -> hysteresis)
pub const OK_SHIFT: f64 = 0.008;
pub const OK_ROT: f64 = 0.6;
pub const OK_SCALE: f64 = 0.015;
pub const OK_CORNER: f64 = 0.012;

pub const MOVED_HOLD: f64 = 3.0; // s the whole background must stay shifted
pub const BLOCKED_HOLD: f64 = 5.0; // s lens covered / dark
pub const LOST_HOLD: f64 = 20.0; // s view unrecognisable
pub const SMOOTH_N: usize = 5; // median of last N checks
pub const REF_CAPTURE_S: f64 = 3.0; // s of frames used to pick static background points
pub const OK_HOLD: f64 = 2.0; // s aligned before "correct"


