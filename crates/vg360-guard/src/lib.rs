//! Camera guard, ported from the `camera_guard` app — now with no OpenCV dependency.
//!
//! * [`aligner`] – measures where the live view sits relative to the reference view
//!   (features → ratio-test matches → robust homography, similarity fallback, "still points" check).
//! * [`grabber`] – RTSP reader: `ffmpeg` subprocess → newest raw BGR frame, auto-reconnect.
//! * [`policy`] – moved / aligned thresholds, guidance text, alarm state machine with hold times.
//!
//! Integration with analytics: a camera that is MOVED, BLOCKED or VIEW LOST has an invalid
//! ground calibration, so its rule engine is paused and its alerts are tagged
//! `calibration_suspended` until the guard reports the view aligned again (or the small
//! measured shift is composed with the ground homography, see `vg360_geom::camera::Camera::ground_through`).
pub mod aligner;
pub mod config;
pub mod grabber;
pub mod policy;
