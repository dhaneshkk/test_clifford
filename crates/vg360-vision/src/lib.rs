//! Pure-Rust replacements for every OpenCV call VisionGuard360 used.
//!
//! | OpenCV (camera_guard / notebook)                 | here                                         |
//! |--------------------------------------------------|----------------------------------------------|
//! | `imread`, `imencode`, `imwrite`                  | [`img::Bgr::load`], [`img::Bgr::encode_jpeg`], [`img::Bgr::save`] |
//! | `resize(INTER_AREA)`, `cvtColor(BGR2GRAY)`       | [`img::Gray::resize_area`], [`img::Bgr::to_gray`] |
//! | `createCLAHE(2.5, 8×8)`                          | [`img::Gray::clahe`]                         |
//! | `SIFT` + `FlannBasedMatcher` + ratio test        | [`features::Detector`] (ORB-style FAST + rotated BRIEF) + [`features::ratio_match`] (exact Hamming kNN) |
//! | `findHomography(USAC_MAGSAC)`                    | [`robust::fit`] with [`robust::Model::Homography`] (MSAC + local optimisation) |
//! | `estimateAffinePartial2D(RANSAC)`                | [`robust::fit`] with [`robust::Model::Similarity`] |
//! | `warpPerspective`, `remap`, `rotate(180)`        | [`img::Bgr::warp_perspective`], [`img::Bgr::remap`], [`img::Bgr::rotate_180`] |
//! | `findHomography` (deck calibration, all points)  | [`robust::homography_dlt`]                   |
//! | `findHomography` + `solvePnP(IPPE)` + focal search | [`calib::calibrate_from_points`] (DLT → decomposition → Levenberg–Marquardt) |
//! | `undistort`, `undistortPoints`                   | [`lens::Pinhole::undistort`], [`lens::Pinhole::undistort_point`] |
//! | `cv2.fisheye.projectPoints` / `undistortPoints`  | [`lens::Fisheye::project`], [`lens::Fisheye::unproject`] |
//! | fisheye → upright virtual views (`remap`)        | [`lens::virtual_view`]                       |
//! | `highgui` window, `putText`, `rectangle`         | replaced by the web console (SVG overlays)   |
pub mod calib;
pub mod features;
pub mod img;
pub mod lens;
pub mod robust;

pub type M3 = [[f64; 3]; 3];
