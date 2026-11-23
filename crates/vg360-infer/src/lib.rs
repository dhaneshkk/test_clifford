//! VisionGuard360 inference on CPU.
//!
//! * [`yolox::Yolox`] — YOLOX-Nano person detector (letterbox, grid decode, NMS, 3×3 tiling).
//! * [`ppe::Ppe`] — hard-hat classifier on head crops (folded Clifford-conv network).
//! * [`engine`] — `Tract` (pure Rust, default) or `Ort` (ONNX Runtime, feature `ort`).
//! * [`fisheye`] — detection on upright virtual pinhole views of a fisheye camera.
//!
//! Golden test: on `testdata/frames/ref.png` the boxes match the notebook's Python path
//! (onnxruntime + cv2) within 1 px — see `tests/golden.rs`.
pub mod engine;
pub mod fisheye;
pub mod ppe;
pub mod pre;
pub mod yolox;
