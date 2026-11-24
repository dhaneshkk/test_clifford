//! Golden test: Rust inference vs the notebook's Python path (onnxruntime + cv2), exported by
//! `scripts/export_golden.py` into `testdata/golden/infer.json`.
//! Needs `models/yolox_nano.onnx` (`scripts/fetch_models.sh`).
use std::time::Instant;

use serde::Deserialize;
use vg360_infer::{ppe::Ppe, pre::letterbox, yolox::{Det, Yolox}};
use vg360_vision::img::Bgr;

#[derive(Deserialize)]
struct Set { boxes: Vec<[f64; 4]>, scores: Vec<f64>, #[serde(default)] ppe_helmet_prob: Vec<Option<f64>> }
#[derive(Deserialize)]
struct Golden { raw_rows: std::collections::BTreeMap<String, Vec<f32>>, single: Set, tiled: Set }

fn root() -> String { format!("{}/../..", env!("CARGO_MANIFEST_DIR")) }
fn golden() -> Golden { serde_json::from_str(&std::fs::read_to_string(format!("{}/testdata/golden/infer.json", root())).unwrap()).unwrap() }
fn yolox() -> Yolox { Yolox::load_tract(format!("{}/models/yolox_nano.onnx", root()), 416).expect("models/yolox_nano.onnx — run scripts/fetch_models.sh") }
fn gray() -> vg360_vision::img::Gray { Bgr::load(format!("{}/testdata/frames/ref.png", root())).unwrap().to_gray() }

fn same_boxes(name: &str, got: &[Det], want: &Set, px: f64) {
    assert_eq!(got.len(), want.boxes.len(), "{name}: {got:?}");
    for (g, (w, s)) in got.iter().zip(want.boxes.iter().zip(&want.scores)) {
        let d = [g.x1 - w[0], g.y1 - w[1], g.x2 - w[2], g.y2 - w[3]].map(f64::abs);
        eprintln!("{name}: max corner diff {:.3} px, score {:.4} vs {:.4}", d.iter().cloned().fold(0.0, f64::max), g.score, s);
        assert!(d.iter().all(|&v| v <= px), "{name}: {g:?} vs {w:?}");
        assert!((g.score - s).abs() < 0.01, "{name}: score {} vs {s}", g.score);
    }
}

#[test]
fn model_matches_onnxruntime_on_identical_input() {
    let lb = Bgr::load(format!("{}/testdata/golden/letterbox_ref.png", root())).unwrap();
    let mut t = vec![0f32; 3 * 416 * 416];
    for (i, p) in lb.data.chunks_exact(3).enumerate() { for c in 0..3 { t[c * 416 * 416 + i] = p[c] as f32; } }
    let raw = yolox().raw(t).unwrap();
    let stride = raw.len() / 3549;
    for (k, want) in golden().raw_rows {
        let i: usize = k.parse().unwrap();
        for (a, b) in raw[i * stride..(i + 1) * stride].iter().zip(&want) { assert!((a - b).abs() < 2e-3, "row {i}: {a} vs {b}"); }
    }
}

#[test]
fn letterbox_matches_cv2_within_one_grey_level() {
    let g = gray();
    let (t, _) = letterbox(&g, 0, 0, g.w, g.h, 416);
    let lb = Bgr::load(format!("{}/testdata/golden/letterbox_ref.png", root())).unwrap();
    let (mut worst, mut off) = (0f32, 0usize);
    for (i, p) in lb.data.chunks_exact(3).enumerate() {
        let d = (t[i] - p[0] as f32).abs();
        worst = worst.max(d);
        if d > 0.0 { off += 1; }
    }
    eprintln!("letterbox: {off} of {} pixels differ, max {worst}", 416 * 416);
    assert!(worst <= 1.0);
}

#[test]
fn detections_and_ppe_match_python_within_1px() {
    let (g, y, want) = (gray(), yolox(), golden());
    y.detect(&g).unwrap(); // warm-up
    let t = Instant::now();
    let single = y.detect(&g).unwrap();
    let t1 = t.elapsed();
    same_boxes("single", &single, &want.single, 1.0);
    let t = Instant::now();
    let tiled = y.detect_tiled(&g, 3, 0.25).unwrap();
    let t2 = t.elapsed();
    same_boxes("tiled 3×3", &tiled, &want.tiled, 1.0);
    let ppe = Ppe::load_tract(format!("{}/models/ppe_clifford_folded.onnx", root()), 48).unwrap();
    let probs = ppe.classify(&g, &tiled).unwrap();
    for (p, w) in probs.iter().zip(&want.tiled.ppe_helmet_prob) {
        eprintln!("ppe: {p:?} vs {w:?}");
        assert_eq!(p.is_some(), w.is_some());
        if let (Some(a), Some(b)) = (p, w) { assert!((a - b).abs() < 0.02, "{a} vs {b}"); }
    }
    eprintln!("engine {} · single {t1:?} · full frame + 3×3 tiles {t2:?}", y.engine_name());
}

#[cfg(feature = "ort")]
#[test]
fn onnxruntime_engine_matches_python_within_1px() {
    let (g, want) = (gray(), golden());
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let y = Yolox::load_ort(format!("{}/models/yolox_nano.onnx", root()), 416, threads).expect("set ORT_DYLIB_PATH");
    y.detect(&g).unwrap(); // warm-up
    let t = Instant::now();
    let single = y.detect(&g).unwrap();
    let t1 = t.elapsed();
    same_boxes("ort single", &single, &want.single, 1.0);
    let t = Instant::now();
    let tiled = y.detect_tiled(&g, 3, 0.25).unwrap();
    let t2 = t.elapsed();
    same_boxes("ort tiled 3×3", &tiled, &want.tiled, 1.0);
    let ppe = Ppe::load_ort(format!("{}/models/ppe_clifford_folded.onnx", root()), 48).unwrap();
    for (p, w) in ppe.classify(&g, &tiled).unwrap().iter().zip(&want.tiled.ppe_helmet_prob) {
        if let (Some(a), Some(b)) = (p, w) { assert!((a - b).abs() < 0.02, "{a} vs {b}"); }
    }
    eprintln!("engine {} · single {t1:?} · full frame + 3×3 tiles {t2:?}", y.engine_name());
}

// ───────────────────────────── PPE model vs the training notebook

#[derive(Deserialize)]
struct PpeCrop { label: u8, u8: Vec<u8>, p_helmet: f64 }
#[derive(Deserialize)]
struct PpeGolden { model_sha256: String, size: usize, crops: Vec<PpeCrop> }

fn ppe_golden() -> PpeGolden { serde_json::from_str(&std::fs::read_to_string(format!("{}/testdata/golden/ppe.json", root())).unwrap()).unwrap() }

/// the committed model is the one the notebook's card and golden vectors describe
#[test]
fn ppe_model_matches_its_card() {
    let model = format!("{}/models/ppe_clifford_folded.onnx", root());
    let card = vg360_infer::card::verify(&model).expect("model card");
    assert_eq!(card.sha256(), ppe_golden().model_sha256);
    let (lo, hi) = card.mcc_ci95().unwrap();
    let mcc = card.test_mcc().unwrap();
    assert!(lo <= mcc && mcc <= hi && (-1.0..=1.0).contains(&mcc));
    eprintln!("PPE card: test MCC {mcc:.3} [{lo:.3}, {hi:.3}] on {} · meets floor: {}", card.data_source(), card.meets_floor());
}

fn check_ppe(ppe: &Ppe, tol: f64) {
    let g = ppe_golden();
    assert_eq!(g.size, ppe.size);
    let mut worst = 0f64;
    for c in &g.crops {
        let p = ppe.p_helmet(c.u8.iter().map(|&v| v as f32 / 255.0).collect()).unwrap();
        worst = worst.max((p - c.p_helmet).abs());
        assert!((p - c.p_helmet).abs() < tol, "label {}: {p} vs {}", c.label, c.p_helmet);
    }
    eprintln!("PPE golden crops: {} crops, max |ΔP| {worst:.2e}", g.crops.len());
}

#[cfg(feature = "tract")]
#[test]
fn ppe_tract_matches_notebook_on_golden_crops() {
    check_ppe(&Ppe::load_tract(format!("{}/models/ppe_clifford_folded.onnx", root()), 48).unwrap(), 1e-4);
}

#[cfg(feature = "ort")]
#[test]
fn ppe_ort_matches_notebook_on_golden_crops() {
    check_ppe(&Ppe::load_ort(format!("{}/models/ppe_clifford_folded.onnx", root()), 48).expect("set ORT_DYLIB_PATH"), 1e-4);
}

// ───────────────────────────── fine-tuned detector vs the training notebook (§10)
// Runs when VG360_DET_FT_DIR points at the notebook's detector output (models/ for real data,
// detector_smoke/ for the procedural smoke set — CI runs the notebook in FAST mode, then this test).

#[derive(Deserialize)]
struct DetFrame { file: String, boxes: Vec<[f64; 4]>, scores: Vec<f64> }
#[derive(Deserialize)]
struct DetGolden { model_sha256: String, size: usize, raw_rows: std::collections::BTreeMap<String, Vec<f32>>, frames: Vec<DetFrame> }

#[cfg(feature = "tract")]
#[test]
fn detector_finetune_matches_notebook() {
    let Ok(dir) = std::env::var("VG360_DET_FT_DIR") else { eprintln!("skipped: set VG360_DET_FT_DIR"); return };
    let model = format!("{dir}/yolox_nano_ft.onnx");
    let card = vg360_infer::card::verify(&model).expect("detector card");
    let g: DetGolden = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/golden/detector.json")).unwrap()).unwrap();
    assert_eq!(card.sha256(), g.model_sha256);
    let y = Yolox::load_tract(&model, g.size).unwrap();
    // raw output on the notebook's own letterboxed tensor
    let lb = Bgr::load(format!("{dir}/golden/letterbox0.png")).unwrap();
    let n = g.size * g.size;
    let mut t = vec![0f32; 3 * n];
    for (i, p) in lb.data.chunks_exact(3).enumerate() { for c in 0..3 { t[c * n + i] = p[c] as f32; } }
    let raw = y.raw(t).unwrap();
    let stride = raw.len() / 3549;
    for (k, want) in &g.raw_rows {
        let i: usize = k.parse().unwrap();
        for (a, b) in raw[i * stride..(i + 1) * stride].iter().zip(want) { assert!((a - b).abs() < 2e-3, "row {i}: {a} vs {b}"); }
    }
    // full runtime path (Rust letterbox → model → decode → NMS) on the golden frames. The model itself is
    // checked exactly above; here the Rust letterbox differs from cv2 by ≤ 1 grey level, which can move a
    // score by ~0.01 and so swap the NMS winner of a box near the threshold. Confident boxes (≥ conf + 0.1)
    // must match within 1 px; borderline ones must overlap a Rust box (IoU ≥ 0.7) when Rust keeps one.
    let mut n_boxes = 0;
    for f in &g.frames {
        let gray = Bgr::load(format!("{dir}/golden/{}", f.file)).unwrap().to_gray();
        let got = y.detect(&gray).unwrap();
        for (w, &s) in f.boxes.iter().zip(&f.scores) {
            let want = Det { x1: w[0], y1: w[1], x2: w[2], y2: w[3], score: s, class: 0 };
            let best = got.iter().max_by(|a, b| a.iou(&want).total_cmp(&b.iou(&want)));
            if s >= y.conf + 0.1 {
                let b = best.unwrap_or_else(|| panic!("{}: confident box {want:?} missing", f.file));
                let d = [b.x1 - w[0], b.y1 - w[1], b.x2 - w[2], b.y2 - w[3]].map(f64::abs);
                eprintln!("{}: confident box, max corner diff {:.3} px, score {:.4} vs {s:.4}", f.file, d.iter().cloned().fold(0.0, f64::max), b.score);
                assert!(d.iter().all(|&v| v <= 1.0) && (b.score - s).abs() < 0.02, "{}: {b:?} vs {want:?}", f.file);
            } else if let Some(b) = best.filter(|b| b.iou(&want) > 0.0) {
                eprintln!("{}: borderline box (score {s:.3}), IoU {:.3} with Rust", f.file, b.iou(&want));
                assert!(b.iou(&want) >= 0.7, "{}: {b:?} vs {want:?}", f.file);
            }
        }
        for b in &got { assert!(b.score < y.conf + 0.1 || f.scores.iter().any(|&s| (s - b.score).abs() < 0.02), "{}: extra confident box {b:?}", f.file); }
        n_boxes += got.len();
    }
    eprintln!("fine-tuned detector: card {} · {} frames, {n_boxes} boxes match the notebook", card.data_source(), g.frames.len());
}
