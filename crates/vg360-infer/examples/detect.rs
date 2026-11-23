//! cargo run --release -p vg360-infer --example detect -- <image> [tiles]
//! (add `--features ort` and set ORT_DYLIB_PATH to use ONNX Runtime). Prints JSON.
use std::time::Instant;
use vg360_infer::{ppe::Ppe, yolox::Yolox};
use vg360_vision::img::Bgr;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let img = args.next().ok_or("usage: detect <image> [tiles]")?;
    let tiles: usize = args.next().map_or(Ok(3), |t| t.parse())?;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");
    #[cfg(feature = "ort")]
    let (det, ppe) = {
        let n = std::thread::available_parallelism().map_or(1, |n| n.get());
        (Yolox::load_ort(format!("{root}/yolox_nano.onnx"), 416, n)?, Ppe::load_ort(format!("{root}/ppe_clifford_folded.onnx"), 48)?)
    };
    #[cfg(not(feature = "ort"))]
    let (det, ppe) = (Yolox::load_tract(format!("{root}/yolox_nano.onnx"), 416)?, Ppe::load_tract(format!("{root}/ppe_clifford_folded.onnx"), 48)?);
    let g = Bgr::load(&img)?.to_gray();
    let t = Instant::now();
    let boxes = det.detect_tiled(&g, tiles, 0.25)?;
    let helmet = ppe.classify(&g, &boxes)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let people: Vec<String> = boxes.iter().zip(&helmet).map(|(b, h)| format!(
        r#"{{"box":[{:.1},{:.1},{:.1},{:.1}],"score":{:.3},"helmet":{}}}"#,
        b.x1, b.y1, b.x2, b.y2, b.score, h.map_or("null".into(), |p| format!("{p:.3}")))).collect();
    println!(r#"{{"engine":"{}","tiles":{tiles},"ms":{ms:.1},"people":[{}]}}"#, det.engine_name(), people.join(","));
    Ok(())
}
