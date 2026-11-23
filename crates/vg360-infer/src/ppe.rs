//! PPE (hard hat) classifier on head crops — the folded Clifford-conv network exported by the notebook.
//! Input crop: 48×48 gray / 255; the graph standardises each crop itself.
use vg360_vision::img::Gray;

use crate::engine::{Engine, Result};
use crate::pre::resize_linear;
use crate::yolox::Det;

pub struct Ppe { engine: Box<dyn Engine>, pub size: usize }

/// Head region of a person box (top ~24 % of the height, square), as in notebook `head_crop`.
pub fn head_crop(g: &Gray, b: &Det) -> Option<(usize, usize, usize, usize)> {
    let h = b.y2 - b.y1;
    let s = (0.24 * h).max(8.0);
    let cx = (b.x1 + b.x2) / 2.0;
    let top = b.y1 - 0.03 * h;
    let xa = (cx - s / 2.0).max(0.0) as usize;
    let ya = top.max(0.0) as usize;
    let xb = (cx + s / 2.0).min(g.w as f64) as usize;
    let yb = (top + s).min(g.h as f64) as usize;
    (xb.saturating_sub(xa) >= 4 && yb.saturating_sub(ya) >= 4).then_some((xa, ya, xb - xa, yb - ya))
}

impl Ppe {
    pub fn new(engine: Box<dyn Engine>, size: usize) -> Self { Self { engine, size } }

    #[cfg(feature = "tract")]
    pub fn load_tract(path: impl AsRef<std::path::Path>, size: usize) -> Result<Self> {
        Ok(Self::new(Box::new(crate::engine::Tract::load(path, [1, 1, size, size])?), size))
    }

    #[cfg(feature = "ort")]
    pub fn load_ort(path: impl AsRef<std::path::Path>, size: usize) -> Result<Self> {
        Ok(Self::new(Box::new(crate::engine::Ort::load(path, 1)?), size))
    }

    /// P(helmet) for each person box; `None` where the head crop is too small.
    pub fn classify(&self, g: &Gray, boxes: &[Det]) -> Result<Vec<Option<f64>>> {
        boxes.iter().map(|b| {
            let Some((x, y, w, h)) = head_crop(g, b) else { return Ok(None) };
            let crop: Vec<f32> = resize_linear(g, x, y, w, h, self.size, self.size).into_iter().map(|v| v / 255.0).collect();
            let lg = self.engine.run(crop, [1, 1, self.size, self.size])?;
            let m = lg[0].max(lg[1]);
            let (e0, e1) = (((lg[0] - m) as f64).exp(), ((lg[1] - m) as f64).exp());
            Ok(Some(e1 / (e0 + e1)))
        }).collect()
    }
}
