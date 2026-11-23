//! YOLOX-Nano person detector: letterbox → model → grid decode → class filter → NMS,
//! plus the notebook's full-frame + 3×3 overlapping-tile detection with merge.
use vg360_vision::img::Gray;

use crate::engine::{Engine, Result};
use crate::pre::letterbox;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Det { pub x1: f64, pub y1: f64, pub x2: f64, pub y2: f64, pub score: f64, pub class: usize }

impl Det {
    pub fn area(&self) -> f64 { (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0) }
    pub fn inter(&self, o: &Det) -> f64 {
        (self.x2.min(o.x2) - self.x1.max(o.x1)).max(0.0) * (self.y2.min(o.y2) - self.y1.max(o.y1)).max(0.0)
    }
    pub fn iou(&self, o: &Det) -> f64 { let i = self.inter(o); i / (self.area() + o.area() - i) }
}

/// `cv2.dnn.NMSBoxes(boxes, scores, score_thr, nms_thr)`: stable sort by score, greedy IoU suppression.
pub fn nms(dets: &[Det], score_thr: f64, nms_thr: f64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..dets.len()).filter(|&i| dets[i].score > score_thr).collect();
    order.sort_by(|&a, &b| dets[b].score.total_cmp(&dets[a].score));
    let mut keep: Vec<usize> = Vec::new();
    for i in order {
        if keep.iter().all(|&k| dets[k].iou(&dets[i]) <= nms_thr) { keep.push(i); }
    }
    keep
}

pub struct Yolox {
    engine: Box<dyn Engine>,
    pub size: usize,
    pub conf: f64,
    pub nms: f64,
    /// COCO class ids to keep (0 = person)
    pub classes: Vec<usize>,
    grid: Vec<(f32, f32, f32)>,
}

impl Yolox {
    pub fn new(engine: Box<dyn Engine>, size: usize) -> Self {
        let mut grid = Vec::new();
        for s in [8usize, 16, 32] {
            let n = size / s;
            for y in 0..n { for x in 0..n { grid.push((x as f32, y as f32, s as f32)); } }
        }
        Self { engine, size, conf: 0.35, nms: 0.45, classes: vec![0], grid }
    }

    #[cfg(feature = "tract")]
    pub fn load_tract(path: impl AsRef<std::path::Path>, size: usize) -> Result<Self> {
        Ok(Self::new(Box::new(crate::engine::Tract::load(path, [1, 3, size, size])?), size))
    }

    #[cfg(feature = "ort")]
    pub fn load_ort(path: impl AsRef<std::path::Path>, size: usize, threads: usize) -> Result<Self> {
        Ok(Self::new(Box::new(crate::engine::Ort::load(path, threads)?), size))
    }

    pub fn engine_name(&self) -> &'static str { self.engine.name() }

    /// Raw model output for an already letterboxed CHW tensor: rows of [cx, cy, w, h, obj, cls…].
    pub fn raw(&self, tensor: Vec<f32>) -> Result<Vec<f32>> { self.engine.run(tensor, [1, 3, self.size, self.size]) }

    /// Detect in the region (x0, y0, w, h) of `g`; boxes in full-image pixels.
    pub fn detect_region(&self, g: &Gray, x0: usize, y0: usize, w: usize, h: usize) -> Result<Vec<Det>> {
        let (t, r) = letterbox(g, x0, y0, w, h, self.size);
        let out = self.raw(t)?;
        let stride = out.len() / self.grid.len();
        let mut dets = Vec::new();
        for (i, &(gx, gy, s)) in self.grid.iter().enumerate() {
            let row = &out[i * stride..(i + 1) * stride];
            let (mut best, mut cls) = (f32::MIN, 0);
            for (c, &p) in row[5..].iter().enumerate() { let v = row[4] * p; if v > best { best = v; cls = c; } }
            if !self.classes.contains(&cls) || (best as f64) <= self.conf { continue; }
            let (cx, cy) = ((row[0] + gx) * s, (row[1] + gy) * s);
            let (bw, bh) = (row[2].exp() * s, row[3].exp() * s);
            dets.push(Det {
                x1: (cx - bw / 2.0) as f64 / r + x0 as f64, y1: (cy - bh / 2.0) as f64 / r + y0 as f64,
                x2: (cx + bw / 2.0) as f64 / r + x0 as f64, y2: (cy + bh / 2.0) as f64 / r + y0 as f64,
                score: best as f64, class: cls,
            });
        }
        Ok(nms(&dets, self.conf, self.nms).into_iter().map(|i| dets[i]).collect())
    }

    pub fn detect(&self, g: &Gray) -> Result<Vec<Det>> { self.detect_region(g, 0, 0, g.w, g.h) }

    /// Full frame + `tiles × tiles` overlapping crops, merged by NMS and by dropping boxes that
    /// sit ≥ 80 % inside a stronger one (people cut by a tile edge). Same as notebook `detect_tiled`.
    pub fn detect_tiled(&self, g: &Gray, tiles: usize, overlap: f64) -> Result<Vec<Det>> {
        let mut all = self.detect(g)?;
        if tiles > 1 {
            let (hh, ww) = (g.h as f64, g.w as f64);
            let th = (hh / tiles as f64 * (1.0 + overlap)) as usize;
            let tw = (ww / tiles as f64 * (1.0 + overlap)) as usize;
            for i in 0..tiles {
                for j in 0..tiles {
                    let y0 = ((i as f64 * hh / tiles as f64) as usize).min(g.h - th);
                    let x0 = ((j as f64 * ww / tiles as f64) as usize).min(g.w - tw);
                    all.extend(self.detect_region(g, x0, y0, tw, th)?);
                }
            }
        }
        let kept: Vec<Det> = nms(&all, self.conf, self.nms).into_iter().map(|i| all[i]).collect();
        let mut order: Vec<usize> = (0..kept.len()).collect();
        order.sort_by(|&a, &b| kept[b].score.total_cmp(&kept[a].score));
        let mut out: Vec<Det> = Vec::new();
        for i in order {
            let a = kept[i];
            if !out.iter().any(|k| a.inter(k) > 0.8 * a.area()) { out.push(a); }
        }
        Ok(out)
    }
}
