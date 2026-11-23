//! Image buffers and the pixel operations VisionGuard360 used from OpenCV:
//! `imread`/`imencode` → [`Bgr::load`]/[`Bgr::encode_jpeg`], `resize(INTER_AREA)` → [`Gray::resize_area`],
//! `cvtColor(BGR2GRAY)` → [`Bgr::to_gray`], `createCLAHE` → [`Gray::clahe`],
//! `warpPerspective`/`remap` → [`Bgr::warp_perspective`]/[`Bgr::remap`], `rotate(180)` → [`Bgr::rotate_180`].
use crate::M3;

/// 8-bit, 3-channel, BGR order (matches ffmpeg `bgr24` and the old OpenCV Mats).
#[derive(Clone, Debug)]
pub struct Bgr { pub w: usize, pub h: usize, pub data: Vec<u8> }

/// 8-bit single channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Gray { pub w: usize, pub h: usize, pub data: Vec<u8> }

impl Bgr {
    pub fn new(w: usize, h: usize) -> Self { Self { w, h, data: vec![0; w * h * 3] } }
    pub fn from_raw(w: usize, h: usize, data: Vec<u8>) -> Self { assert_eq!(data.len(), w * h * 3); Self { w, h, data } }

    pub fn load(path: impl AsRef<std::path::Path>) -> image::ImageResult<Self> {
        let rgb = image::open(path)?.to_rgb8();
        Ok(Self::from_rgb(rgb.width() as usize, rgb.height() as usize, rgb.as_raw()))
    }
    pub fn decode(bytes: &[u8]) -> image::ImageResult<Self> {
        let rgb = image::load_from_memory(bytes)?.to_rgb8();
        Ok(Self::from_rgb(rgb.width() as usize, rgb.height() as usize, rgb.as_raw()))
    }
    fn from_rgb(w: usize, h: usize, rgb: &[u8]) -> Self {
        let mut data = rgb.to_vec();
        for p in data.chunks_exact_mut(3) { p.swap(0, 2); }
        Self { w, h, data }
    }
    fn to_rgb(&self) -> Vec<u8> {
        let mut d = self.data.clone();
        for p in d.chunks_exact_mut(3) { p.swap(0, 2); }
        d
    }
    pub fn encode_jpeg(&self, quality: u8) -> Vec<u8> {
        let mut out = Vec::new();
        let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
        image::ImageEncoder::write_image(enc, &self.to_rgb(), self.w as u32, self.h as u32, image::ExtendedColorType::Rgb8)
            .expect("jpeg encode into memory");
        out
    }
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> image::ImageResult<()> {
        image::save_buffer(path, &self.to_rgb(), self.w as u32, self.h as u32, image::ExtendedColorType::Rgb8)
    }

    /// ITU-R BT.601 luma, same weights as OpenCV's BGR2GRAY (fixed point, rounded).
    pub fn to_gray(&self) -> Gray {
        let data = self.data.chunks_exact(3)
            .map(|p| ((p[0] as u32 * 1868 + p[1] as u32 * 9617 + p[2] as u32 * 4899 + 8192) >> 14) as u8)
            .collect();
        Gray { w: self.w, h: self.h, data }
    }

    pub fn rotate_180(&mut self) {
        let n = self.w * self.h;
        for i in 0..n / 2 {
            for c in 0..3 { self.data.swap(i * 3 + c, (n - 1 - i) * 3 + c); }
        }
    }

    #[inline]
    fn sample(&self, x: f64, y: f64, out: &mut [u8]) -> bool {
        if !(x > -1.0 && y > -1.0 && x < self.w as f64 && y < self.h as f64) { return false; }
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |xi: f64, yi: f64, c: usize| -> f64 {
            let xi = xi.clamp(0.0, self.w as f64 - 1.0) as usize;
            let yi = yi.clamp(0.0, self.h as f64 - 1.0) as usize;
            self.data[(yi * self.w + xi) * 3 + c] as f64
        };
        for c in 0..3 {
            let v = at(x0, y0, c) * (1.0 - fx) * (1.0 - fy) + at(x0 + 1.0, y0, c) * fx * (1.0 - fy)
                + at(x0, y0 + 1.0, c) * (1.0 - fx) * fy + at(x0 + 1.0, y0 + 1.0, c) * fx * fy;
            out[c] = v.round().clamp(0.0, 255.0) as u8;
        }
        true
    }

    /// Generic remap: for every output pixel, `map(x, y)` gives the source position (bilinear,
    /// constant border). Used for lens undistortion and fisheye → virtual pinhole views.
    pub fn remap(&self, w: usize, h: usize, border: [u8; 3], map: impl Fn(f64, f64) -> Option<(f64, f64)>) -> Bgr {
        let mut out = Bgr::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let o = &mut out.data[(y * w + x) * 3..(y * w + x) * 3 + 3];
                let ok = map(x as f64, y as f64).is_some_and(|(sx, sy)| self.sample(sx, sy, o));
                if !ok { o.copy_from_slice(&border); }
            }
        }
        out
    }

    /// `dst(x) = src(H⁻¹ x)`, i.e. OpenCV `warpPerspective(src, H, size)` without WARP_INVERSE_MAP.
    pub fn warp_perspective(&self, h: &M3, w: usize, hh: usize, border: [u8; 3]) -> Bgr {
        let inv = crate::robust::invert(h).unwrap_or([[0.0; 3]; 3]);
        self.remap(w, hh, border, |x, y| {
            let z = inv[2][0] * x + inv[2][1] * y + inv[2][2];
            (z.abs() > 1e-12).then(|| ((inv[0][0] * x + inv[0][1] * y + inv[0][2]) / z, (inv[1][0] * x + inv[1][1] * y + inv[1][2]) / z))
        })
    }

    /// Area-averaging downscale per channel (INTER_AREA for shrink factors ≥ 1).
    pub fn resize_area(&self, w: usize, h: usize) -> Bgr {
        let mut out = Bgr::new(w, h);
        for c in 0..3 {
            let plane = Gray { w: self.w, h: self.h, data: self.data.iter().skip(c).step_by(3).copied().collect() }.resize_area(w, h);
            for (i, v) in plane.data.into_iter().enumerate() { out.data[i * 3 + c] = v; }
        }
        out
    }
}

impl Gray {
    pub fn new(w: usize, h: usize) -> Self { Self { w, h, data: vec![0; w * h] } }
    #[inline] pub fn at(&self, x: usize, y: usize) -> u8 { self.data[y * self.w + x] }

    pub fn mean_std(&self) -> (f64, f64) {
        let n = self.data.len().max(1) as f64;
        let mean = self.data.iter().map(|&b| b as f64).sum::<f64>() / n;
        let var = self.data.iter().map(|&b| (b as f64 - mean).powi(2)).sum::<f64>() / n;
        (mean, var.sqrt())
    }

    /// Exact area-weighted average (OpenCV INTER_AREA semantics for downscaling).
    pub fn resize_area(&self, w: usize, h: usize) -> Gray {
        let (sx, sy) = (self.w as f64 / w as f64, self.h as f64 / h as f64);
        // per-axis tables of (source index, weight)
        let table = |n_out: usize, s: f64, n_in: usize| -> Vec<Vec<(usize, f64)>> {
            (0..n_out).map(|o| {
                let (a, b) = (o as f64 * s, ((o + 1) as f64 * s).min(n_in as f64));
                let mut v = Vec::new();
                let mut i = a.floor() as usize;
                while (i as f64) < b && i < n_in {
                    let wgt = (b.min(i as f64 + 1.0) - a.max(i as f64)).max(0.0);
                    if wgt > 0.0 { v.push((i, wgt)); }
                    i += 1;
                }
                v
            }).collect()
        };
        let (tx, ty) = (table(w, sx, self.w), table(h, sy, self.h));
        let mut out = Gray::new(w, h);
        for (oy, ry) in ty.iter().enumerate() {
            for (ox, rx) in tx.iter().enumerate() {
                let (mut acc, mut wsum) = (0.0, 0.0);
                for &(iy, wy) in ry { for &(ix, wx) in rx { acc += self.at(ix, iy) as f64 * wx * wy; wsum += wx * wy; } }
                out.data[oy * w + ox] = (acc / wsum).round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    /// Contrast-limited adaptive histogram equalisation, OpenCV-style: `tiles × tiles` grid,
    /// clip limit relative to the mean bin count, excess redistributed evenly, bilinear blend
    /// of the four neighbouring tile mappings.
    pub fn clahe(&self, clip_limit: f64, tiles: usize) -> Gray {
        let (tw, th) = (self.w.div_ceil(tiles), self.h.div_ceil(tiles));
        let mut luts = vec![[0u8; 256]; tiles * tiles];
        for ty in 0..tiles {
            for tx in 0..tiles {
                let (x0, y0) = (tx * tw, ty * th);
                let (x1, y1) = ((x0 + tw).min(self.w), (y0 + th).min(self.h));
                let area = ((x1.saturating_sub(x0)) * (y1.saturating_sub(y0))).max(1);
                let mut hist = [0u32; 256];
                for y in y0..y1 { for x in x0..x1 { hist[self.at(x, y) as usize] += 1; } }
                let limit = ((clip_limit * area as f64 / 256.0) as u32).max(1);
                let mut excess = 0u32;
                for h in hist.iter_mut() { if *h > limit { excess += *h - limit; *h = limit; } }
                let (add, rem) = (excess / 256, (excess % 256) as usize);
                for h in hist.iter_mut() { *h += add; }
                if rem > 0 { let step = (256 / rem).max(1); for i in (0..256).step_by(step).take(rem) { hist[i] += 1; } }
                let scale = 255.0 / area as f64;
                let mut sum = 0u32;
                let lut = &mut luts[ty * tiles + tx];
                for (i, h) in hist.iter().enumerate() { sum += h; lut[i] = (sum as f64 * scale).round().min(255.0) as u8; }
            }
        }
        let mut out = Gray::new(self.w, self.h);
        for y in 0..self.h {
            let fy = (y as f64 + 0.5) / th as f64 - 0.5;
            let ty0 = fy.floor().max(0.0) as usize;
            let ty1 = (ty0 + 1).min(tiles - 1);
            let ay = (fy - ty0 as f64).clamp(0.0, 1.0);
            for x in 0..self.w {
                let fx = (x as f64 + 0.5) / tw as f64 - 0.5;
                let tx0 = fx.floor().max(0.0) as usize;
                let tx1 = (tx0 + 1).min(tiles - 1);
                let ax = (fx - tx0 as f64).clamp(0.0, 1.0);
                let v = self.at(x, y) as usize;
                let m = |ty: usize, tx: usize| luts[ty * tiles + tx][v] as f64;
                let r = (m(ty0, tx0) * (1.0 - ax) + m(ty0, tx1) * ax) * (1.0 - ay) + (m(ty1, tx0) * (1.0 - ax) + m(ty1, tx1) * ax) * ay;
                out.data[y * self.w + x] = r.round() as u8;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn area_resize_of_constant_is_constant() {
        let g = Gray { w: 7, h: 5, data: vec![77; 35] };
        assert!(g.resize_area(3, 2).data.iter().all(|&v| v == 77));
    }
    #[test]
    fn area_resize_halves_exactly() {
        let g = Gray { w: 4, h: 2, data: vec![0, 100, 200, 50, 100, 200, 0, 50] };
        assert_eq!(g.resize_area(2, 1).data, vec![100, 75]);
    }
    #[test]
    fn clahe_stretches_low_contrast() {
        let data: Vec<u8> = (0..64 * 64).map(|i| 100 + ((i % 64) % 8) as u8 + ((i / 64) % 8) as u8).collect();
        let g = Gray { w: 64, h: 64, data };
        let (_, s0) = g.mean_std();
        let (_, s1) = g.clahe(2.5, 8).mean_std();
        assert!(s1 > 3.0 * s0, "{s0} -> {s1}");
    }
    #[test]
    fn warp_identity_and_rotate() {
        let mut b = Bgr::new(9, 7);
        for (i, v) in b.data.iter_mut().enumerate() { *v = (i * 7 % 251) as u8; }
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(b.warp_perspective(&id, 9, 7, [0; 3]).data, b.data);
        let mut r = b.clone(); r.rotate_180(); r.rotate_180();
        assert_eq!(r.data, b.data);
    }
}
