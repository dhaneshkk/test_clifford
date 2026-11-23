//! Pre-processing that matches the notebook's cv2 calls.
use vg360_vision::img::Gray;

/// `cv2.resize(..., INTER_LINEAR)` for 8-bit gray: half-pixel centres, edge clamp, no anti-aliasing.
/// Float arithmetic; may differ from OpenCV's fixed-point result by ±1 grey level on some pixels.
pub fn resize_linear(g: &Gray, x0: usize, y0: usize, w_in: usize, h_in: usize, w: usize, h: usize) -> Vec<f32> {
    let (sx, sy) = (w_in as f64 / w as f64, h_in as f64 / h as f64);
    let axis = |n_out: usize, s: f64, n_in: usize| -> Vec<(usize, usize, f32)> {
        (0..n_out).map(|o| {
            let f = (o as f64 + 0.5) * s - 0.5;
            let mut i = f.floor() as isize;
            let mut a = f - i as f64;
            if i < 0 { i = 0; a = 0.0; }
            if i as usize >= n_in - 1 { return (n_in - 1, n_in - 1, 0.0); }
            (i as usize, i as usize + 1, a as f32)
        }).collect()
    };
    let (ax, ay) = (axis(w, sx, w_in), axis(h, sy, h_in));
    let mut out = vec![0f32; w * h];
    for (oy, &(y_a, y_b, fy)) in ay.iter().enumerate() {
        for (ox, &(x_a, x_b, fx)) in ax.iter().enumerate() {
            let p = |x: usize, y: usize| g.at(x0 + x, y0 + y) as f32;
            let v = (p(x_a, y_a) * (1.0 - fx) + p(x_b, y_a) * fx) * (1.0 - fy) + (p(x_a, y_b) * (1.0 - fx) + p(x_b, y_b) * fx) * fy;
            out[oy * w + ox] = v.round().clamp(0.0, 255.0);
        }
    }
    out
}

/// YOLOX letterbox of a gray region (replicated to 3 channels), pad 114, CHW, values 0…255.
/// Returns the tensor and the scale `r` (model px = image px × r).
pub fn letterbox(g: &Gray, x0: usize, y0: usize, w: usize, h: usize, size: usize) -> (Vec<f32>, f64) {
    let r = (size as f64 / h as f64).min(size as f64 / w as f64);
    let (nw, nh) = ((w as f64 * r) as usize, (h as f64 * r) as usize);
    let small = resize_linear(g, x0, y0, w, h, nw, nh);
    let mut t = vec![114f32; 3 * size * size];
    for c in 0..3 {
        for y in 0..nh {
            let row = c * size * size + y * size;
            t[row..row + nw].copy_from_slice(&small[y * nw..y * nw + nw]);
        }
    }
    (t, r)
}
