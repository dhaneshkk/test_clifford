//! Resample the rig demo clip (testdata/clip/f00–f15) into what a 185° fisheye on the same mast would see
//! (notebook §9b), as PNG frames for `testdata/clip/clip_fisheye.mp4`:
//!   cargo run --release -p vg360-pipeline --example fisheye_clip -- /tmp/fe
//!   ffmpeg -framerate 10 -i /tmp/fe/f%02d.png -c:v libx264 -crf 30 -pix_fmt yuv420p testdata/clip/clip_fisheye.mp4
use vg360_geom::camera::Camera;
use vg360_infer::fisheye::FisheyeCam;
use vg360_vision::{img::Bgr, lens::Fisheye};

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/fe".into());
    std::fs::create_dir_all(&out).unwrap();
    let root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
    let pin = Camera { f_px: 640.0 / 35f64.to_radians().tan(), cx: 640.0, cy: 360.0, height_m: 10.0, pitch_deg: 25.0, yaw_deg: 0.0, roll_deg: 0.0, x0: 0.0, y0: 0.0 };
    let fc = FisheyeCam::from_pose(Fisheye { fx: 892.0, fy: 892.0, cx: 1440.0, cy: 1440.0, k: [0.02, -0.01, 0.002, -0.0005] }, &pin);
    for i in 0..16 {
        let f = Bgr::load(format!("{root}/testdata/clip/f{i:02}.jpg")).unwrap();
        fc.render_from_pinhole(&pin, &f, 2880, 2880, [25, 25, 25]).save(format!("{out}/f{i:02}.png")).unwrap();
    }
    eprintln!("16 fisheye frames (2880², Kannala–Brandt f 892 px) → {out}");
}
