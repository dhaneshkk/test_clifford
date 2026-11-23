//! Camera guard on real frames (no OpenCV):
//! * a copy of the reference bumped by +70 px, −25 px and 3.2° must read as MOVED with the right guidance;
//! * a later frame of the same fixed camera (workers have walked around) must read as aligned.
use vg360_guard::{aligner::Aligner, policy};
use vg360_vision::img::Bgr;

fn frame(name: &str) -> Bgr { Bgr::load(format!("{}/../../testdata/frames/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap() }

#[test]
fn bumped_camera_is_moved_with_correct_guidance() {
    let mut a = Aligner::new();
    let n = a.set_reference(&frame("ref.png"), &[frame("later.png")], None);
    assert!(n >= 80, "{n} background points");
    let m = a.measure(&frame("moved.jpg"));
    eprintln!("{:?} inliers {}", m.pose, m.inliers);
    assert!(m.ok && !m.blocked);
    assert!((m.pose.dx - 70.0 / 1280.0).abs() < 0.004, "dx {}", m.pose.dx);
    assert!((m.pose.dy + 25.0 / 1280.0).abs() < 0.004, "dy {}", m.pose.dy);
    assert!((m.pose.angle + 3.2).abs() < 0.3, "angle {}", m.pose.angle);
    assert!(policy::is_moved(&m.pose));
    let g = policy::guidance(&m.pose);
    assert!(g[0].starts_with("Move view RIGHT") && g[1].starts_with("Move view UP") && g[2].starts_with("Rotate view ANTI-CLOCKWISE"), "{g:?}");
}

#[test]
fn people_moving_is_not_a_camera_move() {
    let mut a = Aligner::new();
    a.set_reference(&frame("ref.png"), &[], None);
    let m = a.measure(&frame("later.png"));
    eprintln!("{:?} inliers {}", m.pose, m.inliers);
    assert!(m.ok && policy::is_aligned(&m.pose), "{:?}", m.pose);
}

#[test]
fn black_frame_is_blocked() {
    let mut a = Aligner::new();
    a.set_reference(&frame("ref.png"), &[], None);
    let m = a.measure(&Bgr::new(1280, 720));
    assert!(m.blocked && !m.ok);
}
