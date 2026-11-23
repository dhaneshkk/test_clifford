# VisionGuard360 — Clifford-geometry CCTV safety analytics (Rust runtime)

CPU-only industrial-safety analytics for oil-rig CCTV. Training stays in Python (single-cell Colab
notebook); the runtime, camera guard and operator console are Rust with **no OpenCV dependency**.

```
crates/
  vg360-geom     camera ↔ deck geometry (PGA ray–plane meet), CGA zone tests, MCC,
                 rule timer with gap tolerance, SHA-256 evidence ledger
  vg360-vision   pure-Rust replacements for every OpenCV call we used (see table below)
  vg360-guard    camera guard (port of camera_guard): aligner, ffmpeg RTSP grabber,
                 moved / blocked / view-lost state machine and technician guidance
  vg360-scene    per-camera scene: feet on deck + head height, 3-D pose fit (19 cuboids, rotor tilt),
                 SE(2) motor tracker + Hungarian assignment, CGA zone clearances, fall / intrusion
                 events, Rule Studio engine (gap tolerance, priority cooldowns, SHA-256 ledger)
  vg360-pipeline per-camera worker: camera guard → shift compensation → detection → scene → rules;
                 runs on an RTSP stream or a file replayed as a camera (spawn_stream)
  vg360-infer    YOLOX-Nano person detector + PPE classifier: letterbox, grid decode, NMS,
                 full frame + 3×3 tiles, fisheye virtual views; engines: tract (pure Rust) / ONNX Runtime
apps/
  vg360-server   Topcoat 0.12 operator console + OpenAI-compatible API
models/          ppe_clifford_folded.onnx (committed); yolox_nano.onnx via scripts/fetch_models.sh (SHA-256 checked)
scripts/         fetch_models.sh, export_golden.py, export_golden_scene.py (notebook reference path → testdata/golden)
notebooks/       vg360_clifford_safety_cpu.ipynb (single cell; to be split into training-only in Phase 5)
lean/            CGA.lean — theorems behind the closed forms (not yet machine-checked in CI)
testdata/        notebook exports (rules, ledger) and frames used by the golden tests
```

## Status — Phases 0–3

| Area | State |
|---|---|
| Geometry, zones, MCC, rules, ledger | done, unit + golden tests |
| Python → Rust interop | the notebook's SHA-256 ledger re-verifies in Rust; Rust MCC = sklearn MCC for every rule |
| Scene (`vg360-scene`) | the notebook's 60-s simulated site (600 frames, 8 018 boxes) replays identically: same track ids, feet / head / tilt / clearances within 1e-5 (JSON precision), same 101 fall/intrusion events, same 51 rule alerts, byte-identical ledger |
| OpenCV removal | done for camera_guard and calibration / lens code |
| Camera guard on real frames | known +70 px / −25 px / 3.2° bump recovered as dx 5.47 %, dy −1.98 %, −3.22° |
| Deck calibration | 8 points, ±0.3 px noise → f, height, pitch, yaw, roll; 0.23 px / 0.012 m RMS |
| Console | 6 pages on sample data; CAM-05 is measured live by the guard at start-up |
| Live camera guard + compensation (`vg360-pipeline`) | rig clip knocked mid-run by +32 px / −14 px / 1.5°: analytics continue (mode `Compensated`), people stay on their deck spots (mean error 0.24 m vs 0.85 m uncompensated, 19/21 within 0.5 m), all 4 track ids survive the knock; a 9° / 180 px knock pauses analytics; a covered lens pauses them after the 5 s hold |
| Inference (`vg360-infer`) | done: boxes match the notebook's Python path (onnxruntime + cv2) within 0.125 px, scores within 0.004, PPE probabilities within 0.0005 — both engines |

### Inference speed (YOLOX-Nano 416², 1280×720 frame, 2 vCPU sandbox)

| engine | single pass | full frame + 3×3 tiles (10 passes) |
|---|---|---|
| ONNX Runtime 1.29 (`--features ort`, `ORT_DYLIB_PATH`) | 11 ms | 102 ms |
| tract 0.21 (pure Rust, default) | 84 ms | 863 ms |

Use ONNX Runtime in production; tract is the zero-native-dependency fallback (e.g. ARM boxes
without an ORT build). The engine is a trait, so both run the same pre/post-processing.

### OpenCV → pure Rust

| OpenCV | here |
|---|---|
| `imread` / `imencode` / `imwrite` | `img::Bgr::{load, encode_jpeg, save}` |
| `resize(INTER_AREA)`, `cvtColor(BGR2GRAY)`, `createCLAHE` | `img::Gray::{resize_area, clahe}`, `Bgr::to_gray` |
| `SIFT` + `FlannBasedMatcher` | `features::Detector` (FAST-9 + Harris + grid, rotated BRIEF) + exact Hamming ratio test |
| `findHomography(USAC_MAGSAC)`, `estimateAffinePartial2D` | `robust::fit` (MSAC + local optimisation) |
| `findHomography` + `solvePnP(IPPE)` + focal search | `calib::calibrate_from_points` (DLT → decomposition → Levenberg–Marquardt) |
| `undistort`, `cv2.fisheye.*`, fisheye virtual views | `lens::{Pinhole, Fisheye, virtual_view}` |
| `warpPerspective`, `remap`, `rotate` | `img::Bgr::{warp_perspective, remap, rotate_180}` |
| `highgui` window and drawing | the web console (SVG overlays) |

Known differences: descriptors are ORB-style, not SIFT, and not interchangeable with OpenCV's —
references saved by the old app must be re-captured. The robust estimator does MAGSAC++'s job
(people become outliers) but is not a line-for-line MAGSAC++ port.

## Run

```sh
./scripts/fetch_models.sh                                            # detector weights (3.6 MB)
cargo run --release -p vg360-infer --example detect -- testdata/frames/ref.png 3
ORT_DYLIB_PATH=/path/to/libonnxruntime.so cargo run --release -p vg360-infer --features ort --example detect -- frame.jpg
cargo test -p vg360-geom -p vg360-vision -p vg360-guard --release   # any stable Rust ≥ 1.85
cargo run -p vg360-server --release                                  # needs Rust ≥ 1.99 (Topcoat 0.12)
# → http://127.0.0.1:3000   (HOST / PORT to change)
```

Cameras read their stream URL from `VG360_CAM_<ID>_URL` (e.g. `VG360_CAM_CAM_05_URL`).
**Never commit stream URLs, credentials, or reference / alert snapshots** — they are git-ignored.

### API (OpenAI-compatible)

```sh
curl localhost:3000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"vg360-ops","messages":[{"role":"user","content":"which camera moved?"}]}'
```

`/v1/models`, `/v1/rules`, `/v1/ledger`, `/v1/cameras` are plain JSON. The `vg360-ops` assistant
answers deterministically from site state (no LLM), so every number it gives is auditable.

## Roadmap

1. ~~`vg360-infer`~~ done.
2. ~~Scene in Rust~~ done. The port found a bug in the notebook's motor log (double cover: a track whose
   heading passed ±180° had its correction negated and drifted away from its person); fixed in the
   notebook — simulated identities 19 → 15 tracks for 14 people, fall rule MCC 0.713 → 0.794 (now above
   the P2 floor).
3. ~~Live camera guard + shift compensation~~ done. Limits (≤ 8 % shift, ≤ 5° rotation, ≤ 6 % zoom,
   ≥ 60 matched points) are configurable per camera (`CompLimits`).
4. Live console: per-camera workers, SSE updates, snapshots, streaming chat completions.
5. Training-only single-cell notebook (SAM, MCC, ONNX export, golden vectors), Lean build in CI.

## Caveats

Rule MCCs come from the notebook's simulated site; PPE MCC (0.82) is on synthetic crops; the
Clifford-conv PPE model has not yet been compared against a plain CNN; Lean proofs are not yet
machine-checked. Console thumbnails CAM-01…04 are frames of one simulated feed.
