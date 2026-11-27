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
  vg360-rules    scene state → IOGP facts → iogp-runtime (rules written and approved in iogp-editor,
                 dhaneshkk/test1_logpeditor); per-rule MCC on the simulated site; approval + MCC gate
  vg360-evidence tamper-evident provenance in vesdb (dhaneshkk/test1_vesdb): ruleset, model,
                 calibration, camera-guard and incident events, receipts, spool, offline bundles
  vg360-infer    YOLOX-Nano person detector + PPE classifier: letterbox, grid decode, NMS,
                 full frame + 3×3 tiles, fisheye virtual views; engines: tract (pure Rust) / ONNX Runtime
apps/
  vg360-server   Topcoat 0.12 operator console + OpenAI-compatible API
models/          ppe_clifford_folded.onnx (committed); yolox_nano.onnx via scripts/fetch_models.sh (SHA-256 checked)
scripts/         fetch_models.sh, export_golden.py, export_golden_scene.py (notebook reference path → testdata/golden)
config/          vg360_site_catalog.json (iogp-editor vocabulary for this rig), vg360_rules.ace (the 5 rules in
                 controlled English), vg360_rules.draft.json (iogp-editor export, NOT approved),
                 vg360_rules.validation.json (MCC per rule)
notebooks/       vg360_train_cpu.ipynb — training only (single cell): PPE model, SAM, MCC, ONNX, card, golden, footage audit
                 vg360_clifford_safety_cpu.ipynb — the original research notebook (simulation, scene goldens)
lean/            Lake project (Mathlib v4.36.0), built in CI: VG360/CGA (zone / ray closed forms),
                 VG360/Clifford (Cl(2,0) table), VG360/Train (conv folding, MCC bounds, SAM step),
                 VG360/Detect (IoU and IoU-loss bounds, letterbox round trip),
                 VG360/Calib (focal length from vanishing points, height from people)
testdata/        notebook exports (rules, ledger) and frames used by the golden tests
```

**User guide: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)** (with UI screenshots; printable copy: [docs/VisionGuard360_User_Guide.pdf](docs/VisionGuard360_User_Guide.pdf))

## Status — Phases 0–6 (pipeline) + integration (Rig Twin, Rule Studio, iogp-editor, vesdb)

| Area | State |
|---|---|
| Geometry, zones, MCC, rules, ledger | done, unit + golden tests |
| Python → Rust interop | the notebook's SHA-256 ledger re-verifies in Rust; Rust MCC = sklearn MCC for every rule |
| Scene (`vg360-scene`) | the notebook's 60-s simulated site (600 frames, 8 018 boxes) replays identically: same track ids, feet / head / tilt / clearances within 1e-5 (JSON precision), same 101 fall/intrusion events, same 51 rule alerts, byte-identical ledger |
| OpenCV removal | done for camera_guard and calibration / lens code |
| Camera guard on real frames | known +70 px / −25 px / 3.2° bump recovered as dx 5.47 %, dy −1.98 %, −3.22° |
| Deck calibration | 8 points, ±0.3 px noise → f, height, pitch, yaw, roll; 0.23 px / 0.012 m RMS |
| Console | 7 pages; camera cards on sample data (CAM-05 measured by the guard at start-up); `/live` runs the real pipeline |
| Live console (Phase 4) | one worker thread per stream → SSE (`frame` ≤ 4 Hz/camera, `alert` immediately); latest-frame JPEG; alert snapshots saved as `state/evidence/<incident>.jpg` whose SHA-256 is in the alert and its vesdb incident; site signals (crane load, pipe handling) gate the rules; streaming `/v1/chat/completions` |
| Live camera guard + compensation (`vg360-pipeline`) | rig clip knocked mid-run by +32 px / −14 px / 1.5°: analytics continue (mode `Compensated`), people stay on their deck spots (mean error 0.24 m vs 0.85 m uncompensated, 19/21 within 0.5 m), all 4 track ids survive the knock; a 9° / 180 px knock pauses analytics; a covered lens pauses them after the 5 s hold |
| Fisheye cameras live | twin camera `lens` block (Kannala–Brandt) → 3 upright virtual views, one tracker; demo 185° fisheye (2880²) finds 56 of 57 people the pinhole finds, deck positions within 0.06 m median; moves pause analytics |
| Auto-calibration | `POST /api/calibrate`: vanishing points of two orthogonal deck-edge families → f / FOV, pitch, roll (synthetic test: 70.0°, 32.0°, 0.0°); standing people → mount height (12.2 m for 12 m); live auto-height `VG360_CAM_<ID>_AUTO_HEIGHT=1` (6 m → 9.8 m on the 10 m demo, rules off until set) |
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

### Live analytics

```sh
VG360_DEMO=1 cargo run -p vg360-server --release      # replay testdata/clip/clip.mp4 as camera DEMO
# → http://127.0.0.1:3000/live
VG360_CAM_C5_URL='rtsp://…' cargo run -p vg360-server --release --features ort   # a Rig Twin camera
RTSP_FEED='rtsp://…' cargo run -p vg360-server --release                         # a standalone camera LIVE
```

* A worker starts for every camera of the applied Rig Twin whose `VG360_CAM_<ID>_URL` is set. It uses
  that camera's twin calibration and the twin zones it watches (twin rule `red` → exclusion zone,
  `height` → deck edge, `lift` → load drop zone). `VG360_DEMO=1` adds the simulated clip; `RTSP_FEED` adds a standalone camera `LIVE` instead (generic pinhole from `VG360_LIVE_SIZE` / `_HFOV` / `_PITCH` / `_HEIGHT`, no zones, auto-height). `VG360_TILES`
  (default 1) and `VG360_FPS` (default 10) trade recall for CPU.
* `GET /api/live` (state) · `GET /api/live/events` (SSE: `hello`, `frame`, `alert`, `signals`) ·
  `GET /api/live/{cam}/frame.jpg` · `GET /api/evidence/{incident}/snapshot.jpg` ·
  `POST /api/signals {"suspended_load":true,"active":["pipe-handling"]}`.
* Rules still go through the approval and MCC gates: with the draft config every alert is shadow
  (logged, not sounded). Incidents go to vesdb when `VG360_VESDB_*` is set; the snapshot proves its frame
  with `vesdb-verify bundle --file-hash snapshot.jpg=<event id>:frame_sha256`.
* tract (default) runs ≈ 5 fps on one tile on 2 vCPU; use `--features ort` in production.

**Never commit stream URLs, credentials, or reference / alert snapshots** — they are git-ignored.

### API (OpenAI-compatible)

```sh
curl localhost:3000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"vg360-ops","messages":[{"role":"user","content":"which camera moved?"}]}'
curl -N localhost:3000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"vg360-ops","stream":true,"messages":[{"role":"user","content":"what is live now?"}]}'
```

`"stream": true` returns OpenAI `chat.completion.chunk` events ending in `data: [DONE]`, so the
OpenAI SDKs' streaming mode works unchanged.

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
4. ~~Live console~~ done: per-camera workers, SSE updates, snapshots, site signals, streaming chat completions.
5. ~~Training notebook + Lean in CI~~ done (see *Training* below).
6. ~~Detector fine-tune pipeline~~ done (see *Detector fine-tune*); training it on rig footage needs labelled frames.

## Training (Phase 5)

`notebooks/vg360_train_cpu.ipynb` is one Colab cell and the only place models are trained:

| step | what it does |
|---|---|
| data | `CFG.ppe_dir` (helmet/ no_helmet/ crops) or `CFG.voc_dir` (e.g. SHWD); otherwise 5 000 synthetic crops, a smoke test only. Split 60/20/20, stratified |
| train | Clifford-conv Cl(2,0) net **and** a plain CNN with the same number of conv weights, both with SAM (ρ 0.05, SGD), best epoch picked on val MCC |
| evaluate | test MCC with a bootstrap 95 % CI; sharpness = test-loss increase at the SAM point |
| export | fold → ONNX (opset 17), torch ≡ folded ≡ onnxruntime checks, `models/ppe_clifford_folded.card.json` (SHA-256, metrics, data, SAM settings, MCC gate), `testdata/golden/ppe.json` |
| footage | `VG360_FOOTAGE=clip.mp4` (+ `CFG.footage_crop` for VMS screen recordings): detector + PPE audit, contact sheet, crops and frames for labelling (in `footage_audit/`, git-ignored) |

The Rust side checks the result. `vg360_infer::card::verify` refuses a PPE model whose SHA-256 doesn't
match its card: the console disables PPE and says so, and `/training` shows the card. The golden test
runs the 16 notebook crops through tract (and ONNX Runtime): max |ΔP| 2e-6. With vesdb configured, a
`model-deployed` event records the card.

Current model (synthetic crops, n_test 1002): **Clifford MCC 0.800 [0.764, 0.838]** vs **plain CNN
0.226 [0.169, 0.287]** with the same recipe. The CNN's hyper-parameters were not tuned separately, so
this shows the recipe suits the Clifford net, not that Clifford conv wins in general. Below the P1 floor
(0.85), and synthetic, so the PPE rule stays shadow.

**Lean (CI job `lean`).** `lake build` in `lean/` with the Mathlib cache, and no `sorry`/`admit`.
`Clifford` proves the notebook's product table is Cl(2,0)'s geometric product (`decide`; the notebook
asserts its table equals the Lean one). `Train` proves the folded conv computes the trained Clifford
layer (`fold_correct`, `fold_conv`); that MCC is 1 for a perfect classifier, −1 for an inverted one,
unchanged when the classes are swapped, and always in [−1, 1]; and that SAM's `ε = ρg/‖g‖` has norm ρ and
maximises the linearised loss over the ρ-ball (`sam_eps_maximizes`).

## Detector fine-tune (Phase 6 — pipeline)

Notebook §10 fine-tunes YOLOX-Nano for persons with SAM and exports a drop-in replacement for
`models/yolox_nano.onnx` (same `images` → `output [1,3549,85]` layout, so `vg360-infer` is unchanged).

| step | what it does |
|---|---|
| model | YOLOX-Nano re-implemented in PyTorch in Megvii's layout. It loads `yolox_nano.pth` with `strict=True`, and its outputs match the stock ONNX (max \|Δ\| 9e-5) |
| data | `CFG.det_dir` / `VG360_DET_DIR` in YOLO format (`images/` + `labels/` × train / val / test, class 0 = person), e.g. `footage_audit/frames` labelled in CVAT. Without it: procedural overhead-camera smoke frames (standing, crouching and top-down workers, machinery, helmet-like distractors) |
| train | YOLOX loss with SimOTA assignment (5·IoU + obj + cls), SAM over SGD, frozen backbone and BN statistics, gradient clip 10, best epoch on val cell MCC |
| metric | **cell MCC**: 16-px cells of the 416² letterbox, positive where a person box centre falls. Unlike box AP it has true negatives. Reported with bootstrap 95 % CI, plus P / R / F1 at IoU 0.5 and AP50, before and after fine-tuning |
| export | ONNX + `yolox_nano_ft.card.json` + golden frames and raw output rows. Smoke-set output goes to `detector_smoke/` (git-ignored), never to `models/` |

Rust: `VG360_DETECTOR=models/yolox_nano_ft.onnx` switches the live workers to a fine-tuned detector.
It is served only if it matches its card and was trained on real site frames; a smoke-set model is
refused unless `VG360_ALLOW_SMOKE_MODELS=1`. `/training` shows its card, and vesdb records
`model-deployed`. CI runs the notebook in FAST mode, then
`VG360_DET_FT_DIR=… cargo test -p vg360-infer detector_finetune`. That test checks the raw outputs on
the same tensor (< 2e-3) and the confident boxes within 1 px through the Rust letterbox and NMS.

Smoke set (400 / 100 / 100 frames, 12 epochs, CPU): cell MCC **0.376 → 0.808** [0.747, 0.866], AP50
0.387 → 0.963. This shows the training loop, export and Rust path work end to end; it says nothing
about rig cameras. The PPE model re-trained in the same run is byte-identical to the committed one
(sha256 `8b16c550…`), so the notebook is reproducible.

## Real footage (RIG 2004 drill floor)

76 s screen recording of a wide-angle overhead drill-floor camera at night (1024×576 stream, 2–6 fps).
The footage stays out of the repository. Audit at 1 fps, 75 usable frames:

* **Detector (COCO YOLOX-Nano) is the bottleneck.** At the runtime threshold (0.35) it finds one of the
  two or three workers, and only from ~17 s, when he is more upright. Crouching or top-down workers
  score 0.10–0.20, below yellow machinery at up to 0.46. 6 of the 66 detections at 0.35 are machinery.
  Lowering the threshold does not fix it.
* **PPE on the detected worker** (helmet, 60 frames): P(helmet) 0.69–0.72, all correct. The machinery
  boxes score 0.18–0.37, so they would raise false no-helmet alerts. The pipeline needs a detector that
  knows this camera.
* Next: label person boxes on rig frames (CVAT, YOLO format; `footage_audit/frames`) and run §10 with
  `VG360_DET_DIR`. Real helmet / no-helmet crops then replace the synthetic PPE set.

## Caveats

Rule MCCs come from the notebook's simulated site; PPE MCC (0.800) is on synthetic crops; the
COCO detector misses most workers on real overhead rig footage (see *Real footage*). Console thumbnails CAM-01…04 are frames of one simulated feed.

## Integration: Rig Twin · Rule Studio · iogp-editor · vesdb

**Rules (iogp-editor → iogp-runtime).** The five rules are written in controlled English
(`config/vg360_rules.ace`) against a site catalog that extends iogp-editor's built-in vocabulary
(`fallen-person`, `pipe-handling` signal, `person-down-alert`, `ppe-alert`; `near` = 1 m). Compile:

```sh
iogp --prolog-dir ../ape-rs/prolog-scryer --catalog config/vg360_site_catalog.json \
     compile --file config/vg360_rules.ace > config/vg360_rules.draft.json
cargo run --release -p vg360-rules --example validate      # → config/vg360_rules.validation.json
```

The committed config is a **draft**: no rule is approved, so every rule runs in shadow mode.
To arm rules, an author submits and a *different* approver signs them off in `iogp serve`
(four-eyes), then `/export/config.json` is deployed with `VG360_RULES_CONFIG`. A rule sounds only when
it is approved **and** its MCC clears the floor for its severity.

| rule | severity → floor | MCC (simulated site) | state |
|---|---|---|---|
| LIFT · worker under a suspended load ≥ 3 s | critical 0.85 | 0.771 | below floor |
| RED · in Red-Zone while pipe handling | critical 0.85 | 0.903 | clears floor |
| EDGE · near Deck-Edge (1 m) | critical 0.85 | 0.899 | clears floor |
| FALL · fallen person ≥ 2 s | high 0.75 | 0.798 | clears floor |
| PPE · no hard hat ≥ 2 s | critical 0.85 | 0.800 (synthetic crops, model card) | below floor |

**Evidence (vesdb).** With `VG360_VESDB_URL`, `VG360_VESDB_WRITE_TOKEN`, `VG360_VESDB_AUDIT_TOKEN` set,
each raised alert is an incident (`hazard-detected` + `alert-raised`, frame SHA-256, ruleset / model /
calibration hashes, armed or shadow) opened with `isSubjectPristine`; camera-guard changes, twin
calibrations, model and ruleset deployments are logged too. `cargo test -p vg360-evidence` with
`VESDB_BIN_DIR` runs a real log + witness, survives an outage via the spool, and verifies the bundle
offline with `vesdb-verify` (a swapped frame fails).

**Console.** `/twin` embeds the Rig Twin editor; *Apply to site* turns every twin camera into a runtime
camera (compass yaw = 90° − twin heading, f from FOV and resolution, height above the deck it watches),
checked against the twin's own projection (< 1 px), and logs a `calibration-set` event. `/rules` shows
the IOGP rules with approval and MCC gates and embeds Rule Studio, fed live from `/api/rule-studio`.

**Licence note.** `iogp-runtime` is pulled from `dhaneshkk/test1_logpeditor`, which is LGPL-3.0-or-later
at the pinned commit. The plan is to dual-license it; until that lands there, binaries that link it
carry LGPL relinking obligations.

**Fixed upstream:** iogp-runtime used to keep only an object's first type believed over missed frames, so a
track that is also `fall_pose` lost its fall timer on every miss. Fixed in dhaneshkk/test1_logpeditor#1;
this workspace pins that commit, and the fall pose is again a second type of the track.
