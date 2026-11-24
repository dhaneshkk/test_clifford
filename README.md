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
notebooks/       vg360_clifford_safety_cpu.ipynb (single cell; to be split into training-only in Phase 5)
lean/            CGA.lean — theorems behind the closed forms (not yet machine-checked in CI)
testdata/        notebook exports (rules, ledger) and frames used by the golden tests
```

## Status — Phases 0–4 + integration (Rig Twin, Rule Studio, iogp-editor, vesdb)

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
```

* A worker starts for every camera of the applied Rig Twin whose `VG360_CAM_<ID>_URL` is set. It uses
  that camera's twin calibration and the twin zones it watches (twin rule `red` → exclusion zone,
  `height` → deck edge, `lift` → load drop zone). `VG360_DEMO=1` adds the simulated clip; `VG360_TILES`
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
5. Training-only single-cell notebook (SAM, MCC, ONNX export, golden vectors), Lean build in CI.

## Caveats

Rule MCCs come from the notebook's simulated site; PPE MCC (0.82) is on synthetic crops; the
Clifford-conv PPE model has not yet been compared against a plain CNN; Lean proofs are not yet
machine-checked. Console thumbnails CAM-01…04 are frames of one simulated feed.

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
| PPE · no hard hat ≥ 2 s | critical 0.85 | 0.820 (synthetic crops) | below floor |

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
