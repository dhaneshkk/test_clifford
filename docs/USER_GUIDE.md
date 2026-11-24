# VisionGuard360 — User Guide

VisionGuard360 (VG360) is CPU-only CCTV safety analytics for drilling rigs. It finds people on the
deck, works out where they stand in metres and which hazard zones they are in, and evaluates IOGP Life-Saving
Rules: red zone during pipe handling, deck edge, under a suspended load, fallen worker, no hard hat. Every
alert is written to a tamper-evident log (vesdb) with the frame it was raised on.

Everything that runs on site is Rust. Models are trained in one Python notebook.

Contents

1. [Install and first run](#1-install-and-first-run)
2. [The console](#2-the-console)
3. [Connecting cameras](#3-connecting-cameras)
4. [Calibrating a camera](#4-calibrating-a-camera)
5. [Fisheye cameras](#5-fisheye-cameras)
6. [Camera guard: moved, blocked or lost cameras](#6-camera-guard-moved-blocked-or-lost-cameras)
7. [Rules, approval and site signals](#7-rules-approval-and-site-signals)
8. [Evidence (vesdb)](#8-evidence-vesdb)
9. [APIs and the OpenAI-compatible endpoint](#9-apis-and-the-openai-compatible-endpoint)
10. [Models and training](#10-models-and-training)
11. [Environment variables](#11-environment-variables)
12. [Troubleshooting](#12-troubleshooting)
13. [Privacy and security](#13-privacy-and-security)

---

## 1. Install and first run

**You need:**

- Rust: stable 1.85 or newer for the libraries; **1.99 or newer** for the console (it uses Topcoat 0.12).
- `ffmpeg` on the `PATH`. It reads RTSP streams and video files.
- About 4 MB of model files, downloaded once.
- Optional: ONNX Runtime for faster inference (`--features ort` with `ORT_DYLIB_PATH`).

```sh
git clone https://github.com/dhaneshkk/test_clifford && cd test_clifford
./scripts/fetch_models.sh                       # YOLOX-Nano detector, SHA-256 checked
cargo run --release -p vg360-server             # console on http://127.0.0.1:3000
```

**Demo without cameras.** These commands replay the simulated rig clip as a live camera. The second one
adds the same clip seen through a 185° fisheye lens:

```sh
VG360_DEMO=1 cargo run --release -p vg360-server
VG360_DEMO=1 VG360_DEMO_FISHEYE=1 VG360_FPS=5 cargo run --release -p vg360-server
```

Then open **http://127.0.0.1:3000/live**.

**Production build.** Inference is about 8× faster with ONNX Runtime:

```sh
ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so \
  cargo run --release -p vg360-server --features ort
```

Use `HOST` and `PORT` to change the listening address. `VG360_STATE_DIR` (default `apps/vg360-server/state/`)
holds the applied Rig Twin and the alert snapshots.

---

## 2. The console

| Page | What it is for |
|---|---|
| **Overview** `/` | Camera cards, camera-guard alarms, alert feed, rule status. A green banner links to the live console while streams are running. |
| **Live** `/live` | Every running camera: latest frame with person boxes, track ids, zones and helmet probability. Also the live alert feed (armed or shadow, with the snapshot) and the site-signal switches. |
| **Camera** `/cameras/{id}` | Camera-guard view of one camera: how far it moved, guidance to re-aim it, ground calibration, rules on this camera. |
| **Rules & evidence** `/rules` | IOGP rules with approval state, MCC against the floor, armed or shadow. Also Rule Studio and the vesdb connection. |
| **Site twin** `/twin` | Rig Twin editor: cameras, zones and rules on the deck plan. *Apply to site* makes it the runtime configuration. |
| **Models & training** `/training` | Model registry and the verified model cards: MCC with confidence interval, baseline, SAM settings, gate. |
| **Settings** `/settings` | Cameras and the environment variable that carries each stream URL. |

On `/live`:

- **Box colours:** green = person; amber = inside a zone; dashed yellow = helmet probability below 0.5; red = down.
- **Status line** under each camera: frames per second, milliseconds per frame, people in view and the analytics mode (Aligned, Compensated or Paused). It also shows the mount height in use, or *estimating mount height (rules off)* while auto-height runs.

---

## 3. Connecting cameras

Cameras are defined in the **Rig Twin**, a plan of the rig deck with each camera's position, heading,
tilt, field of view and resolution.

1. Open `/twin`, place or edit the cameras and zones, then press **Apply to site**.
2. Give each camera its stream through an environment variable. **Credentials never go into the repository or the twin.**

   ```sh
   export VG360_CAM_C5_URL='rtsp://user:pass@10.0.0.25:554/stream1'   # camera id C5 in the twin
   ```

   Dashes in camera ids become underscores: `CAM-05` → `VG360_CAM_CAM_05_URL`.
3. Restart the console. A worker starts for every twin camera that has a URL. It uses that camera's
   calibration and only the twin zones the camera watches.

**Zones in the twin map to rules as follows:**

| Twin zone rule | Runtime meaning | IOGP rule |
|---|---|---|
| `red` | exclusion zone (active while pipe handling) | RED |
| `height` | deck edge (`near` = 1 m) | EDGE |
| `lift` | drop zone of the suspended load | LIFT |
| (whole deck) | walkway | PPE, FALL |

**Performance settings:**

- `VG360_FPS` sets frames analysed per second (default 10).
- `VG360_TILES` sets the detection tiles per frame (default 1). Use 2–3 for wide views where people are small; each step costs more CPU.
- On a 2-vCPU box, tract manages about 5 fps on one tile, and ONNX Runtime 30 fps or more.

---

## 4. Calibrating a camera

Calibration tells VG360 how pixels map to metres on the deck. There are three ways to get it.

### 4.1 From deck points (most accurate)

In the Rig Twin, give a camera four or more calibration points: deck spots with known (x, y) metres and
the pixel where each appears. VG360 solves focal length, height, pitch, yaw and roll (DLT and then
Levenberg–Marquardt). It checks the runtime camera against the twin's own projection, to within 1 px.

### 4.2 From vanishing points (no surveyed points)

Use this when you have a frame but no surveyed points. Deck plates, rails, pipe racks and grating give
two families of straight edges at right angles on the deck. Parallel lines meet at one image point (the
vanishing point), and two such points give the camera's focal length, pitch and roll. The formula is
proved in `lean/VG360/Calib.lean` (`focal_from_orthogonal_vps`).

1. On a still frame, mark **2–6 segments** along edges running one way (family A), and 2–6 along edges
   at right angles to them (family B). Long segments spread across the image work best.
2. Optionally, mark boxes around a few **workers standing upright**. Those give the mount height.
3. Send them to the console (the pixel values below are only an example):

```sh
curl -s localhost:3000/api/calibrate -H 'content-type: application/json' -d '{
  "width": 1280, "height": 720,
  "lines_a": [[212,604, 655,401], [402,689, 836,442], [608,712, 1012,470]],
  "lines_b": [[655,401, 1015,488], [430,505, 830,600], [212,604, 600,705]],
  "people":  [[640,300, 664,372], [880,410, 910,500]]
}'
```

The reply contains:

- `f_px`, `hfov_deg`, `pitch_deg`, `roll_deg`;
- `height_m`, when people were given;
- a `twin` block with the values to type into the Rig Twin: `fov`, `pitch` and `h_above_deck`;
- `notes`. For example, it warns when roll is above 2°, because the twin assumes a level camera.

If the families are not really at right angles, or a family's lines are parallel in the image, the reply
says why (`"ok": false`).

### 4.3 Mount height from people, live (auto-height)

If you know the field of view and pitch but not the mount height, let the camera estimate it:

```sh
export VG360_CAM_C5_AUTO_HEIGHT=1
```

- **How it works:** every distance scales linearly with mount height (`Calib.head_height_scales`). The worker collects the first 12 boxes of upright workers and sets the height so their median head is at 1.70 m.
- **No rules while it runs.** Until the height is set, people are tracked but **no rule is evaluated**, and `/live` shows *estimating mount height (rules off)*.
- **Afterwards:** the result is logged to vesdb as a `height-estimated` camera event. Copy it into the twin so the next start does not need to estimate again.
- **Accuracy:** on the demo clip, a camera set to 6 m corrects itself to 9.8 m (true height 10 m, workers rendered 1.75 m tall) after 0.4 s.
- **Fisheye cameras:** auto-height applies to standard (pinhole) cameras only. Fisheye cameras need deck points or a known mount height.

---

## 5. Fisheye cameras

VG360 analyses a fisheye camera (up to about 190°) through **upright virtual views**.

1. Each frame is resampled into three ordinary pinhole views.
2. People are detected in each view and placed on the deck through that view's own camera model.
3. A worker seen by two views is merged (feet within 0.8 m), and one tracker covers the whole fisheye.

Boxes on `/live` are drawn on the fisheye image itself.

**Declaring a fisheye.** Add a `lens` block to the camera in the Rig Twin JSON. The camera's `x`, `y`,
`h`, `yaw` and `pitch` give its position and heading, as for any camera.

```json
{ "id": "C6", "res": "2880x2880", "x": 0, "y": 0, "h": 8, "yaw": 90, "pitch": 90,
  "lens": { "model": "kannala-brandt", "fx": 892, "fy": 892, "cx": 1440, "cy": 1440,
            "k": [0.02, -0.01, 0.002, -0.0005],
            "views": [[0, 35], [120, 35], [240, 35]] } }
```

- **Lens parameters:** `fx`, `fy`, `cx`, `cy` and `k` are the Kannala–Brandt intrinsics (OpenCV's `cv2.fisheye` model), in the camera's own pixels. Take them from the vendor's calibration sheet, or from a checkerboard calibration.
- **Views:** `views` lists `[compass bearing°, pitch°]` pairs and is optional.
  - Camera looking down (pitch ≥ 60°): the default is three views 120° apart at 35° down.
  - Otherwise: views at −45°, 0° and +45° around the camera's heading.
  - Each view is 960×540 with a 70° field of view.
- **Resolution:** use 5 MP or more. On the demo, a 2880² (8 MP) fisheye finds 56 of the 57 people the pinhole camera finds, with deck positions within a median of 0.06 m. At 1280² (1.6 MP), people are too small to be detected reliably.
- **Moved camera:** a moved fisheye cannot be compensated with a homography, so any measured move **pauses** analytics until the camera is put back (see §6).
- **CPU:** each fisheye costs about three pinhole cameras.

---

## 6. Camera guard: moved, blocked or lost cameras

Each worker compares the live view with the reference view captured at start-up, about 4 times a second.

| Situation | Guard status | Analytics |
|---|---|---|
| View matches | POSITION CORRECT | **Aligned**: calibration used as is |
| Small knock (≤ 8 % shift, ≤ 5° rotation, ≤ 6 % zoom, ≥ 60 matched points) | CAMERA MOVED | **Compensated**: detections mapped back through the measured homography; alerts carry `mode: Compensated` |
| Larger move, lens covered, view lost, or any move of a fisheye | CAMERA MOVED / BLOCKED / LOST | **Paused**: no analytics and no alerts |

- **Guidance:** the camera page gives instructions such as *Move view RIGHT 5.5 %, rotate ANTI-CLOCKWISE 3.2°*, with a blue box showing where the correct view now sits in the image.
- **Clearing the alarm:** the alarm clears after the view has been aligned for the hold time.
- **Evidence:** every change of guard status is logged to vesdb.

---

## 7. Rules, approval and site signals

The five rules are written in controlled English in `config/vg360_rules.ace` and compiled by **iogp-editor**.

```
RED  · If a worker is in Red-Zone while pipe-handling …            critical → MCC floor 0.85
EDGE · If a worker is near Deck-Edge …                             critical → 0.85
LIFT · If a worker is under a suspended load for at least 3 s …    critical → 0.85
FALL · If a fallen-person is seen for at least 2 s …               high     → 0.75
PPE  · If a worker is in Deck and does not wear a helmet ≥ 2 s …   critical → 0.85
```

A rule **sounds** (it is *armed*) only when both of these hold:

- it has been **approved** in iogp-editor by someone other than its author (four-eyes sign-off), **and**
- its validated **MCC** is at or above the floor for its severity.

Any other rule runs in **shadow** mode. It is evaluated and logged with `armed: false`, but never sounded.
The committed configuration is a draft, so every rule is shadow until your approvers sign it off.

**To deploy approved rules:**

```sh
iogp serve …            # authors submit, approvers approve
export VG360_RULES_CONFIG=/path/to/export/config.json
```

**Site signals.** Some rules depend on the rig's state. Send it from the control system, or flip the
switches on `/live`:

```sh
curl -s localhost:3000/api/signals -H 'content-type: application/json' \
     -d '{"suspended_load": true, "active": ["pipe-handling"]}'
```

---

## 8. Evidence (vesdb)

Point the console at a vesdb server (`dhaneshkk/test1_vesdb`). Keep tokens in the environment.

```sh
export VG360_VESDB_URL=https://vesdb.rig7.local:3000
export VG360_VESDB_WRITE_TOKEN=…   VG360_VESDB_AUDIT_TOKEN=…
export VG360_RIG=rig7              # subject prefix: /rigs/rig7/…
export VG360_VESDB_SPOOL=/var/lib/vg360/spool   # writes are queued here while vesdb is down
```

**What is written:**

- every raised alert, as an incident (`hazard-detected` and `alert-raised`), with:
  - the frame's SHA-256;
  - the ruleset, model and calibration hashes;
  - whether it was armed;
- camera-guard changes;
- `calibration-set` events from the twin and `height-estimated` events from auto-height;
- `model-deployed` events, carrying the model cards.

**Proving an alert frame.** The snapshot is served at `/api/evidence/{incident}/snapshot.jpg` and kept in
`state/evidence/`. To prove it is the frame the alert was raised on:

```sh
vesdb-verify bundle bundle.json --log-key log.key.pub --witness-key hse.key.pub \
             --file-hash snapshot.jpg=<hazard-detected event id>:frame_sha256
```

A swapped or edited frame fails verification.

---

## 9. APIs and the OpenAI-compatible endpoint

| Method | Path | Returns |
|---|---|---|
| GET | `/api/live` | cameras (fps, frames, latest result, size, fisheye), recent alerts, signals |
| GET | `/api/live/events` | server-sent events: `hello`, `frame` (≤ 4 Hz per camera), `alert`, `signals` |
| GET | `/api/live/{cam}/frame.jpg` | latest frame of a camera |
| GET | `/api/evidence/{incident}/snapshot.jpg` | the frame an alert was raised on |
| POST | `/api/signals` | set site signals |
| POST | `/api/calibrate` | vanishing-point and people calibration (§4.2) |
| GET / POST | `/api/twin` | the applied Rig Twin / apply a new one |
| GET | `/api/evidence` | vesdb connection and checkpoint |
| GET | `/v1/models`, `/v1/rules`, `/v1/ledger`, `/v1/cameras` | JSON |
| POST | `/v1/chat/completions` | OpenAI-compatible assistant (model `vg360-ops`), with optional `"stream": true` |

**The assistant answers from site state, deterministically and without an LLM**, so every number it gives can be audited.
Ask it about cameras ("which camera moved?"), rules ("rules mcc"), alerts or evidence, and what is live now.

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:3000/v1", api_key="unused")
for chunk in c.chat.completions.create(model="vg360-ops", stream=True,
                                       messages=[{"role": "user", "content": "what is live now?"}]):
    print(chunk.choices[0].delta.content or "", end="")
```

---

## 10. Models and training

Training happens only in **`notebooks/vg360_train_cpu.ipynb`**, a single Colab cell. In Colab it clones
the repository and writes its outputs where the Rust side expects them.

**PPE classifier (hard hat).**

- **Data:** your own crops, in folders `helmet/` and `no_helmet/` (`CFG.ppe_dir` or `VG360_PPE_DIR`), or Pascal-VOC head boxes (`CFG.voc_dir`).
- **Training:** a Clifford-convolution network and a plain CNN of equal size, both trained with SAM. The best epoch is picked on validation MCC.
- **Reporting:** test MCC with a bootstrap 95 % confidence interval, and sharpness.
- **Outputs:** `models/ppe_clifford_folded.onnx` and its **model card** (`.card.json`).

**Person detector (YOLOX-Nano fine-tune).**

1. **Label frames.** Draw person boxes on frames from your cameras with CVAT, exported in YOLO format: `images/{train,val,test}` and `labels/{train,val,test}`. A few hundred frames, including crouching and top-down workers, is a good start.
2. **Train.** Run the notebook with `VG360_DET_DIR=/path/to/labels`.
3. **Check the results.** Cell MCC, precision / recall / F1 at IoU 0.5 and AP50, before and after fine-tuning.
4. **Deploy.**

   ```sh
   export VG360_DETECTOR=models/yolox_nano_ft.onnx
   ```

**Model cards are enforced.** At start-up the console checks each model's SHA-256 against its card:

- **PPE:** a model that does not match its card is switched off.
- **Fine-tuned detector:** it runs only if it matches its card and was trained on real site frames. A smoke-set model needs `VG360_ALLOW_SMOKE_MODELS=1`, for testing only.

`/training` shows the cards.

**Auditing site footage.** This runs the detector and PPE model over a clip, and saves a contact sheet,
crops and frames for labelling into `footage_audit/`:

```sh
VG360_FOOTAGE=/path/to/clip.mp4    # set before running the notebook cell
```

Set `CFG.footage_crop` to remove the window border of VMS screen recordings.

**Current status (honest):** both models are trained on synthetic data. The PPE rule is therefore shadow
(MCC 0.800, floor 0.85). On real overhead drill-floor footage (RIG 2004), the stock detector finds only
one of the two or three workers in view. Fine-tune it on labelled frames from your cameras before relying
on it.

---

## 11. Environment variables

| Variable | Meaning |
|---|---|
| `VG360_CAM_<ID>_URL` | stream of twin camera `<ID>` (RTSP or file) |
| `VG360_CAM_<ID>_AUTO_HEIGHT=1` | estimate that camera's mount height from standing workers (§4.3) |
| `VG360_DEMO=1` · `VG360_DEMO_FISHEYE=1` · `VG360_DEMO_CLIP` | demo cameras / replace the demo clip |
| `VG360_FPS` · `VG360_TILES` | analysed frames per second (10) · detection tiles (1) |
| `VG360_DETECTOR` · `VG360_ALLOW_SMOKE_MODELS` | fine-tuned detector · allow smoke-trained models (tests) |
| `VG360_RULES_CONFIG` | approved iogp-editor export (default: the draft) |
| `VG360_VESDB_URL` · `_WRITE_TOKEN` · `_AUDIT_TOKEN` · `_SPOOL` · `VG360_RIG` | evidence log |
| `VG360_STATE_DIR` · `VG360_ROOT` · `HOST` · `PORT` | state folder · app folder · listen address |
| `ORT_DYLIB_PATH` | ONNX Runtime library (`--features ort`) |
| `VG360_GUARD_DEBUG` | show ffmpeg's messages |
| Notebook: `VG360_FAST`, `VG360_OUT`, `VG360_WRITE_REPO`, `VG360_PPE_DIR`, `VG360_DET_DIR`, `VG360_FOOTAGE`, `VG360_URL` | smoke run, output folder, write into repo, data, footage audit, console to query |

---

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `/live` says *No camera streams are running* | No `VG360_CAM_<ID>_URL` matches a camera of the applied twin (check the id spelling and the dash → underscore rule), or `VG360_DEMO` is not set. |
| Camera card shows an error *no frames from stream* | ffmpeg cannot open the URL. Run with `VG360_GUARD_DEBUG=1` to see why: credentials, network, codec. |
| *PPE classifier disabled: … is not the model its card describes* | The ONNX file was replaced without its card. Re-export both from the notebook. |
| Fine-tuned detector *refused* on `/training` | Its card says it was trained on synthetic data, or its SHA-256 does not match. Train on labelled site frames. |
| People missed on overhead / night cameras | Known limit of the stock COCO detector. Raise `VG360_TILES`, and fine-tune on your frames (§10). |
| Analytics paused on a fisheye after a bump | Expected: fisheyes are not compensated. Re-aim the camera until the guard reads POSITION CORRECT. |
| `/api/calibrate` returns `f² ≤ 0` | The two line families are not at right angles on the deck, or both run nearly the same way. Pick edges of rectangular plates. |
| Auto-height never finishes | Fewer than 12 upright workers have been seen. Wait, or set the height in the twin. |
| Every alert is *shadow* | The rules config is a draft, or a rule's MCC is below its floor (§7). |
| Build error mentioning `int_format_into` / Rust version | The console needs Rust ≥ 1.99 (`rustup update`). |

---

## 13. Privacy and security

- **Stream URLs and credentials:** keep them only in environment variables, never in the repository, twin or logs.
- **Footage and snapshots:** site footage, alert snapshots (`state/`), footage audits (`footage_audit/`) and smoke models (`detector_smoke/`) show identifiable workers and are git-ignored. Keep them on site.
- **vesdb tokens:** keep them in the environment.
- **Rule approval:** keep the four-eyes sign-off. No tool in this repository approves rules on anyone's behalf.
