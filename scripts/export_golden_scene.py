"""Export golden vectors for Phase 2 (vg360-scene): the notebook's §8 simulated site, recorded frame by frame.

Executes §0–§5 and §7b of notebooks/vg360_clifford_safety_cpu.ipynb unchanged, runs §8's simulation, and
records for every frame: the person boxes and activity context (inputs) and, for every box, the notebook's
geometry, track id, zone clearances, fall state and events, plus every rule alert and the ledger (outputs).
Usage: python scripts/export_golden_scene.py  → testdata/golden/scene.json
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = "".join(json.loads((ROOT / "notebooks/vg360_clifford_safety_cpu.ipynb").read_text())["cells"][0]["source"])
lines = src.split("\n")
def section(a, b):
    i = next(k for k, l in enumerate(lines) if l.startswith("# ─") and a in l)
    j = next(k for k, l in enumerate(lines) if l.startswith("# ─") and b in l)
    return "\n".join(lines[i:j])
G = {"__name__": "golden"}
exec(section("§0", "§1"), G)
import shutil; shutil.rmtree(G["BASE"], ignore_errors=True)  # §0 creates a work dir; not needed here
exec(section("§2", "self-tests"), G)
exec(section("§7b", "§8"), G)
sim = section("§8", "§9")
sim = sim[: sim.index("EVENT_LOG.clear()")]          # keep the function, not the report printing
exec(sim, G)

frames, rules = [], []
orig_analyse = G["analyse_boxes"]
def analyse(cam, tracker, zones, boxes, t):
    recs = orig_analyse(cam, tracker, zones, boxes, t)
    out = []
    for r in recs:
        if r is None: out.append(None); continue
        out.append({"foot": [float(v) for v in r["foot"][:2]], "head_z": float(r["head_z"]), "tilt": float(r["tilt"]),
                    "iou": float(r["pose_iou"]), "track": int(r["track"]),
                    "zones": {k: float(v) for k, v in r["zones"].items()},
                    "down": bool(r["state"]["down"]), "fallen": bool(r["state"]["fallen"]),
                    "events": [{k: v for k, v in e.items() if k != "hash"} for e in r["events"]]})
    frames.append({"t": t, "boxes": [[float(v) for v in b] for b in boxes], "recs": out})
    return recs
G["analyse_boxes"] = analyse
orig_step = G["RuleEngine"].step
def step(self, key, state, ctx, t, dt, clip="", emit_events=True):
    fired = orig_step(self, key, state, ctx, t, dt, clip, emit_events)
    if self.ledger is not None:
        frames[-1].setdefault("ctx", ctx)
        rules.append({"t": t, "key": key, "fired": fired})
    return fired
G["RuleEngine"].step = step

G["EVENT_LOG"].clear(); G["SIM_LEDGER"] = G["Ledger"](); G["RULE_CMS"] = {}
res = G["simulate_and_score"]()
C = G["CFG"]
out = {"camera": {"f": C.focal_px, "cx": C.img_w / 2, "cy": C.img_h / 2, "h": C.cam_height_m, "pitch": C.pitch_deg, "yaw": C.yaw_deg},
       "dt": 1 / C.sim_fps, "frames": frames, "rule_steps": rules, "ledger": G["SIM_LEDGER"].entries,
       "rule_cms": G["RULE_CMS"], "sim": {k: v for k, v in res.items() if isinstance(v, dict)}}
def rnd(o):
    if isinstance(o, float): return float(f"{o:.7g}")
    if isinstance(o, list): return [rnd(v) for v in o]
    if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
    return o
(ROOT / "testdata/golden/scene.json").write_text(json.dumps(rnd(out), separators=(",", ":")))
# Rule Studio export (rules + MCC + ledger) for the console and the geom golden test; the PPE confusion
# matrix comes from the notebook's PPE test set (no imagery in this simulation), kept from the previous export.
prev = json.loads((ROOT / "testdata/rule_studio_export.json").read_text())
cms = dict(G["RULE_CMS"]); cms["ppe"] = prev["rules"]["ppe"]["cm"]; cms["ppe:src"] = prev["rules"]["ppe"]["validated_on"]
G["export_rule_studio"](cms, G["SIM_LEDGER"], ROOT / "testdata/rule_studio_export.json",
                        "notebook simulated site (noisy projected boxes; bodies share the fall-fitter's shape)")
print(f"{len(frames)} frames, {sum(len(f['boxes']) for f in frames)} boxes, {len(out['ledger'])} ledger entries, "
      f"{sum(len(r['events']) for f in frames for r in f['recs'] if r)} geometry events")
