"""Export golden vectors for the Rust inference crate (vg360-infer).

Runs the notebook's exact Python inference path (onnxruntime + cv2, same functions as §6/§7/§9)
on testdata/frames/ref.png and writes testdata/golden/infer.json plus the letterboxed input.
Usage: python scripts/export_golden.py [path/to/yolox_nano.onnx]
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
DET_CONF, DET_NMS, SIZE, PPE_SIZE = 0.35, 0.45, 416, 48

def letterbox(img, size):
    h, w = img.shape[:2]; r = min(size / h, size / w)
    out = np.full((size, size, 3), 114, np.uint8)
    out[: int(h * r), : int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    return out, r

class YOLOX:
    def __init__(self, path):
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        g, s = [], []
        for st in (8, 16, 32):
            n = SIZE // st; xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            g.append(np.stack((xv, yv), 2).reshape(-1, 2)); s.append(np.full((n * n, 1), st))
        self.grids, self.strides = np.concatenate(g), np.concatenate(s)
    def raw(self, lb):
        return self.sess.run(None, {"images": lb.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
    def __call__(self, gray):
        lb, r = letterbox(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), SIZE)
        out = self.raw(lb)
        xy = (out[:, :2] + self.grids) * self.strides; wh = np.exp(out[:, 2:4]) * self.strides
        sc = out[:, 4:5] * out[:, 5:]; cls = sc.argmax(1); conf = sc.max(1)
        keep = (cls == 0) & (conf > DET_CONF)
        xy, wh, conf = xy[keep], wh[keep], conf[keep]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        if not len(boxes): return np.zeros((0, 4)), np.zeros(0)
        idx = np.array(cv2.dnn.NMSBoxes(np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]].tolist(), conf.tolist(), DET_CONF, DET_NMS)).reshape(-1)
        return boxes[idx], conf[idx]

def detect_tiled(det, gray, tiles=3, overlap=0.25):
    B, S = [], []
    b, c = det(gray); B += list(b); S += list(c)
    H, W = gray.shape[:2]; th, tw = int(H / tiles * (1 + overlap)), int(W / tiles * (1 + overlap))
    for i in range(tiles):
        for j in range(tiles):
            y0 = min(int(i * H / tiles), H - th); x0 = min(int(j * W / tiles), W - tw)
            b, c = det(np.ascontiguousarray(gray[max(y0, 0):y0 + th, max(x0, 0):x0 + tw]))
            B += [bb + np.array([x0, y0, x0, y0]) for bb in b]; S += list(c)
    if not B: return np.zeros((0, 4)), np.zeros(0)
    B, S = np.array(B), np.array(S)
    idx = np.array(cv2.dnn.NMSBoxes(np.c_[B[:, :2], B[:, 2:] - B[:, :2]].tolist(), S.tolist(), DET_CONF, DET_NMS)).reshape(-1)
    B, S = B[idx], S[idx]; order = np.argsort(-S, kind="stable"); keep = []
    for i in order:
        a = B[i]; area = (a[2] - a[0]) * (a[3] - a[1])
        inside = any(max(0, min(a[2], B[k][2]) - max(a[0], B[k][0])) * max(0, min(a[3], B[k][3]) - max(a[1], B[k][1])) > 0.8 * area for k in keep)
        if not inside: keep.append(i)
    return B[keep], S[keep]

def head_crop(gray, box, size):
    x1, y1, x2, y2 = box; h = y2 - y1; s = max(0.24 * h, 8); cx = (x1 + x2) / 2; top = y1 - 0.03 * h
    xa, ya = int(max(cx - s / 2, 0)), int(max(top, 0)); xb, yb = int(min(cx + s / 2, gray.shape[1])), int(min(top + s, gray.shape[0]))
    if xb - xa < 4 or yb - ya < 4: return None
    return cv2.resize(gray[ya:yb, xa:xb], (size, size)).astype(np.float32) / 255

def main():
    det_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models/yolox_nano.onnx"
    det = YOLOX(det_path)
    ppe = ort.InferenceSession(str(ROOT / "models/ppe_clifford_folded.onnx"), providers=["CPUExecutionProvider"])
    gray = cv2.cvtColor(cv2.imread(str(ROOT / "testdata/frames/ref.png")), cv2.COLOR_BGR2GRAY)
    lb, r = letterbox(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), SIZE)
    cv2.imwrite(str(ROOT / "testdata/golden/letterbox_ref.png"), lb)
    raw = det.raw(lb)
    b1, s1 = det(gray)
    bt, st = detect_tiled(det, gray)
    crops = [head_crop(gray, b, PPE_SIZE) for b in bt]
    ok = [i for i, c in enumerate(crops) if c is not None]
    probs = [None] * len(bt)
    if ok:
        lg = ppe.run(None, {"crop": np.stack([crops[i] for i in ok])[:, None]})[0]
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        for i, pi in zip(ok, p[:, 1]): probs[i] = float(pi)
    out = {
        "frame": "testdata/frames/ref.png", "det_conf": DET_CONF, "det_nms": DET_NMS, "size": SIZE, "ratio": r,
        "raw_rows": {str(i): raw[i].tolist() for i in (0, 1, 1000, 2000, 3000, 3500)},
        "raw_max_obj": float(raw[:, 4].max()),
        "single": {"boxes": b1.tolist(), "scores": s1.tolist()},
        "tiled": {"boxes": bt.tolist(), "scores": st.tolist(), "ppe_helmet_prob": probs},
    }
    (ROOT / "testdata/golden/infer.json").write_text(json.dumps(out, indent=1))
    print(f"single {len(b1)} boxes, tiled {len(bt)} boxes, ppe {sum(p is not None for p in probs)} crops")

if __name__ == "__main__":
    main()
