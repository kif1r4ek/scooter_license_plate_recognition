"""Check the detector alone on the CPU, the way the CLI runs it.

    python training/eval_onnx.py --split test --model models/detector.onnx
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
SIZE = 1280
CONF = 0.25
IOU_NMS = 0.3
NAMES = ["front", "rear"]


def letterbox(img):
    h, w = img.shape[:2]
    k = SIZE / max(h, w)
    nw, nh = round(w * k), round(h * k)
    out = np.full((SIZE, SIZE, 3), 114, np.uint8)
    dx, dy = (SIZE - nw) // 2, (SIZE - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, k, dx, dy


def corners(cx, cy, w, h, a):
    c, s = math.cos(a), math.sin(a)
    dx = np.array([c, s]) * w / 2
    dy = np.array([-s, c]) * h / 2
    p = np.array([cx, cy])
    return np.array([p - dx - dy, p + dx - dy, p + dx + dy, p - dx + dy], np.float32)


def iou(a, b):
    a, b = cv2.convexHull(np.float32(a)), cv2.convexHull(np.float32(b))
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0


def detect(sess, name, img):
    inp, k, dx, dy = letterbox(img)
    x = inp[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    out = sess.run(None, {name: x})[0][0].T
    scores = out[:, 4:-1]
    keep = scores.max(1) >= CONF
    out, scores = out[keep], scores[keep]
    order = np.argsort(-scores.max(1))
    dets = []
    for i in order:
        cx, cy, w, h = out[i, :4]
        poly = corners((cx - dx) / k, (cy - dy) / k, w / k, h / k, float(out[i, -1]))
        if all(iou(poly, d["poly"]) < IOU_NMS for d in dets):
            dets.append({"poly": poly, "cls": int(scores[i].argmax()), "conf": float(scores[i].max())})
    return dets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model", type=Path, default=ROOT / "models/detector.onnx")
    args = parser.parse_args()
    SPLIT = args.split
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    data = ROOT / "data" / "yolo"
    tp, fn, fp, wrong_cls, ious, times = 0, 0, 0, 0, [], []
    for path in sorted((data / "images" / SPLIT).glob("*.jpg")):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        gt = []
        for line in (data / "labels" / SPLIT / f"{path.stem}.txt").read_text().split("\n"):
            if line:
                v = list(map(float, line.split()))
                gt.append((int(v[0]), np.array(v[1:]).reshape(4, 2) * [w, h]))
        t = time.perf_counter()
        dets = detect(sess, name, img)
        times.append(time.perf_counter() - t)
        free = list(range(len(dets)))
        for cls, poly in gt:
            best = max(free, key=lambda j: iou(poly, dets[j]["poly"]), default=None)
            score = iou(poly, dets[best]["poly"]) if best is not None else 0
            if score >= 0.3:
                tp += 1
                ious.append(score)
                wrong_cls += dets[best]["cls"] != cls
                free.remove(best)
            else:
                fn += 1
                print("MISSED", path.name, NAMES[cls], round(score, 2))
        for j in free:
            fp += 1
            print("EXTRA", path.name, NAMES[dets[j]["cls"]], round(dets[j]["conf"], 2))
    print(f"{SPLIT}: found {tp}/{tp + fn}, extra {fp}, wrong class {wrong_cls}")
    print(f"mean IoU {np.mean(ious):.3f}, min {np.min(ious):.3f}, below 0.5: {sum(i < 0.5 for i in ious)}")
    print(f"CPU: {np.mean(times[1:]):.2f} s per image, cores {sess.get_session_options().intra_op_num_threads or 'auto'}")


if __name__ == "__main__":
    main()
