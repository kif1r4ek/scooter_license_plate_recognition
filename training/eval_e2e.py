"""Whole pipeline on CPU: photo -> detector -> OCR -> numbers. Port this to C++.

    python training/eval_e2e.py --split test --min-conf 0.3 --models models

A plate counts only if all 6 symbols are right. Labels decide what is on the
photo; plates that cannot be read in full are ignored, neither hit nor error.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import eval_onnx
import ocr_lib
from eval_onnx import detect, iou
from ocr_lib import DIGITS, LETTERS

ROOT = Path(__file__).resolve().parent.parent


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def read_plate(sess, gray, poly, rear):
    """Best reading among the possible directions of an orientation-free box."""
    box = np.float32(poly)
    if np.linalg.norm(box[1] - box[0]) < np.linalg.norm(box[3] - box[0]):
        box = np.roll(box, -1, 0)  # long side first
    turns = [0, 1, 2, 3] if rear else [0, 2]  # a rear plate can be narrower than tall
    gray, box = ocr_lib.shrink(gray, box)
    x = np.stack([ocr_lib.to_input(gray, np.roll(box, -k, 0), rear) for k in turns])[:, None].astype(np.float32)
    letters, digits, upright = sess.run(None, {"image": x})
    chars = np.concatenate([softmax(letters).max(2), softmax(digits).max(2)], 1)
    up = 1 / (1 + np.exp(-upright))
    i = int((np.log(chars).sum(1) + np.log(up)).argmax())
    text = "".join(LETTERS[j] for j in letters[i].argmax(1)) + "".join(DIGITS[j] for j in digits[i].argmax(1))
    return text, float(chars[i].min() * up[i])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--min-conf", type=float, default=0.3, help="drop readings below this (the CLI default)")
    parser.add_argument("--det-conf", type=float, default=eval_onnx.CONF, help="detector score threshold")
    parser.add_argument("--models", type=Path, default=ROOT / "models", help="folder with detector.onnx and ocr.onnx")
    parser.add_argument("--json", type=Path, help="also write the totals here")
    args = parser.parse_args()
    SPLIT, MIN_CONF = args.split, args.min_conf
    eval_onnx.CONF = args.det_conf

    images = json.loads((ROOT / "data/labels/labels.json").read_text())["images"]
    stems = json.loads((ROOT / "data/labels/split.json").read_text())[SPLIT]
    det = ort.InferenceSession(str(args.models / "detector.onnx"), providers=["CPUExecutionProvider"])
    ocr = ort.InferenceSession(str(args.models / "ocr.onnx"), providers=["CPUExecutionProvider"])
    name = det.get_inputs()[0].name

    hit = total = wrong = perfect = 0
    times = []
    for stem in stems:
        item = images[stem]
        img = cv2.imread(str(ROOT / "data/raw" / item["file"]))  # applies EXIF rotation
        t = time.perf_counter()
        dets = detect(det, name, img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        reads = []
        for d in dets:
            text, conf = read_plate(ocr, gray, d["poly"], d["cls"] == 1)
            if conf >= MIN_CONF:
                reads.append((text, conf, d))
        times.append(time.perf_counter() - t)

        truth = [p for p in item["plates"] if p["complete"]]
        ignore = [np.float32(p["pts"]) for p in item["plates"] if not p["complete"]]
        found = {r[0] for r in reads}
        want = {p["text"] for p in truth}
        bad = [r for r in reads if r[0] not in want and not any(iou(q, r[2]["poly"]) > 0.3 for q in ignore)]
        hit += len(want & found)
        total += len(want)
        wrong += len(bad)
        ok = want <= found and not bad
        perfect += ok
        if not ok:
            got = ", ".join(f"{r[0]} {r[1]:.2f}" for r in reads) or "nothing"
            print(f"{stem:24} want {sorted(want)}  got {got}")
    print(f"{SPLIT}: numbers read {hit}/{total} = {hit / max(total, 1):.1%}, wrong numbers {wrong}, "
          f"photos fully right {perfect}/{len(stems)} = {perfect / len(stems):.1%}, "
          f"CPU {np.mean(times[1:]):.2f} s per photo")
    if args.json:
        args.json.write_text(json.dumps({"split": SPLIT, "min_conf": MIN_CONF, "numbers_read": hit, "numbers": total,
                                         "wrong_numbers": wrong, "photos_right": perfect, "photos": len(stems)}, indent=1))


if __name__ == "__main__":
    main()
