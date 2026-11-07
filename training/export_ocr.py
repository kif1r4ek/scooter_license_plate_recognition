"""Cut plates out of the labelled photos for OCR training.

Crops keep a wide margin around the plate, so box jitter can be applied during
training. Uses the detector's split: scooters never cross train/val/test.
Each plate also gets the box the detector really gives for it, because those
are much looser than the labels and the OCR has to read from them.
Output: data/ocr/images/*.jpg and data/ocr/index.json.

    python training/export_ocr.py --detector models/detector.onnx
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from eval_onnx import detect, iou

ROOT = Path(__file__).resolve().parent.parent
MARGIN = 0.8  # of the plate's long side, on each side
MAX_PLATE_SIDE = 480  # px; phone photos are far sharper than the model needs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", type=Path, default=ROOT / "models/detector.onnx",
                        help="the detector the OCR will work with")
    parser.add_argument("--out", type=Path, default=ROOT / "data/ocr")
    args = parser.parse_args()

    images = json.loads((ROOT / "data/labels/labels.json").read_text())["images"]
    split = json.loads((ROOT / "data/labels/split.json").read_text())
    out = args.out
    shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)

    det = ort.InferenceSession(str(args.detector), providers=["CPUExecutionProvider"])
    index = []
    for name, stems in split.items():
        for stem in stems:
            item = images[stem]
            plates = [p for p in item["plates"] if p["complete"] and not p["uncertain"]]
            if not plates:
                continue
            img = ImageOps.exif_transpose(Image.open(ROOT / "data/raw" / item["file"])).convert("L")
            arr = np.asarray(img)
            found = [d["poly"] for d in detect(det, det.get_inputs()[0].name, cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR))]
            for i, p in enumerate(plates):
                pts = np.array(p["pts"], dtype=np.float32)
                side = max(cv2.minAreaRect(pts)[1])
                x0, y0 = np.maximum(pts.min(0) - side * MARGIN, 0).astype(int)
                x1, y1 = (pts.max(0) + side * MARGIN).astype(int)
                crop = arr[y0:y1, x0:x1]
                scale = min(1.0, MAX_PLATE_SIDE / side)
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                file = f"{name}_{re.sub(r'[^A-Za-z0-9_]', '_', stem)}_{i}.jpg"
                cv2.imwrite(str(out / "images" / file), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                poly = max(found, key=lambda q: iou(q, pts), default=None)
                if poly is not None and iou(poly, pts) < 0.25:
                    poly = None
                index.append({
                    "file": file, "split": name, "text": p["text"], "cls": p["cls"],
                    "pts": ((pts - [x0, y0]) * scale).round(1).tolist(),
                    "det": None if poly is None else ((poly - [x0, y0]) * scale).round(1).tolist(),
                })
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(len(index), "plates,", sum(e["det"] is not None for e in index), "with a detector box")


if __name__ == "__main__":
    main()
