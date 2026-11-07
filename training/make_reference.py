"""Answers of the Python pipeline on the test photos, for the C++ parity test.

Run again after replacing a model: tests/data/reference.json.

    python training/make_reference.py --models models
"""
import argparse
import json
from pathlib import Path

import cv2
import onnxruntime as ort

from eval_e2e import read_plate
from eval_onnx import detect

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    images = json.loads((ROOT / "data/labels/labels.json").read_text())["images"]
    stems = json.loads((ROOT / "data/labels/split.json").read_text())["test"]
    det = ort.InferenceSession(str(args.models / "detector.onnx"), providers=["CPUExecutionProvider"])
    ocr = ort.InferenceSession(str(args.models / "ocr.onnx"), providers=["CPUExecutionProvider"])
    out = []
    for stem in stems:
        file = images[stem]["file"]
        img = cv2.imread(str(ROOT / "data/raw" / file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = []
        for d in detect(det, det.get_inputs()[0].name, img):
            text, conf = read_plate(ocr, gray, d["poly"], d["cls"] == 1)
            plates.append({"text": text, "confidence": round(conf, 4), "rear": d["cls"] == 1,
                           "detection": round(d["conf"], 4)})
        out.append({"file": file, "plates": plates})
    path = ROOT / "tests/data/reference.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(path, len(out), "photos")


if __name__ == "__main__":
    main()
