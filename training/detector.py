"""Train the plate detector (YOLO11n OBB) and export it to ONNX.

Runs the same on a CPU, a local GPU or inside Modal (training/modal_app.py).

    python training/detector.py --device cpu --epochs 200 --out runs/manual/detector
    python training/detector.py --device 0   --epochs 200 --out runs/manual/detector

Writes to --out: best.pt, results.csv, detector.onnx, metrics.json.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = 1280


def train(data: Path, out: Path, epochs: int, device: str, batch: int = 6, workers: int = 4, on_save=None):
    """data: folder made by export_yolo.py. on_save: called after files are written (Modal commits its volume)."""
    from ultralytics import YOLO

    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="yolo_"))
    shutil.copytree(data, work / "data")
    yaml = work / "data" / "data.yaml"
    yaml.write_text(yaml.read_text().replace("path: .", f"path: {work / 'data'}"))

    model = YOLO("yolo11n-obb.pt")  # COCO-pretrained weights, downloaded by ultralytics on first use
    model.train(
        data=str(yaml),
        imgsz=IMG_SIZE,
        epochs=epochs,
        patience=50,
        batch=batch,
        workers=workers,
        device=device,
        conf=0.05,  # early epochs emit tens of thousands of boxes and rotated NMS runs out of GPU memory
        degrees=15,
        fliplr=0.5,
        flipud=0.0,
        project=str(work / "runs"),
        name="detector",
        exist_ok=True,
    )
    run = work / "runs" / "detector"
    shutil.copy(run / "weights" / "best.pt", out / "best.pt")
    shutil.copy(run / "results.csv", out / "results.csv")
    if on_save:
        on_save()  # keep the weights even if the export below fails

    best = YOLO(str(out / "best.pt"))
    m = best.val(data=str(yaml), split="test", imgsz=IMG_SIZE, batch=4, conf=0.05, device=device)
    metrics = {"test_map50": float(m.box.map50), "test_map50_95": float(m.box.map),
               "test_ap50_per_class": [float(x) for x in m.box.ap50]}
    print("TEST", metrics, flush=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    onnx = best.export(format="onnx", imgsz=IMG_SIZE, opset=17, simplify=True)
    shutil.copy(onnx, out / "detector.onnx")
    shutil.rmtree(work, ignore_errors=True)
    if on_save:
        on_save()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/yolo")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200, help="early stop after 50 epochs without progress")
    parser.add_argument("--device", default="cpu", help="cpu, or a GPU index such as 0")
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    train(args.data, args.out, args.epochs, args.device, args.batch, args.workers)


if __name__ == "__main__":
    main()
