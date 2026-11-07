"""Everything from labelled photos to installed models, in one command.

    python training/run_all.py --backend cpu            # on this machine's CPU
    python training/run_all.py --backend gpu            # on a local NVIDIA GPU
    python training/run_all.py --backend modal          # on a Modal cloud T4
    python training/run_all.py --backend cpu --smoke    # a few minutes, checks that everything runs

Steps (each one prints what it runs):
  1. export_yolo.py   data/raw + data/labels -> data/yolo (detector dataset, fixed split)
  2. detector         train YOLO11n-OBB              -> runs/<run>/detector/detector.onnx
  3. export_ocr.py    plate crops + the new detector's boxes -> data/ocr
  4. ocr              train the plate reader         -> runs/<run>/ocr/ocr.onnx
  5. evaluate         detector alone and the whole pipeline on the test photos
  6. install          copy the models to models/ and refresh tests/data/reference.json
                      (skipped with --smoke or --no-install)
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable


def sh(*args):
    print("\n+ " + " ".join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["cpu", "gpu", "modal"], required=True)
    parser.add_argument("--run", default=datetime.now().strftime("%Y%m%d_%H%M"), help="name of runs/<run>/")
    parser.add_argument("--smoke", action="store_true", help="1 epoch + 30 steps, do not install the models")
    parser.add_argument("--detector-epochs", type=int, default=200, help="stops early after 50 epochs without progress")
    parser.add_argument("--ocr-steps", type=int, default=10000)
    parser.add_argument("--ocr-select", action="store_true",
                        help="train the reader on train only and keep the step best on val "
                             "(default: train+val, last step, as the shipped model)")
    parser.add_argument("--skip-detector", action="store_true", help="reuse models/detector.onnx, train only the reader")
    parser.add_argument("--no-install", action="store_true", help="leave models/ as it is")
    parser.add_argument("--workers", type=int, default=None, help="data loader processes (default: up to 8)")
    args = parser.parse_args()

    epochs, steps = (1, 30) if args.smoke else (args.detector_epochs, args.ocr_steps)
    run = ("smoke_" if args.smoke else "") + args.run
    out = ROOT / "runs" / run
    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"run {run}: backend {args.backend}, detector {epochs} epochs, reader {steps} steps -> {out}")

    sh(PY, HERE / "export_yolo.py")

    if args.skip_detector:
        shutil.copy(ROOT / "models/detector.onnx", models / "detector.onnx")
    else:
        if args.backend == "modal":
            sh(PY, HERE / "modal_app.py", "detector", "--run", run, "--epochs", epochs)
        else:
            cmd = [PY, HERE / "detector.py", "--out", out / "detector", "--epochs", epochs,
                   "--device", "0" if args.backend == "gpu" else "cpu"]
            if args.workers is not None:
                cmd += ["--workers", args.workers]
            sh(*cmd)
        shutil.copy(out / "detector/detector.onnx", models / "detector.onnx")

    sh(PY, HERE / "export_ocr.py", "--detector", models / "detector.onnx")

    final = [] if args.ocr_select else ["--final"]
    if args.backend == "modal":
        sh(PY, HERE / "modal_app.py", "ocr", "--run", run, "--steps", steps, *final)
    else:
        cmd = [PY, HERE / "ocr_model.py", "--out", out / "ocr", "--steps", steps,
               "--device", "cuda" if args.backend == "gpu" else "cpu", *final]
        if args.workers is not None:
            cmd += ["--workers", args.workers]
        sh(*cmd)
    shutil.copy(out / "ocr/ocr.onnx", models / "ocr.onnx")

    sh(PY, HERE / "eval_onnx.py", "--split", "test", "--model", models / "detector.onnx")
    sh(PY, HERE / "eval_e2e.py", "--split", "test", "--models", models, "--json", out / "e2e_test.json")

    if not (args.smoke or args.no_install):
        for name in ("detector.onnx", "ocr.onnx"):
            shutil.copy(models / name, ROOT / "models" / name)
        if (out / "detector/results.csv").exists():
            shutil.copy(out / "detector/results.csv", ROOT / "models/results.csv")
        shutil.copy(out / "ocr/results.json", ROOT / "models/ocr_results.json")
        sh(PY, HERE / "make_reference.py", "--models", ROOT / "models")
        print("\ninstalled into models/; rebuild the CLI to pick them up (scripts/build_cli.sh)")

    e2e = json.loads((out / "e2e_test.json").read_text())
    print(f"\ndone in {(time.time() - started) / 60:.0f} min. Test photos: "
          f"{e2e['numbers_read']}/{e2e['numbers']} numbers read, {e2e['wrong_numbers']} wrong, "
          f"{e2e['photos_right']}/{e2e['photos']} photos fully right. Everything is in {out}")


if __name__ == "__main__":
    main()
