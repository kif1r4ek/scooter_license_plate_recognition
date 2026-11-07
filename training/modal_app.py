"""Training on a Modal cloud GPU (NVIDIA T4): the same code as on a local machine.

Modal runs detector.py / ocr_model.py in a container with a GPU. Data goes to the
Modal volume "whoosh-lpr", results come back into runs/<run>/. The app is
deployed and the job spawned, so it keeps running if this computer sleeps or
reboots; `download` fetches the results afterwards.

    python training/modal_app.py detector --run r1 --epochs 200
    python training/modal_app.py ocr --run r1 --steps 10000 --final
    python training/modal_app.py download --run r1 --stage ocr     # if the wait above was interrupted

Needs `pip install modal` and `modal setup` (or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
APP = "whoosh-lpr"
VOLUME = "whoosh-lpr"
GPU = "T4"

app = modal.App(APP)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1")
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    .add_local_file(HERE / "detector.py", "/root/detector.py")
    .add_local_file(HERE / "ocr_model.py", "/root/ocr_model.py")
    .add_local_file(HERE / "ocr_lib.py", "/root/ocr_lib.py")
    .add_local_dir(HERE / "fonts", "/root/fonts")
)

# what each stage leaves in /vol/runs/<run>/<stage>/
OUTPUTS = {
    "detector": ["detector.onnx", "best.pt", "results.csv", "metrics.json"],
    "ocr": ["ocr.onnx", "best.pt", "results.json"],
}


@app.function(image=image, gpu=GPU, volumes={"/vol": volume}, timeout=4 * 60 * 60)
def train_detector(run: str, epochs: int):
    import detector

    detector.train(Path("/vol/yolo"), Path(f"/vol/runs/{run}/detector"), epochs, device="0", on_save=volume.commit)


@app.function(image=image, gpu=GPU, cpu=8.0, memory=16384, volumes={"/vol": volume}, timeout=4 * 60 * 60)
def train_ocr(run: str, steps: int, final: bool):
    import ocr_model

    ocr_model.train(Path("/vol/ocr"), Path(f"/vol/runs/{run}/ocr"), steps, final, device="cuda", workers=8,
                    on_save=volume.commit, fonts=Path("/root/fonts"))


# ---------------------------------------------------------------- local side

def sh(*args, check=True):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run([str(a) for a in args], check=check)


def upload(local: Path, remote: str):
    sh("modal", "volume", "rm", "-r", VOLUME, remote, check=False)  # no stale files from an older export
    sh("modal", "volume", "put", "--force", VOLUME, local, remote)


def download(run: str, stage: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS[stage]:
        sh("modal", "volume", "get", "--force", VOLUME, f"/runs/{run}/{stage}/{name}", dest / name)
    print("downloaded to", dest)


def spawn_and_wait(fn_name: str, *args):
    sh("modal", "deploy", HERE / "modal_app.py")
    call = modal.Function.from_name(APP, fn_name).spawn(*args)
    print(f"started {fn_name}{args} on Modal, call {call.object_id}", flush=True)
    print(f"logs: modal app logs {APP}   (the job keeps running if this script stops)", flush=True)
    start = time.time()
    while True:
        try:
            call.get(timeout=60)
            break
        except (TimeoutError, modal.exception.TimeoutError):
            print(f"  ...running {int(time.time() - start) // 60} min", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detector", help="upload data/yolo, train the detector, download the result")
    d.add_argument("--run", required=True)
    d.add_argument("--epochs", type=int, default=200)
    o = sub.add_parser("ocr", help="upload data/ocr, train the reader, download the result")
    o.add_argument("--run", required=True)
    o.add_argument("--steps", type=int, default=10000)
    o.add_argument("--final", action="store_true")
    g = sub.add_parser("download", help="fetch the results of a finished stage")
    g.add_argument("--run", required=True)
    g.add_argument("--stage", choices=OUTPUTS, required=True)
    for p in (d, o, g):
        p.add_argument("--out", type=Path, help="default: runs/<run>/<stage>")
    args = parser.parse_args()

    stage = {"detector": "detector", "ocr": "ocr", "download": getattr(args, "stage", None)}[args.cmd]
    out = args.out or ROOT / "runs" / args.run / stage
    if args.cmd == "detector":
        upload(ROOT / "data/yolo", "/yolo")
        spawn_and_wait("train_detector", args.run, args.epochs)
    elif args.cmd == "ocr":
        upload(ROOT / "data/ocr", "/ocr")
        spawn_and_wait("train_ocr", args.run, args.steps, args.final)
    download(args.run, stage, out)


if __name__ == "__main__":
    sys.exit(main())
