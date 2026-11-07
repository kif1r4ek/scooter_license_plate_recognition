"""Train the plate reader and export it to ONNX.

The format is fixed (2 letters + 4 digits), so instead of a CRNN the network has
one output per position: positions 0-1 can only be letters and 2-5 only digits,
so 0/O and 8/B cannot be confused. A separate head says whether the crop is
upright, because the detector's box does not know the reading direction.

Half of every batch are real crops (data/ocr, made by export_ocr.py), half are
synthetic plates with random numbers (ocr_lib.Synth): the photos show only 78
numbers and 12 letters never appear in them.

    python training/ocr_model.py --device cpu  --out runs/manual/ocr --final
    python training/ocr_model.py --device cuda --out runs/manual/ocr

Without --final the model is trained on train and the step that reads val best
is kept. With --final it also learns from val and keeps the last step: use the
number of steps a run without --final has shown to be enough (10000).

Writes to --out: best.pt, ocr.onnx, results.json.
"""
import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

import ocr_lib
from ocr_lib import DIGITS, H, LETTERS, W

ROOT = Path(__file__).resolve().parent.parent


class Plates(IterableDataset):
    """Endless stream: real crops with detector-like box errors and synthetic plates."""

    def __init__(self, numbers, backgrounds, fonts):
        self.numbers = numbers  # list of lists: every scooter number equally likely
        self.backgrounds = backgrounds
        self.fonts = fonts

    def __iter__(self):
        seed = torch.initial_seed() % 2**32
        rng = np.random.default_rng(seed)
        random.seed(seed)
        cv2.setNumThreads(0)
        synth = ocr_lib.Synth(self.fonts, self.backgrounds)
        while True:
            if rng.random() < 0.5:
                gray, pts, text, rear = synth.sample(rng)
                det = None
            else:  # PH1023 alone is 1/8 of the photos: pick the number first
                e = random.choice(random.choice(self.numbers))
                gray, pts, text, rear, det = e["img"], e["pts"], e["text"], e["rear"], e["det"]
            x, letters, digits, upright = ocr_lib.make_sample(gray, pts, text, rear, rng, det)
            yield x[None].astype(np.float32), np.array(letters), np.array(digits), float(upright)


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b),
        )
        self.skip = nn.Conv2d(a, b, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class PlateNet(nn.Module):
    """Each of the 6 positions attends to its own place on the feature map."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            Block(32, 64), nn.MaxPool2d(2),
            Block(64, 128), nn.MaxPool2d(2),
            Block(128, 192),
            # wide kernels: finding "the 3rd symbol" takes seeing the whole line
            nn.Conv2d(192, 192, (3, 9), padding=(1, 4)), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, (3, 9), padding=(1, 12), dilation=(1, 3)), nn.ReLU(inplace=True),
        )
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H // 8), torch.linspace(-1, 1, W // 8), indexing="ij")
        self.register_buffer("grid", torch.stack([xs, ys])[None])
        self.attention = nn.Sequential(
            nn.Conv2d(194, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 6, 1),
        )
        self.drop = nn.Dropout(0.2)
        self.letters = nn.Linear(192, len(LETTERS))
        self.digits = nn.Linear(192, len(DIGITS))
        self.upright = nn.Linear(192, 1)

    def forward(self, x):
        # per-image contrast: the same plate can be in sun or in shadow
        mean = x.mean((2, 3), keepdim=True)
        std = ((x - mean) ** 2).mean((2, 3), keepdim=True).sqrt()
        f = self.backbone((x - mean) / (std + 1.0))
        grid = self.grid.expand(x.shape[0], -1, -1, -1)
        a = self.attention(torch.cat([f, grid], 1)).flatten(2).softmax(2)
        v = self.drop(a @ f.flatten(2).transpose(1, 2))
        return self.letters(v[:, :2]), self.digits(v[:, 2:]), self.upright(f.mean((2, 3))).squeeze(1)


def load_plates(data: Path):
    index = json.loads((data / "index.json").read_text())
    for e in index:
        e["img"] = cv2.imread(str(data / "images" / e["file"]), cv2.IMREAD_GRAYSCALE)
        e["rear"] = e["cls"] == "rear"
    return {s: [e for e in index if e["split"] == s] for s in ("train", "val", "test")}


def train(data: Path, out: Path, steps: int, final: bool, device: str, batch: int = 192,
          workers: int = 8, on_save=None, fonts: Path = ROOT / "training" / "fonts"):
    """data: folder made by export_ocr.py. on_save: called after files are written (Modal commits its volume)."""
    out.mkdir(parents=True, exist_ok=True)
    by_split = load_plates(data)
    learn = by_split["train"] + (by_split["val"] if final else [])
    numbers = {}
    for e in learn:
        numbers.setdefault(e["text"], []).append(e)

    cuda = device.startswith("cuda")
    model = PlateNet().to(device)
    print("parameters", sum(p.numel() for p in model.parameters()), "device", device, flush=True)

    @torch.no_grad()
    def read(images):
        x = torch.from_numpy(np.stack(images))[:, None].float().to(device)
        l, d, _ = model(x)
        l, d = l.argmax(2).cpu().numpy(), d.argmax(2).cpu().numpy()
        return ["".join(LETTERS[i] for i in a) + "".join(DIGITS[i] for i in b) for a, b in zip(l, d)]

    def evaluate(plates, seed=0, copies=8):
        """Whole-number accuracy on the detector's real box plus jittered ones."""
        rng = np.random.default_rng(seed)
        was_training = model.training
        model.eval()
        hit = chars = total = 0
        wrong = []
        for e in plates:
            box = ocr_lib.oriented_box(e["pts"])
            first = box if e["det"] is None else ocr_lib.align(e["det"], e["pts"])
            boxes = [first] + [ocr_lib.jitter_box(box, rng) for _ in range(copies - 1)]
            texts = read([ocr_lib.to_input(e["img"], b, e["rear"]) for b in boxes])
            good = sum(t == e["text"] for t in texts)
            hit += good
            total += len(texts)
            chars += sum(a == b for t in texts for a, b in zip(t, e["text"]))
            if good < len(texts):
                wrong.append(f'{e["text"]}{"(r)" if e["rear"] else ""} {good}/{len(texts)} '
                             f'{next(t for t in texts if t != e["text"])}')
        model.train(was_training)
        return {"plate": hit / max(total, 1), "char": chars / max(6 * total, 1), "wrong": wrong}

    dataset = Plates(list(numbers.values()), [e["img"] for e in learn], fonts)
    loader = DataLoader(dataset, batch_size=batch, num_workers=workers, persistent_workers=workers > 0,
                        prefetch_factor=4 if workers > 0 else None, pin_memory=cuda)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, 2e-3, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=cuda)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")
    bce = nn.BCEWithLogitsLoss()

    best, log = -1.0, []
    eval_every = max(10, min(500, steps // 2))
    model.train()
    for step, (x, letters, digits, upright) in enumerate(loader, 1):
        x, letters, digits, upright = (t.to(device, non_blocking=True) for t in (x, letters, digits, upright))
        with torch.autocast("cuda", enabled=cuda):
            l, d, u = model(x)
            text = (ce(l.flatten(0, 1), letters.flatten()).view(-1, 2).sum(1)
                    + ce(d.flatten(0, 1), digits.flatten()).view(-1, 4).sum(1))
            # a sideways crop has no meaningful text: only the upright head learns from it
            loss = (text * upright.float()).sum() / upright.sum().clamp(min=1) + bce(u.float(), upright.float())
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        sched.step()
        if step % eval_every == 0 or step == steps:
            val = evaluate(by_split["val"])
            row = {"step": step, "loss": round(float(loss), 3), "val_plate": round(val["plate"], 4),
                   "val_char": round(val["char"], 4)}
            log.append(row)
            print(row, flush=True)
            # with --final val is part of training, so its score says nothing: keep the last step
            if final or val["plate"] > best:
                best = val["plate"]
                torch.save(model.state_dict(), out / "best.pt")
                if on_save:
                    on_save()
        if step == steps:
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    model.eval()
    results = {"final": final, "steps": steps, "val": evaluate(by_split["val"]),
               "test": evaluate(by_split["test"]), "log": log}
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("VAL", results["val"], flush=True)
    print("TEST", results["test"], flush=True)

    path = out / "ocr.onnx"
    model.cpu().eval()
    torch.onnx.export(
        model, torch.zeros(2, 1, H, W), str(path), opset_version=17, input_names=["image"],
        output_names=["letters", "digits", "upright"], dynamic_axes={"image": {0: "batch"}},
    )
    check_onnx(model, path)
    if on_save:
        on_save()


def check_onnx(model, path):
    """The exported graph must answer like PyTorch (catches a model left in train mode)."""
    import onnxruntime as ort

    x = np.random.default_rng(0).integers(0, 255, (2, 1, H, W)).astype(np.float32)
    with torch.no_grad():
        want = [t.numpy() for t in model(torch.from_numpy(x))]
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": x})
    diff = max(float(np.abs(a - b).max()) for a, b in zip(want, got))
    print("onnx max diff", diff, flush=True)
    if diff > 1e-3:
        raise RuntimeError(f"ONNX export differs from PyTorch by {diff}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/ocr")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--final", action="store_true", help="also learn from val, keep the last step")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - args.workers))
    train(args.data, args.out, args.steps, args.final, args.device, args.batch, args.workers)


if __name__ == "__main__":
    main()
