"""Export data/labels/labels.json to a YOLO OBB dataset, split by groups of scooters.

The split is stored in data/labels/split.json and reused on every run, so the
metrics of models trained on different machines are comparable. Delete that
file (or pass --resplit) to split again, e.g. after adding photos.

    python training/export_yolo.py
"""
import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "yolo"
SPLIT = ROOT / "data" / "labels" / "split.json"
CLASSES = {"front": 0, "rear": 1}
MAX_SIDE = 1600
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 7


def real_number(plate, known):
    if plate["complete"]:
        return plate["text"]
    hits = [n for n in known if re.search(plate["text"], n)]
    return hits[0] if len(hits) == 1 else None


def build_groups(images):
    """Photos that share a scooter must stay in one split, otherwise metrics leak."""
    known = {p["text"] for v in images.values() for p in v["plates"] if p["complete"]}
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for stem, v in images.items():
        find(stem)
        for p in v["plates"]:
            number = real_number(p, known)
            if number:
                parent[find(number)] = find(stem)

    groups = defaultdict(list)
    for stem in images:
        groups[find(stem)].append(stem)
    return list(groups.values())


def rear_count(group, images):
    return sum(p["cls"] == "rear" for s in group for p in images[s]["plates"])


def assign_once(groups, images, seed):
    total = sum(len(g) for g in groups)
    order = sorted(groups, key=len, reverse=True)
    biggest, rest = order[0], order[1:]
    random.Random(seed).shuffle(rest)
    split = {name: [] for name in TARGET}
    split["train"].extend(biggest)
    for g in rest:
        name = max(TARGET, key=lambda n: TARGET[n] - len(split[n]) / total)
        split[name].extend(g)
    return split


def imbalance(split, images, groups):
    total = sum(len(g) for g in groups)
    rears = sum(rear_count(g, images) for g in groups)
    score = 0.0
    for name in TARGET:
        score += abs(len(split[name]) / total - TARGET[name])
        score += 0.5 * abs(rear_count(split[name], images) / rears - TARGET[name])
    return score


def assign(groups, images):
    """Group sizes are uneven, so pick the most balanced of many random orders."""
    candidates = [assign_once(groups, images, SEED + i) for i in range(500)]
    return min(candidates, key=lambda s: imbalance(s, images, groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resplit", action="store_true", help="ignore data/labels/split.json and split again")
    args = parser.parse_args()

    images = json.loads((ROOT / "data" / "labels" / "labels.json").read_text())["images"]
    groups = build_groups(images)
    if SPLIT.exists() and not args.resplit:
        split = json.loads(SPLIT.read_text())
        missing = set(images) - {s for v in split.values() for s in v}
        if missing:
            raise SystemExit(f"{len(missing)} labelled photos are not in {SPLIT}, run with --resplit: {sorted(missing)[:5]}")
    else:
        split = assign(groups, images)
        SPLIT.write_text(json.dumps({k: sorted(v) for k, v in split.items()}, ensure_ascii=False, indent=1))

    shutil.rmtree(OUT, ignore_errors=True)
    for name, stems in split.items():
        (OUT / "images" / name).mkdir(parents=True)
        (OUT / "labels" / name).mkdir(parents=True)
        for i, stem in enumerate(sorted(stems)):
            v = images[stem]
            safe = f"{i:03d}_{re.sub(r'[^A-Za-z0-9_]', '_', stem)}"
            im = ImageOps.exif_transpose(Image.open(RAW / v["file"])).convert("RGB")
            w, h = im.size
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            im.save(OUT / "images" / name / f"{safe}.jpg", quality=92)
            lines = []
            for p in v["plates"]:
                xy = [c for x, y in p["pts"] for c in (min(max(x / w, 0), 1), min(max(y / h, 0), 1))]
                lines.append(" ".join([str(CLASSES[p["cls"]])] + [f"{c:.6f}" for c in xy]))
            (OUT / "labels" / name / f"{safe}.txt").write_text("\n".join(lines) + "\n")

    (OUT / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: front\n  1: rear\n"
    )

    print("groups", len(groups), "largest", sorted((len(g) for g in groups), reverse=True)[:6])
    for name, stems in split.items():
        plates = Counter(p["cls"] for s in stems for p in images[s]["plates"])
        print(name, "images", len(stems), dict(plates))


if __name__ == "__main__":
    main()
