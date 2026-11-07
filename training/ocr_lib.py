"""Plate rectification, training augmentation and synthetic plates for the OCR.

No torch here: the same code is used to preview data and to evaluate the ONNX
model on CPU.
"""
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
H, W = 64, 256


def align(box, pts):
    """Start a box at the corner nearest to the text's top-left."""
    box, pts = np.array(box, np.float32), np.array(pts, np.float32)
    best = min(range(4), key=lambda k: np.linalg.norm(np.roll(box, -k, 0) - pts, axis=1).sum())
    return np.roll(box, -best, 0)


def oriented_box(pts):
    """Min-area rectangle of the corners, first corner at the text's top-left.

    The detector predicts exactly this rectangle, so training sees the same
    slant a real parallelogram plate keeps after rectification.
    """
    return align(cv2.boxPoints(cv2.minAreaRect(np.array(pts, np.float32))), pts)


def warp(gray, quad, w, h):
    target = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(np.float32(quad), target)
    return cv2.warpPerspective(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def to_input(gray, box, rear):
    """Rectify a box to H x W. Two-line rear plates are laid out as one line."""
    if not rear:
        return warp(gray, box, W, H)
    tl, tr, br, bl = box
    ml, mr = (tl + bl) / 2, (tr + br) / 2
    top = warp(gray, [tl, tr, mr, ml], W // 2, H)
    bottom = warp(gray, [ml, mr, br, bl], W // 2, H)
    return np.hstack([top, bottom])


def shrink(gray, box, max_side=480):
    """Same scale as training crops. warpPerspective has no area filter, so a
    plate shot from up close would alias badly if warped straight from the photo.
    """
    side = max(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[3] - box[0]))
    k = max_side / side
    if k >= 1:
        return gray, box
    return cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), box * k


def jitter_box(box, rng, loose=True):
    """Shift, resize and turn a box. Loose: as far as a detector box can be from
    the labelled plate (it often takes in the whole stem, IoU 0.4-0.6).
    """
    centre = box.mean(0)
    ux, uy = box[1] - box[0], box[3] - box[0]
    if loose:
        sw, sh = rng.uniform(0.92, 1.3), rng.uniform(0.88, 2.1)
        shift = rng.uniform(-0.06, 0.06) * ux + rng.uniform(-0.2, 0.2) * uy
        a = np.deg2rad(rng.uniform(-6, 6))
    else:
        sw, sh = rng.uniform(0.97, 1.05), rng.uniform(0.95, 1.12)
        shift = rng.uniform(-0.02, 0.02) * ux + rng.uniform(-0.05, 0.05) * uy
        a = np.deg2rad(rng.uniform(-1.5, 1.5))
    turn = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    corners = np.array([-ux * sw - uy * sh, ux * sw - uy * sh, ux * sw + uy * sh, -ux * sw + uy * sh], np.float32) / 2
    return (corners @ turn.T + centre + shift).astype(np.float32)


def photometric(x, rng):
    x = x.astype(np.float32)
    x = 255 * (x / 255) ** rng.uniform(0.6, 1.6)
    x = (x - 128) * rng.uniform(0.6, 1.25) + 128 + rng.uniform(-30, 30)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.6))
    if rng.random() < 0.2:  # motion blur
        n = rng.choice([3, 5, 7])
        k = np.zeros((n, n), np.float32)
        k[n // 2] = 1 / n
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), rng.uniform(0, 180), 1), (n, n))
        x = cv2.filter2D(x, -1, k / max(k.sum(), 1e-3))
    if rng.random() < 0.4:
        x = x + rng.normal(0, rng.uniform(2, 10), x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        _, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 85))])
        x = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return x


class Synth:
    """Draws a plate in one of the three styles seen on real scooters."""

    def __init__(self, font_dir, backgrounds):
        self.fonts = sorted(Path(font_dir).glob("*.ttf"))
        self.backgrounds = backgrounds

    def font(self, path, size):
        font = ImageFont.truetype(str(path), size)
        try:
            font.set_variation_by_axes([font.get_variation_axes()[0]["default"], 700][:len(font.get_variation_axes())])
        except OSError:
            pass  # not a variable font
        return font

    def line(self, text, font, gap, wobble, stencil):
        """Black-on-white mask of one line of text, with per-glyph wobble."""
        glyphs = []
        for ch in text:
            l, t, r, b = font.getbbox(ch)
            g = Image.new("L", (r - l + 8, b - t + 8), 0)
            ImageDraw.Draw(g).text((4 - l, 4 - t), ch, fill=255, font=font)
            if stencil and ch in "03689BDOPQRAE":  # bridges cut through top and bottom
                d = ImageDraw.Draw(g)
                x, y, third = g.width // 2, g.height // 2, g.height // 3
                d.rectangle([x - 2, 0, x + 2, third], fill=0)
                d.rectangle([x - 2, g.height - third, x + 2, g.height], fill=0)
            glyphs.append(g.rotate(random.uniform(-wobble, wobble), Image.BILINEAR, expand=True))
        height = max(g.height for g in glyphs) + 4
        width = sum(g.width for g in glyphs) + gap * (len(glyphs) - 1)
        canvas = Image.new("L", (max(width, 1), height), 0)
        x = 0
        for g in glyphs:
            canvas.paste(255, (x, (height - g.height) // 2 + random.randint(-1, 1)), g)
            x += g.width + gap
        return np.asarray(canvas)

    def sample(self, rng):
        """Returns (gray image, corners, text, rear) for a uniform random number."""
        text = "".join(random.choices(LETTERS, k=2) + random.choices(DIGITS, k=4))
        style = random.choices(["white", "stem", "dark", "rear"], [0.35, 0.35, 0.15, 0.15])[0]
        rear = style == "rear"
        font = self.font(random.choice(self.fonts), 96)

        if rear:
            top = self.line(text[:3], font, 8, 1, False)
            bottom = self.line(text[3:], font, 8, 1, False)
            width = max(top.shape[1], bottom.shape[1])
            top = cv2.resize(top, (width, top.shape[0]))
            bottom = cv2.resize(bottom, (width, bottom.shape[0]))
            mask = np.vstack([top, np.zeros((14, width), np.uint8), bottom])
        else:
            spread = style != "white" and random.random() < 0.8
            gap = random.randint(25, 60) if spread else random.randint(2, 14)
            mask = self.line(text, font, gap, 2.5 if spread else 0.5, spread and random.random() < 0.7)

        # stretch to the proportions of real plates
        ratio = random.uniform(2.0, 5.0) if not rear else random.uniform(1.1, 2.0)
        mh = 120
        mask = cv2.resize(mask, (int(mh * ratio), mh), interpolation=cv2.INTER_AREA)
        pad_x, pad_y = int(mh * random.uniform(0.15, 0.4)), int(mh * random.uniform(0.1, 0.3))
        mask = cv2.copyMakeBorder(mask, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
        ph, pw = mask.shape

        if style in ("white", "rear"):
            plate_v, ink = rng.uniform(170, 250), rng.uniform(0, 55)
        elif style == "stem":
            plate_v, ink = rng.uniform(80, 190), rng.uniform(5, 50)
        else:
            plate_v, ink = rng.uniform(28, 70), rng.uniform(3, 18)

        alpha = (mask / 255).astype(np.float32)
        if random.random() < 0.6:  # worn paint: blotches of missing ink
            noise = cv2.resize(rng.random((ph // 12 + 1, pw // 12 + 1)).astype(np.float32), (pw, ph))
            alpha *= np.clip((noise - rng.uniform(0.1, 0.3)) * 6, 0.1, 1)
        plate = plate_v + (ink - plate_v) * alpha

        if random.random() < 0.5:  # scratches
            scratch = np.zeros((ph, pw), np.float32)
            for _ in range(random.randint(1, 8)):
                p, q = (random.randrange(pw), random.randrange(ph)), (random.randrange(pw), random.randrange(ph))
                cv2.line(scratch, p, q, float(rng.uniform(-60, 60)), random.randint(1, 3), cv2.LINE_AA)
            plate = plate + scratch
        # uneven light across the plate
        grad = np.linspace(rng.uniform(-25, 25), rng.uniform(-25, 25), pw, dtype=np.float32)
        plate = plate + grad

        # place on a real scooter/street texture and tilt
        margin = int(0.8 * max(pw, ph))
        size = (pw + 2 * margin, ph + 2 * margin)
        bg = cv2.resize(self.backgrounds[random.randrange(len(self.backgrounds))], size)
        src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
        tilt = rng.uniform(-0.28, 0.28) * ph * (0.4 if rear else 1)  # shear of the top edge
        persp = rng.uniform(-0.12, 0.12) * ph
        rot = np.deg2rad(rng.uniform(-6, 6))
        dst = src + np.float32([[tilt, persp], [tilt, -persp], [-tilt, persp], [-tilt, -persp]])
        c = dst.mean(0)
        turn = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]], np.float32)
        dst = (dst - c) @ turn.T + c + margin
        matrix = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        layer = cv2.warpPerspective(np.clip(plate, 0, 255).astype(np.uint8), matrix, size, flags=cv2.INTER_LINEAR)
        where = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), matrix, size) > 127
        out = bg.copy()
        out[where] = layer[where]
        return out, dst.astype(np.float32), text, rear


def rotate_label(box, k):
    """Start the box at another corner: an upside-down or sideways view of it."""
    return np.roll(box, -k, 0)


def make_sample(gray, pts, text, rear, rng, det=None):
    """One training example: (image, label positions, upright flag).

    det is the box the detector really gave for this plate, if any.
    """
    if det is not None and rng.random() < 0.5:
        box = jitter_box(align(det, pts), rng, loose=False)
    else:
        box = jitter_box(oriented_box(pts), rng)
    k = 0
    if rng.random() < 0.5:
        k = int(rng.choice([1, 2, 3])) if rear else 2
    x = to_input(gray, rotate_label(box, k), rear)
    x = photometric(x, rng)
    letters = [LETTERS.index(c) for c in text[:2]]
    digits = [DIGITS.index(c) for c in text[2:]]
    return x, letters, digits, int(k == 0)
