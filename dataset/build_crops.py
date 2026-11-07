"""v7 dataset, take two: figure-hugging detail squares.

The first attempt cropped full-width squares and white only fell 87%->82%, because the figure is
tall and narrow and never fills the 896px width. Instead: find the figure's bounding box, then cut
squares the size of the figure's *width* - each square is filled by the character, white becomes
the minority, and head/chest/legs squares differ from each other by construction.

Three squares per pose image (head, torso, legs) with jitter; closeups get one bbox square each.
Captions are rebuilt per framing without the contradictions of attempt one.
"""

import json
import random
from pathlib import Path

from PIL import Image

SRC = Path("/home/bloodycrown/train/omni/dataset")
DST = Path("/home/bloodycrown/train/omni/dataset_crop")

import shutil

shutil.rmtree(DST, ignore_errors=True)
DST.mkdir(parents=True)

random.seed(7)

PARTS = [
    ("hd", 0.02, "head and helmet close-up"),
    ("ch", 0.28, "chest and shoulder armor view"),
    ("lg", 0.58, "legs and boots view"),
]


def bbox_of(im):
    g = im.convert("L")
    w, h = im.size
    px = g.load()
    cols = [x for x in range(w) if any(px[x, y] < 235 for y in range(0, h, 3))]
    rows = [y for y in range(h) if any(px[x, y] < 235 for x in range(0, w, 3))]
    if not cols or not rows:
        return None
    return min(cols), max(cols), min(rows), max(rows)


meta = []
for p in sorted(SRC.iterdir()):
    if p.suffix.lower() != ".png":
        continue
    im = Image.open(p).convert("RGB")
    w, h = im.size
    box = bbox_of(im)
    if box is None:
        continue
    x0, x1, y0, y1 = box
    side = max(180, min(x1 - x0, y1 - y0, 620))  # square side ~= figure width, capped
    idx = int(p.stem)
    context = (SRC / f"{p.stem}.txt").read_text().strip()
    context = context.replace("q21char, ", "").replace(", full body", "").replace(", plain white background", "")

    if idx <= 33:  # pose set: three figure-hugging squares
        for tag, frac, phrase in PARTS:
            cy = min(max(int(y0 + frac * (y1 - y0)), side // 2), h - side // 2)
            cx = min(max((x0 + x1) // 2 + random.randint(-30, 30), side // 2), w - side // 2)
            im.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)).save(
                DST / f"{p.stem}{tag}.png")
            meta.append((f"{p.stem}{tag}.png", f"q21char, {phrase}, {context}"))
    else:  # detail closeups: one square on the subject
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        side = max(240, min(w, h, max(x1 - x0, y1 - y0) + 60))
        cx = min(max(cx, side // 2), w - side // 2)
        cy = min(max(cy, side // 2), h - side // 2)
        im.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)).save(
            DST / f"{p.stem}c.png")
        meta.append((f"{p.stem}c.png", f"q21char, close-up, {context}"))

for name, caption in meta:
    (DST / name.replace(".png", ".txt")).write_text(caption + "\n")

(DST / "_build.json").write_text(json.dumps({"total": len(meta)}))
print(f"total crops: {len(meta)}")

checks = ["0001hd.png", "0001ch.png", "0001lg.png"]
for name in checks:
    path = DST / name
    if path.exists():
        with Image.open(path) as im:
            g = im.convert("L")
            white = sum(1 for v in g.getdata() if v > 235) / (im.width * im.height)
        print(f"{name}: {im.width}x{im.height}, white {white:.0%}")
