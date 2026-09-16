"""Highlight one person in a crowd photo in the deck's green.

Segments people with YOLO, picks one (or --pick N), tints that person green
with a soft glow, and dims/desaturates everyone else. Writes a new file; never
touches the source image.

Usage: uv run --with ultralytics --with pillow scripts/highlight_person.py SRC DST [--pick N]
"""

import argparse

import numpy as np
from PIL import Image, ImageFilter
from ultralytics import YOLO

GREEN = np.array([0x7E, 0xC8, 0x7E], dtype=np.float32)  # $presentation-heading-color


def detect_people(img: Image.Image):
    model = YOLO("yolo11m-seg.pt")
    res = model.predict(img, imgsz=1280, classes=[0], conf=0.35, retina_masks=True, verbose=False)[0]
    if res.masks is None:
        return []
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    masks = res.masks.data.cpu().numpy()
    return [
        {"i": i, "conf": float(c), "box": b, "mask": m > 0.5}
        for i, (b, c, m) in enumerate(zip(boxes, confs, masks))
    ]


def choose(people, w, h):
    """Prefer confident, fully-in-frame, larger people closest to center."""
    margin = 0.02
    cands = [
        p for p in people
        if p["conf"] >= 0.6
        and p["box"][0] > w * margin and p["box"][2] < w * (1 - margin)
        and p["box"][1] > h * margin and p["box"][3] < h * (1 - margin)
    ] or people
    areas = sorted(p["mask"].sum() for p in cands)
    floor = areas[len(areas) // 2]  # upper half by size
    big = [p for p in cands if p["mask"].sum() >= floor]

    def center_dist(p):
        cx, cy = (p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2
        return ((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2

    return min(big, key=center_dist)


def render(img: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(img, dtype=np.float32)
    gray = rgb.mean(axis=2, keepdims=True)

    # Everyone else: half-desaturated and darkened so the highlight pops
    rest = (0.5 * rgb + 0.5 * gray) * 0.55

    # The person: luminance carried into the brand green, a bit of original detail
    tint = np.clip(gray / 255.0 * GREEN * 1.6, 0, 255)
    person = 0.25 * rgb + 0.75 * tint

    m_img = Image.fromarray((mask * 255).astype(np.uint8))
    soft = np.asarray(m_img.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)[..., None] / 255.0
    out = rest * (1 - soft) + person * soft

    # Glow: blurred halo around the silhouette, outside the person only
    radius = max(img.size) / 250
    halo = np.asarray(m_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)[..., None] / 255.0
    glow = np.clip(halo * 1.8, 0, 1) * (1 - soft) * 0.8
    out = out * (1 - glow) + GREEN * glow

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--pick", type=int, default=None)
    args = ap.parse_args()

    img = Image.open(args.src).convert("RGB")
    w, h = img.size
    people = detect_people(img)
    print(f"image {w}x{h}, {len(people)} people detected")
    for p in people:
        x0, y0, x1, y1 = p["box"].astype(int)
        print(f"  #{p['i']:>3} conf={p['conf']:.2f} box=({x0},{y0})-({x1},{y1}) area={int(p['mask'].sum())}")
    if not people:
        raise SystemExit("no people found")

    if args.pick is not None:
        chosen = next(p for p in people if p["i"] == args.pick)
    else:
        chosen = choose(people, w, h)
    print(f"highlighting #{chosen['i']} box={tuple(chosen['box'].astype(int))} ({chosen['mask'].mean():.2%} of image)")
    render(img, chosen["mask"]).save(args.dst, "JPEG", quality=85, optimize=True, progressive=True)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
