"""Render common-format floorplans to PNG with a fixed per-type palette."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .format import ROOM_TYPES, load_sample

PALETTE = {
    "living_room": (238, 77, 77), "kitchen": (192, 192, 224), "bedroom": (255, 165, 0),
    "bathroom": (65, 105, 225), "balcony": (35, 139, 34), "entrance": (220, 220, 50),
    "dining_room": (200, 120, 200), "study": (139, 90, 43), "storage": (150, 150, 150),
    "corridor": (240, 220, 180), "garage": (90, 90, 120), "laundry": (100, 200, 200),
    "closet": (180, 140, 100), "outdoor": (150, 220, 150), "other": (210, 210, 210),
}
assert set(PALETTE) == set(ROOM_TYPES)


def render(sample, size=512, margin=16, rooms_only=False):
    """Render a sample. rooms_only=True skips boundary/doors/windows."""
    pts = [p for r in sample["rooms"] for p in r["polygon"]]
    if sample.get("boundary") and not rooms_only:
        pts += sample["boundary"]
    img = Image.new("RGB", (size, size), (255, 255, 255))
    if not pts:
        return img
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    extent = max(max(xs) - x0, max(ys) - y0, 1e-9)
    s = (size - 2 * margin) / extent

    def tf(poly):
        return [((x - x0) * s + margin, (y - y0) * s + margin) for x, y in poly]

    draw = ImageDraw.Draw(img)
    if sample.get("boundary") and not rooms_only:
        draw.polygon(tf(sample["boundary"]), fill=(245, 245, 245), outline=(0, 0, 0))
    for room in sample["rooms"]:
        if len(room["polygon"]) >= 3:
            draw.polygon(tf(room["polygon"]), fill=PALETTE[room["type"]], outline=(0, 0, 0))
    if rooms_only:
        return img
    for door in sample.get("doors") or []:
        draw.polygon(tf(door["polygon"]), fill=(255, 255, 255), outline=(60, 60, 60))
    for win in sample.get("windows") or []:
        draw.polygon(tf(win["polygon"]), fill=(150, 220, 255), outline=(0, 0, 255))
    return img


def main():
    ap = argparse.ArgumentParser(description="Render samples/*.json to renders/*.png")
    ap.add_argument("sample_dir", help="directory containing samples/ (e.g. outputs/<m>/<v>)")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    root = Path(args.sample_dir)
    out = root / "renders"
    out.mkdir(exist_ok=True)
    for p in sorted((root / "samples").glob("*.json")):
        render(load_sample(p), args.size).save(out / f"{p.stem}.png")


if __name__ == "__main__":
    main()
