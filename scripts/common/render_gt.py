"""Render <variant>/gt_samples/*.json to <variant>/gt_renders/*.png with the shared renderer."""
import sys
from pathlib import Path
from fpeval.format import load_sample
from fpeval.render import render

root = Path(sys.argv[1])
out = root / "gt_renders"
out.mkdir(exist_ok=True)
for p in sorted((root / "gt_samples").glob("*.json")):
    render(load_sample(p)).save(out / f"{p.stem}.png")
