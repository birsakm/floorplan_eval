#!/usr/bin/env bash
# Residential Floorplan Diffusion weights (Google Drive, per upstream README):
#   model_stage1.pth  https://drive.google.com/file/d/1ONAu_i2q0FUGJClArBC4mBMwFAchmo1H
#   model_stage2.pth  https://drive.google.com/file/d/1cVL4dLMM7j0n3nSKkapjdTWgHD9H45rA
# As of 2026-10-04 both files are access-restricted (HTTP 401 / "request access"; see upstream
# GitHub issue #1). If you obtain them, place them in checkpoints/residential_floorplan_diffusion/.
# Inputs: the 11 bundled condition sets in external/methods/residential_floorplan_diffusion/test/stage1_input.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CK="$ROOT/checkpoints/residential_floorplan_diffusion"
mkdir -p "$CK"
ok=1
for pair in "model_stage1.pth 1ONAu_i2q0FUGJClArBC4mBMwFAchmo1H" "model_stage2.pth 1cVL4dLMM7j0n3nSKkapjdTWgHD9H45rA"; do
  set -- $pair
  if [ ! -s "$CK/$1" ]; then
    conda run -n fpe gdown "$2" -O "$CK/$1" || { rm -f "$CK/$1"; ok=0; echo "FAILED: $1 (Drive file $2 is not publicly accessible; request access from the authors)"; }
  fi
done
[ $ok = 1 ] && ls -la "$CK" || exit 1
