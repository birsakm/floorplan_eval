#!/usr/bin/env bash
# Download the HouseDiffusion "temporary model" (README link) into checkpoints/house_diffusion/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/checkpoints/house_diffusion"
mkdir -p "$OUT"
if [ ! -s "$OUT/model.pt" ] && ! ls "$OUT"/*.pt >/dev/null 2>&1; then
  conda run -n fpe gdown 16zKmtxwY5lF6JE-CJGkRf3-OFoD1TrdR -O "$OUT/"
fi
ls -la "$OUT"
