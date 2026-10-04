#!/usr/bin/env bash
# MagicPlan floors released with PuzzleFusion (github.com/sepidsh/PuzzleFussion README),
# Google Drive folder, ~1.9 GB. License: same as PuzzleFusion code (non-commercial research, GPLv3 terms).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/magicplan/raw"; mkdir -p "$OUT"; cd "$OUT"
gdown --folder "https://drive.google.com/drive/folders/15IHJPRVwt32uUdKgHM8EWuI2Aa-PEFgw" -O .
ls -la
