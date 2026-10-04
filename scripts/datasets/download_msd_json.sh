#!/usr/bin/env bash
# Modified Swiss Dwellings JSON (Jabi; Zenodo 17294451, CC BY 4.0, ~0.3 GB) -> data/msd_json/raw/
# Derived from MSD v6 (Kaggle). The original MSD (Kaggle) needs Kaggle credentials.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/msd_json/raw"; mkdir -p "$OUT"; cd "$OUT"
[ -f msd_01_json.zip ] || { wget -c -O msd.part "https://zenodo.org/records/17294451/files/msd_01_json.zip?download=1"; mv msd.part msd_01_json.zip; }
unzip -n -q msd_01_json.zip
