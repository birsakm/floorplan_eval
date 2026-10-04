#!/usr/bin/env bash
# Swiss Dwellings v3.0.0 (Zenodo 7788422, CC BY 4.0, ~0.9 GB) -> data/swiss_dwellings/raw/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/swiss_dwellings/raw"; mkdir -p "$OUT"; cd "$OUT"
[ -f swiss-dwellings-v3.0.0.zip ] || { wget -c -O sd.part "https://zenodo.org/records/7788422/files/swiss-dwellings-v3.0.0.zip?download=1"; mv sd.part swiss-dwellings-v3.0.0.zip; }
md5sum swiss-dwellings-v3.0.0.zip
unzip -n -q swiss-dwellings-v3.0.0.zip
