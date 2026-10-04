#!/usr/bin/env bash
# Modified Swiss Dwellings v1 (ECCV 2024 benchmark) from 4TU.ResearchData
# (DOI 10.4121/e1d89cb5-6872-48fc-be63-aadd687ee6f9.v2, CC BY 4.0, ~5.5 GB) -> data/msd/raw/
# Same data is on Kaggle (caspervanengelenburg/modified-swiss-dwellings), which needs credentials.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/msd/raw"; mkdir -p "$OUT"; cd "$OUT"
B=https://data.4tu.nl/file/e1d89cb5-6872-48fc-be63-aadd687ee6f9
get() { [ -f "$2" ] || { wget -c -O "$2.part" "$B/$1"; mv "$2.part" "$2"; }; }
get 85363e4c-af3d-47e2-9216-09efe7d19a60 README.md
get 7cccae96-99ad-4a11-9526-64c01a4b3f63 modified-swiss-dwellings-v1-test.zip
get 279ef4b4-d3bd-41f4-b0c9-5e9af8cce6f6 modified-swiss-dwellings-v1-train.zip
echo "dfd84215dcb44f16076b29c997e22e24  modified-swiss-dwellings-v1-test.zip
5757e14cb06e4ba96a0420a5b09c4109  modified-swiss-dwellings-v1-train.zip" | md5sum -c
for z in modified-swiss-dwellings-v1-*.zip; do unzip -n -q "$z"; done
