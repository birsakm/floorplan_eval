#!/usr/bin/env bash
# ResPlan (CC BY 4.0, ~100 MB zip). The data zip ships inside the GitHub repo
# (submodule external/datasets/resplan); copy + unzip it into data/resplan/raw/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/external/datasets/resplan"
[ -f "$SRC/ResPlan.zip" ] || git -C "$ROOT" submodule update --init --depth 1 external/datasets/resplan
OUT="$ROOT/data/resplan/raw"; mkdir -p "$OUT"
cp -n "$SRC/ResPlan.zip" "$SRC/split.json" "$OUT/"
cd "$OUT" && unzip -n -q ResPlan.zip && ls -la
