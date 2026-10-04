#!/usr/bin/env bash
# ProcTHOR-10K houses (allenai/procthor-10k, Apache-2.0; git-LFS files, ~0.5 GB).
# Procedurally generated (synthetic) houses, NOT real floorplans; use as a synthetic reference only.
# Equivalent to `prior.load_dataset("procthor-10k")`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/procthor10k/raw"; mkdir -p "$OUT"; cd "$OUT"
REV="${REV:-main}"
for s in train val test; do
  [ -f "$s.jsonl.gz" ] || { wget -q -O "$s.part" "https://media.githubusercontent.com/media/allenai/procthor-10k/$REV/$s.jsonl.gz"; mv "$s.part" "$s.jsonl.gz"; }
done
ls -la
