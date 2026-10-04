#!/usr/bin/env bash
# Download DS2D LoRA weights (Google Drive) and the ProcTHOR converted dataset (HF ludolara/DStruct2Design).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CK="$ROOT/checkpoints/ds2d"; DA="$ROOT/data/method_inputs/ds2d"
mkdir -p "$CK" "$DA"
dl() { # id name
  if [ ! -s "$CK/$2" ]; then conda run -n fpe gdown "$1" -O "$CK/$2"; fi
}
dl 1cAYlEupNUGJefNdwkNaaq7fD3X3_P46D rplan_weights_variants.zip
dl 16cYPK6g_Ho4VbvjvBZIGHMzNTBWzcAZT procthor_weights_BD_variants.zip
dl 13k-pBmhGhYthm4WbHzrRH7WjaSKNkTpq procthor_weights_nonBD_variants.zip
cd "$CK"
# The Drive files are named .zip but are gzipped tarballs. BD and nonBD archives contain the same
# top-level dir names (full_prompt/ mask/ preset_mask/), so extract each into its own folder.
for pair in rplan_weights_variants:rplan procthor_weights_BD_variants:procthor_bd procthor_weights_nonBD_variants:procthor_nonbd; do
  f="${pair%%:*}.zip"; d="${pair##*:}"
  [ -e "$d/.extracted" ] && continue
  mkdir -p "$d"; tar xzf "$f" -C "$d"; touch "$d/.extracted"
done
# ProcTHOR converted data (public HF dataset)
for s in train validation test; do
  [ -s "$DA/procthor/$s.json" ] || { mkdir -p "$DA/procthor"; curl -sL -o "$DA/procthor/$s.json" "https://huggingface.co/datasets/ludolara/DStruct2Design/resolve/main/$s.json"; }
done
ls -la "$CK" "$DA/procthor"
