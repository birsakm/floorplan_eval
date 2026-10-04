#!/usr/bin/env bash
# House-GAN++: weights are bundled in the repo (checkpoints/pretrained.pth) -> copy to
# checkpoints/houseganpp/. RPLAN itself is gated (Google Form), so the only public test inputs
# are the RPLAN graphs bundled with the repos:
#   external/methods/houseganpp/data/json/*.json            (6 RPLAN plans, HG++ JSON)
#   external/data_tools/housegan_data_reader/sample_output/0.json  (1 RPLAN plan, HG++ JSON)
# They are copied to data/method_inputs/<method>/rplan_public/ (+ list.txt).
# If you have the full RPLAN HG++ JSON set, put its test JSONs in a directory and pass
# INPUT_DIR=... to run.sh instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HG="$ROOT/external/methods/houseganpp"
mkdir -p "$ROOT/checkpoints/houseganpp"
cp -n "$HG/checkpoints/pretrained.pth" "$ROOT/checkpoints/houseganpp/pretrained.pth"
for M in houseganpp house_diffusion; do
  D="$ROOT/data/method_inputs/$M/rplan_public"
  mkdir -p "$D"
  cp "$HG"/data/json/*.json "$D/"
  cp "$ROOT/external/data_tools/housegan_data_reader/sample_output/0.json" "$D/0.json"
  ls "$D"/*.json | sort -V > "$D/list.txt"
done
ls -la "$ROOT/checkpoints/houseganpp" "$ROOT/data/method_inputs/houseganpp/rplan_public"
