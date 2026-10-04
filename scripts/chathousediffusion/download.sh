#!/usr/bin/env bash
# ChatHouseDiffusion weights + Tell2Design test inputs (pre-parsed by the authors with moonshot-v1-8k,
# so no LLM API call is needed). Both from Tsinghua cloud (public links in upstream README).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CK="$ROOT/checkpoints/chathousediffusion"; DATA="$ROOT/data/method_inputs/chathousediffusion"
mkdir -p "$CK" "$DATA"
if [ ! -f "$CK/model-98.pt" ]; then
  [ -s "$CK/predict_model.rar" ] || curl -L -o "$CK/predict_model.rar" "https://cloud.tsinghua.edu.cn/f/a01a8205be55462685fd/?dl=1"
  (cd "$CK" && unrar x -o+ predict_model.rar)   # -> model-98.pt, params.pkl
fi
if [ ! -d "$DATA/image_test" ]; then
  [ -s "$DATA/kimi_test_data.rar" ] || curl -L -o "$DATA/kimi_test_data.rar" "https://cloud.tsinghua.edu.cn/f/2844208e0c344d18bd72/?dl=1"
  (cd "$DATA" && unrar x -o+ kimi_test_data.rar)  # -> image_test/ mask_test/ text_test/json.csv (2308) + empty train dirs
fi
ls "$CK"; ls "$DATA/image_test" | wc -l
