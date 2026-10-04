#!/usr/bin/env bash
# House-GAN vectorized LIFULL floorplans (Dropbox link from ennauata/housegan README, ~0.6 GB zip,
# contains housegan_clean_data.npy / train_data.npy + pretrained model). No RGB images.
# Original LIFULL HOME'S data: NII IDR (https://www.nii.ac.jp/dsc/idr/lifull), research use only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/housegan_lifull/raw"; mkdir -p "$OUT"; cd "$OUT"
[ -f house_gan.zip ] || { wget -c -O hg.part "https://www.dropbox.com/sh/p707nojabzf0nhi/AAB4UPwW0EgHhbQuHyq60tCKa?dl=1"; mv hg.part house_gan.zip; }
unzip -n -q house_gan.zip || true   # Dropbox folder zips may warn about empty dir entries
find . -maxdepth 3 | head -50
