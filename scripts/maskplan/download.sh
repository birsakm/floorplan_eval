#!/usr/bin/env bash
# MaskPLAN weights: only All_Base_Deep_cross (the hybrid vec-img "Deep" model used by
# Inference/MaskPLAN_Inference_iterate_cross_Deep.py --model Base) from the authors' Drive folder
# https://drive.google.com/drive/folders/1yvKe9l3l3zk7nM36LqgWmeeBIZTck0kp  (10.4 GB 7z, 11.8 GB unpacked).
# Other files in the folder (not used): All_Base_Deep_vec.7z 1vG71skVD6-vRe8qGsBNAAhbb0uctj1Vu,
#   All_Large_Single_cross.7z 1KkHRhd19TE-oIrgpPlY53TNGP1YT848j, All_Large_Single_vec.7z 1JjtkP7rRz-JiV9mcl7sCPI874WAjIrgL
# VQ-VAE weights (VQ_Pretrained/) and preprocessed RPLAN attributes (Processed_data/) ship in the repo.
# Boundary images: bundled parsed_img/img_room_sqe/0.7z -> data/method_inputs/maskplan/parsed_img.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CK="$ROOT/checkpoints/maskplan"
mkdir -p "$CK/MaskPLAN_Checkpoints"
if [ ! -f "$CK/MaskPLAN_Trained/All_Base_Deep_cross/All.index" ]; then
  [ -f "$CK/MaskPLAN_Checkpoints/All_Base_Deep_cross.7z" ] || \
    conda run -n fpe gdown 1mzZdK313lYybr4GMN6_pF6lW3tRFwLjM -O "$CK/MaskPLAN_Checkpoints/All_Base_Deep_cross.7z"
  7z x -y -o"$CK/MaskPLAN_Trained" "$CK/MaskPLAN_Checkpoints/All_Base_Deep_cross.7z"
fi
DATA="$ROOT/data/method_inputs/maskplan/parsed_img/img_room_sqe"
if [ "$(ls "$DATA/0" 2>/dev/null | wc -l)" -lt 80788 ]; then
  mkdir -p "$DATA"
  7z x -y -o"$DATA" "$ROOT/external/methods/maskplan/parsed_img/img_room_sqe/0.7z" > /dev/null
fi
echo "weights: $(ls "$CK/MaskPLAN_Trained/All_Base_Deep_cross")  boundary pngs: $(ls "$DATA/0" | wc -l)"
