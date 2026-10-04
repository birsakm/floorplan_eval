#!/usr/bin/env bash
# Download iPLAN weights (Google Drive folder from the iPLAN README) and the Graph2Plan-preprocessed
# RPLAN data (public GitHub release of HanHan55/Graph2plan) used as boundary inputs + GT.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p checkpoints/iplan
if [ ! -f checkpoints/iplan/iPLAN/room_partition/G_net_210.pth ]; then
  (cd checkpoints/iplan && conda run -n fpe gdown --folder https://drive.google.com/drive/folders/1TRMKu6zw-pgEpGja2zTCixA2WhhU5KXr)
fi
bash scripts/diffplanner/download_g2p_data.sh
