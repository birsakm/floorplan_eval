#!/usr/bin/env bash
# WallPlan weights: Google Drive file linked in the repo's README.pdf
# (https://drive.google.com/file/d/1Ae9fisgl-r3AJUq_16VSygw0Jeh-XooS/view -- the ID wraps over a line
# break in the PDF; text extraction yields "...Jeh-" + "XooS"). It is a RAR with
# Boundary_constraint/{WindowLiving,WindowOther,LabelNet,GraphNet}.pth (boundary-only models).
# Test inputs: 500 pkls bundled in the repo (test/input) + Graph2Plan RPLAN test boundaries.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p checkpoints/wallplan
if [ ! -f checkpoints/wallplan/Boundary_constraint/GraphNet.pth ]; then
  [ -f checkpoints/wallplan/Boundary_constraint.rar ] || \
    (cd checkpoints/wallplan && conda run -n fpe gdown 1Ae9fisgl-r3AJUq_16VSygw0Jeh-XooS)
  (cd checkpoints/wallplan && unrar x -o+ Boundary_constraint.rar >/dev/null)   # or: 7z x
fi
bash scripts/diffplanner/download_g2p_data.sh
