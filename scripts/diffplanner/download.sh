#!/usr/bin/env bash
# DiffPlanner weights (GitHub release) + inputs (DiffPlanner dataset.zip, Graph2Plan Data.zip, id lists).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p checkpoints/diffplanner
if [ ! -f checkpoints/diffplanner/trained_model/node_diff/scripts/trained_model/b_model300000.pt ]; then
  [ -f checkpoints/diffplanner/trained_model.zip ] || \
    wget -q https://github.com/shidong-wang/DiffPlanner/releases/download/trained_model/trained_model.zip -O checkpoints/diffplanner/trained_model.zip
  unzip -oq checkpoints/diffplanner/trained_model.zip -d checkpoints/diffplanner
fi
bash scripts/diffplanner/download_g2p_data.sh
