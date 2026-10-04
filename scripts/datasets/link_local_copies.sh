#!/usr/bin/env bash
# Link dataset copies that already exist on this machine (read-only) into data/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/data/rplan_graph2plan"
# RPLAN in Graph2Plan format (data_{train,valid,test}.mat); RPLAN itself needs the RPLAN Google Form.
ln -sfn /datawaha/cggroup/datasets/RPLAN/Network/data "$ROOT/data/rplan_graph2plan/raw"
