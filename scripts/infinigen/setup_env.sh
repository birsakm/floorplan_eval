#!/usr/bin/env bash
# Create the infinigen conda env from the pinned submodule (external/methods/infinigen, 59a2574 = package 1.12.3).
# Minimal install (no terrain / OpenGL extras): enough for the floor-plan solver and bpy, which is all
# scripts/infinigen/generate.py uses. Non-editable install, so the submodule stays clean.
# The env used for the results has python 3.11.12, bpy 4.2.0, shapely 2.1.1, numpy 2.2.6, networkx 3.4.2.
set -euo pipefail
export PYTHONNOUSERSITE=1
ENV=infinigen
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  mamba create -y -n "$ENV" -c conda-forge python=3.11 pip
fi
INFINIGEN_MINIMAL_INSTALL=True conda run -n "$ENV" pip install --no-cache-dir "$ROOT/external/methods/infinigen"
conda run -n "$ENV" python -c "import infinigen, bpy, shapely; print(infinigen.__version__, bpy.app.version_string, shapely.__version__)"
