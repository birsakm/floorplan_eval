#!/usr/bin/env bash
# Create the fpe-procthor environment for ProcTHOR's layout stage (external/methods/procthor, 53d5bd4): a venv on top
# of the shared fpe env (numpy, shapely, pandas, networkx) plus attrs and the pure-Python imports of the procthor package. A fresh conda env failed here on a terminfo
# symlink (I/O error on /datawaha), so the venv reuses fpe's packages instead.
# Only the pure-Python layout modules are used (floor plan, interior boundary, walls, door selection), so neither
# ai2thor nor the asset pipeline is installed; scripts/procthor/generate.py stubs the ai2thor/moviepy imports.
# procthor is used via PYTHONPATH (no pip -e, keeps the submodule clean).
set -euo pipefail
export PYTHONNOUSERSITE=1
VENV=${VENV:-/datawaha/cggroup/birsakm/venvs/fpe-procthor}
FPE_PY=$(conda run -n fpe which python)
[ -x "$VENV/bin/python" ] || "$FPE_PY" -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -q attrs trimesh canonicaljson python-sat python-fcl  # imported by procthor/__init__ chain
"$VENV/bin/python" -c "import attrs, shapely, numpy, pandas; print(attrs.__version__, shapely.__version__, numpy.__version__)"
