#!/usr/bin/env bash
# Create the fpe-maskplan conda env (TensorFlow 2.6 + CUDA 11.2/cuDNN 8.1 from conda-forge).
# Linux port of external/methods/maskplan/MaskPLAN.yaml (which is Windows-only).
set -euo pipefail
ENV=fpe-maskplan
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SUB="$ROOT/external/methods/maskplan"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  for i in 1 2 3 4 5; do
    mamba create -y -n "$ENV" -c conda-forge python=3.9 cudatoolkit=11.2 cudnn=8.1 && break
    echo "mamba create failed (cache lock?), retrying in 60s"; sleep 60
  done
fi
conda run -n "$ENV" pip install \
  tensorflow-gpu==2.6.0 keras==2.6.0 tensorflow-estimator==2.6.0 \
  "numpy==1.19.5" "protobuf==3.20.3" "h5py==3.1.0" \
  "opencv-python-headless==4.5.5.64" "shapely==2.0.2" "scipy==1.7.3" "matplotlib==3.5.2" "pillow==9.2.0"

# Activation hook so TF finds the conda CUDA libs.
ACT="$(conda run -n "$ENV" python -c 'import sys;print(sys.prefix)')/etc/conda/activate.d"
mkdir -p "$ACT"
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"' > "$ACT/ld_path.sh"

# Boundary images + weights: see download.sh
conda run -n "$ENV" python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
