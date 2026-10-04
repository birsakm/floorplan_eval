#!/usr/bin/env bash
# Create the fpe-iplan conda env. Original: Python 3.6 / torch 1.7.0 / CUDA 10.1 (no sm_80).
# We use Python 3.8 + torch 1.8.1+cu111 (supports A100/sm_80); numpy<1.24 for np.int.
set -euo pipefail
ENV=fpe-iplan
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  for i in 1 2 3 4 5; do mamba create -y -n "$ENV" python=3.8 pip && break || { echo "retry mamba ($i)"; sleep 60; }; done
fi
conda run -n "$ENV" pip install --no-cache-dir torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
conda run -n "$ENV" pip install --no-cache-dir numpy==1.21.6 scipy==1.7.3 opencv-python-headless==4.5.5.64 \
  pillow==9.5.0 shapely==1.8.5 scikit-image==0.19.3 torchnet==0.0.4 tqdm
conda run -n "$ENV" python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
