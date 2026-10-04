#!/usr/bin/env bash
# Create the fpe-wallplan conda env. Original: Python 3.8 / torch 1.8.1 / numpy 1.19.2 / torchnet 0.0.5.1.
# torch 1.8.1+cu111 wheels support A100 (sm_80).
set -euo pipefail
ENV=fpe-wallplan
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  for i in 1 2 3 4 5; do mamba create -y -n "$ENV" python=3.8 pip && break || { echo "retry mamba ($i)"; sleep 60; }; done
fi
conda run -n "$ENV" pip install --no-cache-dir torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
conda run -n "$ENV" pip install --no-cache-dir numpy==1.19.5 scipy==1.6.3 opencv-python-headless==4.5.5.64 \
  pillow==9.5.0 shapely==1.8.5 scikit-image==0.18.3 networkx==2.6.3 tqdm
conda run -n "$ENV" pip install --no-cache-dir --no-deps torchnet==0.0.4 visdom==0.1.8.9 || true
conda run -n "$ENV" python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
