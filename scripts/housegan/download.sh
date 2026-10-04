#!/usr/bin/env bash
# House-GAN pretrained model + LIFULL vectorized data (public Dropbox folder from the repo README).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
CK=checkpoints/housegan; IN=data/method_inputs/housegan
mkdir -p $CK $IN
if [ ! -f $CK/exp_demo_D_500000.pth ]; then
  wget -q -O $CK/housegan_dropbox.zip "https://www.dropbox.com/sh/p707nojabzf0nhi/AAB4UPwW0EgHhbQuHyq60tCKa?dl=1"
  (cd $CK && unzip -o -q housegan_dropbox.zip -x / || true)
  # zip contains: exp_demo_D_500000.pth, dataset_paper/{train,valid}_data.npy, housegan_clean_data.{npy,pkl}, README/LICENSE
  mv $CK/dataset_paper $CK/housegan_clean_data.* $CK/LICENSE.txt $CK/README.txt $IN/
  rm -f $CK/housegan_dropbox.zip
fi
ls -la $CK $IN
