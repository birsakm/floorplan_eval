#!/usr/bin/env bash
# GSDiff checkpoints (5 Google Drive files from the README) + public RPLAN-derived test inputs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
CK=checkpoints/gsdiff; IN=data/method_inputs/gsdiff
mkdir -p $CK $IN
cd $CK
# unconstrained, topology-constrained, boundary-constrained, boundary-AE CNN, topology-AE transformer
for id in 15gM0GtW2GwHmlpz0r-rpvo-k-BlNy_gu 1pk7SmvLZ8ON3OUL3SNxPRu73ndVKru0z 1puqxXIW4Y7AeQHFuC76PlYpWQm6MD8PS \
          1l6QRpfX5Jtucg3R995HajlwRG8SewUJW 1tExX8LdrFpJfBQH5y2emC6BltBwf9tHx; do
  conda run -n fpe gdown --continue "$id"
done
# unconstrained-params.zip -> outputs/structure-{1,2}; boun-params.zip -> outputs/structure-{81-106-3,56-36-interval1000,...};
# topo-params.zip -> topo-params/structure-{80-106-2,56-35-interval1000,57-16,56-16}; bounae.tar -> structure-78-12; topoae.tar -> structure-57-16
for f in unconstrained-params.zip topo-params.zip boun-params.zip; do unzip -o -q $f; done
for f in bounae.tar topoae.tar; do tar xf $f; done
cd "$ROOT"
# RPLAN-derived test inputs: DiffPlanner's public release of the Graph2Plan RPLAN split (room polygons, types, adjacency)
if [ ! -f $IN/dataset/dataset_json/data_test.json ]; then
  wget -q -O $IN/dataset.zip https://github.com/shidong-wang/DiffPlanner/releases/download/dataset/dataset.zip
  (cd $IN && unzip -o -q dataset.zip)
fi
