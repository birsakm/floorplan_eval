#!/usr/bin/env bash
# Shared inputs for iPLAN / DiffPlanner / WallPlan:
#  - Graph2Plan Data.zip (public GitHub release) -> Network/data/data_{train,valid,test}.mat
#  - DiffPlanner dataset.zip (its own json conversion of the same data, GitHub release)
#  - common id lists (data/method_inputs/common_rplan_test/ids_test_{all,1000}.txt)
set -euo pipefail
cd "$(dirname "$0")/../.."
D=data/method_inputs/diffplanner
mkdir -p $D
[ -f $D/Data.zip ] || wget -q https://github.com/HanHan55/Graph2plan/releases/download/data/Data.zip -O $D/Data.zip
[ -f $D/graph2plan_data/Network/data/data_test.mat ] || unzip -oq $D/Data.zip 'Network/data/*.mat' -d $D/graph2plan_data
[ -f $D/dataset.zip ] || wget -q https://github.com/shidong-wang/DiffPlanner/releases/download/dataset/dataset.zip -O $D/dataset.zip
[ -f $D/dataset/dataset_json/data_test.json ] || unzip -oq $D/dataset.zip -d $D
[ -f data/method_inputs/common_rplan_test/ids_test_1000.txt ] || \
  conda run -n fpe python scripts/diffplanner/make_id_lists.py $D/dataset/dataset_json/data_test.json data/method_inputs/common_rplan_test
