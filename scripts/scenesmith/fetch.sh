#!/bin/bash
# Fetch the layout files of SceneSmith's 31 published house-level scenes into data/scenesmith (~20 MB).
# Each scene's tar (~1.8-6 GB, mostly meshes) is streamed and only the files convert.py reads are kept.
set -e
OUT=$(dirname "$0")/../../data/scenesmith
HF=https://huggingface.co/datasets/nepfaff/scenesmith-example-scenes
mkdir -p $OUT/House
curl -sL $HF/resolve/main/text_prompt_annotations.csv -o $OUT/text_prompt_annotations.csv
for id in $(curl -s "https://huggingface.co/api/datasets/nepfaff/scenesmith-example-scenes/tree/main/House" \
    | python3 -c "import json,sys;print(' '.join(x['path'].split('/')[-1][:-4] for x in json.load(sys.stdin)))"); do
  [ -f $OUT/House/$id/combined_house/house.dmd.yaml ] && continue
  mkdir -p $OUT/House/$id
  curl -sL --retry 3 $HF/resolve/main/House/$id.tar | tar -x -C $OUT/House/$id --wildcards \
    './combined_house/house.dmd.yaml' './room_geometry/*' './floor_plans/*.gltf' 2>/dev/null || true
  echo "$id $(find $OUT/House/$id -type f | wc -l) files"
done
