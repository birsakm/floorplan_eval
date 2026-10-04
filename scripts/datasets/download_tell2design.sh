#!/usr/bin/env bash
# Tell2Design (ACL 2023, CC BY-NC 4.0) Google Drive zip (~0.7 GB, 19.5 GB unzipped incl. a
# 16.6 GB pickle we skip). Contains 80,788 RPLAN-derived floorplan images + text annotations.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/tell2design/raw"; mkdir -p "$OUT"; cd "$OUT"
[ -f Tell2Design.zip ] || gdown 1ZkoDAE72qy-62nO_kul7YSVT0v5Xz87R -O Tell2Design.zip
unzip -n -q Tell2Design.zip -x "Tell2Design Data/General Data/Tell2Design_artificial_all.pkl"
