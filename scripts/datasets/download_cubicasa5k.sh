#!/usr/bin/env bash
# Download CubiCasa5k (Zenodo record 2613548, CC BY-NC-SA 4.0, ~5.5 GB zip) to data/cubicasa5k/raw/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/cubicasa5k/raw"
mkdir -p "$OUT"
cd "$OUT"
if [ ! -f cubicasa5k.zip ]; then
  wget -c -O cubicasa5k.zip.part "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"
  mv cubicasa5k.zip.part cubicasa5k.zip
fi
echo "2613548 md5 check:"; md5sum cubicasa5k.zip
[ -d cubicasa5k ] || unzip -q cubicasa5k.zip
ls cubicasa5k | head
