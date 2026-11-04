"""Download the original RPLAN plans (all 80,788, with interior doors) from the Kaggle upload "RPLAN pickle files"
(https://www.kaggle.com/datasets/mohamedalqblawi/rplan-pickle-files, uploaded 2023-10-20, 21 GB, licence unstated).

  python scripts/datasets/download_rplan_kaggle.py

Needs a Kaggle API token: ~/.kaggle/access_token (or legacy ~/.kaggle/kaggle.json); kaggle.com -> Settings -> API ->
Create New Token. Anonymous single-file downloads work too, but Kaggle blocks them
(404) after a few hundred files, so the whole archive (166 MB zip) is fetched once with the token
and unpacked at idle IO priority.
Output: data/rplan_kaggle/raw/pickle/{train,val}/<RPLAN id>.pkl

The ids are the original RPLAN ids, the same as Graph2Plan's 'name' (data/rplan_graph2plan sample test_<id>);
Kaggle's train/val folders are the uploader's own split, not Graph2Plan's.
Each pickle is a list: [inside mask, boundary (127 = exterior wall, 255 = front door), interior walls,
interior doors, rooms [{category (RPLAN 0-12), centroid (row, col)}]], 256 x 256 px, 18/256 m per px.
Load it with fpeval.datasets.rplan_kaggle.load_pickle (numpy-only unpickler), never with pickle.load.
"""
import json
import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "rplan_kaggle" / "raw"
URL = "https://www.kaggle.com/api/v1/datasets/download/mohamedalqblawi/rplan-pickle-files"
N_FILES = 80788


def curl_auth():
    """curl config line with the Kaggle credentials: ~/.kaggle/access_token (new API tokens, bearer),
    KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json (legacy keys, basic auth)."""
    tok = Path.home() / ".kaggle" / "access_token"
    if os.environ.get("KAGGLE_API_TOKEN") or tok.exists():
        t = os.environ.get("KAGGLE_API_TOKEN") or tok.read_text().strip()
        return f'header = "Authorization: Bearer {t}"\n'
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return f'user = "{os.environ["KAGGLE_USERNAME"]}:{os.environ["KAGGLE_KEY"]}"\n'
    f = Path.home() / ".kaggle" / "kaggle.json"
    if f.exists():
        c = json.load(open(f))
        return f'user = "{c["username"]}:{c["key"]}"\n'
    raise SystemExit("no Kaggle token: ~/.kaggle/access_token or ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    have = len(list((OUT / "pickle").glob("*/*.pkl")))
    if have == N_FILES:
        print(f"all {N_FILES} pickles present")
        return
    z = OUT / "rplan-pickle-files.zip"
    cfg = OUT / ".curl_auth"  # keeps the key off the command line (ps)
    cfg.touch(mode=0o600)
    cfg.write_text(curl_auth())
    try:
        subprocess.run(["ionice", "-c3", "nice", "-n", "19", "curl", "-fL", "-C", "-", "--retry", "5", "-K", str(cfg),
                        "-o", str(z), URL], check=True)
    finally:
        cfg.unlink(missing_ok=True)
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist() if n.endswith(".pkl")]
        print(f"{len(names)} pickles in the archive; unpacking")
        for n in names:
            dst = OUT / n
            if not (dst.exists() and dst.stat().st_size == zf.getinfo(n).file_size):
                zf.extract(n, OUT)
    got = len(list((OUT / "pickle").glob("*/*.pkl")))
    print(f"{got} pickles in {OUT / 'pickle'}")
    if got == N_FILES:
        z.unlink()


if __name__ == "__main__":
    main()
