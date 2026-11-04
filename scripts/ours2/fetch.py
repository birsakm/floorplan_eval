"""Fetch every generated home of one Mason2 generator version through the noface API.

Needs a tunnel to the API (production data is read only through it, never from the live SQLite file):
  ssh -N -L 13021:127.0.0.1:3021 michael@10.68.74.7 &
  python scripts/ours2/fetch.py --version next-living-bay-ad5110bb7b58

Writes data/ours2/<version>/homes/<seed>.json.gz (the full /api/generator/home response) and index.json
(the gallery rows: key, seed, rooms, area, objects, verdict). Serial requests, so production is not loaded.
"""
import argparse, gzip, json, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def get(base, path, **q):
    url = f"{base}{path}?{urllib.parse.urlencode(q)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:13021")
    ap.add_argument("--sleep", type=float, default=0.2)
    a = ap.parse_args()
    out = ROOT / "data" / "ours2" / a.version
    (out / "homes").mkdir(parents=True, exist_ok=True)
    rows, off = [], 0
    while True:
        page = get(a.base, "/api/generator/homes", version=a.version, offset=off, limit=500)
        rows += page["rows"]
        off += len(page["rows"])
        if not page["rows"] or off >= page["total"]:
            break
    json.dump({"version": a.version, "total": page["total"], "rows": rows}, open(out / "index.json", "w"), indent=1)
    print(f"{len(rows)} homes listed")
    for i, r in enumerate(rows):
        f = out / "homes" / f"{r['seed']}.json.gz"
        if f.exists():
            continue
        d = get(a.base, "/api/generator/home", key=r["key"])
        tmp = f.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as g:
            json.dump(d, g)
        tmp.rename(f)
        if i % 50 == 0:
            print(i, r["key"], flush=True)
        time.sleep(a.sleep)
    print("done")


if __name__ == "__main__":
    main()
