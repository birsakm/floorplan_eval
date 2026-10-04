"""Build a DiffPlanner working directory outside the submodule.
The sample scripts use cwd-relative paths ('../../dataset/dataset_json/data_test.json',
'../../output/output_json'), so we create
  <work>/dataset/dataset_json/data_test.json   (DiffPlanner test json filtered/ordered by ids)
  <work>/{node_diff,adjacency_diff,partitioning_diff}/scripts/   (cwd for each stage)
  <work>/output/output_json/
Usage: python prepare_work.py <data_test.json> <ids.txt> <work>"""
import json, os, sys
src, ids, work = sys.argv[1:4]
names = [l.strip() for l in open(ids) if l.strip()]
data = {d["name"]: d for d in json.load(open(src, encoding="utf-8"))}
sel = [data[n] for n in names if n in data]
os.makedirs(f"{work}/dataset/dataset_json", exist_ok=True)
for s in ("node_diff", "adjacency_diff", "partitioning_diff"):
    os.makedirs(f"{work}/{s}/scripts", exist_ok=True)
os.makedirs(f"{work}/output/output_json", exist_ok=True)
json.dump(sel, open(f"{work}/dataset/dataset_json/data_test.json", "w"), ensure_ascii=False)
print(len(sel), "test plans ->", work)
