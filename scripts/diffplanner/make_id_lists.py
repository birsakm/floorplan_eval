"""Write the shared RPLAN test-id lists used by iPLAN / DiffPlanner / WallPlan.
ids_test_all.txt: DiffPlanner's 12002 test plans (Graph2Plan test split minus plans its
data_preparation filters out), in DiffPlanner's order.  ids_test_1000.txt: a fixed random
subset (numpy seed 0) of 1000 of them, sorted by integer name."""
import json, sys, os
import numpy as np
src, outdir = sys.argv[1], sys.argv[2]
names = [d["name"] for d in json.load(open(src))]
os.makedirs(outdir, exist_ok=True)
open(os.path.join(outdir, "ids_test_all.txt"), "w").write("\n".join(names) + "\n")
sub = sorted(np.random.RandomState(0).choice(names, 1000, replace=False).tolist(), key=int)
open(os.path.join(outdir, "ids_test_1000.txt"), "w").write("\n".join(sub) + "\n")
print(len(names), len(sub))
