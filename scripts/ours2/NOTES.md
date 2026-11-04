# Ours, current pipeline (Mason2): run notes

The old 100 apartments + 100 houses (`ours/mason_old_*`, `scripts/ours/`) came from an old pipeline and are
superseded. Only Mason2 results count.

Source: the production Mason2 store on **noface** (`ssh michael@10.68.74.7`), read only through its API, never
from the live `homes.sqlite`. Generator version **`next-living-bay-ad5110bb7b58`** (created 2026-11-03; its source is
`/data/mason/worktrees/LivingBayDepth-source-20261103/`, named by Fedor as the latest run). 1,003 homes, all
single-storey apartments, all passing the generator's own 20 rules.

```bash
ssh -N -L 13021:127.0.0.1:3021 michael@10.68.74.7 &       # tunnel to the API
python scripts/ours2/fetch.py --version next-living-bay-ad5110bb7b58     # -> data/ours2/<version>/ (53 MB)
PYTHONPATH=. conda run -n fpe python scripts/ours2/convert.py --version next-living-bay-ad5110bb7b58
PYTHONPATH=. conda run -n fpe python -m fpeval.evaluate variants ours/mason2_next-living-bay-ad5110bb7b58
PYTHONPATH=. conda run -n fpe python scripts/ours/evaluate_ours.py --kind apartment --variant ours/mason2_next-living-bay-ad5110bb7b58
```

Baselines on the same programs (`--source` picks our variant, `--name` the output variant):
```bash
PYTHONPATH=. $PROCTHOR_PY scripts/procthor/generate.py --kind apartment \
    --source ours/mason2_next-living-bay-ad5110bb7b58 --name apartment_mason2-livingbay_seed0
PYTHONPATH=. $INFINIGEN_PY scripts/infinigen/generate.py --kind apartment \
    --source ours/mason2_next-living-bay-ad5110bb7b58 --name apartment_mason2-livingbay_seed0
PYTHONPATH=. conda run -n fpe python scripts/infinigen/convert.py --kind apartment \
    --source ours/mason2_next-living-bay-ad5110bb7b58 --name apartment_mason2-livingbay_seed0
```

## Re-evaluating a new generator version (one command)
```bash
P=/datawaha/cggroup/birsakm/miniconda3/envs/fpe/bin/python
PYTHONPATH=. $P scripts/ours2/run_all.py list                 # versions on noface (>= 500 homes), newest first
PYTHONPATH=. $P scripts/ours2/run_all.py <version>            # everything; ~4-5 h for ~1,000 homes; resumable
PYTHONPATH=. $P scripts/ours2/run_all.py <version> --status
PYTHONPATH=. $P scripts/ours2/run_all.py <version> --redo infinigen   # a step and the steps that use its output
```
It opens the tunnel itself, fetches, converts, runs ProcTHOR and Infinigen on the new programs, evaluates all three
(variants are registered automatically, `fpeval/eval_config.py: mason2_variants`), rewrites `results/comparison.md`
and the paper's Table 1(b) numbers (`paper/sec/results_auto.tex`, LaTeX macros used by `paper/sec/eval_tables.tex`).
Then compile and commit/push the paper by hand. Logs: `logs/mason2/<version>/`. Not handled: multi-storey homes (the
converter rejects them, and the run stops if < 95% of homes convert) and format changes of the Mason2 state.

The commands below are the individual steps it runs.

## Conversion (see the `convert.py` docstring)
- Net rooms = room polygon minus the wall solids (`wall_envelopes`); net areas then equal the program's targets.
- The `landing` outside the flat is dropped; the door to it is the front door.
- Balconies are kept (native `outdoor`). `small_bathroom` counts as a bathroom.
- Placed objects (class, asset, room, world footprint, height range) go to `objects/<id>.json` for the
  complete-scene evaluation.

## The brief
Mason2 draws its own brief (household, rooms, net areas) from Swiss priors; there is no external brief. `brief` is
therefore the program's room counts and summed indoor target area, and the adherence metrics measure how the room
layout follows the program. For the baselines, the program is written in the old vocabulary (`circulation`,
`living`, `dining`, ...). It has no daylight flag, so living, bedrooms, kitchen, dining and balconies are marked
`needs_exterior` (used by Infinigen).
