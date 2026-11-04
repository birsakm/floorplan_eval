# ProcTHOR (NeurIPS 2022): run notes

Submodule: `external/methods/procthor` (commit 53d5bd4, the last commit, 2022-12-14).

Not to be confused with **ProcTHOR-10k**: that is the dataset of 10k ProcTHOR houses, used here as DS2D's
reference set (`procthor10k:test`, see `scripts/ds2d/`).

Scripts:
- `setup_env.sh` creates the venv `/datawaha/cggroup/birsakm/venvs/fpe-procthor`. It is a venv on top of `fpe` plus
  attrs, trimesh, canonicaljson, python-sat and python-fcl, because a fresh conda env failed on `/datawaha` with an
  I/O error on a terminfo symlink.
- `generate.py` runs ProcTHOR's layout stage on our room programs and writes the common format directly.

```bash
V=/datawaha/cggroup/birsakm/venvs/fpe-procthor/bin/python
PYTHONPATH=. $V scripts/procthor/generate.py --kind apartment
PYTHONPATH=. conda run -n fpe python -m fpeval.evaluate variants procthor/apartment_program_n100_seed0
PYTHONPATH=. conda run -n fpe python scripts/ours/evaluate_ours.py --kind apartment --variant procthor/apartment_program_n100_seed0
```
Same for `--kind house`. One plan per storey, with the same ids and conditions as ours.

## What is ProcTHOR's and what is ours
Only the layout stage runs, which is pure Python. `ai2thor` and `moviepy` are stubbed: they are imported but only
used by the simulator stages.
- **Ours (input):** the `RoomSpec`, built from our program. See the `generate.py` docstring.
  - **Rooms and ratios:** one `LeafRoom` per program room, ratio = target net area.
  - **Door graph as a tree:** a room plus the rooms that hang only off it becomes a `MetaRoom`.
  - **Area:** the grid is sized to the program's area, and the cell scale is set so that the area matches exactly.
  - **Open connections:** taken from our program's open edges.
  - **Front door:** at our entrance room on the ground storey.
- **ProcTHOR's:** everything else. That is the interior boundary with random corner cuts, rectangle and L-shape
  growth, the best of 100 candidate layouts, wall extraction, which neighbour pairs get doors (`select_openings`),
  and door widths from ProcTHOR's Doorway assets.

## Adaptations to disclose
- **Room types:** ProcTHOR's `LeafRoom` only accepts Kitchen, LivingRoom, Bedroom, Bathroom or None. Other types
  (hallway, storage, stairs and so on) are None on ProcTHOR's side, and our type is kept for evaluation. The layout
  algorithm uses types only for the outdoor door and open connections, and both come from our program here.
- **Door graph:** ProcTHOR has no door-graph input. It places doors between siblings of the RoomSpec tree, choosing
  among the pairs that end up as neighbours. So our exact edges are not guaranteed.
- **Walls:** they have no thickness (rooms share edges).
- **Windows:** ProcTHOR places windows as AI2-THOR assets in a later stage that needs the simulator, so there are
  none. Window metrics are reported as not measured (NaN), not as 0.
- **Storeys:** single storey only, so each storey of a house is generated on its own. The stairs room is an ordinary
  room on each storey, and the storeys are not aligned with each other.
- **Grid:** cells are about 1.9–2.0 m (ProcTHOR's `interior_boundary_scale` of U(1.6, 2.2), rescaled to match the
  area), so every room is a whole number of cells.
- **Retries:** the house upper storeys needed more interior-boundary tries than ProcTHOR's default sampling (500 per
  plan). All 100 apartments and all 138 house storeys succeeded.
