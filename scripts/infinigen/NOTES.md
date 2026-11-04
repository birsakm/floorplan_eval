# Infinigen Indoors (CVPR 2024): run notes

Scripts:
- `setup_env.sh` creates the env `infinigen` from the submodule `external/methods/infinigen` (commit 59a2574, package
  version 1.12.3).
- `generate.py` runs Infinigen's floor-plan solver on our room programs and writes `raw/<home>.json` (runs in the
  `infinigen` env).
- `convert.py` writes the common format (runs in `fpe`).

```bash
/datawaha/cggroup/birsakm/miniconda3/envs/infinigen/bin/python scripts/infinigen/generate.py --kind apartment --workers 20
PYTHONPATH=. conda run -n fpe python scripts/infinigen/convert.py --kind apartment
PYTHONPATH=. conda run -n fpe python -m fpeval.evaluate variants infinigen/apartment_program_n100_seed0
PYTHONPATH=. conda run -n fpe python scripts/ours/evaluate_ours.py --kind apartment --variant infinigen/apartment_program_n100_seed0
```
Same for `--kind house`. `--contour free` gives the `_freecontour` variants.

## What is Infinigen's and what is ours
Only the layout stage runs: `FloorPlanSolver` up to, not including, Blender solidification. No furniture, materials
or rendering.
- **Ours (input):** the room graph and the outline area. `GraphMaker.make_graph` returns one `RoomGraph` per storey
  built from our program:
  - our rooms, mapped to Infinigen room types (`TYPES` in `generate.py`)
  - the door and open edges between rooms of that storey
  - the Exterior node, linked to the rooms that need an exterior wall and to the entrance

  `suggest_dimensions` sizes the outline from the program's target net areas times Infinigen's slackness.
- **Infinigen's:** everything else. That is the outline segmentation, room assignment, simulated annealing,
  `contour_factory.decorate`, and door, opening, entrance and window placement. The openings come from the cutter
  functions of `BlueprintSolidifier` and are read back from the Blender cutter objects.

## Two variants
- **Fixed contour (main):** `consgraph.constants.fixed_contour = True`. The outline keeps the brief's area, so the
  net area follows the brief (error 6% / 9%). Outlines are plain rectangles (4 corners).
  - In this mode a two-storey house reuses the ground-floor outline for the upper storey.
- **Free contour (`_freecontour`):** Infinigen's default for homes (`home_room_constraints()` sets
  `fixed_contour = False`). The annealing grows the outline, and homes come out about 2.3× the brief's area.

## Patch
With shapely ≥ 2.1, an annealing move that collapses a room makes `RoomConstants.canonicalize` raise `ValueError`
("linearring requires at least 4 coordinates"). Infinigen expects `NotImplementedError` there, which is its signal
to reject the move. `generate.py` wraps `canonicalize` to turn one into the other. This is done in our script; the
submodule is unmodified.

(The local clone at `/datawaha/cggroup/birsakm/infinigen` that the env was first installed from carries an
uncommitted edit of `constants.py` with the same effect. That edit was not installed: the env's `constants.py`
equals the upstream file at 59a2574.)

## Caveats for the paper
- Wall adjacency is low (0.34–0.42): Infinigen was given only door and open edges.
- Fixed contour breaks more basic rules (29–30%), mostly rooms that are too narrow or too elongated. Free contour
  breaks fewer (3% / 12%) but gets the area wrong.
