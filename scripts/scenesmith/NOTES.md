# SceneSmith (ICML 2026): notes

Submodule: `external/methods/scenesmith` (commit 67cc408). It is not run here. Running it needs an OpenAI key for
gpt-5.2, a Blender render server and Drake. Its paper reports $13.98 per full scene (Appendix N.2). The floor-plan
stage alone would be roughly $2–5 per home, estimated from its config (planner, designer, critic with renders, up to
2 critique rounds).

Instead we evaluate the scenes the authors published:
- `fetch.sh` streams the 31 house-level scenes of HF `nepfaff/scenesmith-example-scenes` (`House/`, 117 GB of tars)
  and keeps only the layout files (~20 MB, in `data/scenesmith/`). It also gets `text_prompt_annotations.csv`.
- `convert.py` reads the layout back from the simulation scene and writes `outputs/scenesmith/house_published`.
  See its docstring for how rooms, doors, windows and the graph are recovered.

```bash
bash scripts/scenesmith/fetch.sh
PYTHONPATH=. conda run -n fpe python scripts/scenesmith/convert.py
PYTHONPATH=. conda run -n fpe python -m fpeval.evaluate variants scenesmith/house_published
```

## Scope
- Only 7 of the 31 prompts are homes: scenes 185, 187, 188, 191, 195, 198, 205.
- 9 are suites: bedroom suites, hotel suites and a dorm (47, 48, 49, 97, 98, 99, 186, 193, 196).
- 15 are non-residential: offices, clinics, a restaurant and so on.
- `condition.category` holds this split.
- Every scene is one storey, and every room is an axis-aligned rectangle.
- The prompts are SceneSmith's, not our briefs. With N = 16 residential plans FID/KID is not meaningful, so
  SceneSmith gets statistics and a qualitative comparison, not a head-to-head score.

## Conversion details
- Open connections versus doors: a gap wider than SceneSmith's largest door (1.9 m), or a wall with no collision at
  all, is an opening. A narrower open connection would be counted as a door. The published scenes have only 2
  openings, both in scene 208.
- The white notches in some outlines are real: every room touches at least one other room.
