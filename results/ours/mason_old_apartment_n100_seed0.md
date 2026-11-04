# Our apartments: evaluation (mason_old, run 20261024-full-n100-seed0)

100 apartments from Mason's old program (state export in `data/ours`, room-plan stage), converted by `scripts/ours/convert.py`. Real baselines: Swiss Dwellings and ResPlan apartments with the same numbers of bedrooms and bathrooms and a balcony or not, 20 per generated plan.

## Validity, input and building norms

| metric (lineage) | all (n=100) |
|---|---|
| plans breaking a basic rule: overlap, holes, pieces, min area/width, max aspect (HouseDiffusion-era validity, GRE-Diff) | 0% |
| room counts exactly as in the brief (DS2D prompt consistency) | 100% |
| net area error vs brief, mean / max (DS2D, RLVR area error) | 0.00% / 0.00% |
| program connections built, recall / precision (DS2D self-consistency) | 1.00 / 1.00 |
| program 'wall' neighbours sharing a wall | 0.996 |
| compatibility: typed graph edit distance program ↔ built connections (House-GAN) | 0.00 |
| all rooms reachable from the front door (GRE-Diff) | 100% |
| habitable rooms only reachable through a bedroom, per home | 0.00 |
| habitable rooms with window area ≥ 1/8 of floor area | 100% |
| plans breaking a binding Austrian rule: Vienna, Lower Austria, Upper Austria | 0% / 0% / 0% |

Program connections and the brief come from the generator's own program stage, so these rows measure internal consistency. Reachability is not compared with real data: the real datasets draw no door for open connections.

## Realism against matched real apartments

| | ours | Swiss Dwellings | ResPlan |
|---|---|---|---|
| net area per plan (m²) | 81.80 | 70.12 | 111.90 |
| rooms | 6.42 | 6.41 | 5.98 |
| corners of the outline | 6.28 | 14.46 | 12.08 |
| outline area / bounding rectangle | 0.89 | 0.82 | 0.82 |
| rooms that are plain rectangles | 0.86 | 0.47 | 0.50 |
| mean room aspect ratio | 1.92 | 1.66 | 1.62 |
| neighbours per room | 2.81 | 2.78 | 2.66 |
| habitable rooms with a window | 1.00 | 0.86 | 0.89 |

Room area in m², median (share of our rooms inside the real p5–p95 range):

| room | ours (n) | Swiss Dwellings | ResPlan |
|---|---|---|---|
| living_room | 26.8 (100) | 24.3 (99%) | 33.5 (97%) |
| kitchen | 10.6 (100) | 7.4 (94%) | 9.6 (100%) |
| bedroom | 15.4 (171) | 14.7 (95%) | 17.2 (100%) |
| bathroom | 5.3 (125) | 3.9 (100%) | 6.3 (100%) |
| balcony | 3.6 (45) | 6.7 (89%) | 4.7 (80%) |
| hallway | 8.2 (100) | 6.8 (100%) | – |

Classifier two-sample test (leave-one-out 1-NN on geometric plan features; 50% = cannot tell apart): ours vs Swiss Dwellings 96% (real vs real 49%), ours vs ResPlan 100% (real vs real 50%); ResPlan vs Swiss Dwellings 99%. The test is very sensitive: two real datasets are about as separable from each other as ours is from either.

Largest differences to Swiss Dwellings (by Cohen's d): frac_rect_rooms 0.86 vs 0.47; outline_corners 6.28 vs 14.37; n_living_room 1.00 vs 0.67; share_living_room 0.33 vs 0.22; mean_room_convexity 0.98 vs 0.96; mean_room_aspect 1.92 vs 1.66.
Largest differences to ResPlan (by Cohen's d): n_hallway 1.00 vs 0.00; share_hallway 0.11 vs 0.00; frac_rect_rooms 0.86 vs 0.50; mean_room_convexity 0.98 vs 0.94; outline_corners 6.28 vs 11.94; share_living_room 0.33 vs 0.42.

## Diversity

Distinct typed room-adjacency topologies per plan: ours 31%, matched Swiss Dwellings 93%, matched ResPlan 72% (mean of 20 draws of 100).

## FID / KID

Inception-V3 features of uniform 256 px layout renders (`fpeval.render.render_layout`: type colours, 4 px walls, no doors or windows), against the matched real plans. Mean ± std over 5 rounds; each round holds out one matched real plan per generated one, and the held-out set is scored the same way as ours. That row is the floor: what a generator that samples the real data would get at this sample size. KID ×1000; lower is better for both.

| reference | Swiss Dwellings (n=1900) | ResPlan (n=1882) |
|---|---|---|
| FID, ours (n=100) | 65.4 ± 0.2 | 42.6 ± 0.1 |
| FID, held-out real of the same dataset (floor) | 28.9 ± 0.7 | 12.3 ± 0.4 |
| FID, held-out real of the other dataset | 43.8 ± 0.5 | 40.4 ± 1.9 |
| KID ×1000, ours | 47.5 ± 0.8 | 34.3 ± 0.3 |
| KID ×1000, held-out real of the same dataset (floor) | -0.1 ± 0.4 | -0.1 ± 0.2 |
| KID ×1000, held-out real of the other dataset | 15.1 ± 1.2 | 12.6 ± 0.7 |

FID at about 100 samples has a large positive bias (the floor is well above 0) and absolute values are not comparable with numbers in papers, which use other renderers and 1,000–12,000 samples. KID is unbiased.

## Not measured

- Human or VLM realism studies (House-GAN-style pairwise realism).
