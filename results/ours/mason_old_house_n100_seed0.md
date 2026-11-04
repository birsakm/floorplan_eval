# Our houses: evaluation (mason_old, run 20261024-full-n100-seed0)

100 houses from Mason's old program (62 one-storey, 38 two-storey), converted by `scripts/ours/convert.py` into one sample per storey (138). In the two-storey houses the State replay had rejected 50 rooms and 97 openings because a newer check counts stair rooms into the net area; the rejected commands carry the original geometry and are restored, so all 100 houses have the generator's full room list. Real baselines: CubiCasa5k and MagicPlan floors with the same numbers of bedrooms and bathrooms, 20 per generated storey. Neither dataset says whether a floor belongs to a house or an apartment.

## Validity, input and building norms

| metric (lineage) | all (n=100) | 1 storey (n=62) | 2 storeys (n=38) |
|---|---|---|---|
| storeys breaking a basic rule: overlap, holes, pieces, min area/width, max aspect (HouseDiffusion-era validity, GRE-Diff) | 0% | | |
| room counts exactly as in the brief (DS2D prompt consistency) | 100% | 100% | 100% |
| net area error vs brief, mean / max (DS2D, RLVR area error) | 0.11% / 0.69% | 0.00% / 0.00% | 0.28% / 0.69% |
| same, counting stair rooms as net area | 4.5% | 0.0% | 11.9% |
| program connections built, recall / precision (DS2D self-consistency) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| program 'wall' neighbours sharing a wall | 1.000 | 1.000 | 1.000 |
| compatibility: typed graph edit distance program ↔ built connections (House-GAN) | 0.00 | 0.00 | 0.00 |
| all rooms reachable from the front door (stairs join storeys) (GRE-Diff) | 100% | 100% | 100% |
| habitable rooms only reachable through a bedroom, per home | 0.00 | 0.00 | 0.00 |
| habitable rooms with window area ≥ 1/8 of floor area | 100% | 100% | 100% |
| storeys breaking a binding Austrian rule: Vienna, Lower Austria, Upper Austria | 0% / 9% / 0% | | |

Building-norm failures: at-rd-noe-wohnraum-10 (9% of storeys): In Niederösterreich every Wohnraum (habitable room of a dwelling) must have a Netto-Grundfläche (net floor area) of at least 10 m²; floor areas with a clear height below 1.5 m are not counted.

Program connections and the brief come from the generator's own program stage, so these rows measure internal consistency. Reachability is not compared with real data: the real datasets draw no door for open connections.

## Realism against matched real floors

| | ours | CubiCasa5k | MagicPlan |
|---|---|---|---|
| net area per storey (m²) | 93.28 | 99.69 | 86.07 |
| rooms | 8.04 | 11.37 | 7.08 |
| corners of the outline | 6.42 | 12.88 | 11.52 |
| outline area / bounding rectangle | 0.84 | 0.83 | 0.85 |
| rooms that are plain rectangles | 0.94 | 0.64 | 0.68 |
| mean room aspect ratio | 2.15 | 1.69 | 1.68 |
| neighbours per room | 2.97 | 3.36 | 2.86 |
| habitable rooms with a window | 1.00 | 0.94 | 0.58 |

Room area in m², median (share of our rooms inside the real p5–p95 range):

| room | ours (n) | CubiCasa5k | MagicPlan |
|---|---|---|---|
| living_room | 23.7 (100) | 19.6 (83%) | 21.7 (100%) |
| kitchen | 12.6 (100) | 10.9 (100%) | 10.0 (100%) |
| bedroom | 14.2 (262) | 11.3 (81%) | 12.4 (99%) |
| bathroom | 5.5 (221) | 3.1 (88%) | 3.7 (100%) |
| hallway | 10.6 (138) | 6.2 (87%) | 6.7 (100%) |
| storage | 5.7 (110) | 3.0 (90%) | 1.6 (95%) |
| study | 10.4 (32) | 9.6 (100%) | 11.2 (100%) |
| laundry | 5.6 (24) | 6.8 (100%) | 4.2 (100%) |
| outdoor | 8.0 (40) | 6.3 (100%) | 4.7 (100%) |

Classifier two-sample test (leave-one-out 1-NN on geometric plan features; 50% = cannot tell apart): ours vs CubiCasa5k 94% (real vs real 48%), ours vs MagicPlan 91% (real vs real 50%); MagicPlan vs CubiCasa5k 91%. The test is very sensitive: two real datasets are about as separable from each other as ours is from either.

Largest differences to CubiCasa5k (by Cohen's d): frac_rect_rooms 0.94 vs 0.64; mean_room_aspect 2.15 vs 1.69; n_outdoor 0.29 vs 1.44; outline_corners 6.42 vs 12.39; share_outdoor 0.02 vs 0.15; n_rooms 8.04 vs 11.37.
Largest differences to MagicPlan (by Cohen's d): mean_room_aspect 2.15 vs 1.68; frac_rect_rooms 0.94 vs 0.68; outline_corners 6.42 vs 11.30; mean_room_convexity 0.99 vs 0.97; n_storage 0.80 vs 0.36; share_storage 0.07 vs 0.02.

## Diversity

Distinct typed room-adjacency topologies per storey: ours 75%, matched CubiCasa5k 99%, matched MagicPlan 98% (mean of 20 draws of 138).

## FID / KID

Inception-V3 features of uniform 256 px layout renders (`fpeval.render.render_layout`: type colours, 4 px walls, no doors or windows), against the matched real storeys. Mean ± std over 5 rounds; each round holds out one matched real storey per generated one, and the held-out set is scored the same way as ours. That row is the floor: what a generator that samples the real data would get at this sample size. KID ×1000; lower is better for both.

| reference | CubiCasa5k (n=2622) | MagicPlan (n=2622) |
|---|---|---|
| FID, ours (n=138) | 86.8 ± 0.1 | 66.3 ± 0.3 |
| FID, held-out real of the same dataset (floor) | 24.9 ± 0.9 | 26.9 ± 0.9 |
| FID, held-out real of the other dataset | 48.6 ± 3.1 | 50.3 ± 3.5 |
| KID ×1000, ours | 80.0 ± 0.6 | 43.8 ± 0.6 |
| KID ×1000, held-out real of the same dataset (floor) | -0.1 ± 0.5 | 0.1 ± 0.6 |
| KID ×1000, held-out real of the other dataset | 29.3 ± 3.2 | 29.5 ± 6.2 |

FID at about 100 samples has a large positive bias (the floor is well above 0) and absolute values are not comparable with numbers in papers, which use other renderers and 1,000–12,000 samples. KID is unbiased.

## Not measured

- Human or VLM realism studies (House-GAN-style pairwise realism).
