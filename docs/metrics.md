# Evaluation metrics

Code: `fpeval/metrics.py` (per-sample and paired), `fpeval/distribution.py` (set-level), `fpeval/rules.py` (thresholds), `fpeval/eval_config.py` (per-variant scale, condition, reference set).

```bash
export PYTHONPATH=. OMP_NUM_THREADS=1
conda run -n fpe python -m fpeval.evaluate refs        # GT reference sets (~2 min)
conda run -n fpe python -m fpeval.evaluate variants    # all generated variants (~15 min, 1 GPU for FID)
conda run -n fpe python -m fpeval.evaluate report      # results/summary.{md,csv}
```

The per-sample CSVs are written to `outputs/<method>/<variant>/eval/` and `data/<dataset>/converted/eval/<split>/`.

## Units and tolerances

All metrics work in meters, using the `scale` (m per coordinate unit) set in `eval_config.py`:

| Sources | Meters per unit |
|---|---|
| RPLAN-derived | 18/256 (MaskPLAN 18/128, ChatHouseDiffusion 18/64) |
| ProcTHOR, CubiCasa5k, ResPlan, Swiss Dwellings, MSD, MagicPlan | 1 (already in meters) |
| LIFULL | unknown; a nominal 18/256 is used only for tolerances, and the metric-unit rules (area, width, m² distances) are skipped |

- **Walls:** rooms in real plans are separated by walls, so two rooms closer than 2 × 0.3 m count as touching, not as having a gap (`WALL_HALF_THICKNESS_M`). Sources derived from RPLAN images (DS2D, Tell2Design, ChatHouseDiffusion, House-GAN++ JSON) leave gaps of up to about 0.6 m between rooms. A smaller tolerance splits their plans into one component per room, while the clean RPLAN polygons stay at 1 component for any tolerance.
- **Adjacency:** two rooms are adjacent when they share at least 0.5 m of wall (`MIN_SHARED_WALL_M`).

## 1. Rule metrics (per sample, no GT needed)

These follow the PI's floorplan rules (size, shape, overlap). The same metrics run on every GT dataset, so each number has a real-data baseline (first table in `results/summary.md`).

| Metric | Definition | Ideal |
|---|---|---|
| `n_rooms` | rooms with non-zero area | ≈ GT |
| `overlap_ratio` | (Σ room areas − union area) / union area | 0 |
| `hole_ratio` | area of enclosed holes wider than a wall / filled footprint | 0 |
| `boundary_coverage` | share of the input boundary covered by rooms (after wall closing); boundary-conditioned only | 1 |
| `outside_boundary_ratio` | share of room area outside the boundary (+ wall tolerance) | 0 |
| `adj_components` | connected components of the room adjacency graph | 1 |
| `frac_isolated_rooms` | rooms adjacent to no other room | 0 |
| `frac_nonrectilinear_rooms` | rooms with < 98% of perimeter within 5° of the plan's dominant orientation. Edges < 0.25 m are ignored. Not reported for raster-output methods (see caveats). | ≈ GT |
| `mean_convexity` | room area / convex hull area | ≈ GT |
| `frac_area_violation` | rooms below the minimum area for their type (`MIN_AREA_M2`) | ≈ GT |
| `frac_width_violation` | rooms whose largest inscribed circle is narrower than the type minimum (`MIN_WIDTH_M`) | ≈ GT |
| `frac_aspect_violation` | rooms whose min-rectangle aspect exceeds the type maximum (`MAX_ASPECT`) | ≈ GT |
| `any_rule_violation` | sample has overlap > 2%, holes > 2%, > 1 component, or any area/width/aspect violation | ≈ GT |

The thresholds are loose "a person or bed must fit" defaults, not a building code. Real RPLAN plans violate them in 17% of samples, mostly through galley kitchens narrower than 1.5 m (about half of the violating rooms) and narrow balconies, so compare against the GT row rather than against 0.

## 2. Paired metrics (sample vs its GT, same input)

Pairing is by sample id. For methods that draw several samples per input, the `_k` suffix is stripped.

| Metric | Definition |
|---|---|
| `yield` | generated samples / (GT inputs × samples per input); captures failed or unparseable generations |
| `room_count_match` | same number of rooms as GT |
| `type_multiset_match` | same multiset of room types as GT |
| `adjacency_f1` | F1 between the multisets of adjacent room-type pairs (e.g. living–bedroom) of generated and GT. No room matching is needed, so it applies to every condition type. |
| `type_miou` / `pixel_type_acc` | per-type IoU / pixel accuracy of rasterized type maps (256²) in the GT frame. Only computed when `frame_aligned` (the boundary was the input). |

## 3. Distribution metrics (generated set vs reference GT set)

The reference is the GT test split of the matching dataset (`ref` in `eval_config.py`).

| Metric | Definition |
|---|---|
| `fid`, `kid_x1000` | FID and unbiased KID (×10³) on Inception-v3 pool features of rooms-only renders (256², same palette, no boundary/doors/windows, each plan scaled to fit) |
| `fid_vs_gt` | FID against the variant's own GT set (same inputs and sample count, so sample-size bias cancels) |
| `fid_gt_vs_ref` | floor: FID of that GT set against the reference |
| `dist_room_count_tv` | total variation distance between room-count histograms |
| `dist_room_type_jsd` | Jensen–Shannon divergence between room-type frequencies |
| `dist_area_frac_w1`, `dist_aspect_w1`, `dist_convexity_w1` | 1-Wasserstein distance between per-room area fraction, aspect (clipped to [1, 10]) and convexity distributions |
| `dist_room_area_m2_w1`, `dist_total_area_m2_w1` | 1-Wasserstein distance between room / plan areas in m² (only when both scales are known) |

## Caveats

- **FID sample size.** FID is biased by sample size. Compare `fid_vs_gt` within a reference set, and prefer KID for small sets.
- **Merged room types.** GSDiff and some datasets merge types (e.g. study → bedroom), so type JSD and the type-match metrics penalise that by design.
- **Raster outputs.** House-GAN, House-GAN++, MaskPLAN and ChatHouseDiffusion output rasters. Their converters vectorize them with `cv2.approxPolyDP`, which cuts staircase corners into long slanted edges. The models' pixel grids are rectilinear, but the polygons read as 25–28% non-rectilinear, so `frac_nonrectilinear_rooms` is blanked for these variants (`raster=True` in `eval_config.py`). Tell2Design GT is vectorized the same way. TODO: one shared pixel-edge polygonizer for all raster converters.
- **Wall gaps.** RPLAN GT polygons from Graph2Plan have no wall gaps. Methods that leave wall-width gaps are handled by the wall tolerance, but wider gaps count as holes.
- **Made-up doors.** For House-GAN++ and HouseDiffusion on the shared 1000-plan RPLAN test set, the interior doors in the input were made up (Graph2Plan data has none). See `scripts/houseganpp/NOTES.md`.
