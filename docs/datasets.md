# Ground-truth floorplan datasets

Real (and a few synthetic) floorplan datasets that serve as ground truth (GT). Each one is converted to the common format (`docs/floorplan_format.md`).

```
data/<dataset>/raw/                    # untouched download (gitignored)
data/<dataset>/converted/samples/*.json  # common format
data/<dataset>/converted/renders/*.png   # python -m fpeval.render data/<dataset>/converted
data/<dataset>/converted/splits.json     # {"counts": {...}, "splits": {split: [ids]}, "note": ...}
```

Each sample's split is also stored in `condition.split`. To reproduce a dataset:

```bash
bash scripts/datasets/download_<name>.sh                  # or link_local_copies.sh
conda run -n fpe python -m fpeval.datasets.<name> --workers 16
conda run -n fpe python -m fpeval.render data/<name>/converted
```

Converters use `fpe` env packages plus `lxml networkx pandas pyarrow matplotlib scipy`, all installed with pip. Set `OMP_NUM_THREADS=1` on shared machines, because OpenCV otherwise oversubscribes cores.

Official code is added as git submodules under `external/datasets/`: `cubicasa5k` (CubiCasa/CubiCasa5k, SVG loader) and `resplan` (m-agour/ResPlan, loader and graph builder, with the data zip in the repo). Related repos that are already submodules: `external/methods/msd` (MSD constants), `external/methods/tell2design`, `external/data_tools/rplan_toolbox`.

## Overview (checked 2026-10-04)

| Dataset | Link | License | Size (raw) | Access | Status | Converted samples (per split) | Units |
|---|---|---|---|---|---|---|---|
| **CubiCasa5k** | [github](https://github.com/CubiCasa/CubiCasa5k), [Zenodo 2613548](https://zenodo.org/records/2613548) | CC BY-NC-SA 4.0 | 5.5 GB zip (17 GB unzipped) | free download | downloaded + converted | 6,103 floors from 4,992 of 5,000 folders: train 5,112 / val 490 / test 501 | m (SVG units are cm) |
| **ResPlan** | [github](https://github.com/m-agour/ResPlan) (also on Kaggle) | CC BY 4.0 (data), MIT (code) | 100 MB zip | in the git repo | downloaded + converted | 17,000: train 13,053 / val 1,632 / test 1,632 / augmented 683 | m (scale = sqrt(area / inner.area)) |
| **House-GAN LIFULL vectors** | Dropbox link in [ennauata/housegan](https://github.com/ennauata/housegan) README | SFU non-commercial research license (in the zip; no redistribution) | 0.6 GB | free download (public Dropbox link) | downloaded + converted | 145,811 (no official split: "all") | px (256 frame) |
| **Swiss Dwellings v3.0.0** | [Zenodo 7788422](https://zenodo.org/records/7788422) | CC BY 4.0 | 0.9 GB zip (2.7 GB csv) | free download | downloaded + converted | 46,830 apartment-floors (no official split: "all"; site ids kept) | m |
| **Modified Swiss Dwellings (MSD v1)** | [4TU.ResearchData](https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9) (also [Kaggle](https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings)) | CC BY 4.0 | 5.5 GB | free download (4TU; Kaggle needs credentials) | downloaded + converted | 4,167 building floors (train; the test release has no public GT) | m |
| MSD JSON (Jabi) | [Zenodo 17294451](https://zenodo.org/records/17294451) | CC BY 4.0 | 0.3 GB | free download | downloaded only (a TopologicPy re-export of MSD v6 with 4,572 plans; redundant with MSD) | - | - |
| **Tell2Design (RPLAN-derived)** | [github](https://github.com/LengSicong/Tell2Design), Google Drive zip | CC BY-NC 4.0 | 0.7 GB zip (2.9 GB extracted, without the 16.6 GB pickle) | free download (Drive, gdown) | downloaded + converted | 80,788: train 76,785 / train_human 1,695 / test 2,308 | px (256 frame) |
| **RPLAN (Graph2Plan .mat)** | local copy `/datawaha/cggroup/datasets/RPLAN/Network/data` | RPLAN terms (form) | 134 MB | local copy (symlink) | converted | 80,436: train 56,305 / val 12,068 / test 12,063 (293 corrupt plans skipped) | px, 18/256 m/px |
| **MagicPlan (PuzzleFusion)** | Drive folder in [sepidsh/PuzzleFussion](https://github.com/sepidsh/PuzzleFussion) README | same as code: non-commercial research, GPLv3 terms | 1.9 GB | free download (Drive folder, gdown) | downloaded + converted (reader-ready "SFU" floors) | 155,229 train + 100 test floors | m |
| ProcTHOR-10K | [allenai/procthor-10k](https://github.com/allenai/procthor-10k) | Apache-2.0 | 63 MB (git-LFS) | free download | downloaded + converted (**synthetic**, procedurally generated) | 12,000: train 10,000 / val 1,000 / test 1,000 | m |
| RPLAN (original PNGs) | [project page](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) | research only | ~80k PNGs | Google Form | documented only (form; the Graph2Plan .mat and Tell2Design copies above cover it) | - | - |
| LIFULL HOME'S (original) / Raster-to-Graph annotations | [NII IDR](https://www.nii.ac.jp/dsc/idr/lifull), [SizheHu/Raster-to-Graph](https://github.com/SizheHu/Raster-to-Graph) | NII IDR research agreement | 5M images | application form | documented only (form) | - | - |
| R2V / Raster-to-Vector | [art-programmer/FloorplanTransformation](https://github.com/art-programmer/FloorplanTransformation) | research (images are LIFULL, not shared) | ~870 annotated + 100k generated vectors (Drive) | annotations in the repo, 100k vectors on Drive | documented only (wall/door line format, no images; House-GAN's vectors above come from the same pipeline) | - | - |
| Structured3D | [github](https://github.com/bertjiazheng/Structured3D) | Structured3D Terms of Use | ~3.5k houses | agreement form | documented only (form) | - | - |
| ZInD (Zillow Indoor) | [github](https://github.com/zillow/zind) | ZInD Terms of Use (academic only) | ~40 GB | Bridge account + terms | documented only (registration and terms) | - | - |
| MLSTRUCT-FP | [github](https://github.com/MLSTRUCT/MLStructFP) | MIT (per repo) | 954 floors | form for the download link | documented only (form) | - | - |
| pseudo-floor-plan-12k | local copy `/datawaha/cggroup/datasets/floor_plans/pseudo-floor-plan-12k` (HF parquet) | n/a | 3.9 GB | local copy | documented only: **procedurally generated** raster images (Grasshopper + PlanFinder), with no room polygons or labels, so not real GT | - | - |
| Vienna apartment JPGs | local copy `/datawaha/cggroup/datasets/floorplans` | unknown | 232 JPGs | local copy | documented only: raster scans with no annotations | - | - |

A local copy of CubiCasa5k also exists at `/datawaha/cggroup/datasets/CubiCasa5k` (archive.zip plus `cubicasa5k/cubicasa5k/` with all 5,000 folders, and a COCO version). We converted from our own Zenodo download, whose md5 `0ce0b203d1e3c125b51087b219bd23b9` matches Zenodo. The local copy has the same folder counts (276 / 992 / 3,732).

Nothing exceeded the 200 GB limit. The largest raw download is CubiCasa5k (17 GB unzipped), followed by MSD (18 GB unzipped).

## Per-dataset notes

### CubiCasa5k (`fpeval/datasets/cubicasa5k.py`)
- There is one sample per `Floorplan Floor-k` group in `model.svg`, with id `<subset>_<n>_f<k>`, because multi-storey houses are drawn side by side. The geometry aligns with `F1_scaled.png`.
- Rooms come from the `Space <Type> [<Subtype>]` polygons. `type_native` holds the full class (e.g. `Outdoor Balcony`) and `room["label"]` holds the visible Finnish name label. Overlay spaces (`Below150cm`, `OpenToBelow`) are dropped. `UserDefined` and `Room` spaces are typed from their Finnish label (`H` maps to bedroom, `TUPA` to living room, and so on).
- Type mapping: `Outdoor` with subtype Balcony, Terrace or Veranda maps to balcony; other `Outdoor` maps to outdoor; `Utility` maps to laundry; `DraughtLobby` and `Entry` map to entrance; `Den` (mostly the fireplace room) maps to living_room.
- About 11k spaces are `Undefined` (their label is literally "UNDEFINED"). They become `other`, which is about 20% of all rooms. Some plans (mostly in the `colorful` subset) are entirely Undefined.
- Walls, railings, doors and windows are kept as polygons. There is no room graph.
- Scale: every space carries a hidden imperial dimension label. Label length divided by extent gives 0.0100 ± 0.0001 m/px on all checked plans, so coordinates are multiplied by 0.01 (metres). The per-floor estimate is stored in `condition.m_per_px_est`.
- Eight folders have no `Space` elements and are skipped. They are listed in `splits.json["failed_folders"]`. Three folders are not listed in any split file.

### ResPlan (`resplan.py`)
- Rooms: living (all parts unioned, as in `resplan_utils.plan_to_graph`), kitchen, bedroom, bathroom, balcony, storage and stair, plus garden, parking and pool as outdoor or garage. Doors, front door, windows, walls and the `inner` boundary are kept.
- Graph: `plan_to_graph` + `add_adjacency_edges` (the paper's definition), with edges labelled door, adjacent or window.
- The geometry is normalized to 256 units of plan width. The metric scale is sqrt(area_m2 / inner.area); with it, 99.3% of wall depths fall in 10–40 cm, as the README states. Coordinates are written in metres, and the native scale is stored in `condition.native_m_per_unit`.

### House-GAN LIFULL vectors (`housegan_lifull.py`)
- Converted from `housegan_clean_data.npy` (145,811 plans; the README says 145,811). `housegan_clean_data.pkl` in the Dropbox zip is truncated upstream. `dataset_paper/{train,valid}_data.npy` hold boxes only, from a different processing run that does not match.
- Room polygons are rebuilt by polygonizing each room's edges, falling back to the bbox. Doors are thin rectangles on door edges. The graph has door edges plus adjacent edges for shared walls.
- Classes: living, kitchen, bedroom, bathroom, missing (mapped to other), closet, balcony, corridor, dining, laundry.
- **License:** the zip contains an SFU license (non-commercial research, no redistribution) that says downloading implies acceptance. Make sure your use complies.

### Swiss Dwellings v3.0.0 (`swiss_dwellings.py`)
- There is one sample per residential `(apartment_id, floor_id)`. Rooms are `area` entities (SHAFT, VOID, LIGHTWELL, OUTDOOR_VOID and AIR are dropped), doors and windows are `opening` entities, and walls are `separator` entities. Coordinates are metres with y flipped.
- `ROOM` (generic "Zimmer") maps to bedroom, following MSD. `LIVING_DINING` maps to living_room. Loggia, terrace and wintergarten map to balcony. `STAIRCASE` maps to other.
- There is no official split. Ids for site, building, plan, unit and apartment are kept in `condition`, so site-disjoint splits are possible.

### MSD (`msd.py`)
- Uses the 4TU release (MSD v1, the ECCV'24 / ICCV'23 challenge). The `graph_out/*.pickle` files are networkx graphs with room polygons in metres and an access graph (door, entrance, passage). The test release has inputs only.
- The pickles store centroids as torch tensors. They are unpickled with torch stubbed out, so torch is not needed.
- Samples are whole building floors (multi-apartment), unlike every other dataset here.

### Tell2Design (`tell2design.py`)
- 80,788 RGB renders of RPLAN plans (same count as RPLAN). The colour-to-type mapping was recovered from the T5 annotation boxes of the 2,308 eval plans, where each colour matched one label in more than 97% of cases.
- Child, second and guest rooms are merged by T2D into "common room", which maps to bedroom. One rare yellow room colour could not be identified and maps to `other` with type_native `unknown_yellow`.
- Interior doors are not drawn, so the graph has adjacent edges only (rooms within one wall). The front door is red.
- Splits: `test` is the eval set (human instructions). `train_human` is the 2nd-finetune set minus the 551 plans that also appear in the eval set. Everything else is `train`.

### RPLAN via Graph2Plan (`rplan_graph2plan.py`)
- Uses Graph2Plan's processed RPLAN: room polygons (`rBoundary`), `rType`, the boundary with the front door in its first two points, and `rEdge` (spatial relations, written as adjacent edges; relation ids are in `condition.relations`).
- Plans with empty or out-of-frame polygons are skipped and counted in `splits.json["skipped_corrupt"]`.
- This is the closest thing to RPLAN GT for the RPLAN-trained methods. Ids `<split>_<rplan_name>` let you join against RPLAN names.

### MagicPlan (`magicplan.py`)
- Floors captured with the MagicPlan AR app. Rooms come with small gaps and overlaps, and the data includes some non-residential spaces (offices, MRI rooms).
- Converted from the reader-ready `SFU_train_samples.pickle` and `sfu_test.pickle`. The larger `SFU_train_set_not_processed.pickle` and the PuzzleFusion npz files are downloaded but not converted.
- y is negated. Doors and windows become 10 cm rectangles. The graph links rooms whose door segments are within 30 cm of each other.

### ProcTHOR-10K (`procthor10k.py`)
- **Synthetic** (procedural). It is included as a reference distribution only. DS2D trains on it.

## Manual steps for datasets that are not auto-downloadable
- RPLAN originals: fill in the Google Form on the RPLAN project page. The Graph2Plan .mat files are already available locally.
- MSD via Kaggle: needs `~/.kaggle/kaggle.json`. This is not needed, because the 4TU mirror is identical.
- Structured3D, ZInD, LIFULL HOME'S (NII IDR) / Raster-to-Graph and MLSTRUCT-FP each require an agreement or registration form, which the user has to fill in.
