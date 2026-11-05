"""What each generated variant is, and which GT set it is compared against.

scale: meters per coordinate unit. scale_known=False means the scale is only a
nominal guess used for tolerances; metric-unit rules are skipped.
frame_aligned: generated and GT samples share a coordinate frame (boundary
given as input), so per-type IoU against GT is meaningful.
raster: output is a raster vectorized with cv2.approxPolyDP, which cuts
staircase corners into slanted edges; rectilinearity is not reported for these.
"""

RPLAN_256 = 18 / 256  # RPLAN images: 256 px = 18 m

REFERENCES = {
    # name: (converted dir, split | [splits] | None for all, scale, scale_known, max samples)
    "rplan_graph2plan:test": ("data/rplan_graph2plan/converted", "test", RPLAN_256, True, 12063),
    "tell2design:test": ("data/tell2design/converted", "test", RPLAN_256, True, 2308),
    "housegan_lifull:all": ("data/housegan_lifull/converted", None, RPLAN_256, False, 10000),
    "procthor10k:test": ("data/procthor10k/converted", "test", 1.0, True, 1000),
    # CubiCasa5k: all 5,000 plans (6,103 floors) as one evaluation set.
    "cubicasa5k:all": ("data/cubicasa5k/converted", None, 1.0, True, 10000),
    # ResPlan without its 683 augmented copies.
    "resplan:all": ("data/resplan/converted", ["train", "val", "test"], 1.0, True, 20000),
    "swiss_dwellings:all": ("data/swiss_dwellings/converted", None, 1.0, True, 50000),
    "msd:train": ("data/msd/converted", "train", 1.0, True, 5000),
    "magicplan:train": ("data/magicplan/converted", "train", 1.0, True, 10000),
}

# Datasets compared in the dataset statistics (results/datasets/), in display order.
# kind: "real" or "synthetic"; unit: what one sample is.
STAT_DATASETS = {
    "cubicasa5k:all": dict(label="CubiCasa5k", kind="real", unit="floor"),
    "resplan:all": dict(label="ResPlan", kind="real", unit="apartment"),
    "rplan_graph2plan:test": dict(label="RPLAN", kind="real", unit="apartment"),
    "swiss_dwellings:all": dict(label="Swiss Dwellings", kind="real", unit="apartment floor"),
    "magicplan:train": dict(label="MagicPlan", kind="real", unit="floor"),
    "msd:train": dict(label="MSD", kind="real", unit="building floor (many apartments)"),
    "housegan_lifull:all": dict(label="LIFULL", kind="real", unit="apartment, scale unknown"),
    "procthor10k:test": dict(label="ProcTHOR-10K", kind="synthetic", unit="house"),
}


# FID/KID reference per evaluation set: a split the generated plans' inputs never come from, so the GT plans of
# those inputs (the floor) are not in the reference. (split, max samples); LIFULL has no splits, so about 3% of
# House-GAN's GT plans are expected in its reference.
FID_N = 1000  # generated plans per variant (random subset of larger variants)
FID_REFS = {
    "rplan_graph2plan:test": ("val", 5000),
    "tell2design:test": ("train", 5000),
    "procthor10k:test": ("train", 5000),
    "housegan_lifull:all": (None, 5000),
    "swiss_dwellings:all": (None, 5000),
    "cubicasa5k:all": (None, 5000),
}
def _v(condition, ref, scale=RPLAN_256, scale_known=True, frame_aligned=False, raster=False):
    return dict(condition=condition, ref=ref, scale=scale, scale_known=scale_known,
                frame_aligned=frame_aligned, raster=raster)


VARIANTS = {
    "housegan/lifull_bubble_testD": _v("bubble", "housegan_lifull:all", scale_known=False, raster=True),
    "houseganpp/rplan-g2p_bubble_test1000_syndoors": _v("bubble", "rplan_graph2plan:test", raster=True),
    # Table 1(a) since 2026-11-04: the same inputs with RPLAN's real interior doors (scripts/rplan/rplan_doors.py)
    "houseganpp/rplan-g2p_bubble_test1000_realdoors": _v("bubble", "rplan_graph2plan:test", raster=True),
    "houseganpp/rplan_bubble_public": _v("bubble", "rplan_graph2plan:test", raster=True),
    "house_diffusion/rplan-g2p_bubble_test1000_syndoors": _v("bubble", "rplan_graph2plan:test"),
    "house_diffusion/rplan-g2p_bubble_test1000_realdoors": _v("bubble", "rplan_graph2plan:test"),
    "house_diffusion/rplan_bubble_public": _v("bubble", "rplan_graph2plan:test"),
    "gsdiff/rplan_uncond": _v("unconditional", "rplan_graph2plan:test"),
    "gsdiff/rplan_bubble_test1000": _v("bubble", "rplan_graph2plan:test"),
    "gsdiff/rplan_boundary_test1000": _v("boundary", "rplan_graph2plan:test", frame_aligned=True),
    "iplan/rplan_boundary_test1000": _v("boundary", "rplan_graph2plan:test", frame_aligned=True),
    "diffplanner/rplan_boundary_test": _v("boundary", "rplan_graph2plan:test", frame_aligned=True),
    "wallplan/rplan_boundary_test1000": _v("boundary", "rplan_graph2plan:test", frame_aligned=True),
    "wallplan/bundled_boundary_test500": _v("boundary", "rplan_graph2plan:test"),
    "maskplan/rplan_boundary": _v("boundary", "rplan_graph2plan:test", scale=18 / 128, frame_aligned=True, raster=True),
    "maskplan/rplan_partial25": _v("boundary+partial", "rplan_graph2plan:test", scale=18 / 128, frame_aligned=True, raster=True),
    "chathousediffusion/t2d_test_textgraph": _v("text+boundary", "tell2design:test", scale=18 / 64, frame_aligned=True, raster=True),
    **{f"ds2d/rplan{n}R_bubble_roomarea_test": _v("bubble+areas", "rplan_graph2plan:test") for n in (5, 6, 7, 8)},
    # Floor-plan RLVR: polygons in metres in the RPLAN image frame, converted to 256 px (scale 18/256).
    # Rooms are drawn with wall gaps (like DS2D); no boundary input, so not frame_aligned.
    "floorplan_rlvr/rplan-g2p_bubble_areas_test1000_syndoors": _v("bubble+areas", "rplan_graph2plan:test"),
    "floorplan_rlvr/rplan-g2p_bubble_areas_test1000_syndoors_sft": _v("bubble+areas", "rplan_graph2plan:test"),
    # Ours (Mason's old program, data/ours): metric polygons, input is a brief (net area + room counts).
    "ours/mason_old_apartment_n100_seed0": _v("brief", "swiss_dwellings:all", scale=1.0),
    # Houses: one sample per storey (<home>_s<k>); rooms the State replay rejected are restored (scripts/ours/convert.py).
    "ours/mason_old_house_n100_seed0": _v("brief", "cubicasa5k:all", scale=1.0),
    # Mason2 (current pipeline): registered automatically per generator version, see mason2_variants() below.
    # Infinigen Indoors floor plan solver given our programs (scripts/infinigen/generate.py): same ids and conditions
    # as ours, so it is evaluated like ours (scripts/ours/evaluate_ours.py --variant infinigen/...).
    "infinigen/apartment_program_n100_seed0": _v("brief+program", "swiss_dwellings:all", scale=1.0),
    "infinigen/house_program_n100_seed0": _v("brief+program", "cubicasa5k:all", scale=1.0),
    # Infinigen's default home setting (free outline): the annealing grows homes to ~2.3x the brief's area.
    "infinigen/apartment_program_freecontour_n100_seed0": _v("brief+program", "swiss_dwellings:all", scale=1.0),
    "infinigen/house_program_freecontour_n100_seed0": _v("brief+program", "cubicasa5k:all", scale=1.0),
    # ProcTHOR layout stage given our programs (scripts/procthor/generate.py): same ids and conditions as ours.
    "procthor/apartment_program_n100_seed0": _v("brief+program", "swiss_dwellings:all", scale=1.0),
    "procthor/house_program_n100_seed0": _v("brief+program", "cubicasa5k:all", scale=1.0),
    # Table 1(a): both baselines on the RPLAN bubble diagrams of the 1,000 Graph2Plan test inputs, with the real rooms'
    # areas as targets (scripts/rplan/make_programs.py, --source rplan/g2p_test1000_programs).
    "procthor/rplan-g2p_program_test1000": _v("bubble+areas", "rplan_graph2plan:test", scale=1.0),
    "infinigen/rplan-g2p_program_test1000": _v("bubble+areas", "rplan_graph2plan:test", scale=1.0),
    # --source rplan/g2p_test1000_realdoors_programs (real RPLAN doors)
    "procthor/rplan-g2p_program_test1000_realdoors": _v("bubble+areas", "rplan_graph2plan:test", scale=1.0),
    "infinigen/rplan-g2p_program_test1000_realdoors": _v("bubble+areas", "rplan_graph2plan:test", scale=1.0),
    # Holodeck floor-plan stages with open-weight LLMs on the same programs, written as text (scripts/holodeck/generate.py)
    "holodeck/rplan-g2p_text_test1000_realdoors_qwen3.8-27b": _v("bubble+areas", "rplan_graph2plan:test", scale=1.0),
    "holodeck/rplan-g2p_text_test1000_realdoors_gemma-4-31b": _v("bubble+areas", "rplan_graph2plan:test", scale=1.0),
    # Blocks (b, c): Holodeck given the brief of a Mason2 version as text (houses: one query per storey)
    "holodeck/apartment_mason2-next-living-bay-ad5110bb7b58_qwen3.8-27b": _v("brief", "swiss_dwellings:all", scale=1.0),
    "holodeck/apartment_mason2-next-living-bay-ad5110bb7b58_gemma-4-31b": _v("brief", "swiss_dwellings:all", scale=1.0),
    "holodeck/house_mason2-next-multistory-nav-4648dcf51a12_qwen3.8-27b": _v("brief", "cubicasa5k:all", scale=1.0),
    "holodeck/house_mason2-next-multistory-nav-4648dcf51a12_gemma-4-31b": _v("brief", "cubicasa5k:all", scale=1.0),
    # SceneSmith's 31 published house-level scenes (scripts/scenesmith/convert.py): text prompts, only 7 of them homes.
    "scenesmith/house_published": _v("text", "cubicasa5k:all", scale=1.0),
    **{f"ds2d/procthor_{c}_test_lora-{l}": _v("constraints", "procthor10k:test", scale=1.0)
       for c in ("bubble_constraints", "constraints") for l in ("fullprompt", "mask", "presetmask")},
}


# --- Mason2 generator versions (scripts/ours2/run_all.py) -----------------------------------------------------------
# Every converted version (outputs/ours/mason2_<version>/) is registered with both baselines run on its programs.
# The reference set is the version's run_info.json "reference" (default: all Swiss Dwellings).
MASON2_DEFAULT_REF = "swiss_dwellings:all"
_MASON2_LEGACY = {"next-living-bay-ad5110bb7b58": "apartment_mason2-livingbay_seed0"}  # first run, old naming


def mason2_variants(version, kind="apartment"):
    """Variant keys of one Mason2 generator version: ours and the two baselines on the same programs."""
    base = _MASON2_LEGACY.get(version, f"{kind}_mason2-{version}_seed0")
    return {"ours": f"ours/mason2_{version}", "procthor": f"procthor/{base}", "infinigen": f"infinigen/{base}"}


def _register_mason2():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for d in sorted((root / "outputs" / "ours").glob("mason2_*")):
        version = d.name[len("mason2_"):]
        try:
            info = json.load(open(d / "run_info.json"))
        except (OSError, ValueError):
            info = {}
        ref = info.get("reference", MASON2_DEFAULT_REF)
        if info.get("benchmark"):  # generated from Table 1(a) programs (scripts/rplan/export_mason_programs.py)
            VARIANTS.setdefault(f"ours/mason2_{version}", _v("bubble+areas", ref, scale=1.0))
            continue
        for m, key in mason2_variants(version, info.get("kind", "apartment")).items():
            VARIANTS.setdefault(key, _v("brief" if m == "ours" else "brief+program", ref, scale=1.0))


_register_mason2()
