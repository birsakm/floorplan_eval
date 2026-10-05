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


def _v(condition, ref, scale=RPLAN_256, scale_known=True, frame_aligned=False, raster=False):
    return dict(condition=condition, ref=ref, scale=scale, scale_known=scale_known,
                frame_aligned=frame_aligned, raster=raster)


VARIANTS = {
    "housegan/lifull_bubble_testD": _v("bubble", "housegan_lifull:all", scale_known=False, raster=True),
    "houseganpp/rplan-g2p_bubble_test1000_syndoors": _v("bubble", "rplan_graph2plan:test", raster=True),
    "houseganpp/rplan_bubble_public": _v("bubble", "rplan_graph2plan:test", raster=True),
    "house_diffusion/rplan-g2p_bubble_test1000_syndoors": _v("bubble", "rplan_graph2plan:test"),
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
    **{f"ds2d/procthor_{c}_test_lora-{l}": _v("constraints", "procthor10k:test", scale=1.0)
       for c in ("bubble_constraints", "constraints") for l in ("fullprompt", "mask", "presetmask")},
}
