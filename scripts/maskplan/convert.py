"""Convert MaskPLAN outputs to the common format.

  conda run -n fpe python scripts/maskplan/convert.py outputs/maskplan/<variant> [--gt]

Generated: raw/native/*/iteration/post/<site>.png (128x128 BGRA, rooms painted with the upstream
T_list colours after the upstream rectangularisation/alignment post-process) -> samples/<site>.json.
Per-room-type masks are vectorized (one polygon per connected component). raw/meta/<site>.npz
(partial input M_* and predicted attributes In_*) is summarised in `condition`.

GT (--gt) -> gt_samples/ + gt_renders/: MaskPLAN site ids are RPLAN file ids, so the exact RPLAN
room polygons are taken from the local Graph2Plan-format RPLAN copy (data_{train,valid,test}.mat,
256 px grid, scaled by 0.5 to MaskPLAN's 128 px frame; full RPLAN room labels). Sites missing from
that copy fall back to MaskPLAN's own preprocessed boxes (Processed_data/RPLAN_input_vec.npz,
20-level quantised boxes, exactly as upstream Inference/Generate_GroundTruth.py --format vec).
Boundary for generated samples = upstream load_boundary_points() on parsed_img/img_room_sqe/0/<site>.png.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # avoid thread oversubscription on the shared machine
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2

cv2.setNumThreads(0)
import numpy as np

from fpeval.format import canonical_room_type, make_sample, save_sample
from fpeval.render import render

SOURCE = "maskplan"
ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "external/methods/maskplan"
# MaskPLAN type ids (Processed_data/RPLAN_DataProcess.ipynb map_type / Names_sim_Lib)
TYPE_NAMES = {1: "Living", 2: "Bathroom", 3: "Storage/closet", 4: "Bed room", 5: "Kitchen", 6: "Dining", 7: "Balcony"}
EXTRA = {"Living": "living_room", "Storage/closet": "storage", "Bed room": "bedroom", "Dining": "dining_room"}
# Inference script T_list (written by cv2 -> read back identically with IMREAD_UNCHANGED)
T_LIST = [[255, 255, 255, 255], [255, 255, 0, 255], [255, 0, 255, 255], [0, 255, 255, 255],
          [0, 0, 255, 255], [255, 0, 0, 255], [0, 255, 0, 255], [127, 127, 255, 255], [127, 255, 127, 255]]
UP = 8  # contour on an 8x nearest-upsampled mask -> polygon follows pixel edges (within 1/16 px)


def mask_polygons(mask, min_px=3, eps=0.75):
    polys = []
    n, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    for k in range(1, n):
        comp = cc == k
        if comp.sum() < min_px:
            continue
        big = cv2.resize(comp.astype(np.uint8), None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), eps * UP, True)[:, 0, :].astype(float)
        if len(c) >= 3:
            polys.append([[round(x, 3), round(y, 3)] for x, y in (c + 0.5) / UP])
    return polys


def load_boundary(site, parsed_img):
    """Same as Inference/inference_utils.load_boundary_points (inlined: no TF import needed)."""
    image = cv2.imread(str(parsed_img / "img_room_sqe/0" / f"{site}.png"), cv2.IMREAD_UNCHANGED)
    ch = image[:, :, -1].copy() if image.ndim == 3 else image.copy()
    ch[ch > 100] = 255
    _, th = cv2.threshold(ch, 120, 255, 0)
    cs, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.squeeze(cv2.approxPolyDP(max(cs, key=cv2.contourArea), ch.shape[0] / 128, True))
    return [[float(x), float(y)] for x, y in pts]


def condition_from_meta(meta_path, variant):
    cond = {"type": "boundary+partial_attributes", "ref_id": f"rplan_{meta_path.stem}",
            "dataset": "RPLAN test split (MaskPLAN Processed_data/Test_set.npy)"}
    if meta_path.exists():
        m = np.load(meta_path)
        given_types = [int(t) for t in m["M_T"][0, 1:] if 0 < t < 8]
        cond["given_types"] = [TYPE_NAMES[t] for t in given_types]
        cond["n_given"] = {"T": int((m["M_T"][0, 1:] > 0).sum() - (m["M_T"][0, 1:] == 8).sum()),
                           "L": int((m["M_L"][0, 1:] != 0).any(-1).sum()),
                           "S": int((m["M_S"][0, 1:] != 0).sum()),
                           "R": int((m["M_R"][0, 1:] != 0).any(-1).sum()),
                           "A_rows": int((m["M_A"][0, 1:] != 0).any(-1).sum())}
        cond["predicted_types"] = [TYPE_NAMES.get(int(t), str(int(t))) for t in m["In_T"][0, 1:] if 0 < t < 8]
    return cond


def gen_sample(post_png, variant, parsed_img, meta_dir):
    site = post_png.stem
    img = cv2.imread(str(post_png), cv2.IMREAD_UNCHANGED)
    rooms = []
    for t in range(1, 8):
        m = np.all(img == np.array(T_LIST[t]), axis=-1)
        for poly in mask_polygons(m):
            rooms.append({"type": canonical_room_type(TYPE_NAMES[t], EXTRA), "type_native": TYPE_NAMES[t], "polygon": poly})
    return make_sample(SOURCE, variant, site, rooms, boundary=load_boundary(site, parsed_img),
                       condition=condition_from_meta(meta_dir / f"{site}.npz", variant))


def gt_sample(site, variant, parsed_img, T, R, bb):
    site = int(site)
    t = T[site]
    num_room = (t == 8).argmax() - 1
    if num_room <= 0:
        num_room = 8
    loc = 20
    c = R[site, 1:num_room + 1].astype(float)
    x_min = ((c[:, 0] + 1) / loc) * (bb[site, 3] - bb[site, 1]) + bb[site, 1]
    y_min = ((c[:, 1] + 1) / loc) * (bb[site, 2] - bb[site, 0]) + bb[site, 0]
    x_max = ((c[:, 2] + 1) / loc) * (bb[site, 3] - bb[site, 1]) + bb[site, 1]
    y_max = ((c[:, 3] + 1) / loc) * (bb[site, 2] - bb[site, 0]) + bb[site, 0]
    rooms = []
    for i in range(num_room):
        ty = int(t[i + 1])
        if ty not in TYPE_NAMES:
            continue
        # upstream draws LineString([[y_min, x_min], [y_max, x_max]]).envelope with cv2 (x, y) = (y_*, x_*)
        X0, X1 = sorted([y_min[i], y_max[i]])
        Y0, Y1 = sorted([x_min[i], x_max[i]])
        if X1 - X0 < 1e-6 or Y1 - Y0 < 1e-6:
            continue
        poly = [[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]]
        rooms.append({"type": canonical_room_type(TYPE_NAMES[ty], EXTRA), "type_native": TYPE_NAMES[ty],
                      "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in poly]})
    return make_sample(SOURCE, variant, str(site), rooms, boundary=load_boundary(site, parsed_img),
                       condition={"type": "ground_truth", "ref_id": f"rplan_{site}",
                                  "note": "room axis-aligned bounding boxes quantised to a 20-level grid (MaskPLAN vec data); rooms may overlap"})


G2P_DIR = Path("/datawaha/cggroup/datasets/RPLAN/Network/data")
RPLAN_LABELS = ["LivingRoom", "MasterRoom", "Kitchen", "Bathroom", "DiningRoom", "ChildRoom", "StudyRoom",
                "SecondRoom", "GuestRoom", "Balcony", "Entrance", "Storage", "Wall-in"]
RPLAN_EXTRA = {"Wall-in": "closet", "ChildRoom": "bedroom", "StudyRoom": "study", "GuestRoom": "bedroom"}


def load_g2p(ids):
    import scipy.io as sio
    out = {}
    for split in ("test", "valid", "train"):
        f = G2P_DIR / f"data_{split}.mat"
        if not f.exists():
            continue
        for r in sio.loadmat(f, squeeze_me=True, struct_as_record=False)["data"]:
            if int(r.name) in ids:
                out[int(r.name)] = (r, split)
    return out


def gt_sample_g2p(site, variant, rec):
    r, split = rec
    rtypes = np.atleast_1d(r.rType)
    rbs = r.rBoundary if len(rtypes) > 1 else [r.rBoundary]
    rooms = []
    for t, rb in zip(rtypes, rbs):
        rb = np.asarray(rb)
        if rb.ndim != 2 or len(rb) < 3:  # a few G2P records have empty room boundaries
            continue
        native = RPLAN_LABELS[int(t)]
        rooms.append({"type": canonical_room_type(native, RPLAN_EXTRA), "type_native": native,
                      "polygon": [[float(x) / 2, float(y) / 2] for x, y in np.asarray(rb)[:, :2]]})
    b = np.asarray(r.boundary)
    door = [[float(x) / 2, float(y) / 2] for x, y in b[b[:, 3] == 1][:, :2]]
    return make_sample(SOURCE, variant, str(site), rooms, boundary=[[float(x) / 2, float(y) / 2] for x, y in b[:, :2]],
                       condition={"type": "ground_truth", "ref_id": f"rplan_{site}",
                                  "note": f"exact RPLAN room polygons from Graph2Plan data_{split}.mat, scaled x0.5 to MaskPLAN 128px frame",
                                  "front_door_segment": door})


def _gen_job(p, variant, parsed_img, vdir):
    save_sample(gen_sample(p, variant, parsed_img, vdir / "raw/meta"), vdir / "samples" / f"{p.stem}.json")


def _gt_job(s, vdir):
    save_sample(s, vdir / "gt_samples" / f"{s['id']}.json")
    render(s).save(vdir / "gt_renders" / f"{s['id']}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--parsed_img", default=str(ROOT / "data/method_inputs/maskplan/parsed_img"))
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    vdir = Path(args.variant_dir)
    variant = vdir.name
    parsed_img = Path(args.parsed_img)
    posts = sorted(vdir.glob("raw/native/*/iteration/post/*.png"), key=lambda p: int(p.stem))
    with Pool(args.workers) as pool:
        pool.map(partial(_gen_job, variant=variant, parsed_img=parsed_img, vdir=vdir), posts, chunksize=16)
    print("samples:", len(posts))
    if args.gt:
        d = np.load(SUB / "Processed_data/RPLAN_input_vec.npz")
        T, R = d["T"], d["R"]
        bb = np.load(SUB / "Processed_data/Boundary_BoundingBox.npy")
        (vdir / "gt_renders").mkdir(parents=True, exist_ok=True)
        g2p = load_g2p({int(p.stem) for p in posts})
        print("GT from Graph2Plan RPLAN copy:", len(g2p), "/", len(posts))
        gts = []
        for p in posts:
            if int(p.stem) in g2p:
                gts.append(gt_sample_g2p(int(p.stem), variant + "_gt", g2p[int(p.stem)]))
            else:
                gts.append(gt_sample(p.stem, variant + "_gt", parsed_img, T, R, bb))
        with Pool(args.workers) as pool:
            pool.map(partial(_gt_job, vdir=vdir), gts, chunksize=16)
        print("gt_samples:", len(posts))


if __name__ == "__main__":
    main()
