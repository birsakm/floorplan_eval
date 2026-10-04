"""Headless batch driver for WallPlan (boundary-only constraint).

Mirrors WallPlan_Main.generate_from_val() in test/Boundary_Test.py but
  * does NOT import Boundary_Test.py (it hard-codes CUDA_VISIBLE_DEVICES=0 at import time),
  * takes absolute checkpoint paths,
  * accepts two input sources:
      --source bundled : the repo's test/input/*.pkl (WallPlan's own 500 test boundaries)
      --source g2p     : Graph2Plan-preprocessed RPLAN test boundaries (data_test.mat), converted
                         into WallPlan's input masks (boundary/inside/door masks at 120 px,
                         start node = lexicographically smallest boundary corner), exactly the
                         way they are derived from the wall graph in the bundled pkls
                         (verified: identical masks for 498/500 bundled samples),
  * saves the vector result (wall graph, room circles, doors, windows) instead of only the
    rendered PNG (the PNG is also saved).
Run in the fpe-wallplan env. Writes raw/<name>.pkl (+ raw/png/<name>.png).
"""
import argparse
import copy
import json
import os
import pickle
import random
import sys
import time

import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WP = os.path.join(REPO, "external", "methods", "wallplan")
sys.path.insert(0, WP)
sys.path.insert(1, os.path.join(WP, "test"))

from train.WallPlan import models  # noqa: E402
from test_process.step1_window_assembly import assembling_windows  # noqa: E402
from test_process.step2_CoupleNet import coupling_networks  # noqa: E402
from test_process.step3_decoration import (clear_graph, get_room_circles, get_door_win_slices,  # noqa: E402
                                           arrange_door_win, put_door_win, floorplan_render)


# ---- helpers copied from test/Boundary_Test.py (unchanged logic)
def check_notalign_nodes(graph, ind1, ind2):
    pos1, pos2 = graph[ind1]['pos'], graph[ind2]['pos']
    return 1 if (pos1[0] != pos2[0] and pos1[1] != pos2[1]) else 0


def check_not_align(graph):
    for node in graph:
        if node is not None:
            for con in node['connect']:
                if con and check_notalign_nodes(graph, node['index'], con):
                    return 1
    return 0


def check_overlap_junction(whole_graph):
    pos_list = [tuple(node['pos']) for node in whole_graph if node is not None]
    return 1 if len(set(pos_list)) != len(pos_list) else 0


# ---- input construction (120 px masks), same drawing primitives as the training data
def boundary_mask_120(poly):  # poly: [[row, col], ...] at 120 px, corners in order
    m = np.zeros((120, 120), np.uint8)
    n = len(poly)
    for i in range(n):
        o, t = poly[i], poly[(i + 1) % n]
        m[o[0] - 1:o[0] + 2, o[1] - 1:o[1] + 2] = 1
        cv2.line(m, (int(o[1]), int(o[0])), (int(t[1]), int(t[0])), 1, 2, 4)
    return m


def boundary_mask_120_5pix(poly):
    m = np.zeros((120, 120), np.uint8)
    n = len(poly)
    for i in range(n):
        o, t = poly[i], poly[(i + 1) % n]
        m[o[0] - 2:o[0] + 3, o[1] - 2:o[1] + 3] = 1
        cv2.line(m, (int(o[1]), int(o[0])), (int(t[1]), int(t[0])), 1, 3, 4)
    return m


def inside_mask_120(poly, bmask):
    m = np.zeros((120, 120), np.uint8)
    cv2.fillPoly(m, [np.array([[p[1], p[0]] for p in poly], np.int32)], 1)
    m[bmask > 0] = 0
    return m


def door_mask_120(door120):
    m = np.zeros((120, 120), np.uint8)
    p = door120['pos']
    if door120['ori'] == 0:
        m[p[0] - 2:p[0] + 3, p[1] - 3:p[1] + 4] = 1
    else:
        m[p[0] - 3:p[0] + 4, p[1] - 2:p[1] + 3] = 1
    return m


def g2p_to_wallplan(boundary256):
    """Graph2Plan boundary rows (x, y, dir, isNew) at 256 px -> WallPlan inputs.
    WallPlan uses (row, col) at 256 px and 120 px = (p - 8) // 2."""
    b = np.asarray(boundary256, dtype=int)
    # door = first two points; ori 0 = horizontal door (on a horizontal wall)
    (x0, y0), (x1, y1) = b[0, :2], b[1, :2]
    ori = 0 if y0 == y1 else 1
    door256 = {'pos': [int((y0 + y1) // 2), int((x0 + x1) // 2)], 'ori': ori}
    # corners only (drop isNew door points and collinear points)
    pts = [(int(r[1]), int(r[0])) for r in b if r[3] == 0]
    keep = []
    n = len(pts)
    for i in range(n):
        a, c, d = pts[i - 1], pts[i], pts[(i + 1) % n]
        if (a[0] == c[0] == d[0]) or (a[1] == c[1] == d[1]):
            continue
        keep.append(c)
    poly120 = [[(r - 8) // 2, (c - 8) // 2] for r, c in keep]
    # dedupe consecutive duplicates produced by the //2
    dedup = []
    for p in poly120:
        if not dedup or dedup[-1] != p:
            dedup.append(p)
    if dedup[0] == dedup[-1]:
        dedup.pop()
    return dedup, door256


def load_models(ckpt):
    nets = {
        'living': models.model(module_name="Living_Win_Net", model_name="dlink34no", num_classes=2, num_channels=4),
        'other': models.model(module_name="Other_WinNet", model_name="dlink34no", num_classes=2, num_channels=5),
        'label': models.model(module_name="LabelNet", model_name="dlink34no", num_classes=8, num_channels=6),
        'graph': models.model(module_name="GraphNet", model_name="dlink34no", num_classes=3, num_channels=8),
    }
    files = {'living': 'WindowLiving.pth', 'other': 'WindowOther.pth', 'label': 'LabelNet.pth', 'graph': 'GraphNet.pth'}
    for k, m in nets.items():
        m.load_model(os.path.join(ckpt, files[k]))
        m.cuda().eval()
    return nets


def generate(nets, boundary_mask_use, inside_mask_use, door_mask_use, b5, start_pos, door_info):
    """Body of WallPlan_Main.generate_from_val (door_info at 256 px)."""
    gen_window_mask, living_windows, other_windows = assembling_windows(
        boundary_mask_use, inside_mask_use, door_mask_use, nets['living'], nets['other'])
    liv_window_para, other_window_para = [], []
    for w in living_windows:
        liv_window_para.append({'pos': [w['para'][0], w['para'][1]], 'ori': w['ori']})
    for w in other_windows:
        if len(w) >= 4:
            other_window_para.append({'pos': [w[0], w[1]], 'ori': w[3]})
    fp_composite = np.zeros((4, 120, 120))
    fp_composite[0] = boundary_mask_use
    fp_composite[1] = inside_mask_use
    fp_composite[2] = door_mask_use
    fp_composite[3] = gen_window_mask
    gen_junction_graph, output_seman, output_seman_8channel = coupling_networks(
        fp_composite, b5, start_pos, nets['label'], nets['graph'])
    gen_junction_graph = clear_graph(gen_junction_graph)
    room_circles = get_room_circles(gen_junction_graph, output_seman, output_seman_8channel)
    door_info_120 = {'pos': (np.array(door_info['pos']) - 8) // 2, 'ori': door_info['ori']}
    if not (gen_junction_graph[1] is not None and check_overlap_junction(gen_junction_graph) == 0
            and check_not_align(gen_junction_graph) == 0):
        raise RuntimeError("generated wall graph rejected (overlap / not aligned / empty)")
    frontdoor_slice, win_slice_room_order, interdoor_slice_room_order, balcony_wins, no_balcony_wins = \
        get_door_win_slices(door_info_120['pos'], gen_junction_graph, room_circles, liv_window_para, other_window_para)
    setted_front_door, setted_inter_door, setted_livwins, setted_wins, special_balcony_doors = arrange_door_win(
        gen_junction_graph, room_circles, frontdoor_slice, win_slice_room_order, interdoor_slice_room_order)
    frontdoor_where, door_where, livwins_where, wins_where = put_door_win(
        gen_junction_graph, room_circles, door_info_120, setted_front_door, setted_inter_door, setted_livwins,
        setted_wins)
    result = copy.deepcopy(dict(graph=gen_junction_graph, room_circles=room_circles, frontdoor=frontdoor_where,
                                doors=door_where, livwins=livwins_where, wins=wins_where))
    fp_mask = floorplan_render(None, gen_junction_graph, room_circles, frontdoor_where, door_where, livwins_where,
                               wins_where, balcony_wins, no_balcony_wins, liv_window_para, special_balcony_doors, 0)
    return result, fp_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bundled", "g2p"], required=True)
    ap.add_argument("--ids", help="g2p: file with RPLAN names")
    ap.add_argument("--g2p_mat", help="g2p: Graph2Plan Network/data/data_test.mat")
    ap.add_argument("--ckpt", required=True, help="checkpoints/wallplan/Boundary_constraint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.join(args.out, "png"), exist_ok=True)

    items = []  # (name, bmask, inside, door_mask, b5, start_pos, door_info256, boundary_poly120)
    if args.source == "bundled":
        d = os.path.join(WP, "test", "input")
        for f in sorted(os.listdir(d), key=lambda s: int(s.split(".")[0])):
            items.append((f[:-4], os.path.join(d, f)))
    else:
        import scipy.io as sio
        data = sio.loadmat(args.g2p_mat, squeeze_me=True, struct_as_record=False)["data"]
        by_name = {str(x.name): x for x in data}
        for n in [l.strip() for l in open(args.ids) if l.strip()]:
            items.append((n, by_name[n].boundary))
    if args.limit:
        items = items[:args.limit]

    nets = load_models(args.ckpt)
    log = {"ok": [], "failed": {}}
    t0 = time.time()
    for k, (name, src) in enumerate(items):
        out_pkl = os.path.join(args.out, f"{name}.pkl")
        if os.path.exists(out_pkl):
            log["ok"].append(name)
            continue
        try:
            if args.source == "bundled":
                with open(src, "rb") as f:
                    wall_graph, door_info, dm, bm, b5, im = pickle.load(f)
                start_pos = [(wall_graph[1]['pos'][0] - 8) // 2, (wall_graph[1]['pos'][1] - 8) // 2]
                # boundary polygon (120 px) for the converter = outline of inside+boundary mask
                inp = dict(wall_graph=wall_graph, door_info=door_info)
            else:
                poly, door_info = g2p_to_wallplan(src)
                bm = boundary_mask_120(poly)
                b5 = boundary_mask_120_5pix(poly)
                im = inside_mask_120(poly, bm)
                dm = door_mask_120({'pos': (np.array(door_info['pos']) - 8) // 2, 'ori': door_info['ori']})
                start_pos = list(min(tuple(p) for p in poly))
                inp = dict(boundary_poly120=poly, door_info=door_info, g2p_boundary=np.asarray(src).tolist())
            res, fp_mask = generate(nets, copy.deepcopy(bm), copy.deepcopy(im), copy.deepcopy(dm), b5,
                                    start_pos, door_info)
            res["input"] = inp
            res["masks120"] = dict(boundary=bm, inside=im, door=dm)
            with open(out_pkl, "wb") as f:
                pickle.dump(res, f, protocol=4)
            cv2.imwrite(os.path.join(args.out, "png", f"{name}.png"), fp_mask)
            log["ok"].append(name)
        except Exception as e:  # noqa
            log["failed"][name] = f"{type(e).__name__}: {e}"
        if (k + 1) % 25 == 0:
            print(f"{k + 1}/{len(items)} ok={len(log['ok'])} fail={len(log['failed'])} "
                  f"{(time.time() - t0) / (k + 1):.2f}s/sample", flush=True)
    log["seconds"] = time.time() - t0
    with open(os.path.join(args.out, "_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print("done", len(log["ok"]), "ok", len(log["failed"]), "failed", f"{log['seconds']:.0f}s")


if __name__ == "__main__":
    main()
