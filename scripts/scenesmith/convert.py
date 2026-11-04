"""Convert SceneSmith's published house-level scenes (HF nepfaff/scenesmith-example-scenes, House/) to the common format.

SceneSmith's layout is not published as a file of its own, so it is read back from the simulation scene:
  rooms    each room is an axis-aligned rectangle: the floor box of room_geometry/room_geometry_<room>.sdf, placed
           at the room frame of combined_house/house.dmd.yaml. Each room has its own walls of WALL_T inside its floor
           box, so the net room is the floor box shrunk by WALL_T.
  doors    gaps in a wall's collision boxes (SceneSmith splits a wall's collision around each door or open
           connection): on an exterior wall front_door, otherwise interior_door, or opening when wider than
           SceneSmith's largest door (1.9 m) or when the wall has no collision at all. Each gap is seen from both
           rooms; they are merged.
  windows  window_<k>_visual poses with width and height from the window's glTF bounds.
  graph    one edge per interior door or opening, between the rooms on its two sides.
The condition is SceneSmith's text prompt (text_prompt_annotations.csv), with the prompt's category: home,
suite (bedroom suites, hotel suites, dorm) or non_residential.

Fetch the layout files first (no meshes, ~20 MB): scripts/scenesmith/fetch.sh
    PYTHONPATH=. conda run -n fpe python scripts/scenesmith/convert.py
"""
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fpeval.format import canonical_room_type, make_sample, save_sample  # noqa: E402

DATA = ROOT / "data" / "scenesmith"
VARIANT = "house_published"
WALL_T = 0.05  # base_floor_plan_agent.yaml: wall_thickness, per room
MAX_DOOR_M = 1.9  # base_floor_plan_agent.yaml: doors.width_range
MIN_GAP_M = 0.3
HOMES = {185, 187, 188, 191, 195, 198, 205}
SUITES = {47, 48, 49, 97, 98, 99, 186, 193, 196}
EXTRA_ALIASES = {"kids_bedroom": "bedroom", "kids_room": "bedroom", "living_kitchen": "living_room",
                 "living_dining_room": "living_room", "storage_room": "storage", "linen_storage": "storage",
                 "props_storage": "storage", "vestibule": "entrance", "entry_vestibule": "entrance",
                 "common_room": "living_room"}


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_multi_constructor("!", lambda loader, suffix, node: loader.construct_mapping(node, deep=True))


def room_frames(dmd):
    """{room: (x, y)} from the room_<room>_frame entries of house.dmd.yaml (translation only; SceneSmith rooms are
    axis-aligned)."""
    out = {}
    for d in yaml.load(open(dmd), Loader=_Loader)["directives"]:
        f = d.get("add_frame") if isinstance(d, dict) else None
        if f and re.fullmatch(r"room_.+_frame", f["name"]):
            assert "rotation" not in f["X_PF"], f
            x, y, _ = f["X_PF"]["translation"]
            out[f["name"][len("room_"):-len("_frame")]] = (x, y)
    return out


def pose(el):
    p = el.find("pose")
    return [float(v) for v in p.text.split()] if p is not None else [0.0] * 6


def gltf_extent(path):
    """(width along the wall, height) of a window glTF (Drake export, z up, wall normal along local x)."""
    g = json.load(open(path))
    acc = g["accessors"][g["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    lo, hi = acc["min"], acc["max"]
    return hi[1] - lo[1], hi[2] - lo[2]


def parse_room(sdf_path):
    """Floor size, wall pieces per side and windows (all in the room frame) of one room_geometry SDF."""
    link = ET.parse(sdf_path).getroot().find("model/link")
    size = None
    walls = {s: [] for s in ("north", "south", "east", "west")}
    exterior, windows = set(), []
    for el in link:
        name = el.get("name", "")
        if el.tag == "collision" and name == "floor_collision":
            size = [float(v) for v in el.find("geometry/box/size").text.split()][:2]
        elif el.tag == "collision" and (m := re.match(r"(north|south|east|west)_wall_collision", name)):
            sx, sy, _ = (float(v) for v in el.find("geometry/box/size").text.split())
            x, y = pose(el)[:2]
            walls[m.group(1)].append((x - sx / 2, x + sx / 2) if m.group(1) in ("north", "south") else
                                     (y - sy / 2, y + sy / 2))
        elif el.tag == "visual" and (m := re.match(r"(north|south|east|west)_wall_exterior_visual", name)):
            exterior.add(m.group(1))
        elif el.tag == "visual" and re.match(r"window_\d+_visual", name):
            uri = el.find("geometry/mesh/uri").text
            width, height = gltf_extent((sdf_path.parent / uri).resolve())
            x, y, z, _, _, yaw = pose(el)
            windows.append((x, y, z, abs(round(yaw / 1.5708)) % 2 == 0, width, height))  # glTF width is local y
    return size, walls, exterior, windows


def gaps(pieces, half):
    """Uncovered intervals of [-half, half] (wider than MIN_GAP_M) given the covered pieces; None if no pieces."""
    if not pieces:
        return None
    out, pos = [], -half
    for a, b in sorted(pieces):
        if a - pos > MIN_GAP_M:
            out.append((pos, a))
        pos = max(pos, b)
    if half - pos > MIN_GAP_M:
        out.append((pos, half))
    return out


def convert_scene(scene_dir, sid, prompt):
    frames = room_frames(scene_dir / "combined_house" / "house.dmd.yaml")
    rooms, floors, cand_doors, windows = [], [], [], []
    for name, (cx, cy) in frames.items():
        size, walls, exterior, wins = parse_room(scene_dir / "room_geometry" / f"room_geometry_{name}.sdf")
        w, h = size
        floors.append(box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        native = re.sub(r"_?\d+$", "", name)
        rooms.append({"type": canonical_room_type(native, EXTRA_ALIASES), "type_native": name,
                      "polygon": [[round(x, 4), round(y, 4)] for x, y in
                                  box(cx - w / 2 + WALL_T, cy - h / 2 + WALL_T, cx + w / 2 - WALL_T,
                                      cy + h / 2 - WALL_T).exterior.coords[:-1]]})
        for side, pieces in walls.items():
            along_x = side in ("north", "south")
            half = (w if along_x else h) / 2
            g = gaps(pieces, half)
            whole = g is None
            for a, b in ([(-half, half)] if whole else g):
                # Door rectangle across both rooms' walls (2 * WALL_T), centred on the shared floor-box edge.
                if along_x:
                    e = cy + (h / 2 if side == "north" else -h / 2)
                    rect = box(cx + a, e - WALL_T, cx + b, e + WALL_T)
                else:
                    e = cx + (w / 2 if side == "east" else -w / 2)
                    rect = box(e - WALL_T, cy + a, e + WALL_T, cy + b)
                cand_doors.append((rect, side in exterior, whole, b - a))
        for x, y, z, rotated, width, height in wins:
            gx, gy = cx + x, cy + y
            rect = (box(gx - WALL_T, gy - width / 2, gx + WALL_T, gy + width / 2) if rotated else
                    box(gx - width / 2, gy - WALL_T, gx + width / 2, gy + WALL_T))
            windows.append({"polygon": [[round(a, 4), round(b, 4)] for a, b in rect.exterior.coords[:-1]],
                            "width_m": round(width, 3), "sill_m": round(z - height / 2, 3),
                            "head_m": round(z + height / 2, 3), "room": len(rooms) - 1})
    # A wall with no collision at all on a shared edge is an open connection over the shared part only.
    doors, edges, done = [], [], []
    for rect, ext, whole, width in cand_doors:
        if any(rect.intersection(d).area > 0.5 * rect.area for d in done):
            continue
        touching = [i for i, f in enumerate(floors) if f.intersects(rect.buffer(-1e-3))]
        if whole and len(touching) == 2:
            shared = floors[touching[0]].buffer(WALL_T, cap_style="flat", join_style="mitre").intersection(
                floors[touching[1]].buffer(WALL_T, cap_style="flat", join_style="mitre"))
            rect, width = shared.envelope, max(shared.bounds[2] - shared.bounds[0], shared.bounds[3] - shared.bounds[1])
        elif whole:
            continue  # exterior wall without collision: none in the published scenes
        done.append(rect)
        if ext and len(touching) == 1:
            dtype = "front_door"
        else:
            dtype = "opening" if whole or width > MAX_DOOR_M + 1e-3 else "interior_door"
        doors.append({"type": dtype, "polygon": [[round(a, 4), round(b, 4)] for a, b in rect.exterior.coords[:-1]],
                      "width_m": round(width, 3)})
        if len(touching) == 2:
            edges.append([touching[0], touching[1], "door" if dtype == "interior_door" else "open"])
    boundary = unary_union(floors)
    num = int(sid.split("_")[1])
    category = "home" if num in HOMES else "suite" if num in SUITES else "non_residential"
    return make_sample("scenesmith", VARIANT, sid, rooms, units="m", scale_m_per_unit=1.0,
                       boundary=[[round(x, 4), round(y, 4)] for x, y in boundary.exterior.coords[:-1]]
                       if boundary.geom_type == "Polygon" else None,
                       doors=doors, windows=windows, graph={"edges": edges},
                       condition={"type": "text", "prompt": prompt, "category": category})


def main():
    prompts = {int(r["ID"]): r["Description"] for r in csv.DictReader(open(DATA / "text_prompt_annotations.csv"))}
    out = ROOT / "outputs" / "scenesmith" / VARIANT
    n = 0
    for d in sorted((DATA / "House").glob("scene_*")):
        s = convert_scene(d, d.name, prompts[int(d.name.split("_")[1])])
        save_sample(s, out / "samples" / f"{d.name}.json")
        n += 1
    json.dump({"source": "SceneSmith published house-level scenes (huggingface.co/datasets/nepfaff/"
                         "scenesmith-example-scenes, House/), layout read back from the scene files",
               "n_samples": n}, open(out / "run_info.json", "w"), indent=1)
    print(f"{VARIANT}: {n} samples")


if __name__ == "__main__":
    main()
