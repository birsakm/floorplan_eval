"""Common floorplan format (see docs/floorplan_format.md)."""
import json
import re
from pathlib import Path

ROOM_TYPES = [
    "living_room", "kitchen", "bedroom", "bathroom", "balcony", "entrance",
    "dining_room", "study", "storage", "corridor", "garage", "laundry",
    "closet", "outdoor", "other",
]

# Normalized native label -> canonical type. Keys are lowercased with
# non-alphanumerics collapsed to "_" (see _norm). Extend as new sources appear.
_ALIASES = {
    "living_room": "living_room", "livingroom": "living_room", "living": "living_room",
    "lounge": "living_room", "draw_room": "living_room", "drawing_room": "living_room",
    "family_room": "living_room", "livingroom_kitchen": "living_room",
    "kitchen": "kitchen", "kitchen_dining": "kitchen",
    "bedroom": "bedroom", "master_room": "bedroom", "masterroom": "bedroom",
    "second_room": "bedroom", "secondroom": "bedroom", "child_room": "bedroom",
    "childroom": "bedroom", "elder_room": "bedroom", "elderroom": "bedroom",
    "guest_room": "bedroom", "guestroom": "bedroom", "master_bedroom": "bedroom",
    "bathroom": "bathroom", "bath": "bathroom", "toilet": "bathroom", "wc": "bathroom",
    "restroom": "bathroom", "sauna": "bathroom",
    "balcony": "balcony", "terrace": "balcony", "loggia": "balcony",
    "entrance": "entrance", "entry": "entrance", "foyer": "entrance", "hall": "entrance",
    "dining_room": "dining_room", "diningroom": "dining_room", "dining": "dining_room",
    "study": "study", "studyroom": "study", "study_room": "study", "office": "study",
    "library": "study",
    "storage": "storage", "store_room": "storage", "storeroom": "storage",
    "utility": "storage", "technical_room": "storage",
    "corridor": "corridor", "hallway": "corridor", "passage": "corridor",
    "garage": "garage", "carport": "garage",
    "laundry": "laundry", "laundry_room": "laundry",
    "closet": "closet", "wardrobe": "closet", "walk_in_closet": "closet",
    "dressing_room": "closet",
    "outdoor": "outdoor", "garden": "outdoor", "yard": "outdoor", "patio": "outdoor",
    "other": "other", "undefined": "other", "unknown": "other", "room": "other",
}


def _norm(label):
    return re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")


def canonical_room_type(native_label, extra_aliases=None):
    """Map a dataset/method-specific room label to the canonical vocabulary.

    extra_aliases: optional {native_label: canonical_type} for source-specific labels.
    Unknown labels map to "other".
    """
    key = _norm(native_label)
    if extra_aliases:
        extra = {_norm(k): v for k, v in extra_aliases.items()}
        if key in extra:
            return extra[key]
    return _ALIASES.get(key, "other")


def make_sample(source, variant, sample_id, rooms, units="px", **optional):
    """Build a sample dict. rooms: list of {"type", "type_native", "polygon"}."""
    sample = {
        "source": source, "variant": variant, "id": str(sample_id),
        "units": units, "scale_m_per_unit": None, "boundary": None,
        "rooms": rooms, "doors": [], "windows": [], "walls": [],
        "graph": None, "condition": None,
    }
    for k, v in optional.items():
        if k not in sample:
            raise KeyError(f"unknown field {k!r}")
        sample[k] = v
    return sample


def validate(sample):
    """Raise ValueError if the sample violates the format."""
    for key in ("source", "variant", "id", "units", "rooms"):
        if key not in sample:
            raise ValueError(f"missing field {key!r}")
    if sample["units"] not in ("px", "m"):
        raise ValueError(f"bad units {sample['units']!r}")
    for i, room in enumerate(sample["rooms"]):
        if room.get("type") not in ROOM_TYPES:
            raise ValueError(f"room {i}: type {room.get('type')!r} not canonical")
        poly = room.get("polygon")
        if not poly or len(poly) < 3 or any(len(p) != 2 for p in poly):
            raise ValueError(f"room {i}: polygon needs >=3 [x, y] points")
    graph = sample.get("graph")
    if graph:
        n = len(sample["rooms"])
        for e in graph.get("edges", []):
            if not (0 <= e[0] < n and 0 <= e[1] < n):
                raise ValueError(f"graph edge {e} out of range")


def save_sample(sample, path):
    validate(sample)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sample, f)


def load_sample(path):
    with open(path) as f:
        return json.load(f)
