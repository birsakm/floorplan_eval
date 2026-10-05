"""Per-room-type architectural rule thresholds (metric units).

These are deliberately loose "a person / a bed must fit" defaults, not a
building code. Evaluate the GT datasets with the same thresholds to see how
often real plans violate them; that is the baseline for generated plans.
"""

# Minimum floor area per room in m^2.
MIN_AREA_M2 = {
    "living_room": 8.0, "kitchen": 3.0, "bedroom": 5.0, "bathroom": 1.5,
    "balcony": 1.0, "entrance": 1.0, "dining_room": 4.0, "study": 4.0,
    "storage": 0.5, "corridor": 1.0, "garage": 10.0, "laundry": 1.0,
    "closet": 0.5, "outdoor": 1.0, "other": 0.5,
}

# Minimum clear width in m, measured as the diameter of the largest inscribed circle.
MIN_WIDTH_M = {
    "living_room": 2.4, "kitchen": 1.5, "bedroom": 2.0, "bathroom": 1.0,
    "balcony": 0.8, "entrance": 0.9, "dining_room": 2.0, "study": 1.8,
    "storage": 0.6, "corridor": 0.8, "garage": 2.4, "laundry": 0.9,
    "closet": 0.6, "outdoor": 0.8, "other": 0.6,
}

# Maximum aspect ratio (long / short side of the minimum rotated rectangle).
# None = unconstrained (corridors are long by design).
MAX_ASPECT = {
    "living_room": 4.0, "kitchen": 4.0, "bedroom": 3.0, "bathroom": 3.5,
    "balcony": 10.0, "entrance": 5.0, "dining_room": 3.5, "study": 3.0,
    "storage": 6.0, "corridor": None, "garage": 4.0, "laundry": 5.0,
    "closet": 6.0, "outdoor": None, "other": None,
}

# Half wall thickness in m. Gaps up to twice this between rooms count as walls, not holes.
# 0.3 because RPLAN-image-derived sources (DS2D, Tell2Design, ChatHouseDiffusion,
# House-GAN++ JSON) leave gaps up to ~0.6 m between rooms.
WALL_HALF_THICKNESS_M = 0.3

# Edges shorter than this are ignored for rectilinearity (raster staircase artifacts).
RECTILINEAR_MIN_EDGE_M = 0.25

# Minimum shared wall length in m for two rooms to count as adjacent.
MIN_SHARED_WALL_M = 0.5
