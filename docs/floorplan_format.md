# Common floorplan format

All generated samples (in `outputs/`) and ground-truth data (in `data/`) are converted to this format so one evaluation pipeline can read every source.

## Layout on disk

```
outputs/<method>/<variant>/
  raw/            # the method's native output, untouched
  samples/<id>.json   # one floorplan in the common format (below)
  renders/<id>.png    # rendering of samples/<id>.json made by our shared renderer
  run_info.json   # commit, checkpoint, command, conditioning, #samples, date, runtime
```

For ground truth, use `data/<dataset>/converted/samples/<id>.json` and `.../renders/<id>.png`.

## Sample JSON

```json
{
  "source": "house_diffusion",
  "variant": "rplan_bubble",
  "id": "000123",
  "units": "px",
  "scale_m_per_unit": null,
  "boundary": [[x, y], ...],
  "rooms": [
    {"type": "bedroom", "type_native": "Bedroom", "polygon": [[x, y], ...]}
  ],
  "doors":   [{"type": "interior_door", "polygon": [[x, y], ...]}],
  "windows": [{"polygon": [[x, y], ...]}],
  "walls":   [{"polygon": [[x, y], ...]}],
  "graph": {"edges": [[i, j, "door"]]},
  "condition": {"type": "bubble_diagram", "ref_id": "rplan_000123"}
}
```

- `units`: `"px"` when the data has no metric scale, `"m"` when it does. Set `scale_m_per_unit` when the scale is known.
- `boundary`, `doors`, `windows`, `walls`, `graph` and `condition` are optional. Use `null` or `[]` when they don't apply.
- Polygons are simple polygons with y pointing down, in pixel or meter coordinates. They are not closed (the first point is not repeated). Raster outputs are vectorized from contours, one polygon per connected component.
- `graph.edges` index into `rooms`.
- `type` must come from this canonical vocabulary. Keep the original label in `type_native`.

```
living_room, kitchen, bedroom, bathroom, balcony, entrance, dining_room,
study, storage, corridor, garage, laundry, closet, outdoor, other
```

Canonicalization is done by `fpeval/format.py` (`canonical_room_type`), and rendering by `fpeval/render.py`.
