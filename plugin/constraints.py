"""Constraint-JSON reader: load a CadMust-Autopilot constraint file (schema v1
or v2) and apply it to a BoardModel.

Pure Python, stdlib only, no pcbnew.

    cset = load_constraints("board.json")
    report = apply_constraints(model, cset)   # mutates the model
    print(report.summary())

Conventions (shared with CadMust-Autopilot's check_constraints.py, so the
engine and the checker agree):

* Coordinates in mm are BOARD-RELATIVE: origin = (outline_xmin, outline_ymin)
  of the model's outline bounding box, x to the right, y down.
* "Footprint centre" is the footprint position (KiCad origin). It is used for
  fixed positions, templates and zone preferences. Edge distances and
  clearances use the footprint bbox.
* Zones are the cells of a 3x3 grid over the outline bounding box.
* Edge distance is bbox edge to outline-bbox edge.
* Net names match exactly first, then with KiCad's escaping normalised: a
  leading "/" is dropped and "{slash}" reads as "/".
* A reference that occurs more than once on the board resolves to its first
  occurrence (as in the checker); a warning is logged.

Mapping onto the engine:

    fixed_positions      position set, footprint locked
    back_side            Footprint.is_front = False (static; no flip move,
                         pad geometry is not mirrored)
    component_groups     with template: members placed at the template offsets
                         from the anchor, registered as a rigid ComponentGroup.
                         Offsets are in the ANCHOR frame (at anchor angle 0):
                         the board-frame offset is (dx, dy) rotated by the
                         anchor's angle_deg exactly like Pad.abs_position.
                         Any group with max_span_mm (template or not) also
                         gets the soft cohesion term over all present members.
    net_priorities       Net.weight (critical/high), Net.is_excluded (exclude)
    edge_locked          Footprint.edge_lock, snapped to the edge; slides along it
    zone_preferences     BoardModel.zone_prefs (soft distance term)
    keep_out_zones       Footprint.clearance / clearance_side (overlap inflation);
                         empty layers or both F.Cu and B.Cu = ring on both
                         sides (opposite-side bodies count too), a single
                         copper layer = that side only. An item with an
                         "antenna" object is no ring: it attaches one local
                         rect (Footprint.keepout_rects, antenna_keepout_rect)
                         beyond the antenna end of the module; same layer rule
    meta                 board size sanity check (warning only)
    signal_classes, design_notes, req_ids, source: ignored by the engine

Antenna keep-out (schema v2 ``keep_out_zones[].antenna``), resolved by
resolve_antenna() / antenna_keepout_rect() in the footprint's local frame at 0
degrees, from pad CENTRES (the model stores no pad sizes):

* side: "pad_free" or "+x" / "-x" / "+y" / "-y". Every pad belongs to the
  local side it is nearest to, in coordinates normalised to the pad-centre
  extent (pad_sides; a corner pad goes to its longer row, inner pads to no
  side); "pad_free" =
  the one side no pad belongs to, zero or several such sides is an
  AntennaError. Explicit sides are in the model's local frame, which for a
  back-side part is the mirrored (as-placed) frame.
* pad_edge = outermost pad centre on the side; antenna end = pad_edge +
  module_end_beyond_pads_mm (datasheet), or the courtyard bbox edge when the
  item does not give it; antenna strip = [end - depth_mm, end] across the
  pad-centre extent (recorded, see below).
* keep-out rect: outward from the bbox edge to end + clearance_mm, sideways
  the pad-centre extent widened by clearance_mm on both sides. Clipped to
  the region beyond the bbox edge (what the courtyard already covers adds
  nothing; a zero-depth result means the courtyard already encodes it and
  nothing is attached). The rect starts AT the bbox edge even when the strip
  begins further out: the module body between the courtyard and the strip
  is part of the physical module and must stay free too, so depth_mm does
  not move the rect; it is reported and cross-checked only.

constrained_refs() lists every reference the file places; unlock_refs()
clears the board lock on those parts (batch_runner --unlock constrained).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .board_model import BoardModel, ComponentGroup, Pad
from .moves import resnap_edges, _get_group_members

MM = 1_000_000  # nanometres per millimetre

EDGES = ("top", "bottom", "left", "right")
ZONE_ROWS = ("top", "center", "bottom")
ZONE_COLS = ("left", "center", "right")


def zone_name(row: int, col: int) -> str:
    if row == 1 and col == 1:
        return "center"
    return "%s-%s" % (ZONE_ROWS[row], ZONE_COLS[col])


ZONES = [zone_name(r, c) for r in range(3) for c in range(3)]

_TOP_KEYS = ("meta", "component_groups", "net_priorities",
             "placement_constraints", "signal_classes", "design_notes")
_REQUIRED_TOP = ("meta", "component_groups", "net_priorities",
                 "placement_constraints")
_PLACEMENT_KEYS = ("edge_locked", "zone_preferences", "keep_out_zones",
                   "fixed_positions", "back_side")


class ConstraintError(ValueError):
    """The constraint file is unusable (unreadable, not JSON, or malformed)."""


def norm_net(name: str) -> str:
    """KiCad net-name normalisation used for the second matching pass."""
    return name.replace("{slash}", "/").lstrip("/")


# ---------------------------------------------------------------------------
# Loading and light validation
# ---------------------------------------------------------------------------

@dataclass
class ConstraintSet:
    """A parsed, structurally validated constraint file."""
    data: dict
    path: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def meta(self) -> dict:
        return self.data.get("meta") or {}

    @property
    def component_groups(self) -> list:
        return self.data.get("component_groups") or []

    @property
    def net_priorities(self) -> dict:
        return self.data.get("net_priorities") or {}

    @property
    def placement(self) -> dict:
        return self.data.get("placement_constraints") or {}


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_str_list(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _validate(data) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings)."""
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(data, dict):
        return ["top level must be an object"], warnings
    for key in _REQUIRED_TOP:
        if key not in data:
            errors.append("missing required top-level key '%s'" % key)
    for key in data:
        if key not in _TOP_KEYS:
            warnings.append("unknown top-level key '%s' (ignored)" % key)

    meta = data.get("meta")
    if meta is not None:
        if not isinstance(meta, dict):
            errors.append("meta must be an object")
        else:
            dims = meta.get("board_dimensions_mm")
            if dims is not None and not (isinstance(dims, list) and len(dims) == 2
                                         and all(_is_num(x) for x in dims)):
                errors.append("meta.board_dimensions_mm must be [width, height] numbers")

    groups = data.get("component_groups")
    if groups is not None:
        if not isinstance(groups, list):
            errors.append("component_groups must be an array")
        else:
            for i, g in enumerate(groups):
                where = "component_groups[%d]" % i
                if not isinstance(g, dict):
                    errors.append("%s must be an object" % where)
                    continue
                if not isinstance(g.get("id"), str):
                    errors.append("%s.id must be a string" % where)
                if not _is_str_list(g.get("members")):
                    errors.append("%s.members must be an array of strings" % where)
                if "max_span_mm" in g and not (_is_num(g["max_span_mm"])
                                               and g["max_span_mm"] >= 0):
                    errors.append("%s.max_span_mm must be a number >= 0" % where)
                t = g.get("template")
                if t is not None:
                    if not isinstance(t, dict) or not isinstance(t.get("anchor"), str):
                        errors.append("%s.template.anchor must be a string" % where)
                        continue
                    rp = t.get("relative_positions", {})
                    if not isinstance(rp, dict):
                        errors.append("%s.template.relative_positions must be an object"
                                      % where)
                        continue
                    for ref, pos in rp.items():
                        if not (isinstance(pos, dict) and _is_num(pos.get("dx_mm"))
                                and _is_num(pos.get("dy_mm"))):
                            errors.append("%s.template.relative_positions.%s needs "
                                          "numeric dx_mm and dy_mm" % (where, ref))

    np_ = data.get("net_priorities")
    if np_ is not None:
        if not isinstance(np_, dict):
            errors.append("net_priorities must be an object")
        else:
            for tier, body in np_.items():
                where = "net_priorities.%s" % tier
                if tier not in ("critical", "high", "normal", "exclude"):
                    warnings.append("unknown net priority tier '%s' (ignored)" % tier)
                    continue
                if not isinstance(body, dict):
                    errors.append("%s must be an object" % where)
                    continue
                if "nets" in body and not _is_str_list(body["nets"]):
                    errors.append("%s.nets must be an array of strings" % where)
                if "weight" in body and not (_is_num(body["weight"]) and body["weight"] >= 0):
                    errors.append("%s.weight must be a number >= 0" % where)

    pc = data.get("placement_constraints")
    if pc is not None:
        if not isinstance(pc, dict):
            errors.append("placement_constraints must be an object")
        else:
            required = {
                "edge_locked": (("ref", "str"), ("edge", "str")),
                "zone_preferences": (("ref", "str"), ("zone", "str")),
                "keep_out_zones": (("ref", "str"), ("clearance_mm", "num")),
                "fixed_positions": (("ref", "str"), ("x_mm", "num"), ("y_mm", "num")),
                "back_side": (("ref", "str"),),
            }
            for key, items in pc.items():
                if key not in required:
                    warnings.append("unknown placement_constraints key '%s' (ignored)" % key)
                    continue
                if not isinstance(items, list):
                    errors.append("placement_constraints.%s must be an array" % key)
                    continue
                for i, it in enumerate(items):
                    where = "placement_constraints.%s[%d]" % (key, i)
                    if not isinstance(it, dict):
                        errors.append("%s must be an object" % where)
                        continue
                    for fname, ftype in required[key]:
                        v = it.get(fname)
                        ok = isinstance(v, str) if ftype == "str" else _is_num(v)
                        if not ok:
                            errors.append("%s.%s must be a %s" % (
                                where, fname, "string" if ftype == "str" else "number"))
                    if key == "edge_locked" and isinstance(it.get("edge"), str) \
                            and it["edge"] not in EDGES:
                        errors.append("%s.edge must be one of %s" % (where, list(EDGES)))
                    if key == "zone_preferences" and isinstance(it.get("zone"), str) \
                            and it["zone"] not in ZONES:
                        errors.append("%s.zone must be one of %s" % (where, ZONES))
                    if key == "keep_out_zones":
                        if _is_num(it.get("clearance_mm")) and it["clearance_mm"] < 0:
                            errors.append("%s.clearance_mm must be >= 0" % where)
                        if "layers" in it and not _is_str_list(it["layers"]):
                            errors.append("%s.layers must be an array of strings" % where)
                        if "antenna" in it:
                            errors.extend(validate_antenna(it["antenna"], where + ".antenna"))

    sc = data.get("signal_classes")
    if sc is not None and not isinstance(sc, list):
        errors.append("signal_classes must be an array")
    dn = data.get("design_notes")
    if dn is not None and not isinstance(dn, list):
        errors.append("design_notes must be an array")
    return errors, warnings


def parse_constraints(data, path: str = "") -> ConstraintSet:
    """Validate an already-decoded constraint dict. Raises ConstraintError."""
    errors, warnings = _validate(data)
    if errors:
        raise ConstraintError("%s: %d error(s) in constraint file:\n  %s"
                              % (path or "<constraints>", len(errors),
                                 "\n  ".join(errors)))
    return ConstraintSet(data=data, path=path, warnings=warnings)


def load_constraints(path) -> ConstraintSet:
    """Read and validate a constraint JSON file. Raises ConstraintError."""
    path = str(path)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ConstraintError("constraint file not found: %s" % path)
    except json.JSONDecodeError as e:
        raise ConstraintError("%s: not valid JSON (line %d, column %d): %s"
                              % (path, e.lineno, e.colno, e.msg))
    except OSError as e:
        raise ConstraintError("cannot read constraint file %s: %s" % (path, e))
    return parse_constraints(data, path)


# ---------------------------------------------------------------------------
# Antenna keep-out resolution (pure geometry on one Footprint, no position)
# ---------------------------------------------------------------------------

ANTENNA_SIDES = ("+x", "-x", "+y", "-y")
PAD_SIDE_TIE = 0.02     # normalised units: |u| and |v| this close = corner pad
PAD_SIDE_BAND = 0.4     # pads with max(|u|, |v|) below this are inner pads
PAD_ROW_TOL = 500_000   # nm: pad centres this close to an extreme form that side's row
COURTYARD_REPEAT_TOL = 500_000  # nm: a drawing this close to the bbox edge repeats the courtyard


class AntennaError(ValueError):
    """The antenna side cannot be resolved for this footprint."""


def validate_antenna(a, where: str = "antenna") -> List[str]:
    """Structural errors of a keep_out_zones[].antenna object."""
    if not isinstance(a, dict):
        return ["%s must be an object" % where]
    errors = []
    side = a.get("side")
    if side not in ("pad_free",) + ANTENNA_SIDES:
        errors.append("%s.side must be one of %s" % (where, ["pad_free"] + list(ANTENNA_SIDES)))
    for k in ("module_end_beyond_pads_mm", "depth_mm"):
        if k in a and not (_is_num(a[k]) and a[k] >= 0):
            errors.append("%s.%s must be a number >= 0" % (where, k))
    for k in a:
        if k not in ("side", "module_end_beyond_pads_mm", "depth_mm"):
            errors.append("%s: unknown key '%s'" % (where, k))
    return errors


def _axis_sign(side: str) -> Tuple[int, int]:
    return (0 if side[1] == "x" else 1), (1 if side[0] == "+" else -1)


def pad_centre_extent(fp) -> Optional[Tuple[int, int, int, int]]:
    """Local (xmin, ymin, xmax, ymax) of the pad centres, None without pads."""
    if not fp.pads:
        return None
    xs = [p.offset_x for p in fp.pads]
    ys = [p.offset_y for p in fp.pads]
    return (min(xs), min(ys), max(xs), max(ys))


def pad_sides(fp) -> Dict[str, int]:
    """Pad count per local side ("+x", "-x", "+y", "-y"), placement-blind.

    Each pad centre is normalised to the pad-centre extent (u = (dx - cx) /
    half-width, v = (dy - cy) / half-height, an axis with zero extent reads
    0) and belongs to the side it is uniquely nearest to: +/-x when |u| > |v|
    + PAD_SIDE_TIE, +/-y when |v| > |u| + PAD_SIDE_TIE. A corner pad (|u| and
    |v| within the tie) belongs to whichever of its two sides has the longer
    pad row (pads within PAD_ROW_TOL of that side's outermost centre), to
    both when the rows are equal; inner pads (max(|u|, |v|) <
    PAD_SIDE_BAND, e.g. a thermal pad) belong to no side. Normalising to the pad extent, not the
    courtyard, keeps the answer independent of how far the courtyard is drawn
    (a courtyard that already carries the antenna clearance)."""
    out = {s: 0 for s in ANTENNA_SIDES}
    ext = pad_centre_extent(fp)
    if ext is None:
        return out
    cx = (ext[0] + ext[2]) / 2.0
    cy = (ext[1] + ext[3]) / 2.0
    hw = (ext[2] - ext[0]) / 2.0
    hh = (ext[3] - ext[1]) / 2.0
    edge = {"+x": ext[2], "-x": ext[0], "+y": ext[3], "-y": ext[1]}

    def row_len(side):
        axis = 0 if side[1] == "x" else 1
        return sum(1 for q in fp.pads
                   if abs((q.offset_x, q.offset_y)[axis] - edge[side]) <= PAD_ROW_TOL)

    for p in fp.pads:
        u = (p.offset_x - cx) / hw if hw > 0 else 0.0
        v = (p.offset_y - cy) / hh if hh > 0 else 0.0
        if max(abs(u), abs(v)) < PAD_SIDE_BAND:
            continue
        sx = "+x" if u > 0 else "-x"
        sy = "+y" if v > 0 else "-y"
        if abs(u) - abs(v) > PAD_SIDE_TIE:
            out[sx] += 1
        elif abs(v) - abs(u) > PAD_SIDE_TIE:
            out[sy] += 1
        else:
            # corner pad: it belongs to the side with the longer pad row
            # (a module's pin 1 to its long pin column, a SOT-23 outer pin
            # to its two-pin row); equal rows give it to both (conservative:
            # it can only remove a pad-free side, never invent one)
            lx, ly = row_len(sx), row_len(sy)
            if lx >= ly:
                out[sx] += 1
            if ly >= lx:
                out[sy] += 1
    return out


def pad_free_sides(fp) -> List[str]:
    """Local sides (of "+x", "-x", "+y", "-y") that no pad belongs to
    (pad_sides). All four for a footprint without pads."""
    counts = pad_sides(fp)
    return [s for s in ANTENNA_SIDES if not counts[s]]


def resolve_antenna_side(fp, side: str) -> str:
    """An explicit side as given; "pad_free" resolved to the one pad-free
    side. Raises AntennaError when that is not exactly one side."""
    if side in ANTENNA_SIDES:
        return side
    if side != "pad_free":
        raise AntennaError("unknown antenna side %r" % (side,))
    if not fp.pads:
        raise AntennaError("%s has no pads, so no pad-free side" % fp.reference)
    free = pad_free_sides(fp)
    if len(free) != 1:
        raise AntennaError("%s: pad_free is ambiguous, %d pad-free sides (%s)"
                           % (fp.reference, len(free), ", ".join(free) or "none"))
    return free[0]


def _from_outward(axis: int, sign: int, u0: int, u1: int, p0: int, p1: int):
    """Local rect from an outward interval [u0, u1] along (axis, sign) and a
    perpendicular interval [p0, p1]."""
    a0, a1 = (u0, u1) if sign > 0 else (-u1, -u0)
    if axis == 0:
        return (a0, p0, a1, p1)
    return (p0, a0, p1, a1)


@dataclass
class AntennaResolution:
    """Everything resolve_antenna derives; nm, local frame at 0 degrees."""
    side: str                                  # resolved local side
    rect: Tuple[int, int, int, int]            # keep-out rect (may have zero area)
    strip: Tuple[int, int, int, int]           # antenna strip (record only)
    pad_edge: int                              # outward coord of the outermost pad centre
    bbox_edge: int                             # outward coord of the courtyard bbox edge
    antenna_end: int                           # outward coord of the module's antenna end
    module_end_source: str                     # "datasheet" or "courtyard"
    notes: List[str] = field(default_factory=list)

    @property
    def zero_area(self) -> bool:
        r = self.rect
        return r[2] <= r[0] or r[3] <= r[1]


def resolve_antenna(fp, side: str, clearance_nm: int,
                    module_end_beyond_pads_nm: Optional[int] = None,
                    depth_nm: Optional[int] = None) -> AntennaResolution:
    """Resolve an antenna keep-out on one footprint (see the module doc).
    Uses sizes and pad offsets only, never the position."""
    rs = resolve_antenna_side(fp, side)
    ext = pad_centre_extent(fp)
    if ext is None:
        raise AntennaError("%s has no pads" % fp.reference)
    axis, sign = _axis_sign(rs)
    lb = fp.local_bbox
    pad_edge = sign * (ext[2 + axis] if sign > 0 else ext[axis])
    bbox_edge = sign * (lb[2 + axis] if sign > 0 else lb[axis])
    notes = []
    if module_end_beyond_pads_nm is None:
        end = bbox_edge
        source = "courtyard"
        notes.append("no module_end_beyond_pads_mm: the courtyard bbox edge is taken "
                     "as the module's antenna end")
    else:
        end = pad_edge + int(module_end_beyond_pads_nm)
        source = "datasheet"
    depth = int(depth_nm or 0)
    p0 = ext[1 - axis] - int(clearance_nm)
    p1 = ext[3 - axis] + int(clearance_nm)
    strip = _from_outward(axis, sign, end - depth, end, ext[1 - axis], ext[3 - axis])
    outer = max(bbox_edge, end + int(clearance_nm))
    rect = _from_outward(axis, sign, bbox_edge, outer, p0, p1)
    if outer == bbox_edge:
        notes.append("zero area: the courtyard already covers the antenna end plus "
                     "the clearance on this side")
    if source == "datasheet" and end < bbox_edge:
        notes.append("the antenna end lies %.2f mm inside the courtyard bbox"
                     % ((bbox_edge - end) / MM))
    if depth and end - depth < pad_edge:
        notes.append("depth_mm reaches %.2f mm past the outermost pad centre into the pads"
                     % ((pad_edge - (end - depth)) / MM))
    return AntennaResolution(side=rs, rect=rect, strip=strip, pad_edge=pad_edge,
                             bbox_edge=bbox_edge, antenna_end=end,
                             module_end_source=source, notes=notes)


def antenna_keepout_rect(fp, side: str, clearance_nm: int,
                         module_end_beyond_pads_nm: Optional[int] = None,
                         depth_nm: Optional[int] = None) -> Tuple[int, int, int, int]:
    """The local keep-out rect (nm, 0 degrees, relative to the fp origin) of
    an antenna keep-out; zero area when the courtyard already encodes it.
    Raises AntennaError when the side cannot be resolved."""
    return resolve_antenna(fp, side, clearance_nm, module_end_beyond_pads_nm,
                           depth_nm).rect


def resolve_antenna_item(fp, item: dict) -> AntennaResolution:
    """resolve_antenna for a keep_out_zones item that has an "antenna" object."""
    a = item.get("antenna") or {}
    me = a.get("module_end_beyond_pads_mm")
    d = a.get("depth_mm")
    return resolve_antenna(
        fp, a.get("side", "pad_free"), int(round(float(item.get("clearance_mm", 0)) * MM)),
        None if me is None else int(round(me * MM)),
        None if d is None else int(round(d * MM)))


def drawing_crosscheck(fp, res: AntennaResolution, tol_nm: int = MM) -> Optional[dict]:
    """Compare the datasheet antenna end with the drawn outline on that side.

    The drawn end is taken from the drawing layer (Footprint.drawing_rects)
    that extends furthest beyond the outermost pad centre, ignoring layers
    whose outline on that side lies within COURTYARD_REPEAT_TOL of the
    courtyard bbox edge (an outline drawn as large as the courtyard repeats
    the courtyard, it does not show the module). A drawing beyond the
    courtyard counts (the Olimex U3 case). Returns None when the model has
    no per-layer drawings or no layer qualifies, else {layer,
    drawn_end_beyond_pads_mm, datasheet_end_beyond_pads_mm, difference_mm,
    ok}."""
    rects = getattr(fp, "drawing_rects", None) or {}
    axis, sign = _axis_sign(res.side)
    best = None
    for layer, r in sorted(rects.items()):
        u = sign * (r[2 + axis] if sign > 0 else r[axis])
        if u <= res.pad_edge or abs(u - res.bbox_edge) <= COURTYARD_REPEAT_TOL:
            continue
        if best is None or u > best[1]:
            best = (layer, u)
    if best is None:
        return None
    diff = best[1] - res.antenna_end
    return {"layer": best[0],
            "drawn_end_beyond_pads_mm": round((best[1] - res.pad_edge) / MM, 3),
            "datasheet_end_beyond_pads_mm": round((res.antenna_end - res.pad_edge) / MM, 3),
            "difference_mm": round(diff / MM, 3),
            "ok": abs(diff) <= tol_nm}


# ---------------------------------------------------------------------------
# Constrained references and selective unlock
# ---------------------------------------------------------------------------

def constrained_refs(constraints) -> set:
    """References the constraint file places: members of component_groups
    (including template anchors and template members) and every per-ref entry
    of placement_constraints (edge_locked, zone_preferences, keep_out_zones,
    fixed_positions, back_side). Accepts a ConstraintSet or the decoded dict.

    Used by `batch_runner.py --unlock constrained` (and reusable by the plugin
    GUI): a board lock on a part the file places is older information than
    the file, so these parts lose their board lock."""
    data = constraints.data if isinstance(constraints, ConstraintSet) else (constraints or {})
    refs = set()
    for g in data.get("component_groups") or []:
        if not isinstance(g, dict):
            continue
        refs.update(r for r in (g.get("members") or []) if isinstance(r, str))
        t = g.get("template")
        if isinstance(t, dict):
            if isinstance(t.get("anchor"), str):
                refs.add(t["anchor"])
            rp = t.get("relative_positions")
            if isinstance(rp, dict):
                refs.update(r for r in rp if isinstance(r, str))
    pc = data.get("placement_constraints") or {}
    if isinstance(pc, dict):
        for key in _PLACEMENT_KEYS:
            for it in pc.get(key) or []:
                if isinstance(it, dict) and isinstance(it.get("ref"), str):
                    refs.add(it["ref"])
    return refs


def unlock_refs(model: BoardModel, refs) -> List[str]:
    """Clear the locked flag on the footprints named in `refs` (first
    occurrence of a reference, as in apply_constraints) and on the KiCad
    groups that contain them. A rigid group moves as one body and stays
    locked while any member is locked, so every member of such a group is
    unlocked too. moveable_indices is rebuilt: unlocked footprints (one
    representative, the first member, per unlocked group) become moveable,
    nothing else changes. Returns the references whose lock flag was cleared,
    sorted; footprints that were not locked are not listed."""
    fps = model.footprints
    by_ref: Dict[str, int] = {}
    for i, f in enumerate(fps):
        by_ref.setdefault(f.reference, i)
    targets = set(by_ref[r] for r in refs if r in by_ref)
    groups = set()
    for idx in list(targets):
        gi = model.fp_to_group.get(idx)
        if gi is not None:
            groups.add(gi)
            targets.update(model.component_groups[gi].member_indices)
    cleared = []
    for idx in sorted(targets):
        if fps[idx].locked:
            fps[idx].locked = False
            cleared.append(fps[idx].reference)
    for gi in groups:
        model.component_groups[gi].locked = False
    candidates = set(model.moveable_indices) | targets
    excluded = set()
    for grp in model.component_groups:
        if grp.locked or any(fps[i].locked for i in grp.member_indices):
            excluded.update(grp.member_indices)
        else:
            excluded.update(grp.member_indices[1:])
            if targets.intersection(grp.member_indices):
                candidates.add(grp.member_indices[0])
    model.moveable_indices = [i for i, f in enumerate(fps)
                              if i in candidates and not f.locked and i not in excluded]
    return sorted(cleared)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ConstraintReport:
    """What apply_constraints did. Nothing is dropped silently: every item
    that was not applied appears in `skipped` with a reason."""
    applied: Dict[str, int] = field(default_factory=dict)
    skipped: List[Tuple[str, str, str]] = field(default_factory=list)  # (category, item, reason)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unknown_refs: List[str] = field(default_factory=list)
    unknown_nets: List[str] = field(default_factory=list)
    ignored: Dict[str, int] = field(default_factory=dict)

    def _applied(self, cat: str, n: int = 1) -> None:
        self.applied[cat] = self.applied.get(cat, 0) + n

    def _skip(self, cat: str, item: str, reason: str) -> None:
        self.skipped.append((cat, item, reason))

    def _unknown_ref(self, cat: str, ref: str) -> None:
        if ref not in self.unknown_refs:
            self.unknown_refs.append(ref)
        self._skip(cat, ref, "reference not on the board")

    def _unknown_net(self, cat: str, name: str) -> None:
        if name not in self.unknown_nets:
            self.unknown_nets.append(name)
        self._skip(cat, name, "net name not found on the board")

    def counts(self) -> dict:
        skipped_by_cat: Dict[str, int] = {}
        for cat, _item, _reason in self.skipped:
            skipped_by_cat[cat] = skipped_by_cat.get(cat, 0) + 1
        return {
            "applied": dict(self.applied),
            "skipped": skipped_by_cat,
            "unknown_refs": list(self.unknown_refs),
            "unknown_nets": list(self.unknown_nets),
            "ignored": dict(self.ignored),
            "warnings": len(self.warnings),
        }

    def as_dict(self) -> dict:
        d = self.counts()
        d["skipped_items"] = [{"category": c, "item": i, "reason": r}
                              for c, i, r in self.skipped]
        d["notes"] = list(self.notes)
        d["warning_messages"] = list(self.warnings)
        return d

    def summary(self) -> str:
        lines = []
        applied = ", ".join("%s %d" % kv for kv in self.applied.items()) or "nothing"
        lines.append("[CadMust] Constraints applied: %s" % applied)
        if self.skipped:
            lines.append("[CadMust] Constraints skipped (%d):" % len(self.skipped))
            for cat, item, reason in self.skipped:
                lines.append("    %s %s: %s" % (cat, item, reason))
        for n in self.notes:
            lines.append("[CadMust] Note: %s" % n)
        for w in self.warnings:
            lines.append("[CadMust] Warning: %s" % w)
        if self.ignored:
            lines.append("[CadMust] Ignored by the engine: %s" % ", ".join(
                "%s %d" % kv for kv in self.ignored.items()))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def zone_rect_nm(model: BoardModel, zone: str) -> Tuple[float, float, float, float]:
    """Rectangle of a zone of the 3x3 grid over the outline bbox (nm, absolute)."""
    ox1, oy1 = model.outline_xmin, model.outline_ymin
    w = (model.outline_xmax - ox1) / 3.0
    h = (model.outline_ymax - oy1) / 3.0
    idx = ZONES.index(zone)
    row, col = idx // 3, idx % 3
    return (ox1 + col * w, oy1 + row * h, ox1 + (col + 1) * w, oy1 + (row + 1) * h)


def _count_provenance(obj) -> int:
    """Count objects that carry req_ids or source (ignored by the engine)."""
    n = 0
    if isinstance(obj, dict):
        if "req_ids" in obj or "source" in obj:
            n += 1
        for v in obj.values():
            n += _count_provenance(v)
    elif isinstance(obj, list):
        for v in obj:
            n += _count_provenance(v)
    return n


def apply_constraints(model: BoardModel, cset: ConstraintSet,
                      log: Optional[Callable[[str], None]] = print) -> ConstraintReport:
    """Apply a constraint set to the model (in place) and report what happened.

    Order: meta, fixed_positions, back_side, net_priorities, template groups
    (rigid), soft groups, edge_locked (after templates so a rigid group with an
    edge-locked member is snapped as a whole), zone_preferences,
    keep_out_zones; finally moveable_indices is rebuilt.
    """
    rep = ConstraintReport()
    rep.warnings.extend(cset.warnings)
    fps = model.footprints
    ox, oy = model.outline_xmin, model.outline_ymin

    # Reference lookup: first occurrence wins (as in the checker)
    by_ref: Dict[str, int] = {}
    dup_refs = set()
    for i, f in enumerate(fps):
        if f.reference in by_ref:
            dup_refs.add(f.reference)
        else:
            by_ref[f.reference] = i
    warned_dups = set()

    def lookup(cat: str, ref: str) -> Optional[int]:
        idx = by_ref.get(ref)
        if idx is None:
            rep._unknown_ref(cat, ref)
            return None
        if ref in dup_refs and ref not in warned_dups:
            warned_dups.add(ref)
            rep.warnings.append("reference %s occurs more than once; the first "
                                "occurrence is used" % ref)
        return idx

    # Net lookup: exact, then normalised
    net_exact: Dict[str, object] = {}
    net_norm: Dict[str, object] = {}
    for net in model.nets.values():
        net_exact.setdefault(net.net_name, net)
        net_norm.setdefault(norm_net(net.net_name), net)

    def find_net(name: str):
        net = net_exact.get(name)
        if net is None:
            net = net_norm.get(norm_net(name))
        return net

    pc = cset.placement

    # --- meta ---
    dims = cset.meta.get("board_dimensions_mm")
    if dims:
        w = (model.outline_xmax - model.outline_xmin) / MM
        h = (model.outline_ymax - model.outline_ymin) / MM
        err = max(abs(w - dims[0]), abs(h - dims[1]))
        if err > 1.0:
            rep.warnings.append(
                "board outline bbox %.2f x %.2f mm differs from declared "
                "%.2f x %.2f mm by %.2f mm" % (w, h, dims[0], dims[1], err))

    # --- fixed positions ---
    for it in pc.get("fixed_positions", []) or []:
        idx = lookup("fixed_positions", it["ref"])
        if idx is None:
            continue
        f = fps[idx]
        f.x = ox + int(round(it["x_mm"] * MM))
        f.y = oy + int(round(it["y_mm"] * MM))
        f.locked = True
        rep._applied("fixed_positions")

    # --- back side ---
    for it in pc.get("back_side", []) or []:
        idx = lookup("back_side", it["ref"])
        if idx is None:
            continue
        fps[idx].is_front = False
        rep._applied("back_side")

    # --- net priorities ---
    np_ = cset.net_priorities
    for tier in ("high", "critical"):   # critical last, so it wins on overlap
        body = np_.get(tier) or {}
        names = body.get("nets", []) or []
        if not names:
            continue
        weight = body.get("weight")
        if weight is None:
            for name in names:
                rep._skip("net_priorities", name, "tier %s has no weight" % tier)
            continue
        for name in names:
            net = find_net(name)
            if net is None:
                rep._unknown_net("net_priorities", name)
                continue
            if net.is_excluded:
                rep.notes.append("net %s (%s) is excluded from HPWL; its %s weight "
                                 "has no effect" % (name, net.net_name, tier))
            net.weight = float(weight)
            rep._applied("net_weights")
    ex = np_.get("exclude") or {}
    for name in ex.get("nets", []) or []:
        net = find_net(name)
        if net is None:
            rep._unknown_net("net_exclude", name)
            continue
        net.is_excluded = True
        rep._applied("net_excluded")

    # --- component groups ---
    soft_groups: List[Tuple[str, List[int], int]] = []
    for g in cset.component_groups:
        gid = g["id"]
        t = g.get("template")
        if t:
            _apply_template(model, g, t, lookup, rep)
        present: List[int] = []
        for ref in g["members"]:
            idx = lookup("component_groups", ref)
            if idx is not None and idx not in present:
                present.append(idx)
        if "max_span_mm" not in g:
            rep._skip("component_groups", gid, "no max_span_mm, no cohesion term")
            continue
        if len(present) < 2:
            rep._skip("component_groups", gid,
                      "only %d member(s) on the board, need 2" % len(present))
            continue
        soft_groups.append((gid, present, int(round(g["max_span_mm"] * MM))))
        rep._applied("soft_groups")
    model.soft_groups = soft_groups or None

    # --- edge locks ---
    edge_set = set(model.edge_locked_indices or ())
    for it in pc.get("edge_locked", []) or []:
        ref, edge = it["ref"], it["edge"]
        idx = lookup("edge_locked", ref)
        if idx is None:
            continue
        f = fps[idx]
        gi = model.fp_to_group.get(idx)
        if f.locked or (gi is not None and model.component_groups[gi].locked):
            rep._skip("edge_locked", ref, "footprint is locked; left where it is")
            continue
        f.edge_lock = edge
        edge_set.add(idx)
        rep._applied("edge_locked")
    model.edge_locked_indices = edge_set or None
    if edge_set:
        # Detect conflicting edges on the same axis inside one rigid group
        seen_groups = set()
        for idx in sorted(edge_set):
            gi = model.fp_to_group.get(idx)
            if gi is not None:
                if gi in seen_groups:
                    continue
                seen_groups.add(gi)
                locked = [i for i in model.component_groups[gi].member_indices
                          if i in edge_set]
                axes: Dict[str, str] = {}
                for i in locked:
                    e = fps[i].edge_lock
                    ax = "x" if e in ("left", "right") else "y"
                    if ax in axes and axes[ax] != e:
                        rep.warnings.append(
                            "rigid group with %s has conflicting edge locks on one "
                            "axis; the first member decides" % ", ".join(
                                fps[k].reference for k in locked))
                    axes.setdefault(ax, e)
            resnap_edges(model, _get_group_members(model, idx))

    # --- zone preferences ---
    zone_prefs: Dict[int, Tuple[float, float, float, float]] = dict(model.zone_prefs or {})
    for it in pc.get("zone_preferences", []) or []:
        ref = it["ref"]
        idx = lookup("zone_preferences", ref)
        if idx is None:
            continue
        if fps[idx].locked:
            rep._skip("zone_preferences", ref, "footprint is locked; no zone term")
            continue
        zone_prefs[idx] = zone_rect_nm(model, it["zone"])
        rep._applied("zone_preferences")
    model.zone_prefs = zone_prefs or None

    # --- keep-out zones (component clearance) ---
    for it in pc.get("keep_out_zones", []) or []:
        ref = it["ref"]
        idx = lookup("keep_out_zones", ref)
        if idx is None:
            continue
        layers = set(it.get("layers") or [])
        side: Optional[bool] = None
        cu = layers & {"F.Cu", "B.Cu"}
        if cu == {"F.Cu"}:
            side = True
        elif cu == {"B.Cu"}:
            side = False
        f = fps[idx]
        if "antenna" in it:
            try:
                res = resolve_antenna_item(f, it)
            except AntennaError as e:
                rep._skip("keep_out_zones", ref, "antenna: %s" % e)
                continue
            if res.module_end_source == "courtyard":
                rep.warnings.append("keep_out_zones %s: antenna end taken at the courtyard "
                                    "bbox edge (no module_end_beyond_pads_mm)" % ref)
            if res.zero_area:
                rep._skip("keep_out_zones", ref, "antenna: courtyard already covers the "
                          "keep-out (zero-area rect)")
                continue
            f.keepout_rects = list(getattr(f, "keepout_rects", ()) or ()) + [res.rect]
            f.clearance_side = side
            rep._applied("keep_out_zones")
            continue
        c = int(round(it["clearance_mm"] * MM))
        if c <= 0:
            rep._skip("keep_out_zones", ref, "clearance is 0")
            continue
        if f.clearance and f.clearance >= c:
            rep._skip("keep_out_zones", ref, "a larger clearance is already set")
            continue
        f.clearance = c
        f.clearance_side = side
        rep._applied("keep_out_zones")

    # --- moveable list (fixed positions and rigid groups changed it) ---
    _rebuild_moveable(model)

    # --- ignored sections ---
    sc = cset.data.get("signal_classes") or []
    dn = cset.data.get("design_notes") or []
    prov = _count_provenance(cset.data)
    if sc:
        rep.ignored["signal_classes"] = len(sc)
    if dn:
        rep.ignored["design_notes"] = len(dn)
    if prov:
        rep.ignored["req_ids/source"] = prov

    if log is not None:
        log(rep.summary())
    return rep


def template_position(anchor, dx_mm: float, dy_mm: float) -> Tuple[int, int]:
    """Board position of a template member: the anchor-frame offset (dx, dy)
    rotated by the anchor's angle with the same transform as Pad.abs_position."""
    off = Pad(net_code=0, net_name="", offset_x=int(round(dx_mm * MM)),
              offset_y=int(round(dy_mm * MM)))
    return off.abs_position(anchor.x, anchor.y, anchor.angle_deg)


def anchor_frame_offset(anchor, member) -> Tuple[float, float]:
    """Inverse of template_position: the member's offset from the anchor in
    the anchor frame, in mm."""
    import math
    dx = member.x - anchor.x
    dy = member.y - anchor.y
    rad = math.radians(anchor.angle_deg)   # inverse of abs_position's radians(-angle)
    c, s_ = math.cos(rad), math.sin(rad)
    return ((dx * c - dy * s_) / MM, (dx * s_ + dy * c) / MM)


def _apply_template(model: BoardModel, g: dict, t: dict, lookup, rep: ConstraintReport) -> None:
    """Place template members relative to the anchor and register a rigid group."""
    fps = model.footprints
    gid = g["id"]
    anchor_idx = lookup("component_groups", t["anchor"])
    if anchor_idx is None:
        rep._skip("component_groups", gid, "template anchor %s not on the board"
                  % t["anchor"])
        return
    if anchor_idx in model.fp_to_group:
        rep._skip("component_groups", gid, "anchor %s is already in a rigid group"
                  % t["anchor"])
        return
    anchor = fps[anchor_idx]
    rp = t.get("relative_positions", {}) or {}
    rigid = [anchor_idx]
    for ref, pos in rp.items():
        idx = lookup("component_groups", ref)
        if idx is None:
            continue
        if idx in rigid:
            rep._skip("component_groups", "%s.%s" % (gid, ref), "listed twice")
            continue
        if idx in model.fp_to_group:
            rep._skip("component_groups", "%s.%s" % (gid, ref),
                      "already in another rigid group; not part of this template")
            continue
        if fps[idx].locked:
            rep._skip("component_groups", "%s.%s" % (gid, ref),
                      "footprint is locked; not moved and not part of the rigid body")
            continue
        f = fps[idx]
        f.x, f.y = template_position(anchor, pos["dx_mm"], pos["dy_mm"])
        rigid.append(idx)
    in_template = {t["anchor"]} | set(rp.keys())
    loose = [r for r in g["members"] if r not in in_template]
    if loose:
        rep.notes.append("group %s: %d member(s) not in the template are not moved "
                         "and not part of the rigid body%s: %s"
                         % (gid, len(loose),
                            " (the cohesion term covers them)" if "max_span_mm" in g else "",
                            ", ".join(loose)))
    if len(rigid) < 2:
        rep._skip("component_groups", gid, "template has fewer than 2 placeable members")
        return
    gi = len(model.component_groups)
    model.component_groups.append(ComponentGroup(member_indices=rigid,
                                                 locked=anchor.locked))
    for idx in rigid:
        model.fp_to_group[idx] = gi
    rep._applied("rigid_groups")
    rep._applied("rigid_group_members", len(rigid))


def _rebuild_moveable(model: BoardModel) -> None:
    """Recompute moveable_indices like extract_board_model does: locked
    footprints and locked groups out, one representative (the first member)
    per unlocked group. Never adds an index that was not moveable before."""
    fps = model.footprints
    old = set(model.moveable_indices)
    excluded = set()
    for grp in model.component_groups:
        if grp.locked or any(fps[i].locked for i in grp.member_indices):
            grp.locked = True
            excluded.update(grp.member_indices)
        else:
            excluded.update(grp.member_indices[1:])
    model.moveable_indices = [i for i, f in enumerate(fps)
                              if not f.locked and i not in excluded and i in old]
