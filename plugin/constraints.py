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
                         copper layer = that side only
    meta                 board size sanity check (warning only)
    signal_classes, design_notes, req_ids, source: ignored by the engine
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
