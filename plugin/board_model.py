"""Extract KiCad board data into pure Python structures for the optimizer."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Sequence


# Power net detection — net name patterns that are unambiguously power/ground.
# Strips a leading '/' (KiCad hierarchical prefix) before matching.
# Used to exclude power nets from HPWL: their contribution is a large constant
# that can't be optimised and dominates the incremental HPWL computation.
_POWER_NAME_PREFIXES = (
    'GND', 'AGND', 'DGND', 'PGND', 'SGND',  # ground variants
    'VSS',                                     # ground (CMOS)
    'VCC', 'VDD', 'VEE',                       # supply rails
    'VBAT', 'VBUS',                            # battery / USB bus
)
_POWER_VOLTAGE_RE = re.compile(r'^[+\-]\d[\d.]*V', re.IGNORECASE)


def _is_power_net_name(name: str) -> bool:
    """Return True if the net name looks like a power or ground net."""
    n = name.lstrip('/').upper()
    if any(n.startswith(p) for p in _POWER_NAME_PREFIXES):
        return True
    # +3V3, +5V, +3.3V, -12V, etc.
    return bool(_POWER_VOLTAGE_RE.match(n))


# All positions/dimensions in nanometers (KiCad native units: 1 mm = 1_000_000 nm)

# Through-hole pad rects closer than this gap are merged into one rect —
# large enough to fuse standard header/connector pitches (up to ~3.96mm
# with typical pad sizes) into a single strip, small enough to keep e.g.
# DIP rows (7.62mm apart) as separate strips.
TH_MERGE_GAP = 2_000_000  # 2 mm


def merge_pad_rects(rects: List[Tuple[int, int, int, int]],
                    gap: int = TH_MERGE_GAP) -> List[Tuple[int, int, int, int]]:
    """Merge rectangles that are within `gap` of each other into their
    union bbox, repeated until stable. O(k² · passes), k = rect count.

    Turns a pin-header row into one strip while keeping well-separated
    pads (e.g. corner mounting holes) as individual rects.
    """
    rects = list(rects)
    merged = True
    while merged:
        merged = False
        out: List[Tuple[int, int, int, int]] = []
        for r in rects:
            for oi, o in enumerate(out):
                if (r[0] - gap < o[2] and r[2] + gap > o[0] and
                        r[1] - gap < o[3] and r[3] + gap > o[1]):
                    out[oi] = (min(r[0], o[0]), min(r[1], o[1]),
                               max(r[2], o[2]), max(r[3], o[3]))
                    merged = True
                    break
            else:
                out.append(r)
        rects = out
    return rects

@dataclass
class Pad:
    """A single pad on a footprint."""
    net_code: int               # 0 = unconnected
    net_name: str
    offset_x: int              # pad offset from footprint center at 0° rotation (nm)
    offset_y: int
    # Copper geometry for the pad-level clearance term (2026-11-06). Plain
    # defaults are class attributes, so older pickles read "no data".
    # rect: local bbox at 0 degrees relative to the fp origin (nm), the
    # th_rects convention; side: "F", "B", "TH" (both sides) or "" (no
    # outer copper); clearance: pad local clearance, else the footprint's,
    # else 0 (nm).
    rect: Optional[Tuple[int, int, int, int]] = None
    side: str = ""
    clearance: int = 0

    def abs_position(self, fp_x: int, fp_y: int, angle_deg: float) -> Tuple[int, int]:
        """Compute absolute pad position given footprint center and rotation.

        KiCad uses clockwise rotation (Y-axis points down), so we negate
        the angle to convert to standard math (CCW) trigonometry.
        """
        rad = math.radians(-angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        rx = int(self.offset_x * cos_a - self.offset_y * sin_a)
        ry = int(self.offset_x * sin_a + self.offset_y * cos_a)
        return (fp_x + rx, fp_y + ry)


def _flip_layer_name(name: str) -> str:
    """F.Fab <-> B.Fab etc.; layers without a side prefix stay."""
    if name.startswith("F."):
        return "B." + name[2:]
    if name.startswith("B."):
        return "F." + name[2:]
    return name


def anchor_line_world(fp, side: str, end: int, span: Tuple[int, int]
                      ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """World end points of a module end line: the local line at outward
    distance `end` along `side` ("+x", "-x", "+y", "-y"), running over
    `span` (lo, hi) on the other local axis, rotated with fp's angle like
    Pad.abs_position. Shared by the snap (Footprint.edge_anchor_span_world)
    and the CadMust-Autopilot checker."""
    sgn = 1 if side[0] == "+" else -1
    rad = math.radians(-fp.angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    out = []
    for u in span:
        lx, ly = (sgn * end, u) if side[1] == "x" else (u, sgn * end)
        out.append((fp.x + int(round(lx * c - ly * s)),
                    fp.y + int(round(lx * s + ly * c))))
    return out[0], out[1]


@dataclass
class Footprint:
    """A component on the board."""
    reference: str             # "U1", "R3"
    index: int                 # position in the footprints list
    x: int                     # center x (nm)
    y: int                     # center y (nm)
    angle_deg: float           # rotation in degrees (any angle)
    width: int                 # bounding box width at 0° rotation (nm)
    height: int                # bounding box height at 0° rotation (nm)
    locked: bool
    pads: List[Pad] = field(default_factory=list)
    net_codes: Set[int] = field(default_factory=set)
    cx_offset: int = 0         # bbox center x offset from fp origin at 0° (nm)
    cy_offset: int = 0         # bbox center y offset from fp origin at 0° (nm)
    uuid_str: str = ""         # KiCad footprint UUID (unique even with duplicate refs)
    is_front: bool = True      # False = mounted on the back side (B.Cu)
    # Merged rects covering through-hole pads, local coords at 0° rotation
    # relative to the fp origin (empty = all-SMD footprint). TH pads exist
    # on both board sides, so these rects also block the opposite side
    # (unlike the body bbox).
    th_rects: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Constraint-file attributes (set by constraints.apply_constraints).
    # Plain defaults become class attributes, so BoardModels pickled before
    # these fields existed still read the defaults.
    edge_lock: Optional[str] = None      # "top"/"bottom"/"left"/"right": slides along that edge
    clearance: int = 0                   # component-centred keep-out ring (nm), 0 = none
    clearance_side: Optional[bool] = None  # None = both sides, True = front only, False = back only
    # Attached keep-out rects (constraints.antenna_keepout_rect), local coords
    # at 0 degrees relative to the fp origin, nm, the th_rects convention.
    # They rotate with the part and count in the overlap test only (not in
    # the boundary penalty, not in HPWL). The tuple default is a class
    # attribute, so pickles from before the field read "none".
    keepout_rects: Sequence[Tuple[int, int, int, int]] = ()
    # Local bbox at 0 degrees of the footprint's drawn graphics (lines, rects,
    # polys, circles, arcs on Fab, User and SilkS layers; no text), nm. A
    # cross-check for the datasheet antenna end, never the keep-out itself.
    # None = no such graphics, or a model extracted before the field existed.
    drawing_rect: Optional[Tuple[int, int, int, int]] = None
    # The same per drawing layer ("F.Fab", "Dwgs.User", "F.SilkS", ...), for
    # cross-checks (a Fab outline may be drawn as large as the courtyard).
    # Never defines a keep-out. None = none / old model.
    drawing_rects: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    # True when apply_constraints moved the part to the other board side
    # (Footprint.flip_side); apply_model.py then flips it on the board too.
    # Class default, so older pickles read False.
    side_flipped: bool = False
    # Edge lock details (set by constraints.apply_constraints; class defaults,
    # so older pickles read "none"). overhang_ok: the part may reach past its
    # locked edge (the boundary penalty and the off-board test clip its bbox
    # at that edge, moves/cost_function.locked_edge_clip). antenna_side: the
    # resolved local antenna side ("+x", "-x", "+y", "-y") of an
    # antenna_outward edge lock, None otherwise. edge_anchor_offset_nm: the
    # outward distance (nm, local frame at 0 degrees) from the origin to the
    # module end along antenna_side; resnap_edges puts that line on the
    # locked edge instead of the bbox. 0 = no anchor (bbox snap).
    # edge_anchor_span_nm: the module's own width across antenna_side, as
    # (lo, hi) of the pad-centre extent along the other local axis (nm, 0
    # degrees); the snap takes the outermost outline point over that width
    # as the edge reference (cost_function.anchor_edge_reference). None =
    # no width known (older pickles): the bbox-span innermost reference.
    overhang_ok: bool = False
    antenna_side: Optional[str] = None
    edge_anchor_offset_nm: int = 0
    edge_anchor_span_nm: Optional[Tuple[int, int]] = None
    # Cached trig values for abs_position (updated via set_angle)
    _cos_a: float = field(init=False, repr=False, default=1.0)
    _sin_a: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        self._update_trig()

    def _update_trig(self):
        rad = math.radians(-self.angle_deg)
        self._cos_a = math.cos(rad)
        self._sin_a = math.sin(rad)

    def set_angle(self, angle_deg: float):
        """Set rotation angle and update cached trig values."""
        self.angle_deg = angle_deg
        self._update_trig()

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return (xmin, ymin, xmax, ymax) accounting for rotation.

        The bbox center may not coincide with the footprint origin (e.g. for
        asymmetric connectors whose courtyard extends further on one side).
        cx_offset / cy_offset are the bbox-center offsets at 0° rotation;
        abs_position() rotates them into world space.
        """
        cx = self.x + int(self.cx_offset * self._cos_a - self.cy_offset * self._sin_a)
        cy = self.y + int(self.cx_offset * self._sin_a + self.cy_offset * self._cos_a)
        if int(self.angle_deg) % 180 == 90:
            hw, hh = self.height // 2, self.width // 2
        else:
            hw, hh = self.width // 2, self.height // 2
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    @property
    def th_bboxes(self) -> List[Tuple[int, int, int, int]]:
        """World-space rects of through-hole pad clusters (empty if all-SMD).

        Same rotation handling as bbox, applied per rect. TH pads (and
        NPTH holes) drill through the board, so these areas are occupied
        on both sides.
        """
        out = []
        swap = int(self.angle_deg) % 180 == 90
        for (x1, y1, x2, y2) in self.th_rects:
            rcx = (x1 + x2) // 2
            rcy = (y1 + y2) // 2
            cx = self.x + int(rcx * self._cos_a - rcy * self._sin_a)
            cy = self.y + int(rcx * self._sin_a + rcy * self._cos_a)
            if swap:
                hw, hh = (y2 - y1) // 2, (x2 - x1) // 2
            else:
                hw, hh = (x2 - x1) // 2, (y2 - y1) // 2
            out.append((cx - hw, cy - hh, cx + hw, cy + hh))
        return out

    @property
    def local_bbox(self) -> Tuple[int, int, int, int]:
        """The bbox in the local frame at 0 degrees, relative to the origin."""
        hw, hh = self.width // 2, self.height // 2
        return (self.cx_offset - hw, self.cy_offset - hh,
                self.cx_offset + hw, self.cy_offset + hh)

    def local_rect_to_world(self, r: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """World-space axis-aligned bbox of a local rect (rotated like the pads).

        Exact for multiples of 90 degrees; for other angles the bbox of the
        rotated rect.
        """
        c, s = self._cos_a, self._sin_a
        xs = []
        ys = []
        for lx, ly in ((r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])):
            xs.append(lx * c - ly * s)
            ys.append(lx * s + ly * c)
        return (self.x + int(round(min(xs))), self.y + int(round(min(ys))),
                self.x + int(round(max(xs))), self.y + int(round(max(ys))))

    def flip_side(self) -> None:
        """Move the part to the other board side, in place.

        Mirrors the footprint exactly as KiCad's FOOTPRINT::Flip(position,
        FLIP_DIRECTION_TOP_BOTTOM) does around the footprint's own origin:
        the position stays, every local y (pads, courtyard bbox offset, TH
        rects, attached keep-out rects, drawing rects) is negated, and the
        orientation becomes -angle. A model extracted from the flipped board
        then has the same local geometry and angle, because extraction
        un-rotates the as-placed (already mirrored) pad positions. The bbox
        width/height do not change. Toggles is_front and side_flipped.
        """
        def my(r):
            return (r[0], -r[3], r[2], -r[1])
        for pad in self.pads:
            pad.offset_y = -pad.offset_y
            r = getattr(pad, 'rect', None)
            if r is not None:
                pad.rect = my(r)
            ps = getattr(pad, 'side', "")
            if ps in ("F", "B"):
                pad.side = "B" if ps == "F" else "F"
        self.cy_offset = -self.cy_offset
        self.th_rects = [my(r) for r in (getattr(self, 'th_rects', None) or [])]
        ko = getattr(self, 'keepout_rects', ())
        if ko:
            self.keepout_rects = tuple(my(r) for r in ko)
        dr = getattr(self, 'drawing_rect', None)
        if dr is not None:
            self.drawing_rect = my(dr)
        drs = getattr(self, 'drawing_rects', None)
        if drs:
            self.drawing_rects = {_flip_layer_name(k): my(v) for k, v in drs.items()}
        a_side = getattr(self, 'antenna_side', None)
        if a_side in ("+y", "-y"):
            self.antenna_side = "-y" if a_side == "+y" else "+y"
        span = getattr(self, 'edge_anchor_span_nm', None)
        if span is not None and a_side in ("+x", "-x"):
            # a +/-x antenna side: the width runs along local y, mirrored
            self.edge_anchor_span_nm = (-span[1], -span[0])
        self.set_angle((-self.angle_deg) % 360.0)
        self.is_front = not getattr(self, 'is_front', True)
        self.side_flipped = not getattr(self, 'side_flipped', False)

    def edge_anchor_span_world(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """World end points of the module end line over the module's own
        width (edge_anchor_span_nm), None without an anchor or a width."""
        off = getattr(self, 'edge_anchor_offset_nm', 0)
        side = getattr(self, 'antenna_side', None)
        span = getattr(self, 'edge_anchor_span_nm', None)
        if not off or side is None or span is None:
            return None
        return anchor_line_world(self, side, off, span)

    def edge_anchor_world(self) -> Optional[Tuple[int, int]]:
        """World position of the edge anchor point (the module end on the
        antenna axis through the origin), None without an anchor."""
        off = getattr(self, 'edge_anchor_offset_nm', 0)
        side = getattr(self, 'antenna_side', None)
        if not off or side is None:
            return None
        sgn = 1 if side[0] == "+" else -1
        lx, ly = (sgn * off, 0) if side[1] == "x" else (0, sgn * off)
        c, s = self._cos_a, self._sin_a
        return (self.x + int(round(lx * c - ly * s)),
                self.y + int(round(lx * s + ly * c)))

    @property
    def keepout_bboxes(self) -> List[Tuple[int, int, int, int]]:
        """World-space rects of the attached keep-outs (empty if none)."""
        rects = getattr(self, 'keepout_rects', ())
        if not rects:
            return []
        return [self.local_rect_to_world(r) for r in rects]


@dataclass
class Net:
    """A net with references to its pads."""
    net_code: int
    net_name: str
    pad_refs: List[Tuple[int, int]] = field(default_factory=list)  # (fp_index, pad_index)
    is_excluded: bool = False  # True → excluded from HPWL (large constant, can't be optimised)
    weight: float = 1.0        # HPWL weight in the total cost (constraint-file net priority)


@dataclass
class KeepOut:
    """A rectangular footprint keep-out zone (bbox of a KiCad rule area).

    ``layers`` records the copper layers the rule area is on: ``"all"`` when
    it covers both outer copper layers (or none could be read), otherwise the
    comma-joined layer names, e.g. ``"F.Cu"``. The cost function and the move
    filter still treat every keep-out as applying to both sides; the field is
    there so a single-layer keep-out can later apply to that side only. Old
    pickles without the field read the class default ``"all"``.
    """
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    layers: str = "all"


@dataclass
class ComponentGroup:
    """A group of footprints that must move as a rigid body."""
    member_indices: List[int]   # footprint indices in this group
    locked: bool = False

@dataclass
class BoardModel:
    """Complete board state for the optimizer — pure Python, no pcbnew."""
    footprints: List[Footprint]
    nets: Dict[int, Net]
    outline_xmin: int
    outline_ymin: int
    outline_xmax: int
    outline_ymax: int
    keepouts: List[KeepOut] = field(default_factory=list)
    moveable_indices: List[int] = field(default_factory=list)
    outline_polygon: Optional[List[Tuple[int, int]]] = None  # board outline vertices (nm)
    component_groups: List[ComponentGroup] = field(default_factory=list)
    # Map from fp_index → group index (None if not grouped)
    fp_to_group: Dict[int, int] = field(default_factory=dict)
    # Constraint-file state (set by constraints.apply_constraints; None = none).
    # Plain None defaults are class attributes, so old pickles keep working.
    # Footprint indices with an edge_lock (fast membership test in moves).
    edge_locked_indices: Optional[Set[int]] = None
    # Soft cohesion groups: list of (group_id, member_indices, max_span_nm).
    soft_groups: Optional[List[Tuple[str, List[int], int]]] = None
    # Zone preferences: fp_index -> preferred zone rect (xmin, ymin, xmax, ymax) in nm.
    zone_prefs: Optional[Dict[int, Tuple[float, float, float, float]]] = None


def _opt_nm(v) -> int:
    """KiCad 9+ returns std::optional<int> (None or int), older an int."""
    try:
        return max(0, int(v)) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def _pad_copper(pad, pos, cos_a, sin_a, has_hole, fp_clearance):
    """(rect, side, clearance) of one pad for the pad-clearance term.

    rect: the pad's world bbox (pad.GetBoundingBox, rotation included)
    un-rotated into the footprint frame at 0 degrees, as a bbox; exact for
    parts at multiples of 90 degrees.
    """
    import pcbnew as pcbnew_mod
    bb = pad.GetBoundingBox()
    xs, ys = [], []
    for wx, wy in ((bb.GetLeft(), bb.GetTop()), (bb.GetRight(), bb.GetTop()),
                   (bb.GetRight(), bb.GetBottom()), (bb.GetLeft(), bb.GetBottom())):
        dx = wx - pos.x
        dy = wy - pos.y
        xs.append(dx * cos_a - dy * sin_a)
        ys.append(dx * sin_a + dy * cos_a)
    rect = (int(round(min(xs))), int(round(min(ys))),
            int(round(max(xs))), int(round(max(ys))))
    if has_hole:
        side = "TH"
    elif pad.IsOnLayer(pcbnew_mod.F_Cu):
        side = "F"
    elif pad.IsOnLayer(pcbnew_mod.B_Cu):
        side = "B"
    else:
        side = ""
    clr = _opt_nm(pad.GetLocalClearance())
    if not clr:
        clr = fp_clearance
    return rect, side, clr


def _rule_area_forbids_footprints(zone) -> bool:
    """True when a KiCad rule area forbids footprints.

    KiCad 6+ exposes ``GetDoNotAllowFootprints()``. A zone object without
    that method (very old KiCad, or a stub) is treated as a footprint
    keep-out, which was the behaviour before this filter existed.
    """
    getter = getattr(zone, "GetDoNotAllowFootprints", None)
    if getter is None:
        return True
    try:
        return bool(getter())
    except Exception:
        return True


def _rule_area_layers(board, zone) -> str:
    """Copper layers of a rule area as ``"all"`` or comma-joined names."""
    try:
        names = [board.GetLayerName(l) for l in zone.GetLayerSet().Seq()]
    except Exception:
        return "all"
    names = [n for n in names if n.endswith(".Cu")]
    if not names or ("F.Cu" in names and "B.Cu" in names):
        return "all"
    return ",".join(sorted(names))


def extract_keepouts(board) -> List[KeepOut]:
    """Footprint keep-outs from the board's rule areas, as bboxes.

    Only rule areas that forbid footprints count: a rule area that only
    forbids copper pour, tracks or vias (e.g. "(copperpour not_allowed)
    (footprints allowed)" under magnetics or a mains section) says nothing
    about placement.
    """
    keepouts: List[KeepOut] = []
    for zone in board.Zones():
        if not (zone.GetIsRuleArea() and _rule_area_forbids_footprints(zone)):
            continue
        outline = zone.Outline()
        if outline:
            zb = outline.BBox()
            keepouts.append(KeepOut(
                xmin=zb.GetX(),
                ymin=zb.GetY(),
                xmax=zb.GetX() + zb.GetWidth(),
                ymax=zb.GetY() + zb.GetHeight(),
                layers=_rule_area_layers(board, zone),
            ))
    return keepouts


# Layers whose graphics describe the physical part outline (drawing_rect).
_DRAWING_LAYER_NAMES = ("F_Fab", "B_Fab", "Dwgs_User", "Cmts_User",
                        "F_SilkS", "B_SilkS")


def _shape_points_world(item, pcbnew_mod) -> List[Tuple[float, float]]:
    """Geometric points (world coords, no stroke width) that bound one
    graphic shape: segment ends, rect corners, circle bbox corners, arc
    samples, polygon vertices, bezier control points. Text returns []."""
    try:
        cls = item.GetClass()
    except Exception:
        cls = ""
    if "TEXT" in str(cls).upper():
        return []
    if not hasattr(item, "GetShape"):
        return []
    pts: List[Tuple[float, float]] = []
    shape = item.GetShape()

    def const(name):
        return getattr(pcbnew_mod, name, object())

    try:
        if shape == const("SHAPE_T_CIRCLE"):
            c = item.GetCenter()
            r = item.GetRadius()
            pts += [(c.x - r, c.y - r), (c.x + r, c.y + r),
                    (c.x - r, c.y + r), (c.x + r, c.y - r)]
        elif shape == const("SHAPE_T_ARC"):
            c = item.GetCenter()
            r = float(item.GetRadius())
            a0 = math.atan2(item.GetStart().y - c.y, item.GetStart().x - c.x)
            am = math.atan2(item.GetArcMid().y - c.y, item.GetArcMid().x - c.x)
            a1 = math.atan2(item.GetEnd().y - c.y, item.GetEnd().x - c.x)
            # sweep from a0 to a1 through am
            def norm(a):
                return a % (2 * math.pi)
            sweep = norm(a1 - a0)
            if norm(am - a0) > sweep:      # mid lies on the other way round
                sweep = sweep - 2 * math.pi
            for k in range(33):
                a = a0 + sweep * k / 32
                pts.append((c.x + r * math.cos(a), c.y + r * math.sin(a)))
        elif shape == const("SHAPE_T_RECTANGLE") or shape == const("SHAPE_T_RECT"):
            try:
                pts += [(p.x, p.y) for p in item.GetRectCorners()]
            except Exception:
                a, b = item.GetStart(), item.GetEnd()
                pts += [(a.x, a.y), (b.x, b.y), (a.x, b.y), (b.x, a.y)]
        elif shape == const("SHAPE_T_POLY"):
            poly = item.GetPolyShape()
            for oi in range(poly.OutlineCount()):
                ol = poly.Outline(oi)
                for vi in range(ol.PointCount()):
                    p = ol.CPoint(vi)
                    pts.append((p.x, p.y))
        elif shape == const("SHAPE_T_BEZIER"):
            for getter in ("GetStart", "GetBezierC1", "GetBezierC2", "GetEnd"):
                p = getattr(item, getter)()
                pts.append((p.x, p.y))
        else:  # segment and anything unknown: the two end points
            a, b = item.GetStart(), item.GetEnd()
            pts += [(a.x, a.y), (b.x, b.y)]
    except Exception:
        return []
    return pts


def extract_drawing_rects(fp, pcbnew_mod) -> Dict[str, Tuple[int, int, int, int]]:
    """Per-layer local bboxes at 0 degrees (relative to the fp origin, nm) of
    the footprint's drawn graphics on the Fab, User and SilkS layers, keyed
    by KiCad layer name ("F.Fab", "Dwgs.User", ...). Layers without shapes
    are absent.

    World points are un-rotated like the pad offsets, so a footprint on the
    back side gives its mirrored local outline, consistent with its pads."""
    layer_ids = {}
    for name in _DRAWING_LAYER_NAMES:
        v = getattr(pcbnew_mod, name, None)
        if v is not None:
            layer_ids[v] = name.replace("_", ".")
    pos = fp.GetPosition()
    angle = fp.GetOrientationDegrees() % 360.0
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    out: Dict[str, List[int]] = {}
    for item in fp.GraphicalItems():
        try:
            lname = layer_ids.get(item.GetLayer())
        except Exception:
            continue
        if lname is None:
            continue
        for (wx, wy) in _shape_points_world(item, pcbnew_mod):
            tx = wx - pos.x
            ty = wy - pos.y
            lx = int(round(tx * cos_a - ty * sin_a))
            ly = int(round(tx * sin_a + ty * cos_a))
            r = out.get(lname)
            if r is None:
                out[lname] = [lx, ly, lx, ly]
            else:
                r[0] = min(r[0], lx); r[1] = min(r[1], ly)
                r[2] = max(r[2], lx); r[3] = max(r[3], ly)
    return {k: (v[0], v[1], v[2], v[3]) for k, v in out.items()
            if not (v[0] == v[2] and v[1] == v[3])}


def union_rect(rects) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of a collection of rects, None when empty."""
    rects = list(rects)
    if not rects:
        return None
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))


def _courtyard_points_world(item, pcbnew_mod, pos, cos_a, sin_a
                            ) -> List[Tuple[float, float]]:
    """World points that bound one courtyard / fab graphic for the placement
    bbox. Start/end are only geometry for segments, rects, arcs (plus the arc
    mid) and beziers. A polygon (fp_poly) returns (0, 0) from GetStart() and
    GetEnd() in world coordinates, so it contributes its outline vertices
    only; using start/end there stretched e.g. a U.FL courtyard to the board
    origin. Circles give the corners of their local (0 degree) bbox mapped
    back to world, as before."""
    def const(*names):
        vals = set()
        for n in names:
            v = getattr(pcbnew_mod, n, None)
            if v is not None:
                vals.add(v)
        return vals

    pts: List[Tuple[float, float]] = []
    try:
        shape_val = item.GetShape() if hasattr(item, "GetShape") else None
    except Exception:
        shape_val = None

    if shape_val is not None and shape_val in const("SHAPE_T_CIRCLE", "S_CIRCLE"):
        try:
            c = item.GetCenter()
            r = item.GetRadius()
        except AttributeError:
            return pts
        tx_c = c.x - pos.x
        ty_c = c.y - pos.y
        lcx = tx_c * cos_a - ty_c * sin_a
        lcy = tx_c * sin_a + ty_c * cos_a

        def local_to_world(lx, ly):
            return (pos.x + lx * cos_a + ly * sin_a, pos.y - lx * sin_a + ly * cos_a)
        for (lx, ly) in ((lcx - r, lcy - r), (lcx + r, lcy - r),
                         (lcx - r, lcy + r), (lcx + r, lcy + r)):
            pts.append(local_to_world(lx, ly))
        return pts

    if shape_val is not None and shape_val in const("SHAPE_T_POLY", "S_POLYGON"):
        try:
            poly = item.GetPolyShape()
            for oi in range(poly.OutlineCount()):
                outline = poly.Outline(oi)
                for vi in range(outline.PointCount()):
                    p = outline.CPoint(vi)
                    pts.append((p.x, p.y))
        except AttributeError:
            pass
        return pts

    # Segment, rect, arc, bezier (and unknown shapes): start and end.
    try:
        a, b = item.GetStart(), item.GetEnd()
        pts += [(a.x, a.y), (b.x, b.y)]
    except AttributeError:
        return pts
    if shape_val is not None and shape_val in const("SHAPE_T_ARC", "S_ARC"):
        try:
            m = item.GetArcMid()
            pts.append((m.x, m.y))
        except AttributeError:
            pass
    elif shape_val is not None and shape_val in const("SHAPE_T_BEZIER", "S_CURVE"):
        for getter in ("GetBezierC1", "GetBezierC2"):
            try:
                p = getattr(item, getter)()
                pts.append((p.x, p.y))
            except AttributeError:
                pass
    return pts


def extract_board_model(board, selected_only: bool = False) -> BoardModel:
    """
    Extract board data from a pcbnew.BOARD into a BoardModel.

    This is the ONLY function in the optimizer that reads pcbnew APIs.
    It runs once at the start of optimization.

    If selected_only is True, only footprints currently selected in the
    KiCad editor will be moveable; all others are treated as locked.
    """
    footprints = []
    nets: Dict[int, Net] = {}

    # Margin around pad extents for the placement bounding box (0.25 mm).
    # With pad physical sizes included in the extent calculation,
    # 0.25 mm closely matches IPC courtyard envelopes.
    PAD_MARGIN = 250_000

    for i, fp in enumerate(board.GetFootprints()):
        pos = fp.GetPosition()
        angle = fp.GetOrientationDegrees() % 360.0

        pads_list = []
        fp_net_codes: Set[int] = set()

        # Track extents at 0° rotation to compute a tight bounding box.
        # Initialise to None; set from first pad or courtyard data.
        ext_xmin = ext_xmax = ext_ymin = ext_ymax = None

        # Per-pad rects for through-hole pads (incl. NPTH holes) — these
        # occupy both board sides, unlike the SMD body. Merged into
        # cluster rects after the pad loop.
        th_pad_rects: List[Tuple[int, int, int, int]] = []

        # KiCad uses CW rotation (Y-down), so un-rotate with +angle
        # (not -angle) to recover canonical 0° pad offsets.
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        fp_clearance = _opt_nm(getattr(fp, "GetLocalClearance", lambda: None)())
        for j, pad in enumerate(fp.Pads()):
            pad_pos = pad.GetPosition()
            nc = pad.GetNetCode()

            # Un-rotate pad offset to get canonical offset at 0°
            dx = pad_pos.x - pos.x
            dy = pad_pos.y - pos.y
            offset_x = int(dx * cos_a - dy * sin_a)
            offset_y = int(dx * sin_a + dy * cos_a)

            # Pad physical size — use max dimension as conservative extent
            # (handles pads with custom rotation relative to footprint)
            pad_size = pad.GetSize()
            pad_half = max(pad_size.x, pad_size.y) // 2

            # Expand extents to include this pad
            if ext_xmin is None:
                ext_xmin = offset_x - pad_half
                ext_xmax = offset_x + pad_half
                ext_ymin = offset_y - pad_half
                ext_ymax = offset_y + pad_half
            else:
                ext_xmin = min(ext_xmin, offset_x - pad_half)
                ext_xmax = max(ext_xmax, offset_x + pad_half)
                ext_ymin = min(ext_ymin, offset_y - pad_half)
                ext_ymax = max(ext_ymax, offset_y + pad_half)

            # Through-hole detection: HasHole() (KiCad 8+), fall back to
            # a nonzero drill size on older APIs.
            try:
                has_hole = pad.HasHole()
            except AttributeError:
                drill = pad.GetDrillSize()
                has_hole = drill.x > 0 or drill.y > 0
            if has_hole:
                th_pad_rects.append((
                    offset_x - pad_half - PAD_MARGIN,
                    offset_y - pad_half - PAD_MARGIN,
                    offset_x + pad_half + PAD_MARGIN,
                    offset_y + pad_half + PAD_MARGIN,
                ))

            p = Pad(
                net_code=nc,
                net_name=pad.GetNetname(),
                offset_x=offset_x,
                offset_y=offset_y,
            )
            try:
                p.rect, p.side, p.clearance = _pad_copper(
                    pad, pos, cos_a, sin_a, has_hole, fp_clearance)
            except Exception:
                pass   # no copper data: the pad-clearance term skips the pad
            pads_list.append(p)

            if nc > 0:
                fp_net_codes.add(nc)
                if nc not in nets:
                    nets[nc] = Net(net_code=nc, net_name=pad.GetNetname())

        # Also expand extents from outline graphics.
        # Prefer F_CrtYd/B_CrtYd; fall back to F_Fab/B_Fab for footprints
        # that omit a courtyard (the fab outline reflects the physical body).
        try:
            import pcbnew as _pcbnew
            ext_snap = (ext_xmin, ext_ymin, ext_xmax, ext_ymax)  # snapshot before graphics
            for layer_set in (
                (_pcbnew.F_CrtYd, _pcbnew.B_CrtYd),
                (_pcbnew.F_Fab,   _pcbnew.B_Fab),
            ):
                for item in fp.GraphicalItems():
                    if item.GetLayer() not in layer_set:
                        continue
                    pts_world = _courtyard_points_world(
                        item, _pcbnew, pos, cos_a, sin_a)
                    for (px, py) in pts_world:
                        tx = px - pos.x
                        ty = py - pos.y
                        lx = int(tx * cos_a - ty * sin_a)
                        ly = int(tx * sin_a + ty * cos_a)
                        if ext_xmin is None:
                            ext_xmin = lx; ext_xmax = lx
                            ext_ymin = ly; ext_ymax = ly
                        else:
                            ext_xmin = min(ext_xmin, lx); ext_xmax = max(ext_xmax, lx)
                            ext_ymin = min(ext_ymin, ly); ext_ymax = max(ext_ymax, ly)
                # Stop after courtyard if it contributed anything
                if (ext_xmin, ext_ymin, ext_xmax, ext_ymax) != ext_snap:
                    break
        except Exception:
            pass  # graphics not available; pad extents alone are used

        if ext_xmin is None:
            ext_xmin = ext_xmax = ext_ymin = ext_ymax = 0

        # Bounding box = union of pad + courtyard extents, plus margin.
        # PAD_MARGIN is added symmetrically so it doesn't shift the center.
        fp_width = (ext_xmax - ext_xmin) + 2 * PAD_MARGIN
        fp_height = (ext_ymax - ext_ymin) + 2 * PAD_MARGIN
        # Ensure minimum size for single-pad / no-pad footprints
        fp_width  = max(fp_width,  2 * PAD_MARGIN)
        fp_height = max(fp_height, 2 * PAD_MARGIN)

        # Center of the bbox may not coincide with the fp origin for asymmetric
        # footprints (e.g. connectors with courtyard offset to one side).
        # Store the local (0°) offset so bbox property can apply abs_position().
        fp_cx_offset = (ext_xmin + ext_xmax) // 2
        fp_cy_offset = (ext_ymin + ext_ymax) // 2

        # Merge per-pad TH rects into cluster rects (header row → one strip,
        # well-separated pads → individual rects)
        fp_th_rects = merge_pad_rects(th_pad_rects)

        try:
            import pcbnew as _pcbnew
            fp_drawing_rects = extract_drawing_rects(fp, _pcbnew)
        except Exception:
            fp_drawing_rects = {}
        fp_drawing_rect = union_rect(fp_drawing_rects.values())

        is_locked = fp.IsLocked() or (selected_only and not fp.IsSelected())
        try:
            uuid = fp.m_Uuid.AsString()
        except Exception:
            uuid = ""
        try:
            is_front = not fp.IsFlipped()
        except Exception:
            is_front = True
        f = Footprint(
            reference=fp.GetReference(),
            index=i,
            x=pos.x,
            y=pos.y,
            angle_deg=angle,
            width=fp_width,
            height=fp_height,
            cx_offset=fp_cx_offset,
            cy_offset=fp_cy_offset,
            locked=is_locked,
            pads=pads_list,
            net_codes=fp_net_codes,
            uuid_str=uuid,
            is_front=is_front,
            th_rects=fp_th_rects,
            drawing_rect=fp_drawing_rect,
            drawing_rects=fp_drawing_rects or None,
        )
        footprints.append(f)

    # Build pad_refs for each net
    for fi, fp in enumerate(footprints):
        for pi, pad in enumerate(fp.pads):
            if pad.net_code > 0:
                nets[pad.net_code].pad_refs.append((fi, pi))

    # Board outline bounding box
    bbox = board.GetBoardEdgesBoundingBox()
    ox = bbox.GetX()
    oy = bbox.GetY()

    keepouts = extract_keepouts(board)

    # Extract component groups
    component_groups: List[ComponentGroup] = []
    fp_to_group: Dict[int, int] = {}
    try:
        import pcbnew as _pcbnew
        # Build UUID → footprint index map (uuid_str is now stored on Footprint)
        uuid_to_idx: Dict[str, int] = {
            f.uuid_str: f.index for f in footprints if f.uuid_str
        }

        for grp in board.Groups():
            member_uuids = []
            for item in grp.GetItems():
                uid = str(item.m_Uuid.AsString())
                if uid in uuid_to_idx:
                    member_uuids.append(uid)

            if len(member_uuids) >= 2:
                indices = [uuid_to_idx[u] for u in member_uuids]
                grp_locked = grp.IsLocked() or any(
                    footprints[i].locked for i in indices)
                gi = len(component_groups)
                component_groups.append(ComponentGroup(
                    member_indices=indices, locked=grp_locked))
                for idx in indices:
                    fp_to_group[idx] = gi
    except Exception:
        pass  # group extraction not available

    # Build moveable list: exclude locked footprints and locked-group members.
    # For unlocked groups, include only ONE representative per group to avoid
    # selecting the same group multiple times.
    locked_by_group: Set[int] = set()
    group_reps: Set[int] = set()
    for grp in component_groups:
        if grp.locked:
            locked_by_group.update(grp.member_indices)
        else:
            # Use the first member as the group representative
            group_reps.add(grp.member_indices[0])
            # Exclude other members from moveable
            for idx in grp.member_indices[1:]:
                locked_by_group.add(idx)

    moveable = [i for i, fp in enumerate(footprints)
                if not fp.locked and i not in locked_by_group]

    # Extract board outline polygon for non-rectangular boards
    outline_polygon: Optional[List[Tuple[int, int]]] = None
    try:
        import pcbnew as _pcbnew
        poly_set = _pcbnew.SHAPE_POLY_SET()
        try:
            # KiCad 10 added a required aInferOutlineIfNecessary argument.
            board.GetBoardPolygonOutlines(poly_set, True)
        except TypeError:
            board.GetBoardPolygonOutlines(poly_set)  # KiCad 7 to 9 signature
        if poly_set.OutlineCount() > 0:
            outline = poly_set.Outline(0)
            pts = []
            for vi in range(outline.PointCount()):
                pt = outline.CPoint(vi)
                pts.append((pt.x, pt.y))
            if len(pts) >= 3:
                outline_polygon = pts
    except Exception:
        pass  # polygon extraction not available; rectangular bbox used

    # Mark power/ground nets — excluded from HPWL (their contribution is a large
    # constant and can't be reduced by placement).
    #
    # Two complementary methods:
    #   1. #PWR phantom footprints (KiCad 5/6 boards): any net referenced by a
    #      footprint whose reference starts with '#PWR' is a power net.
    #   2. Net name pattern matching (all boards including KiCad 7): net names
    #      matching common power/ground patterns (GND*, VCC*, +nV, etc.).
    for fp in footprints:
        if fp.reference.startswith('#PWR'):
            for nc in fp.net_codes:
                if nc in nets:
                    nets[nc].is_excluded = True
    for nc, net in nets.items():
        if not net.is_excluded and _is_power_net_name(net.net_name):
            net.is_excluded = True

    return BoardModel(
        footprints=footprints,
        nets=nets,
        outline_xmin=ox,
        outline_ymin=oy,
        outline_xmax=ox + bbox.GetWidth(),
        outline_ymax=oy + bbox.GetHeight(),
        keepouts=keepouts,
        moveable_indices=moveable,
        outline_polygon=outline_polygon,
        component_groups=component_groups,
        fp_to_group=fp_to_group,
    )
