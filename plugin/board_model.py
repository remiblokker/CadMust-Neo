"""Extract KiCad board data into pure Python structures for the optimizer."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional


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

@dataclass
class Pad:
    """A single pad on a footprint."""
    net_code: int               # 0 = unconnected
    net_name: str
    offset_x: int              # pad offset from footprint center at 0° rotation (nm)
    offset_y: int

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


@dataclass
class Net:
    """A net with references to its pads."""
    net_code: int
    net_name: str
    pad_refs: List[Tuple[int, int]] = field(default_factory=list)  # (fp_index, pad_index)
    is_excluded: bool = False  # True → excluded from HPWL (large constant, can't be optimised)


@dataclass
class KeepOut:
    """A rectangular keep-out zone."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int


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

        # KiCad uses CW rotation (Y-down), so un-rotate with +angle
        # (not -angle) to recover canonical 0° pad offsets.
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

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

            p = Pad(
                net_code=nc,
                net_name=pad.GetNetname(),
                offset_x=offset_x,
                offset_y=offset_y,
            )
            pads_list.append(p)

            if nc > 0:
                fp_net_codes.add(nc)
                if nc not in nets:
                    nets[nc] = Net(net_code=nc, net_name=pad.GetNetname())

        # Also expand extents from the courtyard outline.
        # fp.GraphicalItems() includes FP_SHAPE items on F_CrtYd / B_CrtYd.
        # Their GetStart()/GetEnd() coordinates are in BOARD (world) space,
        # so we apply the same (translate → un-rotate) transform used for pads
        # above to get footprint-local (0°) coordinates.
        # This catches components (inductors, connectors) whose body extends
        # well beyond their pad extents.
        try:
            import pcbnew as _pcbnew
            for item in fp.GraphicalItems():
                if item.GetLayer() not in (_pcbnew.F_CrtYd, _pcbnew.B_CrtYd):
                    continue
                pts_world = []
                try:
                    # For circles, GetStart() is the center and GetEnd() is a point on the perimeter.
                    # Just adding them would only bound a single quadrant! 
                    is_circle = False
                    if hasattr(item, 'GetShape'):
                        shape_val = item.GetShape()
                        is_circle = shape_val in (getattr(_pcbnew, 'S_CIRCLE', -1), getattr(_pcbnew, 'SHAPE_T_CIRCLE', -1))
                    
                    if is_circle and hasattr(item, 'GetCenter') and hasattr(item, 'GetRadius'):
                        c = item.GetCenter()
                        r = item.GetRadius()
                        class Pt:
                            def __init__(self, x, y):
                                self.x = x
                                self.y = y
                        
                        # Calculate local center coordinates
                        tx_c = c.x - pos.x
                        ty_c = c.y - pos.y
                        lcx = tx_c * cos_a - ty_c * sin_a
                        lcy = tx_c * sin_a + ty_c * cos_a
                        
                        # Generate world points that un-rotate perfectly to the local 2r x 2r bounding box
                        def local_to_world(lx, ly):
                            return Pt(pos.x + lx * cos_a + ly * sin_a, pos.y - lx * sin_a + ly * cos_a)
                            
                        pts_world.append(local_to_world(lcx - r, lcy - r))
                        pts_world.append(local_to_world(lcx + r, lcy - r))
                        pts_world.append(local_to_world(lcx - r, lcy + r))
                        pts_world.append(local_to_world(lcx + r, lcy + r))
                    else:
                        pts_world.append(item.GetStart())
                        pts_world.append(item.GetEnd())
                except AttributeError:
                    pass
                try:
                    shape = item.GetPolyShape()
                    if shape.OutlineCount() > 0:
                        outline = shape.Outline(0)
                        for vi in range(outline.PointCount()):
                            pts_world.append(outline.CPoint(vi))
                except AttributeError:
                    pass
                for pt in pts_world:
                    tx = pt.x - pos.x
                    ty = pt.y - pos.y
                    lx = int(tx * cos_a - ty * sin_a)
                    ly = int(tx * sin_a + ty * cos_a)
                    if ext_xmin is None:
                        ext_xmin = lx; ext_xmax = lx
                        ext_ymin = ly; ext_ymax = ly
                    else:
                        ext_xmin = min(ext_xmin, lx); ext_xmax = max(ext_xmax, lx)
                        ext_ymin = min(ext_ymin, ly); ext_ymax = max(ext_ymax, ly)
        except Exception:
            pass  # courtyard graphics not available; pad extents alone are used

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

        is_locked = fp.IsLocked() or (selected_only and not fp.IsSelected())
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

    # Extract keep-out zones
    keepouts: List[KeepOut] = []
    for zone in board.Zones():
        if zone.GetIsRuleArea():
            outline = zone.Outline()
            if outline:
                zb = outline.BBox()
                keepouts.append(KeepOut(
                    xmin=zb.GetX(),
                    ymin=zb.GetY(),
                    xmax=zb.GetX() + zb.GetWidth(),
                    ymax=zb.GetY() + zb.GetHeight(),
                ))

    # Extract component groups
    component_groups: List[ComponentGroup] = []
    fp_to_group: Dict[int, int] = {}
    try:
        import pcbnew as _pcbnew
        # Build UUID → footprint index map
        uuid_to_idx: Dict[str, int] = {}
        for fp_obj in board.GetFootprints():
            uid = str(fp_obj.m_Uuid.AsString())
            ref = fp_obj.GetReference()
            for f in footprints:
                if f.reference == ref:
                    uuid_to_idx[uid] = f.index
                    break

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
        board.GetBoardPolygonOutlines(poly_set)
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
