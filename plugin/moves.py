"""Move operators for the SA optimizer."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from .board_model import BoardModel
from .cost_function import point_in_polygon


@dataclass
class MoveUndo:
    """Information needed to revert a move."""
    move_type: str                # "translate", "swap", "rotate", "median"
    # All footprints involved and their pre-move state
    old_states: List[Tuple[int, int, int, float]] = field(default_factory=list)
    # (fp_index, old_x, old_y, old_angle)

    # Legacy accessors for backward compatibility
    @property
    def fp_index(self) -> int:
        return self.old_states[0][0] if self.old_states else -1

    @property
    def fp2_index(self) -> Optional[int]:
        if self.move_type == "swap" and len(self.old_states) >= 2:
            # Return the first index of the second group/footprint
            # For a simple swap, states[1] is the second footprint
            n = len(self.old_states)
            return self.old_states[n // 2][0]
        return None


def _get_group_members(model: BoardModel, fp_index: int) -> List[int]:
    """Get all footprint indices in the same group (or just [fp_index] if ungrouped)."""
    gi = model.fp_to_group.get(fp_index)
    if gi is not None:
        return list(model.component_groups[gi].member_indices)
    return [fp_index]


def _fp_half_dims(fp) -> Tuple[int, int]:
    """Return (half_width, half_height) accounting for rotation."""
    if int(fp.angle_deg) % 180 == 90:
        return fp.height // 2, fp.width // 2
    return fp.width // 2, fp.height // 2


def _clamp_group_to_board(model: BoardModel, members: List[int]) -> None:
    """Clamp member positions so all bboxes stay inside the board outline.

    Computes the aggregate bbox of all members and determines the minimum
    (dx, dy) correction needed, then applies it uniformly to all members.
    """
    # Compute aggregate bbox using fp.bbox (correctly handles asymmetric footprints)
    agg_xmin = agg_ymin = float('inf')
    agg_xmax = agg_ymax = float('-inf')
    for idx in members:
        fp = model.footprints[idx]
        x1, y1, x2, y2 = fp.bbox
        agg_xmin = min(agg_xmin, x1)
        agg_ymin = min(agg_ymin, y1)
        agg_xmax = max(agg_xmax, x2)
        agg_ymax = max(agg_ymax, y2)

    # Compute correction
    cx = 0
    cy = 0
    if agg_xmin < model.outline_xmin:
        cx = model.outline_xmin - agg_xmin
    elif agg_xmax > model.outline_xmax:
        cx = model.outline_xmax - agg_xmax
    if agg_ymin < model.outline_ymin:
        cy = model.outline_ymin - agg_ymin
    elif agg_ymax > model.outline_ymax:
        cy = model.outline_ymax - agg_ymax

    if cx != 0 or cy != 0:
        for idx in members:
            model.footprints[idx].x += cx
            model.footprints[idx].y += cy


def _hits_new_keepout(model: BoardModel, members: List[int],
                      old_states: List[Tuple[int, int, int, float]]) -> bool:
    """Check if any member entered a keepout zone it wasn't already in.

    Components already inside a keepout are allowed to stay/move within it
    (the penalty system will push them out). Only NEW violations are blocked.
    """
    if not model.keepouts:
        return False

    old_lookup = {s[0]: (s[1], s[2], s[3]) for s in old_states}

    for idx in members:
        fp = model.footprints[idx]
        new_xmin, new_ymin, new_xmax, new_ymax = fp.bbox

        # Compute old bbox given saved (ox, oy, oa) — cx/cy offsets don't change
        ox, oy, oa = old_lookup[idx]
        oa_cos = math.cos(math.radians(-oa))
        oa_sin = math.sin(math.radians(-oa))
        old_cx = ox + int(fp.cx_offset * oa_cos - fp.cy_offset * oa_sin)
        old_cy = oy + int(fp.cx_offset * oa_sin + fp.cy_offset * oa_cos)
        if int(oa) % 180 == 90:
            ohw, ohh = fp.height // 2, fp.width // 2
        else:
            ohw, ohh = fp.width // 2, fp.height // 2
        old_xmin = old_cx - ohw
        old_ymin = old_cy - ohh
        old_xmax = old_cx + ohw
        old_ymax = old_cy + ohh

        for ko in model.keepouts:
            # AABB overlap check — new position
            new_hit = (new_xmin < ko.xmax and new_xmax > ko.xmin and
                       new_ymin < ko.ymax and new_ymax > ko.ymin)
            if not new_hit:
                continue
            # Was it already overlapping this keepout?
            old_hit = (old_xmin < ko.xmax and old_xmax > ko.xmin and
                       old_ymin < ko.ymax and old_ymax > ko.ymin)
            if not old_hit:
                return True  # new violation
    return False


def _hits_polygon_cutout(model: BoardModel, members: List[int],
                         old_states: List[Tuple[int, int, int, float]]) -> bool:
    """Return True if any member bbox corner moves from inside to outside the board polygon.

    Uses was-inside logic: if a corner was already outside (SA left the component in a
    cutout region via soft penalty), moving it to remain outside is allowed — the cost
    function boundary penalty provides the gradient to migrate it back in. Only NEW
    violations (inside → outside) are blocked.

    Returns False immediately if outline_polygon is None (rectangular board).
    """
    poly = model.outline_polygon
    if poly is None:
        return False

    old_lookup = {s[0]: (s[1], s[2], s[3]) for s in old_states}

    for idx in members:
        fp = model.footprints[idx]
        x1, y1, x2, y2 = fp.bbox
        new_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        new_outside = [not point_in_polygon(nx, ny, poly) for nx, ny in new_corners]
        if not any(new_outside):
            continue

        # Compute old bbox corners
        ox, oy, oa = old_lookup[idx]
        oa_cos = math.cos(math.radians(-oa))
        oa_sin = math.sin(math.radians(-oa))
        old_bcx = ox + int(fp.cx_offset * oa_cos - fp.cy_offset * oa_sin)
        old_bcy = oy + int(fp.cx_offset * oa_sin + fp.cy_offset * oa_cos)
        if int(oa) % 180 == 90:
            ohw, ohh = fp.height // 2, fp.width // 2
        else:
            ohw, ohh = fp.width // 2, fp.height // 2
        old_corners = [
            (old_bcx - ohw, old_bcy - ohh),
            (old_bcx + ohw, old_bcy - ohh),
            (old_bcx + ohw, old_bcy + ohh),
            (old_bcx - ohw, old_bcy + ohh),
        ]

        for ci, is_new_outside in enumerate(new_outside):
            if is_new_outside and point_in_polygon(old_corners[ci][0], old_corners[ci][1], poly):
                return True  # corner was inside, now outside: new violation

    return False


def _kickout_polygon_violations(
    model: BoardModel,
    moved_indices: List[int],
    cost_state,
) -> int:
    """Force any component whose bbox corners are outside the polygon to the nearest
    valid position.  Called once at the start of greedy refinement to fix components
    that SA (soft-penalty) left in a polygon cutout.

    Uses binary search in each of 8 directions to find the minimum displacement that
    gets all corners inside the polygon, then picks the shortest option.

    Returns the number of components that were relocated.
    """
    poly = model.outline_polygon
    if poly is None:
        return 0

    # Use half the shorter board dimension as search limit.  The full diagonal
    # overshoots: a component near the top edge moved south by board_diag lands
    # beyond the bottom edge, so the feasibility check fails for every direction
    # and nothing gets kicked.  Half the shorter side is always enough to escape
    # any realistic cutout while staying on the board.
    board_diag = min(model.outline_xmax - model.outline_xmin,
                     model.outline_ymax - model.outline_ymin) // 2
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    count = 0

    for mi in moved_indices:
        members = _get_group_members(model, mi)

        # Collect all bbox corners for the whole group
        all_corners = []
        for idx in members:
            x1, y1, x2, y2 = model.footprints[idx].bbox
            all_corners.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        if all(point_in_polygon(cx, cy, poly) for cx, cy in all_corners):
            continue  # already inside — nothing to do

        best_dx, best_dy = None, None
        best_dist = float('inf')

        for ddx, ddy in directions:
            # Quick check: can this direction ever fix the violation?
            if not all(point_in_polygon(cx + ddx * board_diag,
                                        cy + ddy * board_diag, poly)
                       for cx, cy in all_corners):
                continue

            # Binary search for minimum distance in this direction
            lo, hi = 0, board_diag
            for _ in range(25):  # 2^25 steps → ~1 nm precision
                mid = (lo + hi) // 2
                if all(point_in_polygon(cx + ddx * mid, cy + ddy * mid, poly)
                       for cx, cy in all_corners):
                    hi = mid
                else:
                    lo = mid

            if hi < best_dist:
                best_dist = hi
                # Add 0.5 mm margin so the component sits comfortably inside
                # the polygon, not right at the boundary edge.
                margin = 500_000  # 0.5 mm in nm
                best_dx, best_dy = ddx * (hi + margin), ddy * (hi + margin)

        if best_dx is not None:
            for idx in members:
                model.footprints[idx].x += best_dx
                model.footprints[idx].y += best_dy
            _clamp_group_to_board(model, members)
            count += 1

    if count > 0:
        cost_state._compute_all()

    return count


def select_move_type(t_ratio: float) -> str:
    """
    Select a move type based on precomputed temperature ratio (0=cold, 1=hot).

    High T: more swaps and rotates (exploration).
    Low T: mostly translates and median moves (refinement).
    """
    # Interpolate probabilities between high-T and low-T
    p_translate = 0.70 + (0.55 - 0.70) * t_ratio   # 0.70 at low T, 0.55 at high T
    p_median = 0.15 + (0.05 - 0.15) * t_ratio       # 0.15 at low T, 0.05 at high T
    p_swap = 0.05 + (0.20 - 0.05) * t_ratio          # 0.05 at low T, 0.20 at high T
    # p_rotate = remainder (~0.10 at low T, ~0.20 at high T)

    r = random.random()
    if r < p_translate:
        return "translate"
    elif r < p_translate + p_median:
        return "median"
    elif r < p_translate + p_median + p_swap:
        return "swap"
    else:
        return "rotate"


def _compute_t_ratio(temperature: float, t0: float) -> float:
    """Compute temperature ratio (0 at T=0, 1 at T=T0) using log scaling."""
    if t0 > 0 and temperature > 0:
        return max(0.0, min(1.0, math.log(temperature + 1) / math.log(t0 + 1)))
    return 0.0


def do_translate(model: BoardModel, t_ratio: float, window: int) -> MoveUndo:
    """
    Translate a random moveable footprint (and its group) by a
    temperature-dependent distance.

    window: precomputed move window in nm (0.1mm at low T, half-board at high T).
    All group members move by the same (dx, dy).
    """
    mi = random.choice(model.moveable_indices)
    members = _get_group_members(model, mi)

    undo = MoveUndo(move_type="translate")
    for idx in members:
        fp = model.footprints[idx]
        undo.old_states.append((idx, fp.x, fp.y, fp.angle_deg))

    dx = random.randint(-window, window)
    dy = random.randint(-window, window)

    for idx in members:
        fp = model.footprints[idx]
        fp.x += dx
        fp.y += dy

    return undo


def do_median(model: BoardModel, t_ratio: float, noise: int) -> MoveUndo:
    """
    Move a footprint toward the weighted center of its connected pads.

    Computes the centroid of all pads on the same nets as this footprint
    (excluding the footprint's own pads), then moves the footprint a
    fraction of the way toward that centroid. The fraction depends on
    t_ratio: full step at low T, partial at high T (with noise).

    t_ratio: precomputed temperature ratio (0=cold, 1=hot).
    noise: precomputed max noise in nm (board_w * 0.02 * t_ratio).

    This is a net-aware "force-directed" move that directly targets
    HPWL reduction.
    """
    mi = random.choice(model.moveable_indices)
    members = _get_group_members(model, mi)
    member_set = set(members)

    undo = MoveUndo(move_type="median")
    for idx in members:
        fp = model.footprints[idx]
        undo.old_states.append((idx, fp.x, fp.y, fp.angle_deg))

    # Compute centroid of connected pads (excluding our own footprints)
    sum_x = 0
    sum_y = 0
    count = 0
    fps = model.footprints
    for idx in members:
        fp = fps[idx]
        for nc in fp.net_codes:
            net = model.nets.get(nc)
            if net is None:
                continue
            for fi, pi in net.pad_refs:
                if fi in member_set:
                    continue
                other_fp = fps[fi]
                pad = other_fp.pads[pi]
                cos_a = other_fp._cos_a
                sin_a = other_fp._sin_a
                ax = other_fp.x + int(pad.offset_x * cos_a - pad.offset_y * sin_a)
                ay = other_fp.y + int(pad.offset_x * sin_a + pad.offset_y * cos_a)
                sum_x += ax
                sum_y += ay
                count += 1

    if count == 0:
        # No connected pads outside the group — fall back to small random translate
        dx = random.randint(-100_000, 100_000)
        dy = random.randint(-100_000, 100_000)
        for idx in members:
            fps[idx].x += dx
            fps[idx].y += dy
        return undo

    target_x = sum_x // count
    target_y = sum_y // count

    # Compute centroid of our group
    cx = sum(fps[i].x for i in members) // len(members)
    cy = sum(fps[i].y for i in members) // len(members)

    # Move fraction: at high T, move 30-70% of the way (with noise).
    # At low T, move 50-100% of the way (more precise).
    base_frac = 0.5 + 0.3 * (1.0 - t_ratio)  # 0.5 at high T, 0.8 at low T
    frac = base_frac * (0.6 + 0.8 * random.random())  # randomize ±40%
    frac = min(frac, 1.0)

    dx = int((target_x - cx) * frac)
    dy = int((target_y - cy) * frac)

    # Add small noise proportional to temperature
    if noise > 0:
        dx += random.randint(-noise, noise)
        dy += random.randint(-noise, noise)

    for idx in members:
        fp = fps[idx]
        fp.x += dx
        fp.y += dy

    return undo


def do_swap(model: BoardModel) -> Optional[MoveUndo]:
    """Swap positions of two random moveable footprints (or groups).

    If both are ungrouped, swap their positions directly.
    If either is grouped, swap the centroids of the two groups and
    translate all members accordingly.
    Does not swap within the same group.
    """
    if len(model.moveable_indices) < 2:
        return None

    # Pick two different moveable indices that are NOT in the same group
    for _ in range(10):  # try up to 10 times to find a valid pair
        mi1, mi2 = random.sample(model.moveable_indices, 2)
        g1 = model.fp_to_group.get(mi1)
        g2 = model.fp_to_group.get(mi2)
        if g1 is not None and g1 == g2:
            continue  # same group — skip
        break
    else:
        return None  # couldn't find a valid pair

    members1 = _get_group_members(model, mi1)
    members2 = _get_group_members(model, mi2)

    undo = MoveUndo(move_type="swap")
    for idx in members1 + members2:
        fp = model.footprints[idx]
        undo.old_states.append((idx, fp.x, fp.y, fp.angle_deg))

    # Compute centroids
    cx1 = sum(model.footprints[i].x for i in members1) // len(members1)
    cy1 = sum(model.footprints[i].y for i in members1) // len(members1)
    cx2 = sum(model.footprints[i].x for i in members2) // len(members2)
    cy2 = sum(model.footprints[i].y for i in members2) // len(members2)

    # Move group1 members by (cx2 - cx1, cy2 - cy1) and vice versa
    dx = cx2 - cx1
    dy = cy2 - cy1
    for idx in members1:
        model.footprints[idx].x += dx
        model.footprints[idx].y += dy
    for idx in members2:
        model.footprints[idx].x -= dx
        model.footprints[idx].y -= dy

    return undo


def do_rotate(model: BoardModel) -> MoveUndo:
    """Rotate a random moveable footprint (and its group) by 90, 180, or 270 degrees.

    For ungrouped footprints, rotates in place.
    For grouped footprints, rotates all members around the group centroid.
    """
    mi = random.choice(model.moveable_indices)
    members = _get_group_members(model, mi)

    undo = MoveUndo(move_type="rotate")
    for idx in members:
        fp = model.footprints[idx]
        undo.old_states.append((idx, fp.x, fp.y, fp.angle_deg))

    rotation = random.choice([90.0, 180.0, 270.0])

    if len(members) == 1:
        # Simple single-footprint rotation
        fp = model.footprints[members[0]]
        fp.set_angle((fp.angle_deg + rotation) % 360.0)
    else:
        # Rotate all group members around the group centroid
        cx = sum(model.footprints[i].x for i in members) // len(members)
        cy = sum(model.footprints[i].y for i in members) // len(members)

        # KiCad uses CW rotation (Y-down), negate angle for standard trig
        rad = math.radians(-rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)

        for idx in members:
            fp = model.footprints[idx]
            dx = fp.x - cx
            dy = fp.y - cy
            fp.x = cx + int(dx * cos_r - dy * sin_r)
            fp.y = cy + int(dx * sin_r + dy * cos_r)
            fp.set_angle((fp.angle_deg + rotation) % 360.0)

    return undo


def revert_move(model: BoardModel, undo: MoveUndo):
    """Restore the board state to before the move."""
    for fp_idx, old_x, old_y, old_angle in undo.old_states:
        fp = model.footprints[fp_idx]
        fp.x = old_x
        fp.y = old_y
        fp.set_angle(old_angle)


def affected_indices(undo: MoveUndo) -> Set[int]:
    """Return the set of footprint indices affected by this move."""
    return {s[0] for s in undo.old_states}
