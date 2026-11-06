"""Cost function: HPWL + overlap penalty + boundary penalty + keep-out penalty
+ optional congestion penalty.

Supports incremental updates: when footprints move, only affected pairs/items
are recomputed, reducing per-move cost from O(n²) to O(n).

Overlap detection uses flat bbox arrays plus a sorted xmin list for O(log n + k)
neighbor scans, where k is the number of footprints whose left edge is left of
the query's right edge. The bisect narrows the scan vs the previous O(n) flat scan.

Congestion estimation divides the board into routing bins and counts how many net
bounding boxes cross each bin.  Bins where demand exceeds capacity incur a
quadratic penalty.  Disabled when congestion_weight == 0 (zero overhead).

Constraint-file terms (see constraints.py; all inactive, with no per-move work,
when the model carries no constraints):
  * per-net HPWL weight (Net.weight): the weighted excess over the raw HPWL is
    kept as a separate float, so `hpwl` stays the raw value for reporting;
  * soft group cohesion: penalty when a group's union-bbox diagonal exceeds
    its max span;
  * zone preference: distance from the footprint position to its preferred
    zone rectangle;
  * component clearance (Footprint.clearance): bbox inflated in the overlap test,
    against bodies on both sides unless restricted to one side;
  * attached keep-out rects (Footprint.keepout_rects, e.g. an antenna area):
    world rects that collide with other bodies like a clearance ring (same
    side rule, same-group pairs exempt); not in the boundary term or HPWL.
"""
from __future__ import annotations

import bisect
import math
import time
from typing import Dict, List, Optional, Set, Tuple
from .board_model import BoardModel, Net


OVERLAP_WEIGHT = 50.0       # per-pair, multiplied by overlap distance (nm)
BOUNDARY_WEIGHT = 50.0      # per-footprint, multiplied by distance outside rect bbox (nm)
POLYGON_BOUNDARY_WEIGHT = 500.0  # polygon cutout violations (nm); 10× BOUNDARY_WEIGHT
                            # so that at high T (penalty_scale_min=0.1) the effective
                            # weight equals OVERLAP_WEIGHT — preventing SA from trading
                            # cutout placement for overlap relief even at high temperature.
KEEPOUT_WEIGHT = 100.0      # per-violation, multiplied by overlap distance (nm);
                            # flat per part since 2026-11-06 (was x net count), so
                            # at T -> 0 a keep-out intrusion costs every part 2x
                            # an overlap of the same depth (OVERLAP_WEIGHT, flat)
COHESION_WEIGHT = 1.0       # per soft group, times excess span (nm): 1 mm excess = 1 mm wire
ZONE_WEIGHT = 1.0           # per footprint, times distance to preferred zone (nm)
# Both are defaults only; CostState(cohesion_weight=, zone_weight=) and the
# SAConfig fields of the same name override them per run (0 disables the term).

# Default congestion bin capacity: assumes 2-layer board, 0.4mm track pitch,
# both horizontal and vertical routing.  capacity = (bin_size / pitch) * layers * 2.
_DEFAULT_CONGESTION_LAYERS = 2
_DEFAULT_TRACK_PITCH_NM = 400_000  # 0.4 mm


class _CongestionGrid:
    """Uniform routing-demand grid for congestion estimation.

    The board is divided into rectangular bins of *bin_size* nm.  For each
    non-excluded net with >= 2 pads, the HPWL bounding box determines which
    bins the net's routing crosses.  Each covered bin gets +1 demand.  Bins
    where demand > capacity incur a quadratic penalty.

    All state is stored in flat Python lists for C-level indexing speed.
    """

    __slots__ = ('bin_size', 'nx', 'ny', 'ox', 'oy',
                 'demand', 'capacity', 'net_bins', '_cap_val')

    def __init__(self, ox: int, oy: int, board_w: int, board_h: int,
                 bin_size: int, layers: int = _DEFAULT_CONGESTION_LAYERS):
        self.bin_size = bin_size
        self.ox = ox
        self.oy = oy
        self.nx = max(1, (board_w + bin_size - 1) // bin_size)
        self.ny = max(1, (board_h + bin_size - 1) // bin_size)
        n_bins = self.nx * self.ny
        self.demand: List[int] = [0] * n_bins
        # Capacity per bin
        self._cap_val = max(1, int(
            (bin_size / _DEFAULT_TRACK_PITCH_NM) * layers * 2
        ))
        self.capacity: List[int] = [self._cap_val] * n_bins
        # net_code -> list of bin indices covered by its HPWL bbox
        self.net_bins: Dict[int, List[int]] = {}

    def bbox_to_bins(self, xmin: int, ymin: int,
                     xmax: int, ymax: int) -> List[int]:
        """Return list of bin indices covered by a bounding box."""
        bs = self.bin_size
        bx0 = max(0, (xmin - self.ox) // bs)
        by0 = max(0, (ymin - self.oy) // bs)
        bx1 = min(self.nx - 1, (xmax - self.ox) // bs)
        by1 = min(self.ny - 1, (ymax - self.oy) // bs)
        nx = self.nx
        bins = []
        for by in range(by0, by1 + 1):
            row = by * nx
            for bx in range(bx0, bx1 + 1):
                bins.append(row + bx)
        return bins

    def add_demand(self, bins: List[int]) -> None:
        """Increment demand for each bin in the list."""
        demand = self.demand
        for b in bins:
            demand[b] += 1

    def remove_demand(self, bins: List[int]) -> None:
        """Decrement demand for each bin in the list."""
        demand = self.demand
        for b in bins:
            demand[b] -= 1

    def compute_penalty(self, weight: float) -> float:
        """Sum quadratic excess penalty across all bins.

        penalty = sum( max(0, demand - capacity)^2 ) * weight * bin_size
        """
        demand = self.demand
        cap = self.capacity
        total = 0
        for i in range(len(demand)):
            excess = demand[i] - cap[i]
            if excess > 0:
                total += excess * excess
        return total * weight * self.bin_size

    def remove_demand_delta(self, bins: List[int], weight: float) -> float:
        """Remove demand from bins and return the penalty delta."""
        demand = self.demand
        cap = self.capacity
        delta = 0
        for b in bins:
            old_d = demand[b]
            old_e = old_d - cap[b]
            new_e = old_e - 1
            demand[b] = old_d - 1
            if old_e > 0:
                delta -= old_e * old_e
            if new_e > 0:
                delta += new_e * new_e
        return delta * weight * self.bin_size

    def add_demand_delta(self, bins: List[int], weight: float) -> float:
        """Add demand to bins and return the penalty delta."""
        demand = self.demand
        cap = self.capacity
        delta = 0
        for b in bins:
            old_d = demand[b]
            old_e = old_d - cap[b]
            new_e = old_e + 1
            demand[b] = old_d + 1
            if old_e > 0:
                delta -= old_e * old_e
            if new_e > 0:
                delta += new_e * new_e
        return delta * weight * self.bin_size


def point_in_polygon(px: int, py: int,
                     polygon: List[Tuple[int, int]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _dist_to_segment_sq(px: int, py: int,
                        ax: int, ay: int,
                        bx: int, by: int) -> float:
    """Squared distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        ex, ey = px - ax, py - ay
        return float(ex * ex + ey * ey)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    nx = ax + t * dx
    ny = ay + t * dy
    ex, ey = px - nx, py - ny
    return ex * ex + ey * ey


def dist_to_polygon(px: int, py: int,
                    polygon: List[Tuple[int, int]]) -> float:
    """Distance from point to nearest polygon edge (nm)."""
    n = len(polygon)
    min_d_sq = float('inf')
    for i in range(n):
        j = (i + 1) % n
        d_sq = _dist_to_segment_sq(px, py,
                                   polygon[i][0], polygon[i][1],
                                   polygon[j][0], polygon[j][1])
        if d_sq < min_d_sq:
            min_d_sq = d_sq
    return math.sqrt(min_d_sq)


def _polygon_is_rectangular(poly: List[Tuple[int, int]],
                             xmin: int, ymin: int,
                             xmax: int, ymax: int,
                             tol: int = 1000) -> bool:
    """Return True if every vertex of poly lies on the bounding-box perimeter.

    Used to skip the polygon boundary check for rectangular boards — the simple
    bbox dx/dy check is sufficient and the polygon check would be wasted work
    (point_in_polygon always returns True for points already inside a rectangle).
    Tolerance of 1000 nm (1 µm) handles rounding in KiCad's polygon extraction.
    """
    for x, y in poly:
        if not (abs(x - xmin) <= tol or abs(x - xmax) <= tol
                or abs(y - ymin) <= tol or abs(y - ymax) <= tol):
            return False
    return True


# ---------------------------------------------------------------------------
# Outline-polygon edge profiles (edge locks on non-rectangular boards)
# ---------------------------------------------------------------------------
#
# For edge "top" the boundary at a vertical line x is the topmost (smallest y)
# crossing of that line with the outline polygon. Over a footprint's x-span
# [x1, x2] the snap target is the innermost (largest y) value of that boundary.
# "bottom" mirrors it (largest-y crossing, innermost = smallest), and "left" /
# "right" do the same with x and y swapped. Between two consecutive vertex
# coordinates the boundary is one polygon edge (a simple polygon has no
# crossing edges), so the boundary is piecewise linear and the extreme over a
# span lies at x1, x2 or a vertex coordinate inside it, approached from the
# side of the interval; this is exact for axis-aligned and diagonal edges.
# Both the CadMust-Neo edge snap (moves.resnap_edges) and the
# CadMust-Autopilot checker use this one definition.

_EDGE_PROFILE_CACHE: Dict[Tuple[int, str], Tuple[object, object]] = {}


def _build_edge_profile(poly: List[Tuple[int, int]], edge: str):
    """Return (breakpoints, lines) for one edge of the polygon.

    breakpoints: sorted unique vertex coordinates along the edge direction.
    lines[i]: (u_a, v_a, u_b, v_b) of the outermost polygon edge over the open
    interval (breakpoints[i], breakpoints[i+1]), or None if no edge spans it.
    u runs along the edge (x for top/bottom, y for left/right), v is the
    perpendicular coordinate.
    """
    along_x = edge in ("top", "bottom")
    pts = [(x, y) if along_x else (y, x) for x, y in poly]
    outer_is_min = edge in ("top", "left")
    bps = sorted(set(u for u, _ in pts))
    segs = []
    n = len(pts)
    for i in range(n):
        ua, va = pts[i]
        ub, vb = pts[(i + 1) % n]
        if ua == ub:
            continue  # parallel to v: no crossing in an open interval
        if ua > ub:
            ua, va, ub, vb = ub, vb, ua, va
        segs.append((ua, va, ub, vb))
    lines = []
    for i in range(len(bps) - 1):
        lo, hi = bps[i], bps[i + 1]
        mid = (lo + hi) / 2.0
        best = None
        best_v = None
        for s in segs:
            if s[0] <= lo and s[2] >= hi:
                v = s[1] + (s[3] - s[1]) * (mid - s[0]) / (s[2] - s[0])
                if best is None or (v < best_v if outer_is_min else v > best_v):
                    best, best_v = s, v
        lines.append(best)
    return bps, lines


def _edge_profile(poly: List[Tuple[int, int]], edge: str):
    """Cached _build_edge_profile, keyed on id(poly); the cache keeps the
    polygon object so a recycled id can never return a stale profile."""
    key = (id(poly), edge)
    hit = _EDGE_PROFILE_CACHE.get(key)
    if hit is not None and hit[0] is poly:
        return hit[1]
    prof = _build_edge_profile(poly, edge)
    _EDGE_PROFILE_CACHE[key] = (poly, prof)
    return prof


def polygon_edge_limit(poly: List[Tuple[int, int]], edge: str,
                       lo: int, hi: int, outermost: bool = False) -> Optional[float]:
    """Innermost boundary coordinate of the polygon's `edge` side over [lo, hi].

    outermost=True returns the outermost one instead (smallest top-boundary
    y, largest bottom-boundary y, and so on): the reference for an anchored
    part's module end (anchor_edge_reference).

    edge "top"/"bottom": [lo, hi] is an x-span, the result is a y (largest
    top-boundary y, resp. smallest bottom-boundary y). edge "left"/"right":
    [lo, hi] is a y-span, the result is an x (largest left-boundary x, resp.
    smallest right-boundary x). The span is clipped to the polygon's extent;
    None when it does not overlap the polygon at all.
    """
    bps, lines = _edge_profile(poly, edge)
    if not bps or hi < bps[0] or lo > bps[-1]:
        return None
    inner_is_max = edge in ("top", "left")
    if outermost:
        inner_is_max = not inner_is_max
    best = None
    # intervals overlapping [lo, hi]
    i0 = max(0, bisect.bisect_right(bps, lo) - 1)
    for i in range(i0, len(lines)):
        a, b = bps[i], bps[i + 1]
        if a > hi:
            break
        if b < lo:
            continue
        s = lines[i]
        if s is None:
            continue
        u1 = a if a > lo else lo
        u2 = b if b < hi else hi
        if u1 > u2:
            continue
        for u in (u1, u2):
            v = s[1] + (s[3] - s[1]) * (u - s[0]) / (s[2] - s[0])
            if best is None or (v > best if inner_is_max else v < best):
                best = v
    return best


def anchor_edge_reference(poly: Optional[List[Tuple[int, int]]], edge: str,
                          lo: int, hi: int, outline_ref: int) -> int:
    """Edge reference (nm) for the module end of an anchored part (an
    antenna_outward lock): the outermost outline point along `edge` over
    [lo, hi], the module's own width (pad-centre extent across the antenna
    side, Footprint.edge_anchor_span_world), not the courtyard. Without a
    polygon (rectangular board) the outline bbox edge `outline_ref`.

    On a notch in the locked edge: a module wider than the notch, or one
    that reaches past a notch side, gets the outer line, so the module end
    goes on the outer line and the body spans the notch. That a notch
    narrower than the module is spanned is the designer's call, not the
    engine's; the courtyard clip (locked_edge_clip) keeps the innermost
    reference, so the off-board tests still see the notch. A module whose
    whole width lies inside the notch only sees the notch floor and goes
    there. Plain edge locks (no anchor) keep the innermost reference over
    the bbox span (polygon_edge_limit). Snap (moves.resnap_edges), GUI
    summary (constraint_summary) and the CadMust-Autopilot checker share
    this one definition."""
    if poly is None:
        return outline_ref
    lim = polygon_edge_limit(poly, edge, lo, hi, outermost=True)
    return outline_ref if lim is None else int(round(lim))


def locked_edge_clip(fp, bb: Tuple[int, int, int, int],
                     poly: Optional[List[Tuple[int, int]]],
                     oxmin: int, oymin: int, oxmax: int, oymax: int
                     ) -> Optional[Tuple[int, int, int, int]]:
    """The bbox the boundary tests use for footprint fp.

    For an edge-locked part that may overhang its locked edge
    (Footprint.overhang_ok, set from the file's overhang_ok and for every
    antenna_outward lock) the bbox side on that edge is pulled back to the
    edge reference: the outline bbox edge, on a polygonal outline (poly not
    None) the innermost polygon boundary point over the bbox span
    (polygon_edge_limit, the snap target). The other three sides are
    unchanged. The CadMust-Autopilot checker clips the same way. Every
    other part: bb as given. None when nothing is left on the board side."""
    edge = fp.edge_lock if getattr(fp, "overhang_ok", False) else None
    if edge is None:
        return bb
    x1, y1, x2, y2 = bb
    if edge == "top":
        ref = oymin
    elif edge == "bottom":
        ref = oymax
    elif edge == "left":
        ref = oxmin
    else:
        ref = oxmax
    if poly is not None:
        lo, hi = (x1, x2) if edge in ("top", "bottom") else (y1, y2)
        lim = polygon_edge_limit(poly, edge, lo, hi)
        if lim is not None:
            ref = lim
    if edge == "top":
        y1 = max(y1, ref)
    elif edge == "bottom":
        y2 = min(y2, ref)
    elif edge == "left":
        x1 = max(x1, ref)
    else:
        x2 = min(x2, ref)
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def outline_is_polygonal(model) -> bool:
    """True when the model's outline polygon differs from its bounding box
    (an L-shape, notch or bevel). False for no polygon or a rectangle."""
    poly = getattr(model, "outline_polygon", None)
    if poly is None:
        return False
    return not _polygon_is_rectangular(poly, model.outline_xmin, model.outline_ymin,
                                       model.outline_xmax, model.outline_ymax)


def bbox_outside_polygon(bbox: Tuple[int, int, int, int],
                         poly: List[Tuple[int, int]]) -> float:
    """How far a rectangle leaves the polygon (nm); 0.0 when it is inside.

    Outside when a bbox corner lies outside the polygon (excess = that
    corner's distance to the polygon) or a polygon vertex lies strictly inside
    the bbox (a notch corner poking in; excess = the vertex's depth, its
    distance to the nearest bbox side). Returns the largest such excess.
    """
    x1, y1, x2, y2 = bbox
    excess = 0.0
    for cx, cy in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        if not point_in_polygon(cx, cy, poly):
            # a corner exactly on the outline (d == 0) counts as inside
            excess = max(excess, dist_to_polygon(cx, cy, poly))
    for vx, vy in poly:
        if x1 < vx < x2 and y1 < vy < y2:
            excess = max(excess, float(min(vx - x1, x2 - vx, vy - y1, y2 - vy)))
    return excess


def _pair_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


class CostState:
    """
    Maintains the current cost and supports incremental updates.

    Total cost = sum_of_net_hpwl + overlap_penalty + boundary_penalty + keepout_penalty

    Per-net HPWL values are cached so that when a footprint moves,
    only its connected nets need recomputation.

    Per-pair overlap values and per-footprint boundary/keepout penalties
    are cached so incremental updates are O(n) instead of O(n²).
    """

    def __init__(self, model: BoardModel, quiet: bool = False,
                 congestion_weight: float = 0.0,
                 congestion_bin_size: int = 5_000_000,
                 cohesion_weight: float = COHESION_WEIGHT,
                 zone_weight: float = ZONE_WEIGHT,
                 pad_clearance: bool = False):
        self.model = model
        self.net_hpwl: Dict[int, int] = {}
        self._total_hpwl: int = 0
        self._overlap_penalty: float = 0.0
        self._boundary_penalty: float = 0.0
        self._keepout_penalty: float = 0.0
        self._congestion_penalty: float = 0.0
        self._hpwl_extra: float = 0.0       # sum (weight - 1) * net_hpwl over weighted nets
        self._cohesion_penalty: float = 0.0
        self._zone_penalty: float = 0.0
        self._soft_penalty: float = 0.0     # cohesion + zone (one load in total_cost)
        self._congestion_weight: float = congestion_weight
        self._congestion_bin_size: int = congestion_bin_size
        # Soft-term weights (constraint file). 0 disables the term entirely.
        self._cohesion_weight: float = cohesion_weight
        self._zone_weight: float = zone_weight
        self.penalty_scale: float = 1.0  # scales boundary+keepout+congestion (not overlap)

        # Caches for incremental updates
        self._pair_overlaps: Dict[Tuple[int, int], float] = {}
        self._fp_boundary: Dict[int, float] = {}
        self._fp_keepout: Dict[int, float] = {}

        # Flat bbox arrays + sorted xmin list for O(log n + k) overlap neighbor scans
        n = len(model.footprints)
        self._bx1 = [0] * n
        self._by1 = [0] * n
        self._bx2 = [0] * n
        self._by2 = [0] * n
        self._xmin_items: List[Tuple[int, int]] = []  # sorted (xmin, fp_index)
        # Attached keep-out rects: fps that carry them and their world rects
        # (refreshed with the bbox arrays). Empty for boards without them.
        self._ko_fps: List[int] = [
            i for i, fp in enumerate(model.footprints)
            if getattr(fp, 'keepout_rects', None)]
        self._ko_world: Dict[int, List[Tuple[int, int, int, int]]] = {}
        # Pad-level clearance (docs/PAD_CLEARANCE_DESIGN.md): per-fp local pad
        # tuples (x1, y1, x2, y2, side mask, clearance, net), world copies
        # cached per position, and the local pad reach (pads inflated by
        # their clearance) of the fps whose reach can exceed their bbox.
        self._pad_on: bool = bool(pad_clearance)
        self._pad_local: Dict[int, List[tuple]] = {}
        self._pad_world: Dict[int, tuple] = {}
        self._pr_local: Dict[int, Tuple[int, int, int, int]] = {}
        self._pr_world: Dict[int, Tuple[int, int, int, int]] = {}
        self._pad_maxc: Dict[int, int] = {}
        if self._pad_on:
            self._build_pad_data()
        self._rebuild_bbox_arrays()

        # Sub-timers for profiling incremental_update breakdown
        self.t_hpwl = 0.0
        self.t_overlap = 0.0
        self.t_boundary = 0.0
        self.t_keepout = 0.0
        self.t_congestion = 0.0
        self.t_snapshot = 0.0

        # Precomputed per-footprint net scale (max(1, len(net_codes))).
        # Net codes never change during a run, so this is safe to cache once.
        self._fp_net_scale: List[int] = [
            max(1, len(fp.net_codes)) for fp in model.footprints
        ]

        # Precomputed per-footprint board side. Phase 1 has no flip move, so
        # sides never change during a run. getattr default keeps pickled
        # BoardModels from before the is_front field working (batch corpus).
        self._fp_is_front: List[bool] = [
            getattr(fp, 'is_front', True) for fp in model.footprints
        ]
        # Backfill through-hole rects on models pickled before they existed,
        # so the th_bboxes property works without per-call getattr.
        for fp in model.footprints:
            if not hasattr(fp, 'th_rects'):
                fp.th_rects = []

        # --- Constraint-file terms (all empty / zero without constraints) ---
        # Net weights: only nets with weight != 1 are tracked.
        self._net_w: Dict[int, float] = {}
        for nc, net in model.nets.items():
            w = getattr(net, 'weight', 1.0)
            if w != 1.0 and not net.is_excluded:
                self._net_w[nc] = w
        # Soft cohesion groups: (member_indices, max_span_nm)
        self._soft_groups: List[Tuple[List[int], int]] = [
            (list(idx), int(span))
            for _gid, idx, span in (getattr(model, 'soft_groups', None) or [])
            if len(idx) >= 2
        ]
        self._fp_soft: Dict[int, List[int]] = {}
        for gi, (idx, _span) in enumerate(self._soft_groups):
            for fi in idx:
                self._fp_soft.setdefault(fi, []).append(gi)
        self._group_cost: List[float] = [0.0] * len(self._soft_groups)
        # Zone preferences: fp_index -> rect
        self._zone_rects: Dict[int, Tuple[float, float, float, float]] = dict(
            getattr(model, 'zone_prefs', None) or {})
        self._fp_zone: Dict[int, float] = {}
        # One flag for the cohesion + zone terms (checked once per update)
        self._extra_terms: bool = bool(self._soft_groups or self._zone_rects)
        # Component clearances: per-fp ring (nm) and side restriction
        self._fp_clear: List[int] = [
            int(getattr(fp, 'clearance', 0) or 0) for fp in model.footprints
        ]
        self._fp_clear_side: List[Optional[bool]] = [
            getattr(fp, 'clearance_side', None) for fp in model.footprints
        ]
        self._max_clear: int = max(self._fp_clear) if self._fp_clear else 0
        self._clear_fps: List[int] = [i for i, c in enumerate(self._fp_clear) if c > 0]
        # Fps whose overlap reach exceeds their bbox (ring or attached rects);
        # the neighbour scan tests them explicitly.
        self._ext_fps: List[int] = sorted(set(self._clear_fps) | set(self._ko_fps)
                                          | set(self._pr_local))
        if set(self._clear_fps) | set(self._ko_fps):
            # Shadow the plain pair test with the clearance-aware one; boards
            # without clearances or attached rects keep the original method.
            self._compute_pair_overlap = self._compute_pair_overlap_clear
        if self._pad_on and self._pad_local:
            self._pair_base = self._compute_pair_overlap
            self._compute_pair_overlap = self._compute_pair_overlap_pad

        # Cached board extents — avoid self.model.X attribute chain in hot path.
        self._oxmin: int = model.outline_xmin
        self._oymin: int = model.outline_ymin
        self._oxmax: int = model.outline_xmax
        self._oymax: int = model.outline_ymax
        self._keepouts = model.keepouts

        # Polygon boundary check — skip for rectangular boards.
        # For a rectangle the simple bbox dx/dy check (below) is sufficient;
        # point_in_polygon would always return True for any interior point,
        # so the polygon loop would run ~4 * n_vertices iterations per move
        # for zero benefit.
        poly = model.outline_polygon
        if poly is not None and _polygon_is_rectangular(
                poly,
                model.outline_xmin, model.outline_ymin,
                model.outline_xmax, model.outline_ymax):
            poly = None  # treat as rectangular: bbox check is sufficient
        self._outline_polygon = poly
        # Edge-locked parts that may overhang their locked edge
        # (Footprint.overhang_ok): the boundary penalty clips their bbox there.
        self._overhang_fps: Set[int] = {
            i for i, f in enumerate(model.footprints)
            if getattr(f, "edge_lock", None) and getattr(f, "overhang_ok", False)}

        # Congestion grid — None when disabled (zero overhead)
        self._cong_grid: Optional[_CongestionGrid] = None
        if congestion_weight > 0:
            board_w = model.outline_xmax - model.outline_xmin
            board_h = model.outline_ymax - model.outline_ymin
            self._cong_grid = _CongestionGrid(
                model.outline_xmin, model.outline_ymin,
                board_w, board_h, congestion_bin_size,
            )

        # Log excluded power nets once at startup (suppressed for preview runs)
        if not quiet:
            power_nets = [net for net in model.nets.values() if net.is_excluded]
            if power_nets:
                names = ', '.join(n.net_name for n in power_nets[:10])
                suffix = f' (+{len(power_nets)-10} more)' if len(power_nets) > 10 else ''
                print(f'[CadMust] Excluding {len(power_nets)} power net(s) from HPWL: {names}{suffix}')
            if self._cong_grid is not None:
                g = self._cong_grid
                print(f'[CadMust] Congestion grid: {g.nx}x{g.ny} bins, '
                      f'{congestion_bin_size/1e6:.1f}mm bin size, '
                      f'capacity {g._cap_val}/bin, weight {congestion_weight}')

        self._compute_all()

    def _rebuild_bbox_arrays(self) -> None:
        """Populate flat bbox arrays and sorted xmin list from current footprint positions."""
        for i, fp in enumerate(self.model.footprints):
            x1, y1, x2, y2 = fp.bbox
            self._bx1[i] = x1
            self._by1[i] = y1
            self._bx2[i] = x2
            self._by2[i] = y2
        self._xmin_items = sorted((self._bx1[i], i)
                                  for i in range(len(self.model.footprints)))
        for i in self._ko_fps:
            self._ko_world[i] = self.model.footprints[i].keepout_bboxes
        for i, r in self._pr_local.items():
            self._pr_world[i] = self.model.footprints[i].local_rect_to_world(r)

    _PAD_MASK = {"F": 1, "B": 2, "TH": 3}

    def _build_pad_data(self) -> None:
        """Collect the pads with copper data; fps whose pad reach (pad rect
        inflated by its clearance) leaves the bbox get a reach rect."""
        for i, fp in enumerate(self.model.footprints):
            rows = []
            for p in fp.pads:
                r = getattr(p, 'rect', None)
                m = self._PAD_MASK.get(getattr(p, 'side', ""), 0)
                if r is None or not m:
                    continue
                rows.append((r[0], r[1], r[2], r[3], m,
                             int(getattr(p, 'clearance', 0) or 0), p.net_code))
            if not rows:
                continue
            self._pad_local[i] = rows
            self._pad_maxc[i] = max(t[5] for t in rows)
            if any(t[5] for t in rows):
                rx1 = min(t[0] - t[5] for t in rows); ry1 = min(t[1] - t[5] for t in rows)
                rx2 = max(t[2] + t[5] for t in rows); ry2 = max(t[3] + t[5] for t in rows)
                lb = fp.local_bbox
                if rx1 < lb[0] or ry1 < lb[1] or rx2 > lb[2] or ry2 > lb[3]:
                    self._pr_local[i] = (rx1, ry1, rx2, ry2)

    def _pads_world(self, fi: int) -> List[tuple]:
        fp = self.model.footprints[fi]
        key = (fp.x, fp.y, fp.angle_deg)
        c = self._pad_world.get(fi)
        if c is not None and c[0] == key:
            return c[1]
        to_w = fp.local_rect_to_world
        rows = []
        for t in self._pad_local[fi]:
            w = to_w((t[0], t[1], t[2], t[3]))
            rows.append((w[0], w[1], w[2], w[3], t[4], t[5], t[6]))
        self._pad_world[fi] = (key, rows)
        return rows

    def pad_conflict_depth(self, i: int, j: int) -> int:
        """Largest (ox + oy) of a pad of i inflated by the required clearance
        against a pad of j (shared copper side, not the same non-zero net);
        0 when none. Same-group pairs are not exempt here (callers decide)."""
        if i not in self._pad_local or j not in self._pad_local:
            return 0
        pi = self._pads_world(i)
        pj = self._pads_world(j)
        # Exact prefilter: a pad can only conflict with a pad of the other
        # part if, inflated by the largest clearance that can apply, it
        # reaches the other part's reach rect (which contains all its pads).
        ri = self._reach(i)
        rj = self._reach(j)
        cj = self._pad_maxc[j]
        ci = self._pad_maxc[i]
        pi = [a for a in pi
              if a[0] - (a[5] if a[5] > cj else cj) < rj[2]
              and a[2] + (a[5] if a[5] > cj else cj) > rj[0]
              and a[1] - (a[5] if a[5] > cj else cj) < rj[3]
              and a[3] + (a[5] if a[5] > cj else cj) > rj[1]]
        if not pi:
            return 0
        pj = [b for b in pj
              if b[0] - (b[5] if b[5] > ci else ci) < ri[2]
              and b[2] + (b[5] if b[5] > ci else ci) > ri[0]
              and b[1] - (b[5] if b[5] > ci else ci) < ri[3]
              and b[3] + (b[5] if b[5] > ci else ci) > ri[1]]
        v = 0
        for a in pi:
            ax1, ay1, ax2, ay2, am, ac, an = a
            for b in pj:
                if not (am & b[4]):
                    continue
                if an and an == b[6]:
                    continue
                req = ac if ac > b[5] else b[5]
                ox = min(ax2 + req, b[2]) - max(ax1 - req, b[0])
                if ox <= 0:
                    continue
                oy = min(ay2 + req, b[3]) - max(ay1 - req, b[1])
                if oy <= 0:
                    continue
                if ox + oy > v:
                    v = ox + oy
        return v

    def _compute_pair_overlap_pad(self, i: int, j: int) -> float:
        """Pair value with the pad-level clearance term: max(courtyard/ring
        value, pad conflict depth * OVERLAP_WEIGHT); same-group pairs exempt."""
        base = self._pair_base(i, j)
        if i not in self._pad_local or j not in self._pad_local:
            return base
        ri = self._pr_world.get(i) or (self._bx1[i], self._by1[i], self._bx2[i], self._by2[i])
        rj = self._pr_world.get(j) or (self._bx1[j], self._by1[j], self._bx2[j], self._by2[j])
        if not (ri[0] < rj[2] and rj[0] < ri[2] and ri[1] < rj[3] and rj[1] < ri[3]):
            return base
        if self._same_group(i, j):
            return base
        v = self.pad_conflict_depth(i, j) * OVERLAP_WEIGHT
        return v if v > base else base

    def _update_bbox(self, fi: int) -> None:
        """Update bbox arrays and sorted xmin list for a single footprint."""
        # Remove old xmin entry from sorted list
        old_x1 = self._bx1[fi]
        pos = bisect.bisect_left(self._xmin_items, (old_x1, fi))
        if pos < len(self._xmin_items) and self._xmin_items[pos] == (old_x1, fi):
            self._xmin_items.pop(pos)
        # Update flat arrays
        x1, y1, x2, y2 = self.model.footprints[fi].bbox
        self._bx1[fi] = x1
        self._by1[fi] = y1
        self._bx2[fi] = x2
        self._by2[fi] = y2
        # Insert new xmin entry
        bisect.insort(self._xmin_items, (x1, fi))
        if fi in self._ko_world:
            self._ko_world[fi] = self.model.footprints[fi].keepout_bboxes
        r = self._pr_local.get(fi)
        if r is not None:
            self._pr_world[fi] = self.model.footprints[fi].local_rect_to_world(r)

    def _reach(self, fi: int) -> Tuple[int, int, int, int]:
        """Overlap reach of one fp: its bbox inflated by its clearance ring,
        united with its attached keep-out rects (world space)."""
        c = self._fp_clear[fi]
        x1 = self._bx1[fi] - c; y1 = self._by1[fi] - c
        x2 = self._bx2[fi] + c; y2 = self._by2[fi] + c
        pr = self._pr_world.get(fi)
        if pr is not None:
            if pr[0] < x1: x1 = pr[0]
            if pr[1] < y1: y1 = pr[1]
            if pr[2] > x2: x2 = pr[2]
            if pr[3] > y2: y2 = pr[3]
        for r in self._ko_world.get(fi, ()):
            if r[0] < x1: x1 = r[0]
            if r[1] < y1: y1 = r[1]
            if r[2] > x2: x2 = r[2]
            if r[3] > y2: y2 = r[3]
        return x1, y1, x2, y2

    def _compute_net_hpwl(self, net: Net) -> int:
        if net.is_excluded or len(net.pad_refs) < 2:
            return 0
        fps = self.model.footprints
        xmin = ymin = float('inf')
        xmax = ymax = float('-inf')
        for fi, pi in net.pad_refs:
            fp = fps[fi]
            pad = fp.pads[pi]
            # Inline abs_position using cached trig (avoids math.cos/sin per pad)
            cos_a = fp._cos_a
            sin_a = fp._sin_a
            ax = fp.x + int(pad.offset_x * cos_a - pad.offset_y * sin_a)
            ay = fp.y + int(pad.offset_x * sin_a + pad.offset_y * cos_a)
            if ax < xmin: xmin = ax
            if ax > xmax: xmax = ax
            if ay < ymin: ymin = ay
            if ay > ymax: ymax = ay
        return int(xmax - xmin) + int(ymax - ymin)

    def _compute_net_hpwl_bbox(self, net: Net):
        """Compute HPWL and bounding box in a single pad iteration.

        Returns (hpwl, (xmin, ymin, xmax, ymax)) or (0, None).
        Used when congestion is enabled to avoid duplicate pad iteration.
        """
        if net.is_excluded or len(net.pad_refs) < 2:
            return 0, None
        fps = self.model.footprints
        xmin = ymin = float('inf')
        xmax = ymax = float('-inf')
        for fi, pi in net.pad_refs:
            fp = fps[fi]
            pad = fp.pads[pi]
            cos_a = fp._cos_a
            sin_a = fp._sin_a
            ax = fp.x + int(pad.offset_x * cos_a - pad.offset_y * sin_a)
            ay = fp.y + int(pad.offset_x * sin_a + pad.offset_y * cos_a)
            if ax < xmin: xmin = ax
            if ax > xmax: xmax = ax
            if ay < ymin: ymin = ay
            if ay > ymax: ymax = ay
        return (int(xmax - xmin) + int(ymax - ymin),
                (int(xmin), int(ymin), int(xmax), int(ymax)))

    def _compute_all_hpwl(self):
        self._total_hpwl = 0
        for nc, net in self.model.nets.items():
            h = self._compute_net_hpwl(net)
            self.net_hpwl[nc] = h
            self._total_hpwl += h
        extra = 0.0
        for nc, w in self._net_w.items():
            extra += (w - 1.0) * self.net_hpwl.get(nc, 0)
        self._hpwl_extra = extra

    def _same_group(self, i: int, j: int) -> bool:
        """Check if footprints i and j are in the same component group."""
        gi = self.model.fp_to_group.get(i)
        return gi is not None and gi == self.model.fp_to_group.get(j)

    @staticmethod
    def _rect_overlap(a: Tuple[int, int, int, int],
                      b: Tuple[int, int, int, int]) -> int:
        """Return ox + oy for two overlapping rects, 0 if disjoint."""
        ox = min(a[2], b[2]) - max(a[0], b[0])
        oy = min(a[3], b[3]) - max(a[1], b[1])
        if ox > 0 and oy > 0:
            return ox + oy
        return 0

    def _compute_pair_overlap(self, i: int, j: int) -> float:
        """Compute overlap penalty for a single pair (i, j).

        Pairs on opposite board sides collide only via through-hole pads:
        a back-side component's body does not occupy front-side area (e.g.
        a display module mounted on the back must not block front-side
        placement), but its TH pads drill through and block both sides.

        Skips pairs in the same component group — the designer placed
        them intentionally and the group moves as a rigid body.
        """
        fps = self.model.footprints
        if self._fp_is_front[i] != self._fp_is_front[j]:
            if self._same_group(i, j):
                return 0.0
            total = 0
            th_j = fps[j].th_bboxes
            if th_j:
                bbox_i = fps[i].bbox
                for r in th_j:
                    total += self._rect_overlap(bbox_i, r)
            th_i = fps[i].th_bboxes
            if th_i:
                bbox_j = fps[j].bbox
                for r in th_i:
                    total += self._rect_overlap(r, bbox_j)
            return total * OVERLAP_WEIGHT
        if self._same_group(i, j):
            return 0.0
        x1min, y1min, x1max, y1max = fps[i].bbox
        x2min, y2min, x2max, y2max = fps[j].bbox
        ox = max(0, min(x1max, x2max) - max(x1min, x2min))
        oy = max(0, min(y1max, y2max) - max(y1min, y2min))
        if ox > 0 and oy > 0:
            return (ox + oy) * OVERLAP_WEIGHT
        return 0.0

    @staticmethod
    def _inflate(r: Tuple[int, int, int, int], c: int) -> Tuple[int, int, int, int]:
        return (r[0] - c, r[1] - c, r[2] + c, r[3] + c)

    def _compute_pair_overlap_clear(self, i: int, j: int) -> float:
        """Pair overlap with component clearances (keep-out rings) and
        attached keep-out rects (Footprint.keepout_rects).

        A footprint's ring applies to a pair when its side restriction is
        None (both sides) or equals the OTHER footprint's side, as in the
        checker. When it applies, the footprint's bbox is inflated by its
        clearance and tested against the other's raw bbox, on either side of
        the board (a both-sides antenna keep-out also excludes back-side
        bodies). The larger of the ring overlaps and the plain pair overlap
        (same-side body or opposite-side through-hole interaction) counts.
        Attached rects follow the same side rule and are tested in world
        space against the other's raw bbox; the largest single intrusion
        (ring or rect, ox + oy) counts. Same-group pairs stay exempt.
        """
        ci = self._fp_clear[i]
        cj = self._fp_clear[j]
        ki = self._ko_world.get(i)
        kj = self._ko_world.get(j)
        if ci or ki:
            si = self._fp_clear_side[i]
            if si is not None and si != self._fp_is_front[j]:
                ci = 0
                ki = None
        if cj or kj:
            sj = self._fp_clear_side[j]
            if sj is not None and sj != self._fp_is_front[i]:
                cj = 0
                kj = None
        plain = CostState._compute_pair_overlap(self, i, j)
        if not ci and not cj and not ki and not kj:
            return plain
        if self._same_group(i, j):
            return 0.0
        fps = self.model.footprints
        ro = self._rect_overlap
        bi = fps[i].bbox
        bj = fps[j].bbox
        v = 0
        if ci:
            v = ro(self._inflate(bi, ci), bj)
        if cj:
            v = max(v, ro(bi, self._inflate(bj, cj)))
        # Attached keep-out rects: each against the other's body bbox
        if ki:
            for r in ki:
                v = max(v, ro(r, bj))
        if kj:
            for r in kj:
                v = max(v, ro(bi, r))
        ring = v * OVERLAP_WEIGHT
        return ring if ring > plain else plain

    def _compute_overlap_penalty(self) -> float:
        """O(n^2) bounding-box overlap check. Populates per-pair cache."""
        fps = self.model.footprints
        n = len(fps)
        self._pair_overlaps.clear()
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                v = self._compute_pair_overlap(i, j)
                if v > 0:
                    self._pair_overlaps[(i, j)] = v
                    total += v
        return total

    def _compute_fp_boundary(self, fp_idx: int) -> float:
        """Compute boundary penalty for a single footprint.

        Penalty scales with the footprint's net count so that
        high-pin-count components get proportionally stronger boundary
        penalties, matching their larger HPWL pull.

        Uses cached bbox arrays (_bx1/_by1/_bx2/_by2), precomputed net scale,
        and cached board extents to avoid redundant attribute lookups.
        """
        fp = self.model.footprints[fp_idx]
        if fp.locked:
            return 0.0
        net_scale = self._fp_net_scale[fp_idx]
        xmin = self._bx1[fp_idx]
        ymin = self._by1[fp_idx]
        xmax = self._bx2[fp_idx]
        ymax = self._by2[fp_idx]
        if fp_idx in self._overhang_fps:
            # the part may reach past its locked edge: clip the bbox there
            cb = locked_edge_clip(fp, (xmin, ymin, xmax, ymax), self._outline_polygon,
                                  self._oxmin, self._oymin, self._oxmax, self._oymax)
            if cb is None:
                return 0.0
            xmin, ymin, xmax, ymax = cb
        dx = max(0, self._oxmin - xmin) + max(0, xmax - self._oxmax)
        dy = max(0, self._oymin - ymin) + max(0, ymax - self._oymax)
        total = (dx + dy) * BOUNDARY_WEIGHT * net_scale
        poly = self._outline_polygon
        if poly is not None and dx == 0 and dy == 0:
            for cx, cy in ((xmin, ymin), (xmax, ymin),
                           (xmax, ymax), (xmin, ymax)):
                if not point_in_polygon(cx, cy, poly):
                    d = dist_to_polygon(cx, cy, poly)
                    total += d * POLYGON_BOUNDARY_WEIGHT * net_scale
        return total

    def _compute_boundary_penalty(self) -> float:
        """Compute boundary penalties for all footprints. Populates per-fp cache."""
        self._fp_boundary.clear()
        total = 0.0
        for i, fp in enumerate(self.model.footprints):
            v = self._compute_fp_boundary(i)
            if v > 0:
                self._fp_boundary[i] = v
                total += v
        return total

    def _compute_fp_keepout(self, fp_idx: int) -> float:
        """Compute keepout penalty for a single footprint.

        (ox + oy) * KEEPOUT_WEIGHT per intruded keep-out, the same per-nm
        form as the overlap term and flat per part. Until 2026-11-06 it was
        also multiplied by the part's net count, so a many-net part paid up
        to 100 x nets per nm for a keep-out against a flat 50 for a
        collision and preferred the collision. penalty_scale still applies
        (total_cost), as before.

        Uses cached bbox arrays.
        """
        if self.model.footprints[fp_idx].locked or not self._keepouts:
            return 0.0
        fxmin = self._bx1[fp_idx]
        fymin = self._by1[fp_idx]
        fxmax = self._bx2[fp_idx]
        fymax = self._by2[fp_idx]
        total = 0.0
        # Every keep-out applies to both sides for now, also one whose
        # KeepOut.layers names a single copper layer.
        for ko in self._keepouts:
            ox = max(0, min(fxmax, ko.xmax) - max(fxmin, ko.xmin))
            oy = max(0, min(fymax, ko.ymax) - max(fymin, ko.ymin))
            if ox > 0 and oy > 0:
                total += (ox + oy) * KEEPOUT_WEIGHT
        return total

    def _compute_keepout_penalty(self) -> float:
        """Compute keepout penalties for all footprints. Populates per-fp cache."""
        self._fp_keepout.clear()
        total = 0.0
        for i, fp in enumerate(self.model.footprints):
            v = self._compute_fp_keepout(i)
            if v > 0:
                self._fp_keepout[i] = v
                total += v
        return total

    # --- Soft group cohesion and zone preference (constraint file) ---

    def _compute_group_cohesion(self, gi: int) -> float:
        """max(0, union-bbox diagonal - max_span) * cohesion weight for one group."""
        idx, span = self._soft_groups[gi]
        bx1 = self._bx1; by1 = self._by1; bx2 = self._bx2; by2 = self._by2
        i0 = idx[0]
        x1 = bx1[i0]; y1 = by1[i0]; x2 = bx2[i0]; y2 = by2[i0]
        for k in idx:
            if bx1[k] < x1: x1 = bx1[k]
            if by1[k] < y1: y1 = by1[k]
            if bx2[k] > x2: x2 = bx2[k]
            if by2[k] > y2: y2 = by2[k]
        excess = math.hypot(x2 - x1, y2 - y1) - span
        return excess * self._cohesion_weight if excess > 0 else 0.0

    def _compute_fp_zone(self, fi: int) -> float:
        """Distance from the footprint position to its preferred zone rect, times the zone weight."""
        r = self._zone_rects.get(fi)
        if r is None or self._zone_weight == 0:
            return 0.0
        fp = self.model.footprints[fi]
        x = fp.x
        y = fp.y
        dx = max(r[0] - x, 0, x - r[2])
        dy = max(r[1] - y, 0, y - r[3])
        if dx == 0 and dy == 0:
            return 0.0
        return math.hypot(dx, dy) * self._zone_weight

    def _compute_extra_all(self) -> None:
        """Full recompute of the cohesion and zone terms (bbox arrays must be current)."""
        total = 0.0
        for gi in range(len(self._soft_groups)):
            v = self._compute_group_cohesion(gi)
            self._group_cost[gi] = v
            total += v
        self._cohesion_penalty = total
        self._fp_zone.clear()
        total = 0.0
        for fi in self._zone_rects:
            v = self._compute_fp_zone(fi)
            if v > 0:
                self._fp_zone[fi] = v
                total += v
        self._zone_penalty = total
        self._soft_penalty = self._cohesion_penalty + total

    def _update_extra(self, moved_fp_indices: Set[int]) -> None:
        """Incremental cohesion (groups containing a moved fp) and zone update."""
        fp_soft = self._fp_soft
        if fp_soft:
            groups: Set[int] = set()
            for fi in moved_fp_indices:
                gl = fp_soft.get(fi)
                if gl:
                    groups.update(gl)
            if groups:
                gc = self._group_cost
                pen = self._cohesion_penalty
                for gi in groups:
                    v = self._compute_group_cohesion(gi)
                    pen += v - gc[gi]
                    gc[gi] = v
                self._cohesion_penalty = pen
        zr = self._zone_rects
        if zr:
            fz = self._fp_zone
            pen = self._zone_penalty
            for fi in moved_fp_indices:
                if fi in zr:
                    old_v = fz.pop(fi, 0.0)
                    v = self._compute_fp_zone(fi)
                    if v > 0:
                        fz[fi] = v
                    pen += v - old_v
            self._zone_penalty = pen
        self._soft_penalty = self._cohesion_penalty + self._zone_penalty

    # --- Congestion ---

    def _compute_net_bbox(self, net: Net):
        """Return (xmin, ymin, xmax, ymax) of pad positions, or None."""
        if net.is_excluded or len(net.pad_refs) < 2:
            return None
        fps = self.model.footprints
        xmin = ymin = float('inf')
        xmax = ymax = float('-inf')
        for fi, pi in net.pad_refs:
            fp = fps[fi]
            pad = fp.pads[pi]
            cos_a = fp._cos_a
            sin_a = fp._sin_a
            ax = fp.x + int(pad.offset_x * cos_a - pad.offset_y * sin_a)
            ay = fp.y + int(pad.offset_x * sin_a + pad.offset_y * cos_a)
            if ax < xmin: xmin = ax
            if ax > xmax: xmax = ax
            if ay < ymin: ymin = ay
            if ay > ymax: ymax = ay
        return (int(xmin), int(ymin), int(xmax), int(ymax))

    def _build_congestion(self) -> None:
        """Build congestion grid demand from scratch."""
        grid = self._cong_grid
        if grid is None:
            return
        # Reset demand
        for i in range(len(grid.demand)):
            grid.demand[i] = 0
        grid.net_bins.clear()
        # Populate from all nets
        for nc, net in self.model.nets.items():
            bbox = self._compute_net_bbox(net)
            if bbox is None:
                continue
            bins = grid.bbox_to_bins(*bbox)
            grid.net_bins[nc] = bins
            grid.add_demand(bins)
        self._congestion_penalty = grid.compute_penalty(self._congestion_weight)

    def _update_congestion(self, affected_nets: Set[int],
                           net_bboxes: Dict[int, Optional[Tuple[int, int, int, int]]]) -> None:
        """Incrementally update congestion for nets whose pads moved.

        Uses per-bin penalty deltas instead of rescanning all bins.
        net_bboxes: pre-computed from the HPWL pass (avoids duplicate pad iteration).
        """
        grid = self._cong_grid
        if grid is None:
            return
        weight = self._congestion_weight
        penalty = self._congestion_penalty
        for nc in affected_nets:
            # Remove old demand (with delta tracking)
            old_bins = grid.net_bins.get(nc)
            if old_bins is not None:
                penalty += grid.remove_demand_delta(old_bins, weight)
            # Use pre-computed bbox
            bbox = net_bboxes.get(nc)
            if bbox is None:
                if old_bins is not None:
                    del grid.net_bins[nc]
                continue
            new_bins = grid.bbox_to_bins(*bbox)
            grid.net_bins[nc] = new_bins
            penalty += grid.add_demand_delta(new_bins, weight)
        self._congestion_penalty = penalty

    def _compute_all(self):
        self._rebuild_bbox_arrays()
        self._compute_all_hpwl()
        self._overlap_penalty = self._compute_overlap_penalty()
        self._boundary_penalty = self._compute_boundary_penalty()
        self._keepout_penalty = self._compute_keepout_penalty()
        self._build_congestion()
        if self._extra_terms:
            self._compute_extra_all()

    def update_penalty_scale(self, scale: float) -> None:
        """Set the penalty scale for boundary and keepout penalties.

        Called once per temperature step. Overlap is always at full weight.
        """
        self.penalty_scale = scale

    @property
    def total_cost(self) -> float:
        """Cost with current penalty_scale applied to boundary+keepout+congestion
        (and the constraint-file cohesion and zone terms)."""
        return (float(self._total_hpwl) + self._hpwl_extra + self._overlap_penalty
                + (self._boundary_penalty + self._keepout_penalty
                   + self._congestion_penalty + self._soft_penalty)
                * self.penalty_scale)

    @property
    def normalized_cost(self) -> float:
        """Cost with all penalties at full weight (penalty_scale=1.0).

        Used for best-solution tracking so solutions are compared fairly
        regardless of when they were found during the anneal.
        """
        return (float(self._total_hpwl) + self._hpwl_extra + self._overlap_penalty
                + self._boundary_penalty + self._keepout_penalty
                + self._congestion_penalty + self._soft_penalty)

    @property
    def hpwl(self) -> int:
        """Raw (unweighted) HPWL sum; net weights only affect the total cost."""
        return self._total_hpwl

    @property
    def weighted_hpwl(self) -> float:
        """HPWL with per-net weights applied (equals hpwl without weights)."""
        return float(self._total_hpwl) + self._hpwl_extra

    def snapshot(self, moved_fp_indices: Set[int]) -> dict:
        """Save cost state for affected footprints, so it can be restored
        cheaply on move rejection instead of recomputing.

        Saves all existing overlapping pairs involving moved footprints,
        plus bbox array values for moved footprints.
        """
        _ts = time.perf_counter()
        fps = self.model.footprints

        # Affected nets
        affected_nets: Set[int] = set()
        for fi in moved_fp_indices:
            affected_nets.update(fps[fi].net_codes)

        # Save existing overlap pairs involving moved footprints
        saved_pairs: Dict[Tuple[int, int], float] = {}
        for k, v in self._pair_overlaps.items():
            if k[0] in moved_fp_indices or k[1] in moved_fp_indices:
                saved_pairs[k] = v

        # Save bbox values for moved footprints
        saved_bbox = {}
        for fi in moved_fp_indices:
            saved_bbox[fi] = (self._bx1[fi], self._by1[fi],
                              self._bx2[fi], self._by2[fi])

        # Save congestion state for affected nets
        saved_congestion = None
        grid = self._cong_grid
        if grid is not None:
            saved_net_bins = {}
            for nc in affected_nets:
                old_bins = grid.net_bins.get(nc)
                saved_net_bins[nc] = list(old_bins) if old_bins is not None else None
            saved_congestion = {
                'penalty': self._congestion_penalty,
                'net_bins': saved_net_bins,
            }

        self.t_snapshot += time.perf_counter() - _ts
        snap = {
            'hpwl': self._total_hpwl,
            'overlap': self._overlap_penalty,
            'boundary': self._boundary_penalty,
            'keepout': self._keepout_penalty,
            'congestion': saved_congestion,
            'net_hpwl': {nc: self.net_hpwl.get(nc, 0) for nc in affected_nets},
            'pair_overlaps': saved_pairs,
            'fp_boundary': {fi: self._fp_boundary.get(fi, 0.0) for fi in moved_fp_indices},
            'fp_keepout': {fi: self._fp_keepout.get(fi, 0.0) for fi in moved_fp_indices},
            'saved_bbox': saved_bbox,
            'moved_fp_indices': moved_fp_indices,
        }
        if self._net_w:
            snap['hpwl_extra'] = self._hpwl_extra
        if self._extra_terms:
            groups: Set[int] = set()
            fp_soft = self._fp_soft
            for fi in moved_fp_indices:
                gl = fp_soft.get(fi)
                if gl:
                    groups.update(gl)
            gc = self._group_cost
            fz = self._fp_zone
            snap['extra'] = (
                self._cohesion_penalty,
                {gi: gc[gi] for gi in groups},
                self._zone_penalty,
                {fi: fz.get(fi, 0.0) for fi in moved_fp_indices
                 if fi in self._zone_rects},
            )
        return snap

    def restore(self, snap: dict) -> None:
        """Restore cost state from a snapshot. O(k) — no recomputation.

        Purges all _pair_overlaps entries involving moved footprints,
        then restores saved pairs and bbox values.
        """
        self._total_hpwl = snap['hpwl']
        self._overlap_penalty = snap['overlap']
        self._boundary_penalty = snap['boundary']
        self._keepout_penalty = snap['keepout']
        for nc, h in snap['net_hpwl'].items():
            self.net_hpwl[nc] = h
        if self._net_w:
            self._hpwl_extra = snap['hpwl_extra']
        if self._extra_terms:
            coh, gsaved, zpen, zsaved = snap['extra']
            self._cohesion_penalty = coh
            gc = self._group_cost
            for gi, v in gsaved.items():
                gc[gi] = v
            self._zone_penalty = zpen
            self._soft_penalty = coh + zpen
            fz = self._fp_zone
            for fi, v in zsaved.items():
                if v > 0:
                    fz[fi] = v
                else:
                    fz.pop(fi, None)

        # Purge ALL overlap pairs involving moved fps, then restore saved ones
        moved = snap.get('moved_fp_indices', set())
        keys_to_remove = [k for k in self._pair_overlaps
                          if k[0] in moved or k[1] in moved]
        for k in keys_to_remove:
            del self._pair_overlaps[k]
        for k, v in snap['pair_overlaps'].items():
            if v > 0:
                self._pair_overlaps[k] = v

        for fi, v in snap['fp_boundary'].items():
            if v > 0:
                self._fp_boundary[fi] = v
            else:
                self._fp_boundary.pop(fi, None)
        for fi, v in snap['fp_keepout'].items():
            if v > 0:
                self._fp_keepout[fi] = v
            else:
                self._fp_keepout.pop(fi, None)

        # Restore congestion grid state: undo the incremental update by
        # removing current (post-move) demand and re-adding old (pre-move) demand.
        saved_cong = snap.get('congestion')
        if saved_cong is not None and self._cong_grid is not None:
            grid = self._cong_grid
            self._congestion_penalty = saved_cong['penalty']
            for nc, old_bins in saved_cong['net_bins'].items():
                # Remove demand that was added by incremental_update
                cur_bins = grid.net_bins.get(nc)
                if cur_bins is not None:
                    grid.remove_demand(cur_bins)
                # Restore old net_bins and demand
                if old_bins is not None:
                    grid.net_bins[nc] = old_bins
                    grid.add_demand(old_bins)
                else:
                    grid.net_bins.pop(nc, None)

        # Restore bbox arrays and sorted xmin list (footprint positions already reverted)
        for fi, bbox in snap.get('saved_bbox', {}).items():
            # Remove current (post-move) xmin from sorted list
            cur_x1 = self._bx1[fi]
            pos = bisect.bisect_left(self._xmin_items, (cur_x1, fi))
            if pos < len(self._xmin_items) and self._xmin_items[pos] == (cur_x1, fi):
                self._xmin_items.pop(pos)
            # Restore flat arrays
            self._bx1[fi] = bbox[0]
            self._by1[fi] = bbox[1]
            self._bx2[fi] = bbox[2]
            self._by2[fi] = bbox[3]
            # Insort restored xmin
            bisect.insort(self._xmin_items, (bbox[0], fi))
            if fi in self._ko_world:
                self._ko_world[fi] = self.model.footprints[fi].keepout_bboxes

    def incremental_update(self, moved_fp_indices: Set[int]) -> float:
        """
        Recompute cost after footprints moved.

        HPWL: only recomputes nets connected to moved footprints.
        Overlap: flat bbox scan — O(n) per moved fp with C-level list ops.
        Boundary/keepout: only recomputes moved footprints — O(k).
        """
        fps = self.model.footprints
        n = len(fps)

        # --- HPWL (incremental, nets only; power nets excluded) ---
        # When congestion is enabled, compute bbox alongside HPWL in one
        # pad iteration (avoids duplicate _compute_net_bbox calls).
        _t0 = time.perf_counter()
        affected_nets: Set[int] = set()
        for fi in moved_fp_indices:
            affected_nets.update(fps[fi].net_codes)
        net_w = self._net_w
        if net_w:
            nh = self.net_hpwl
            w_before = 0.0
            for nc in affected_nets:
                w = net_w.get(nc)
                if w is not None:
                    w_before += (w - 1.0) * nh.get(nc, 0)
        cong_enabled = self._cong_grid is not None
        net_bboxes: Dict[int, Optional[Tuple[int, int, int, int]]] = {}
        if cong_enabled:
            for nc in affected_nets:
                net = self.model.nets[nc]
                if net.is_excluded:
                    continue
                old_h = self.net_hpwl.get(nc, 0)
                new_h, bbox = self._compute_net_hpwl_bbox(net)
                self._total_hpwl += (new_h - old_h)
                self.net_hpwl[nc] = new_h
                net_bboxes[nc] = bbox
        else:
            for nc in affected_nets:
                net = self.model.nets[nc]
                if net.is_excluded:
                    continue
                old_h = self.net_hpwl.get(nc, 0)
                new_h = self._compute_net_hpwl(net)
                self._total_hpwl += (new_h - old_h)
                self.net_hpwl[nc] = new_h
        if net_w:
            w_after = 0.0
            for nc in affected_nets:
                w = net_w.get(nc)
                if w is not None:
                    w_after += (w - 1.0) * nh.get(nc, 0)
            self._hpwl_extra += w_after - w_before
        _t1 = time.perf_counter()
        self.t_hpwl += _t1 - _t0

        # --- Overlap (incremental, flat bbox scan) ---
        # Update bbox arrays for moved footprints
        for fi in moved_fp_indices:
            self._update_bbox(fi)

        # Collect all pairs to check using sorted xmin list for O(log n + k) scan:
        # bisect finds all fps with xmin < mi_x2 (candidate left edges), then
        # check remaining 3 conditions on that reduced candidate set.
        pairs_to_update: Set[Tuple[int, int]] = set()
        bx1 = self._bx1; by1 = self._by1
        bx2 = self._bx2; by2 = self._by2
        xmin_items = self._xmin_items
        # Clearance rings: a moved fp's own ring widens its window; the few
        # fps that carry a ring are tested explicitly below (cheaper than
        # widening every scan by the largest ring).
        # Attached keep-out rects: the scan window of a moved fp is the union
        # of its (ring-inflated) bbox and its rects; the fps with a ring or
        # rects are tested explicitly with their own reach.
        ext_fps = self._ext_fps
        for mi in moved_fp_indices:
            mi_x1 = bx1[mi]; mi_y1 = by1[mi]
            mi_x2 = bx2[mi]; mi_y2 = by2[mi]
            if ext_fps:
                mi_x1, mi_y1, mi_x2, mi_y2 = self._reach(mi)
                for j in ext_fps:
                    if j != mi and j not in moved_fp_indices:
                        jx1, jy1, jx2, jy2 = self._reach(j)
                        if (jx1 < mi_x2 and jx2 > mi_x1
                                and jy1 < mi_y2 and jy2 > mi_y1):
                            pairs_to_update.add(_pair_key(mi, j))
            # All fps with xmin < mi_x2 are candidates (left edge left of query right edge)
            i1 = bisect.bisect_left(xmin_items, (mi_x2, -1))
            for k in range(i1):
                j = xmin_items[k][1]
                if j != mi and j not in moved_fp_indices:
                    if bx2[j] > mi_x1 and by1[j] < mi_y2 and by2[j] > mi_y1:
                        pairs_to_update.add(_pair_key(mi, j))
            # Also add existing overlapping pairs (might have moved apart)
            # — these may not pass the new bbox check above
        for k in list(self._pair_overlaps):
            if k[0] in moved_fp_indices or k[1] in moved_fp_indices:
                pairs_to_update.add(k)

        # Between moved fps themselves
        moved_list = list(moved_fp_indices)
        for idx_a in range(len(moved_list)):
            for idx_b in range(idx_a + 1, len(moved_list)):
                pairs_to_update.add(_pair_key(moved_list[idx_a], moved_list[idx_b]))

        # Recompute affected pairs
        for key in pairs_to_update:
            old_v = self._pair_overlaps.pop(key, 0.0)
            self._overlap_penalty -= old_v
            new_v = self._compute_pair_overlap(key[0], key[1])
            if new_v > 0:
                self._pair_overlaps[key] = new_v
            self._overlap_penalty += new_v
        _t2 = time.perf_counter()
        self.t_overlap += _t2 - _t1

        # --- Boundary (incremental, moved fps only) ---
        for fi in moved_fp_indices:
            old_v = self._fp_boundary.pop(fi, 0.0)
            self._boundary_penalty -= old_v
            new_v = self._compute_fp_boundary(fi)
            if new_v > 0:
                self._fp_boundary[fi] = new_v
            self._boundary_penalty += new_v
        _t3 = time.perf_counter()
        self.t_boundary += _t3 - _t2

        # --- Keepout (incremental, moved fps only) ---
        for fi in moved_fp_indices:
            old_v = self._fp_keepout.pop(fi, 0.0)
            self._keepout_penalty -= old_v
            new_v = self._compute_fp_keepout(fi)
            if new_v > 0:
                self._fp_keepout[fi] = new_v
            self._keepout_penalty += new_v
        _t4 = time.perf_counter()
        self.t_keepout += _t4 - _t3

        # --- Congestion (incremental, affected nets only) ---
        if self._cong_grid is not None:
            self._update_congestion(affected_nets, net_bboxes)
            self.t_congestion += time.perf_counter() - _t4

        # --- Cohesion + zone (constraint file; skipped without constraints) ---
        if self._extra_terms:
            self._update_extra(moved_fp_indices)

        return self.total_cost
