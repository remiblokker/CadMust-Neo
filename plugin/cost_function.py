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
KEEPOUT_WEIGHT = 100.0      # per-violation, multiplied by overlap distance (nm)

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
                 congestion_bin_size: int = 5_000_000):
        self.model = model
        self.net_hpwl: Dict[int, int] = {}
        self._total_hpwl: int = 0
        self._overlap_penalty: float = 0.0
        self._boundary_penalty: float = 0.0
        self._keepout_penalty: float = 0.0
        self._congestion_penalty: float = 0.0
        self._congestion_weight: float = congestion_weight
        self._congestion_bin_size: int = congestion_bin_size
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
        if self.model.footprints[fp_idx].locked:
            return 0.0
        net_scale = self._fp_net_scale[fp_idx]
        xmin = self._bx1[fp_idx]
        ymin = self._by1[fp_idx]
        xmax = self._bx2[fp_idx]
        ymax = self._by2[fp_idx]
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

        Penalty scales with the footprint's net count so that
        high-pin-count components get proportionally stronger keepout
        penalties, matching their larger HPWL pull.

        Uses cached bbox arrays and precomputed net scale.
        """
        if self.model.footprints[fp_idx].locked or not self._keepouts:
            return 0.0
        net_scale = self._fp_net_scale[fp_idx]
        fxmin = self._bx1[fp_idx]
        fymin = self._by1[fp_idx]
        fxmax = self._bx2[fp_idx]
        fymax = self._by2[fp_idx]
        total = 0.0
        for ko in self._keepouts:
            ox = max(0, min(fxmax, ko.xmax) - max(fxmin, ko.xmin))
            oy = max(0, min(fymax, ko.ymax) - max(fymin, ko.ymin))
            if ox > 0 and oy > 0:
                total += (ox + oy) * KEEPOUT_WEIGHT * net_scale
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

    def update_penalty_scale(self, scale: float) -> None:
        """Set the penalty scale for boundary and keepout penalties.

        Called once per temperature step. Overlap is always at full weight.
        """
        self.penalty_scale = scale

    @property
    def total_cost(self) -> float:
        """Cost with current penalty_scale applied to boundary+keepout+congestion."""
        return (float(self._total_hpwl) + self._overlap_penalty
                + (self._boundary_penalty + self._keepout_penalty
                   + self._congestion_penalty)
                * self.penalty_scale)

    @property
    def normalized_cost(self) -> float:
        """Cost with all penalties at full weight (penalty_scale=1.0).

        Used for best-solution tracking so solutions are compared fairly
        regardless of when they were found during the anneal.
        """
        return (float(self._total_hpwl) + self._overlap_penalty
                + self._boundary_penalty + self._keepout_penalty
                + self._congestion_penalty)

    @property
    def hpwl(self) -> int:
        return self._total_hpwl

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
        return {
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
        for mi in moved_fp_indices:
            mi_x1 = bx1[mi]; mi_y1 = by1[mi]
            mi_x2 = bx2[mi]; mi_y2 = by2[mi]
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

        return self.total_cost
