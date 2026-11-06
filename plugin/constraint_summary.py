"""Constraint-satisfaction summary of a placement, for the plugin results dialog.

Pure Python, no wx, no pcbnew. Measures only the constraint kinds the engine
itself models, with the engine's own geometry, so a FAIL here means the cost
term (or the snap / lock) did not get what it asked for:

    kind         measured                               pass when
    group_span   union-bbox diagonal of the soft group  diagonal <= max_span_mm
                 (CostState._compute_group_cohesion)    (the term is 0 exactly then)
    zone         distance from the footprint position   distance == 0, edges
                 to the zone rect (_compute_fp_zone)    inclusive (term is 0 then)
    edge_lock    bbox edge to the outline edge, on a    distance <= EDGE_LOCK_MARGIN
                 polygonal outline the polygon edge     (1.0 mm, where the snap puts
                 over the bbox span (moves.resnap_edges, it) + 1 um rounding;
                 cost_function.polygon_edge_limit)      a part past the edge passes
                 antenna_outward lock: the module end   module end within 1 um of
                 (Footprint.edge_anchor_world) to the   the edge, and the antenna
                 edge, plus the antenna direction       side faces the locked edge
    fixed        footprint position vs the file's x/y   error <= 1 um (apply sets it
                 (board-relative, apply_constraints)    exactly, then locks it)

Everything else (overlaps, boundary, keep-outs, back side, net priorities,
signal classes, templates) is not measured here; the full checker is
CadMust-Autopilot check_constraints.py. Note that the checker applies its own
tolerances (zone 0.5 mm, edge 3.0 mm, fixed 0.5 mm), so it can pass an item
this summary fails, never the other way round.

    summary = summarize(model, cset)   # model AFTER the run, constraints applied
    text = format_text(summary, report)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .constraints import MM, ZONES, zone_rect_nm, facing_edge
from .cost_function import polygon_edge_limit
from .moves import EDGE_LOCK_MARGIN, _edge_snap_polygon

# apply_constraints rounds a fixed position to whole nm and resnap_edges
# rounds the polygon edge to whole nm; 1 um absorbs that and nothing else.
ROUNDING_TOL_NM = 1_000

KIND_LABELS = (
    ("group_span", "Group spans"),
    ("zone", "Zone preferences"),
    ("edge_lock", "Edge locks"),
    ("fixed", "Fixed positions"),
)

NOT_MEASURED = ("overlaps, boundary, keep-outs, back side, net priorities, "
                "templates, signal classes")
FULL_CHECK = "Full check: CadMust-Autopilot check_constraints.py"


@dataclass
class Check:
    kind: str           # group_span / zone / edge_lock / fixed
    item: str           # group id or reference
    passed: bool
    measured_mm: float
    limit_mm: float
    detail: str


@dataclass
class ConstraintSummary:
    checks: List[Check] = field(default_factory=list)

    def totals(self) -> Dict[str, Tuple[int, int]]:
        """kind -> (n_pass, n_fail), only kinds that have checks."""
        out: Dict[str, Tuple[int, int]] = {}
        for c in self.checks:
            p, f = out.get(c.kind, (0, 0))
            out[c.kind] = (p + 1, f) if c.passed else (p, f + 1)
        return out

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]


def _first_index(model) -> Dict[str, int]:
    by_ref: Dict[str, int] = {}
    for i, f in enumerate(model.footprints):
        by_ref.setdefault(f.reference, i)
    return by_ref


def measure_group_spans(model) -> List[Check]:
    """One check per soft group the engine carries (model.soft_groups)."""
    fps = model.footprints
    out = []
    for gid, idx, span in (getattr(model, "soft_groups", None) or []):
        if len(idx) < 2:
            continue
        bbs = [fps[i].bbox for i in idx]
        x1 = min(b[0] for b in bbs); y1 = min(b[1] for b in bbs)
        x2 = max(b[2] for b in bbs); y2 = max(b[3] for b in bbs)
        diag = math.hypot(x2 - x1, y2 - y1)
        ok = diag - span <= 0
        out.append(Check("group_span", gid, ok, diag / MM, span / MM,
                         "span %.2f mm %s max %.2f mm (%d members)"
                         % (diag / MM, "<=" if ok else ">", span / MM, len(idx))))
    return out


def _zone_names(model, cset) -> Dict[int, str]:
    """index -> preferred zone name from the file (first occurrence), else by
    matching the stored rect against the 3x3 grid."""
    names: Dict[int, str] = {}
    if cset is not None:
        by_ref = _first_index(model)
        for it in (cset.placement.get("zone_preferences") or []):
            i = by_ref.get(it.get("ref"))
            if i is not None:
                names[i] = it.get("zone", "?")
    return names


def _zone_from_rect(model, rect) -> str:
    for z in ZONES:
        if all(abs(a - b) < 1 for a, b in zip(zone_rect_nm(model, z), rect)):
            return z
    return "?"


def measure_zones(model, cset=None) -> List[Check]:
    """One check per zone preference the engine carries (model.zone_prefs)."""
    fps = model.footprints
    names = _zone_names(model, cset)
    out = []
    for fi, r in sorted((getattr(model, "zone_prefs", None) or {}).items()):
        f = fps[fi]
        dx = max(r[0] - f.x, 0, f.x - r[2])
        dy = max(r[1] - f.y, 0, f.y - r[3])
        d = math.hypot(dx, dy)
        zone = names.get(fi) or _zone_from_rect(model, r)
        ok = d == 0
        out.append(Check("zone", f.reference, ok, d / MM, 0.0,
                         "centre in %s" % zone if ok else
                         "centre %.2f mm outside %s" % (d / MM, zone)))
    return out


def edge_distance_nm(model, fi: int) -> Optional[float]:
    """Distance from footprint fi's bbox to its locked edge, measured the way
    resnap_edges snaps it (positive = inside the board). None if not locked."""
    f = model.footprints[fi]
    edge = getattr(f, "edge_lock", None)
    if edge is None:
        return None
    poly = _edge_snap_polygon(model)
    x1, y1, x2, y2 = f.bbox
    anc = f.edge_anchor_world()
    ref = {"left": model.outline_xmin, "right": model.outline_xmax,
           "top": model.outline_ymin, "bottom": model.outline_ymax}[edge]
    if poly is not None:
        lo, hi = (y1, y2) if edge in ("left", "right") else (x1, x2)
        lim = polygon_edge_limit(poly, edge, lo, hi)
        if lim is not None:
            ref = int(round(lim))
    if anc is not None:     # antenna_outward: the module end, not the bbox
        x1 = x2 = anc[0]
        y1 = y2 = anc[1]
    return {"left": x1 - ref, "right": ref - x2,
            "top": y1 - ref, "bottom": ref - y2}[edge]


def measure_edge_locks(model) -> List[Check]:
    """One check per edge-locked footprint (model.edge_locked_indices)."""
    out = []
    limit = EDGE_LOCK_MARGIN + ROUNDING_TOL_NM
    for fi in sorted(getattr(model, "edge_locked_indices", None) or ()):
        d = edge_distance_nm(model, fi)
        if d is None:
            continue
        f = model.footprints[fi]
        side = getattr(f, "antenna_side", None)
        if side is not None and f.edge_anchor_world() is not None:
            ok = abs(d) <= ROUNDING_TOL_NM
            out.append(Check("edge_lock", f.reference, ok, d / MM, 0.0,
                             "module end %.2f mm from the %s edge" % (d / MM, f.edge_lock)))
            faces = facing_edge(side, f.angle_deg)
            out.append(Check("edge_lock", f.reference + " antenna", faces == f.edge_lock,
                             0.0, 0.0, "antenna side %s faces the %s edge" % (side, faces)))
            continue
        ok = d <= limit
        past = " (reaches past the edge)" if d < -ROUNDING_TOL_NM else ""
        out.append(Check("edge_lock", f.reference, ok, d / MM, EDGE_LOCK_MARGIN / MM,
                         "%.2f mm from the %s edge%s%s"
                         % (d / MM, f.edge_lock, past,
                            "" if ok else ", snap distance is %.2f mm"
                            % (EDGE_LOCK_MARGIN / MM))))
    return out


def measure_fixed(model, cset) -> List[Check]:
    """One check per fixed_positions item whose reference is on the board."""
    if cset is None:
        return []
    by_ref = _first_index(model)
    ox, oy = model.outline_xmin, model.outline_ymin
    out = []
    for it in (cset.placement.get("fixed_positions") or []):
        i = by_ref.get(it.get("ref"))
        if i is None:
            continue    # reported as skipped by apply_constraints
        f = model.footprints[i]
        tx = ox + int(round(it["x_mm"] * MM))
        ty = oy + int(round(it["y_mm"] * MM))
        err = math.hypot(f.x - tx, f.y - ty)
        ok = err <= ROUNDING_TOL_NM
        out.append(Check("fixed", f.reference, ok, err / MM, 0.0,
                         "at (%.2f, %.2f) mm" % ((f.x - ox) / MM, (f.y - oy) / MM)
                         + ("" if ok else ", %.2f mm from (%.2f, %.2f) mm"
                            % (err / MM, it["x_mm"], it["y_mm"]))))
    return out


def summarize(model, cset=None) -> ConstraintSummary:
    """Measure the final placement. `model` must carry the applied constraints
    (soft_groups, zone_prefs, edge_locked_indices) and the final positions."""
    s = ConstraintSummary()
    s.checks.extend(measure_group_spans(model))
    s.checks.extend(measure_zones(model, cset))
    s.checks.extend(measure_edge_locks(model))
    s.checks.extend(measure_fixed(model, cset))
    return s


def format_text(summary: ConstraintSummary, report=None, *, path: str = "",
                weights: Optional[Tuple[float, float]] = None,
                unlocked: Optional[List[str]] = None,
                max_unlocked: int = 20) -> str:
    """Plain text for the results dialog: what was applied and skipped (from
    the ConstraintReport), then the measured totals and every FAIL line."""
    lines: List[str] = []
    if path:
        lines.append("File: %s" % path)
    if weights is not None:
        lines.append("Weights: cohesion %g, zone %g" % weights)
    if report is not None:
        c = report.counts()
        applied = ", ".join("%s %d" % kv for kv in c["applied"].items()) or "nothing"
        lines.append("Applied: %s" % applied)
    if unlocked is not None:
        if unlocked:
            shown = ", ".join(unlocked[:max_unlocked])
            more = " (+%d more)" % (len(unlocked) - max_unlocked) \
                if len(unlocked) > max_unlocked else ""
            lines.append("Board lock cleared (%d): %s%s" % (len(unlocked), shown, more))
        else:
            lines.append("Board lock cleared: none")
    if report is not None:
        if report.skipped:
            lines.append("Skipped (%d):" % len(report.skipped))
            for cat, item, reason in report.skipped:
                lines.append("  %s %s: %s" % (cat, item, reason))
        else:
            lines.append("Skipped: none")
        for w in report.warnings:
            lines.append("Warning: %s" % w)
    lines.append("")
    lines.append("Measured on the final placement: %d pass, %d FAIL"
                 % (summary.n_pass, summary.n_fail))
    totals = summary.totals()
    for kind, label in KIND_LABELS:
        if kind in totals:
            p, f = totals[kind]
            lines.append("  %-17s %3d pass  %3d FAIL" % (label, p, f))
    fails = summary.failures()
    if fails:
        lines.append("")
        lines.append("FAIL:")
        for c in fails:
            lines.append("  %-10s %-14s %s" % (c.kind, c.item, c.detail))
    lines.append("")
    lines.append("Not measured here: %s." % NOT_MEASURED)
    lines.append("Zone and edge use the engine's tolerance (0 mm and the 1.0 mm snap);")
    lines.append("the checker's tolerances are looser. %s." % FULL_CHECK)
    return "\n".join(lines)
