"""Antenna window record: where the antenna strip of an antenna_outward part
lies relative to the board outline after placement. A measurement, not a
placement rule; it does not change what the engine does.

An antenna_outward part has its module end on the outer envelope of the
locked edge (cost_function.anchor_edge_reference). Under the antenna strip
the outline can then be:

* void      (a notch under the module: the antenna window, or the module
             hanging past a lower part of a stepped edge),
* material  (a straight edge: the strip lies on the board, as on a module
             whose antenna sits over a copper-free board area),
* both      (the module straddles a notch wall or a step).

antenna_window() measures that with the real outline polygon (not its
bbox) and classifies it. Geometry only: a "pass" says the geometry is one
of the two unambiguous cases and records which; it says nothing about RF
performance or the clearance the datasheet asks for (unknown RF
requirements are never turned into a pass).

Definitions (nm, world frame, KiCad y down):

* strip: the antenna strip of the antenna item (AntennaResolution.strip:
  [module end - depth_mm, module end] outward, across the pad-centre extent
  of the antenna side). Without depth_mm the strip is taken from the
  outermost pad centre to the module end (strip_source says which).
* material fraction: area(strip intersected with the outline polygon) /
  area(strip). Without a polygon (rectangular board) the outline bbox.
* levels: the outline levels of the locked edge across the strip's width
  (the per-interval outer boundary of cost_function._edge_profile, read at
  the middle of each interval the width overlaps); stepped = more than one
  level (a notch wall or a step under the module).
* notch: one level under the module that lies inward of the outline on
  BOTH sides of the module along the edge (the edge comes back out on the
  left and on the right): a window. Inward on one side only = a step.

* envelope offset: |module end - outer envelope| on the edge normal. The
  engine snaps an antenna_outward module end onto the envelope exactly; a
  hand-placed board (the checker also runs on designer boards) may not.
  window and material need the offset within ENVELOPE_TOL_NM (the engine's
  edge-lock snap distance, moves.EDGE_LOCK_MARGIN, 1.0 mm); otherwise the
  class is off_envelope (internal review #1, 2026-11-06: a module 2.8 mm
  past the outline over a notch read as a window).
* outline: the single outer contour of the board (board_model stores
  poly_set.Outline(0) only); holes and cut-outs inside it are not
  represented. When the extracted model counts holes (outline_hole_count >
  0) the material fraction cannot be trusted and the class is unknown.

Classification, in this order:

    no_strip      strip has zero area                      warn
    unknown       the outline has holes (not represented)  warn
    mixed         0 < material fraction < 1                warn
    stepped       more than one level under the module    warn
    off_envelope  wholly over material or wholly over void warn
                  in a notch, but the module end is more
                  than ENVELOPE_TOL_NM off the envelope
    material      strip wholly over board material         pass
    window        strip wholly over void in a notch        pass
    overhang      strip wholly over void, no notch         warn
                  (past a lower step or past the outline)

Shared by plugin.constraint_summary (GUI results) and the
CadMust-Autopilot checker (record keepout.<ref>.antenna_window).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .constraints import MM, _axis_sign, _from_outward
from .cost_function import anchor_edge_reference, _edge_profile
from .moves import EDGE_LOCK_MARGIN

# area fractions within this of 0 or 1 count as wholly void / material
FRACTION_TOL = 1e-3
# two outline levels closer than this are one level (10 um)
LEVEL_TOL_NM = 10_000
# module end vs outer envelope: window/material only within this (the
# engine's edge-lock snap distance; the checker uses the same number, it
# calls this function)
ENVELOPE_TOL_NM = EDGE_LOCK_MARGIN

SINGLE_CONTOUR_NOTE = "single outer contour, holes not represented"

CLASS_STATUS = {"no_strip": "warn", "mixed": "warn", "stepped": "warn",
                "material": "pass", "window": "pass", "overhang": "warn",
                "off_envelope": "warn", "unknown": "warn"}

CLASS_TEXT = {
    "no_strip": "no antenna strip to measure",
    "mixed": "strip partly over board material, partly over void",
    "stepped": "the locked edge has more than one level under the module",
    "material": "plain edge: strip wholly over board material",
    "window": "antenna window: strip wholly over void in a notch",
    "overhang": "strip wholly over void without a notch around it "
                "(past a lower step of the edge or past the outline)",
    "off_envelope": "module end off the outer envelope of the locked edge",
    "unknown": "outline has holes the model does not represent; strip not classified",
}


@dataclass
class AntennaWindow:
    ref: str
    edge: str
    classification: str
    status: str                                 # "pass" or "warn"
    module_end_nm: float                        # module end, coordinate along the edge normal
    envelope_nm: float                          # outer envelope of the locked edge, same axis
    strip_world: Tuple[int, int, int, int]      # world rect of the antenna strip
    strip_source: str                           # "depth_mm" or "pad_edge_to_module_end"
    strip_area_mm2: float
    material_fraction: float
    levels_nm: List[float] = field(default_factory=list)   # outline levels under the strip width
    stepped: bool = False
    notch: Optional[bool] = None                # None unless a single inward level
    envelope_offset_nm: float = 0.0             # |module end - envelope|
    hole_count: int = 0                         # outline holes in the model (not represented)

    @property
    def void_fraction(self) -> float:
        return 1.0 - self.material_fraction

    @property
    def text(self) -> str:
        if self.classification == "off_envelope":
            return "%s by %.2f mm (tolerance %.2f mm)" % (
                CLASS_TEXT["off_envelope"], self.envelope_offset_nm / MM, ENVELOPE_TOL_NM / MM)
        if self.classification == "unknown":
            return "outline has %d hole(s), not represented; strip not classified" \
                % self.hole_count
        return CLASS_TEXT[self.classification]


def _clip_polygon(poly, rect):
    """Sutherland-Hodgman: poly (any simple polygon) clipped to an axis-
    aligned rect. For a concave poly the result may carry zero-width
    bridges along the rect border; its shoelace area is still exact."""
    x1, y1, x2, y2 = rect
    out = [(float(x), float(y)) for x, y in poly]
    for inside, cross in (
            (lambda p: p[0] >= x1, lambda a, b: (x1, a[1] + (b[1] - a[1]) * (x1 - a[0]) / (b[0] - a[0]))),
            (lambda p: p[0] <= x2, lambda a, b: (x2, a[1] + (b[1] - a[1]) * (x2 - a[0]) / (b[0] - a[0]))),
            (lambda p: p[1] >= y1, lambda a, b: (a[0] + (b[0] - a[0]) * (y1 - a[1]) / (b[1] - a[1]), y1)),
            (lambda p: p[1] <= y2, lambda a, b: (a[0] + (b[0] - a[0]) * (y2 - a[1]) / (b[1] - a[1]), y2))):
        if not out:
            break
        src, out = out, []
        prev = src[-1]
        for cur in src:
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
            prev = cur
    return out


def _area(poly) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        xa, ya = poly[i]
        xb, yb = poly[(i + 1) % n]
        s += xa * yb - xb * ya
    return abs(s) / 2.0


def material_fraction(strip, poly) -> float:
    """Fraction of the rect `strip` that lies inside the polygon `poly`."""
    a = float(strip[2] - strip[0]) * float(strip[3] - strip[1])
    if a <= 0:
        return 0.0
    return min(1.0, _area(_clip_polygon(poly, strip)) / a)


def _levels(poly, edge, lo, hi):
    """Outline levels of `edge` over the along-edge interval (lo, hi), plus the
    outermost level strictly before lo and strictly after hi (None if the
    edge has no boundary there)."""
    bps, lines = _edge_profile(poly, edge)
    outer_is_min = edge in ("top", "left")

    def v_at(s, u):
        return s[1] + (s[3] - s[1]) * (u - s[0]) / (s[2] - s[0])

    def outer(a, b):
        if a is None:
            return b
        return min(a, b) if outer_is_min else max(a, b)

    under, before, after = [], None, None
    for i in range(len(bps) - 1):
        a, b = bps[i], bps[i + 1]
        s = lines[i]
        if s is None:
            continue
        if b <= lo:
            before = outer(before, v_at(s, (a + b) / 2.0))
            continue
        if a >= hi:
            after = outer(after, v_at(s, (a + b) / 2.0))
            continue
        u1, u2 = max(a, lo), min(b, hi)
        if u2 > u1:
            under.append(v_at(s, (u1 + u2) / 2.0))
        if a < lo:
            before = outer(before, v_at(s, (a + lo) / 2.0))
        if b > hi:
            after = outer(after, v_at(s, (hi + b) / 2.0))
    distinct: List[float] = []
    for v in sorted(under):
        if not distinct or v - distinct[-1] > LEVEL_TOL_NM:
            distinct.append(v)
    return distinct, before, after


def antenna_window(fp, res, edge: str, poly, outline, holes: int = 0) -> AntennaWindow:
    """Measure the antenna strip of footprint fp (final position) against the
    outline. res: the AntennaResolution of its antenna item; edge: the locked
    edge; poly: the outline polygon, or None for a rectangular board;
    outline: (xmin, ymin, xmax, ymax) of the outline bbox; holes: the number
    of holes in the board outline (BoardModel.outline_hole_count), which the
    polygon does not carry."""
    axis, sign = _axis_sign(res.side)
    strip_local = res.strip
    source = "depth_mm"
    lo_out, hi_out = (strip_local[axis], strip_local[axis + 2])
    if hi_out - lo_out <= 0:
        # no depth_mm: from the outermost pad centre to the module end
        p0, p1 = strip_local[1 - axis], strip_local[3 - axis]
        strip_local = _from_outward(axis, sign, res.pad_edge, res.antenna_end, p0, p1)
        source = "pad_edge_to_module_end"
    sw = fp.local_rect_to_world(strip_local)
    area = float(sw[2] - sw[0]) * float(sw[3] - sw[1])

    xmin, ymin, xmax, ymax = outline
    outline_ref = {"top": ymin, "bottom": ymax, "left": xmin, "right": xmax}[edge]
    envelope = anchor_edge_reference(poly, edge, outline_ref)
    # module end on the antenna axis through the origin (as Footprint.edge_anchor_world)
    sgn = 1 if res.side[0] == "+" else -1
    lx, ly = (sgn * res.antenna_end, 0) if res.side[1] == "x" else (0, sgn * res.antenna_end)
    wx = fp.x + lx * fp._cos_a - ly * fp._sin_a
    wy = fp.y + lx * fp._sin_a + ly * fp._cos_a
    module_end = wy if edge in ("top", "bottom") else wx

    shape = poly if poly else [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    frac = material_fraction(sw, shape)

    lo, hi = (sw[0], sw[2]) if edge in ("top", "bottom") else (sw[1], sw[3])
    if poly:
        levels, before, after = _levels(poly, edge, lo, hi)
    else:
        levels, before, after = [float(outline_ref)], float(outline_ref), float(outline_ref)
    stepped = len(levels) > 1
    notch = None
    if len(levels) == 1:
        lev = levels[0]
        inward = (lambda v: v is not None and v < lev - LEVEL_TOL_NM) \
            if edge in ("top", "left") else \
            (lambda v: v is not None and v > lev + LEVEL_TOL_NM)
        notch = inward(before) and inward(after)

    offset = abs(float(module_end) - float(envelope))
    on_envelope = offset <= ENVELOPE_TOL_NM
    holes = int(holes or 0)
    if area <= 0:
        cls = "no_strip"
    elif holes > 0:
        cls = "unknown"
    elif FRACTION_TOL < frac < 1.0 - FRACTION_TOL:
        cls = "mixed"
    elif stepped:
        cls = "stepped"
    elif frac >= 1.0 - FRACTION_TOL:
        cls = "material" if on_envelope else "off_envelope"
    elif notch:
        cls = "window" if on_envelope else "off_envelope"
    else:
        cls = "overhang"
    return AntennaWindow(ref=fp.reference, edge=edge, classification=cls,
                         status=CLASS_STATUS[cls], module_end_nm=module_end,
                         envelope_nm=float(envelope), strip_world=sw,
                         strip_source=source, strip_area_mm2=area / (MM * MM),
                         material_fraction=frac, levels_nm=levels,
                         stepped=stepped, notch=notch, envelope_offset_nm=offset,
                         hole_count=holes)


def summary_line(w: AntennaWindow) -> str:
    """One line for the GUI results summary."""
    return ("%s (%s edge): %s, %.0f%% of the strip over board material%s; "
            "geometry recorded, RF clearance not assessed"
            % (w.ref, w.edge, w.text, 100.0 * w.material_fraction,
               ", %d edge levels under the module" % len(w.levels_nm) if w.stepped else ""))
