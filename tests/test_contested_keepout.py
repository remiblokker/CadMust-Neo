"""Contested keep-out fixture (CadMust-Autopilot external review 2026-11-06,
section 2.4): a part whose nets pull it INTO a footprint keep-out (a KiCad
rule area that forbids footprints, BoardModel.keepouts). Does the flat
keep-out cost keep it out after a short seeded SA run?

What "100 per mm" measures (plugin/cost_function.py, _compute_fp_keepout):
for every keep-out rect the part's courtyard bbox intersects,

    penalty = (ox + oy) * KEEPOUT_WEIGHT        KEEPOUT_WEIGHT = 100 per nm

where ox and oy are the widths of the intersection rectangle along x and y
(the overlap extents, half the perimeter of the intersection). It is
neither the intrusion depth (the distance the part must move to get out)
nor the distance to the keep-out edge nor the overlap area. Consequences:

* the cost jumps as soon as the bbox touches the keep-out: entering 0.1 mm
  across the full width of a 2 x 1 mm part already costs (2 + 0.1) mm x 100;
* the cost is BOUNDED: a part wholly inside a keep-out larger than itself
  pays (w + h) x 100 wherever it stands, so inside there is no gradient
  towards the exit, and a part whose HPWL gain from entering exceeds
  (w + h) x 100 is better off inside (TestPenaltyBound);
* flat per part since 2026-11-06: the net count does not scale it;
* penalty_scale applies (0.1 at T0 rising to 1.0); the best state is
  tracked at scale 1.0;
* KeepOut.layers is recorded but not used: a keep-out on F.Cu only also
  keeps out back-side parts (TestExemptedLayer documents that);
* locked parts pay nothing (an infeasible lock is reported by the
  CostState per-part keep-out cache, which the GUI shows as "Keep-out
  violations").

Intrusion is reported in mm as the depth: the shortest axis move that
takes the bbox out of the keep-out (0 when outside; geometric, it ignores
the board edge), plus ox + oy and the overlap area for reference.

Fixture: 60 x 40 mm board, keep-out 10 x 10 mm at (25..35, 15..25),
part P 2 x 1 mm, four locked 1 x 1 mm anchors N (30, 5), S (30, 35),
W (10, 20), E (50, 20); every anchor carries k two-pin nets to P, so the
HPWL optimum is the keep-out centre: HPWL = const + 2k|y - 20| (+ 2k|x - 30|
off the vertical axis). From the nearest legal position (y 14.5) to the
centre the gain is 11k mm against a full-inside penalty of 300 mm, so the
penalty holds up to k = 27 nets per anchor (108 nets on the part).

Run with: python3 tests/test_contested_keepout.py -v
       or python3 tests/test_contested_keepout.py --table  (the measured table)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugin.board_model import BoardModel, Footprint, Pad, Net, KeepOut  # noqa: E402
from plugin.annealer import run_sa, SAConfig  # noqa: E402
from plugin.cost_function import CostState, KEEPOUT_WEIGHT  # noqa: E402
from plugin.constraints import parse_constraints, apply_constraints  # noqa: E402

MM = 1_000_000
BW, BH = 60, 40
KO = (25, 15, 35, 25)
ANCHORS = {"N": (30, 5), "S": (30, 35), "W": (10, 20), "E": (50, 20)}
SEED = 1
CFG = dict(max_iterations=60, reheat_count=1, seed=SEED)


def ko(rect, layers="all"):
    return KeepOut(*(int(v * MM) for v in rect), layers=layers)


def build(k=1, start=(45, 32), keepouts=(KO,), anchors=ANCHORS, part_front=True,
          ko_layers="all"):
    """Model with P (index 0) and the locked anchors; k nets per anchor."""
    fps = [Footprint(reference="P", index=0, x=int(start[0] * MM), y=int(start[1] * MM),
                     angle_deg=0.0, width=2 * MM, height=1 * MM, locked=False, pads=[],
                     net_codes=set(), is_front=part_front)]
    for i, (name, (x, y)) in enumerate(anchors.items(), start=1):
        fps.append(Footprint(reference=name, index=i, x=int(x * MM), y=int(y * MM),
                             angle_deg=0.0, width=1 * MM, height=1 * MM, locked=True,
                             pads=[], net_codes=set()))
    nets = {}
    nc = 0
    for ai in range(1, len(fps)):
        for _ in range(k):
            nc += 1
            name = "N%d" % nc
            fps[0].pads.append(Pad(nc, name, 0, 0))
            fps[0].net_codes.add(nc)
            fps[ai].pads.append(Pad(nc, name, 0, 0))
            fps[ai].net_codes.add(nc)
            nets[nc] = Net(nc, name, pad_refs=[(0, len(fps[0].pads) - 1),
                                               (ai, len(fps[ai].pads) - 1)])
    return BoardModel(footprints=fps, nets=nets, outline_xmin=0, outline_ymin=0,
                      outline_xmax=BW * MM, outline_ymax=BH * MM,
                      moveable_indices=[0],
                      keepouts=[ko(r, ko_layers) for r in keepouts])


def intrusion(fp_bbox, rect_mm):
    """(depth mm, ox + oy mm, area mm2) of a bbox (nm) in a keep-out (mm)."""
    x1, y1, x2, y2 = (v / MM for v in fp_bbox)
    kx1, ky1, kx2, ky2 = rect_mm
    ox = max(0.0, min(x2, kx2) - max(x1, kx1))
    oy = max(0.0, min(y2, ky2) - max(y1, ky1))
    if ox <= 0 or oy <= 0:
        return 0.0, 0.0, 0.0
    depth = min(x2 - kx1, kx2 - x1, y2 - ky1, ky2 - y1)
    return round(depth, 3), round(ox + oy, 3), round(ox * oy, 3)


def run(m, **cfg):
    c = dict(CFG)
    c.update(cfg)
    run_sa(m, config=SAConfig(**c))
    return m.footprints[0]


def measure(m, rects):
    f = m.footprints[0]
    worst = max((intrusion(f.bbox, r) for r in rects), key=lambda t: t[0])
    cs = CostState(m, quiet=True)
    return {"x": round(f.x / MM, 2), "y": round(f.y / MM, 2), "depth_mm": worst[0],
            "ox_plus_oy_mm": worst[1], "area_mm2": worst[2],
            "keepout_cost": cs._fp_keepout.get(0, 0.0),
            "flagged": 0 in cs._fp_keepout, "hpwl_mm": round(cs.hpwl / MM, 2)}


class TestCostDefinition(unittest.TestCase):
    """The term itself, on fixed positions (no SA)."""

    def _cost_at(self, x, y):
        m = build(start=(x, y))
        return CostState(m, quiet=True)._fp_keepout.get(0, 0.0) / MM

    def test_outside_is_zero(self):
        self.assertEqual(self._cost_at(30, 14.4), 0.0)

    def test_half_perimeter_of_the_intersection(self):
        # bbox y 14.6..15.6 at x 29..31: ox 2, oy 0.6 -> (2 + 0.6) x 100
        self.assertAlmostEqual(self._cost_at(30, 15.1), 2.6 * KEEPOUT_WEIGHT, places=6)

    def test_bounded_inside(self):
        # wholly inside: (w + h) x 100 = 300 wherever the part stands
        for x, y in ((30, 20), (27, 17), (33, 23.5)):
            self.assertAlmostEqual(self._cost_at(x, y), 3 * KEEPOUT_WEIGHT, places=6)

    def test_not_scaled_by_net_count(self):
        a = CostState(build(k=1, start=(30, 20)), quiet=True)._fp_keepout[0]
        b = CostState(build(k=10, start=(30, 20)), quiet=True)._fp_keepout[0]
        self.assertEqual(a, b)


class TestContestedKeepout(unittest.TestCase):
    """Short seeded SA; P must end outside the keep-out. Broken case: with
    the keep-out removed the same run puts P inside the rect (the nets do
    pull it in)."""

    def assertOutside(self, m, rects):
        r = measure(m, rects)
        self.assertEqual(r["depth_mm"], 0.0, r)
        self.assertFalse(r["flagged"], r)
        return r

    def test_broken_case_without_keepout_part_goes_in(self):
        m = build(keepouts=())
        run(m)
        self.assertGreater(intrusion(m.footprints[0].bbox, KO)[0], 0.5)

    def test_few_nets(self):
        m = build(k=1)
        run(m)
        self.assertOutside(m, [KO])

    def test_many_nets(self):
        m = build(k=10)          # 40 nets on a 2 x 1 mm part, gain 110 < 300
        run(m)
        self.assertOutside(m, [KO])

    def test_start_inside_gets_out(self):
        m = build(k=1, start=(30, 20))
        run(m)
        self.assertOutside(m, [KO])

    def test_two_intersecting_keepouts(self):
        rects = [(25, 15, 35, 25), (28, 10, 40, 22)]
        m = build(k=3, keepouts=rects)
        run(m)
        self.assertOutside(m, rects)

    def test_keepout_touching_the_board_edge(self):
        # keep-out along the top edge, anchors pull P up to (30, 2)
        rect = (20, 0, 40, 8)
        anchors = {"N": (30, 0.5), "W": (10, 2), "E": (50, 2), "S": (30, 12)}
        m = build(k=2, keepouts=[rect], anchors=anchors, start=(45, 30))
        run(m)
        self.assertOutside(m, [rect])
        f = m.footprints[0]
        x1, y1, x2, y2 = f.bbox
        self.assertTrue(0 <= x1 and x2 <= BW * MM and 0 <= y1 and y2 <= BH * MM)


class TestExemptedLayer(unittest.TestCase):
    """KeepOut.layers is recorded but the cost ignores it (cost_function:
    "Every keep-out applies to both sides for now"). A keep-out on F.Cu
    only still keeps a back-side part out. This documents the current
    behaviour; it is not an exemption."""

    def test_front_only_keepout_also_keeps_a_back_part_out(self):
        m = build(k=1, part_front=False, ko_layers="F.Cu")
        self.assertGreater(CostState(build(start=(30, 20), part_front=False,
                                           ko_layers="F.Cu"), quiet=True)._fp_keepout[0], 0)
        run(m)
        r = measure(m, [KO])
        self.assertEqual(r["depth_mm"], 0.0, r)


class TestEdgeLockAgainstKeepout(unittest.TestCase):
    def _locked(self, rect, anchors, k=1):
        m = build(k=k, keepouts=[rect], anchors=anchors, start=(45, 30))
        cset = parse_constraints({"meta": {}, "component_groups": [], "net_priorities": {},
                                  "placement_constraints": {
                                      "edge_locked": [{"ref": "P", "edge": "top"}]}})
        apply_constraints(m, cset, log=None)
        run(m)
        return m

    def test_feasible_slides_along_the_edge_outside(self):
        rect = (20, 0, 40, 8)
        anchors = {"N": (30, 10), "W": (5, 2), "E": (55, 2), "S": (30, 30)}
        m = self._locked(rect, anchors)
        r = measure(m, [rect])
        self.assertEqual(r["depth_mm"], 0.0, r)
        self.assertLessEqual(m.footprints[0].bbox[1], 1 * MM + 1000)   # still on the edge

    def test_infeasible_is_reported_not_hidden(self):
        # the keep-out covers the whole top band: the edge lock forces P in
        rect = (0, 0, 60, 5)
        anchors = {"N": (30, 10), "W": (5, 8), "E": (55, 8), "S": (30, 30)}
        m = self._locked(rect, anchors)
        r = measure(m, [rect])
        self.assertLessEqual(m.footprints[0].bbox[1], 1 * MM + 1000)   # lock holds
        self.assertGreater(r["depth_mm"], 0.0)
        # bbox y 1..2 in a band 0..5: the shortest geometric exit is 2 mm up
        # (through the board edge; depth ignores the board), 4 mm down
        self.assertAlmostEqual(r["depth_mm"], 2.0, places=2)
        self.assertTrue(r["flagged"], r)        # CostState counts it: the GUI shows it


class TestPenaltyBound(unittest.TestCase):
    """Finding, not a desired behaviour: the term is bounded by (w + h) x 100
    = 300 mm for P, the HPWL gain grows with the net count (11 mm per net
    per anchor here). Above the threshold (k > 27) the part ends inside.
    Recorded so a change of the term shows up here; see the module doc."""

    def test_threshold_from_the_formula(self):
        full = CostState(build(k=1, start=(30, 20)), quiet=True)._fp_keepout[0] / MM
        self.assertAlmostEqual(full, 300.0, places=6)
        for k in (27, 28):
            h_in = CostState(build(k=k, start=(30, 20)), quiet=True)
            h_out = CostState(build(k=k, start=(30, 14.5)), quiet=True)
            gain = (h_out.hpwl - h_in.hpwl) / MM
            self.assertAlmostEqual(gain, 11.0 * k, places=3)
            self.assertEqual(gain > full, k == 28)

    def test_many_nets_enter(self):
        m = build(k=40)          # 160 nets, gain 440 > 300
        run(m)
        r = measure(m, [KO])
        self.assertGreater(r["depth_mm"], 0.0, r)
        self.assertTrue(r["flagged"], r)     # reported, not hidden


CASES = [
    ("no keep-out (broken case)", lambda: build(keepouts=()), [KO]),
    ("few nets k=1 (4 nets)", lambda: build(k=1), [KO]),
    ("many nets k=10 (40 nets)", lambda: build(k=10), [KO]),
    ("start inside, k=1", lambda: build(k=1, start=(30, 20)), [KO]),
    ("two intersecting, k=3", lambda: build(k=3, keepouts=[(25, 15, 35, 25), (28, 10, 40, 22)]),
     [(25, 15, 35, 25), (28, 10, 40, 22)]),
    ("touching the top edge, k=2", lambda: build(
        k=2, keepouts=[(20, 0, 40, 8)], start=(45, 30),
        anchors={"N": (30, 0.5), "W": (10, 2), "E": (50, 2), "S": (30, 12)}), [(20, 0, 40, 8)]),
    ("F.Cu-only keep-out, back part, k=1", lambda: build(k=1, part_front=False,
                                                         ko_layers="F.Cu"), [KO]),
    ("bound: k=40 (160 nets)", lambda: build(k=40), [KO]),
]


def table():
    print("| case | P x, y mm | depth mm | ox+oy mm | area mm2 | flagged | HPWL mm |")
    print("|---|---|---|---|---|---|---|")
    for name, mk, rects in CASES:
        m = mk()
        run(m)
        r = measure(m, rects)
        print("| %s | %.2f, %.2f | %.2f | %.2f | %.2f | %s | %.1f |"
              % (name, r["x"], r["y"], r["depth_mm"], r["ox_plus_oy_mm"], r["area_mm2"],
                 r["flagged"], r["hpwl_mm"]))
    for name, rect, anchors in (
            ("edge lock, feasible", (20, 0, 40, 8),
             {"N": (30, 10), "W": (5, 2), "E": (55, 2), "S": (30, 30)}),
            ("edge lock, infeasible", (0, 0, 60, 5),
             {"N": (30, 10), "W": (5, 8), "E": (55, 8), "S": (30, 30)})):
        t = TestEdgeLockAgainstKeepout()
        m = t._locked(rect, anchors)
        r = measure(m, [rect])
        print("| %s | %.2f, %.2f | %.2f | %.2f | %.2f | %s | %.1f |"
              % (name, r["x"], r["y"], r["depth_mm"], r["ox_plus_oy_mm"], r["area_mm2"],
                 r["flagged"], r["hpwl_mm"]))


if __name__ == '__main__':
    if "--table" in sys.argv:
        table()
    else:
        unittest.main()
