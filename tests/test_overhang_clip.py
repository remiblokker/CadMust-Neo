"""overhang_ok in the engine (CadMust-Autopilot design note 2026-11-06): the
boundary penalty and the off-board tests (reject excess, clamp, polygon
cutout, greedy kickout) clip the bbox of an edge-locked part that may
overhang its locked edge at that edge; the other three sides count as
before. No KiCad dependency.

Run with: python3 tests/test_overhang_clip.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugin.board_model import BoardModel, Footprint, Pad, Net
from plugin.cost_function import CostState, locked_edge_clip
from plugin.moves import (_group_bbox_excess, _offboard_rejects, _clamp_group_to_board,
                          _kickout_polygon_violations, _hits_polygon_cutout)

MM = 1_000_000
BW, BH = 60 * MM, 40 * MM
# L-shaped outline: a 20 x 10 mm notch at the top-left
L_POLY = [(0, 10 * MM), (20 * MM, 10 * MM), (20 * MM, 0), (BW, 0), (BW, BH), (0, BH)]


def part(ref, i, x, y, w=10.0, h=6.0, edge=None, overhang=False):
    f = Footprint(reference=ref, index=i, x=int(x * MM), y=int(y * MM), angle_deg=0.0,
                  width=int(w * MM), height=int(h * MM), locked=False,
                  pads=[Pad(1, "N1", 0, 0)], net_codes={1})
    f.edge_lock = edge
    f.overhang_ok = overhang
    return f


def model(fps, poly=None):
    for i, f in enumerate(fps):
        f.index = i
    nets = {1: Net(1, "N1", pad_refs=[(i, 0) for i in range(len(fps))])}
    return BoardModel(footprints=fps, nets=nets, outline_xmin=0, outline_ymin=0,
                      outline_xmax=BW, outline_ymax=BH,
                      moveable_indices=list(range(len(fps))),
                      outline_polygon=list(poly) if poly else None,
                      edge_locked_indices={i for i, f in enumerate(fps) if f.edge_lock})


class TestClipHelper(unittest.TestCase):
    def test_only_the_locked_edge_is_clipped(self):
        f = part("J1", 0, 30, 1, edge="top", overhang=True)       # bbox y -2 .. 4
        self.assertEqual(locked_edge_clip(f, f.bbox, None, 0, 0, BW, BH),
                         (25 * MM, 0, 35 * MM, 4 * MM))
        g = part("J2", 0, 30, 1, edge="top", overhang=False)
        self.assertEqual(locked_edge_clip(g, g.bbox, None, 0, 0, BW, BH), g.bbox)
        h = part("J3", 0, 30, 1, edge="left", overhang=True)      # locked elsewhere
        self.assertEqual(locked_edge_clip(h, h.bbox, None, 0, 0, BW, BH), h.bbox)

    def test_polygon_reference(self):
        # top-locked over the notch: the reference is the notch floor (y 10)
        f = part("J1", 0, 18, 9, edge="top", overhang=True)       # x 13..23, y 6..12
        self.assertEqual(locked_edge_clip(f, f.bbox, L_POLY, 0, 0, BW, BH)[1], 10 * MM)


class TestBoundaryPenalty(unittest.TestCase):
    def test_overhang_on_locked_edge_is_free(self):
        m = model([part("J1", 0, 30, 1, edge="top", overhang=True), part("R1", 1, 30, 30)])
        self.assertEqual(CostState(m, quiet=True)._boundary_penalty, 0.0)

    def test_without_the_flag_it_counts(self):
        # broken case: the same overhang without overhang_ok
        m = model([part("J1", 0, 30, 1, edge="top"), part("R1", 1, 30, 30)])
        self.assertGreater(CostState(m, quiet=True)._boundary_penalty, 0.0)

    def test_other_sides_still_count(self):
        # past the top (exempt) and past the left side (counted)
        m = model([part("J1", 0, 3, 1, edge="top", overhang=True), part("R1", 1, 30, 30)])
        cs = CostState(m, quiet=True)
        self.assertGreater(cs._boundary_penalty, 0.0)
        m2 = model([part("J1", 0, 6, 1, edge="top", overhang=True), part("R1", 1, 30, 30)])
        self.assertEqual(CostState(m2, quiet=True)._boundary_penalty, 0.0)

    def test_incremental_matches_full(self):
        m = model([part("J1", 0, 30, 1, edge="top", overhang=True), part("R1", 1, 30, 30)],
                  L_POLY)
        cs = CostState(m, quiet=True)
        for x in (25, 15, 5, 40):
            m.footprints[0].x = x * MM
            inc = cs.incremental_update({0})
            full = CostState(m, quiet=True).total_cost
            self.assertAlmostEqual(inc, full, delta=1e-6 * max(1.0, abs(full)))

    def test_polygon_notch_on_locked_edge_is_exempt(self):
        # top-locked, overhanging the top at x 22..32 (no notch there): free;
        # slid into the notch region its left corners leave the polygon
        m = model([part("J1", 0, 27, 1, edge="top", overhang=True)], L_POLY)
        self.assertEqual(CostState(m, quiet=True)._boundary_penalty, 0.0)
        m.footprints[0].x = 18 * MM     # x 13..23 over the notch; clip at y 10
        self.assertEqual(CostState(m, quiet=True)._boundary_penalty, 0.0)
        m.footprints[0].y = 12 * MM     # bbox y 9..15: clipped to 10..15, inside
        self.assertEqual(CostState(m, quiet=True)._boundary_penalty, 0.0)


class TestOffboard(unittest.TestCase):
    def test_excess_ignores_the_locked_edge_only(self):
        f = part("J1", 0, 30, 1, edge="top", overhang=True)
        m = model([f])
        self.assertEqual(_group_bbox_excess(m, [0]), 0)
        f.x = 3 * MM                     # 2 mm past the left side
        self.assertEqual(_group_bbox_excess(m, [0]), 2 * MM)
        f.overhang_ok = False            # broken case: no flag, top counts too
        f.x = 30 * MM
        self.assertEqual(_group_bbox_excess(m, [0]), 2 * MM)

    def test_reject_mode_keeps_overhang(self):
        # an anchor-less overhang part stays where resnap puts it (1 mm inside),
        # so give it an anchor by hand: snap puts its anchor on the edge
        f = part("J1", 0, 30, 1, edge="top", overhang=True)
        f.antenna_side = "-y"
        f.edge_anchor_offset_nm = 1 * MM      # anchor 1 mm above the origin
        m = model([f])
        self.assertFalse(_offboard_rejects(m, [0], 0, "reject"))
        self.assertEqual(f.y, 1 * MM)        # anchor at y 0, bbox top at -2 mm
        self.assertLess(f.bbox[1], 0)

    def test_clamp_leaves_the_locked_edge_overhang(self):
        f = part("J1", 0, 30, 1, edge="top", overhang=True)
        m = model([f])
        _clamp_group_to_board(m, [0])
        self.assertEqual(f.y, 1 * MM)
        g = part("J2", 0, 30, 1, edge="top")     # broken case: no flag, clamped in
        m = model([g])
        _clamp_group_to_board(m, [0])
        self.assertEqual(g.bbox[1], 0)

    def test_kickout_leaves_the_overhang_alone(self):
        f = part("J1", 0, 40, 1, edge="top", overhang=True)
        m = model([f, part("R1", 1, 40, 30)], L_POLY)
        cs = CostState(m, quiet=True)
        self.assertEqual(_kickout_polygon_violations(m, [0], cs), 0)
        self.assertEqual((f.x, f.y), (40 * MM, 1 * MM))
        g = part("J2", 0, 40, 1, edge="top")     # broken case: kicked inside
        m = model([g, part("R1", 1, 40, 30)], L_POLY)
        cs = CostState(m, quiet=True)
        self.assertEqual(_kickout_polygon_violations(m, [0], cs), 1)

    def test_cutout_test_uses_the_clipped_box(self):
        f = part("J1", 0, 40, 4, edge="top", overhang=True)    # inside, y 1..7
        m = model([f], L_POLY)
        old = [(0, f.x, f.y, f.angle_deg)]
        f.y = 1 * MM                                            # now past the top
        self.assertFalse(_hits_polygon_cutout(m, [0], old))
        f.overhang_ok = False                                   # broken case
        self.assertTrue(_hits_polygon_cutout(m, [0], old))


if __name__ == "__main__":
    unittest.main()
