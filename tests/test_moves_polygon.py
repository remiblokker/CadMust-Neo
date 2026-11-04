"""Edge locks on a non-rectangular outline: polygon edge profiles
(cost_function.polygon_edge_limit, bbox_outside_polygon) and the polygon-aware
moves.resnap_edges. No KiCad dependency.

Run with: python3 -m unittest discover -s tests

Synthetic board, 60 x 40 mm at KiCad origin (50, 20) mm, with
  * a notch at top-left: off the board for x < 20 and y < 10 (board-relative),
  * a 45-degree chamfer at bottom-right from (60, 30) to (50, 40),
  * a 2 mm wide, 5 mm deep slot in the bottom edge at x 30..32.
"""
import copy
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugin.board_model import BoardModel, Footprint, Pad, Net
from plugin.cost_function import (CostState, polygon_edge_limit,
                                  bbox_outside_polygon, outline_is_polygonal,
                                  point_in_polygon)
from plugin.moves import (resnap_edges, do_translate, do_median,
                          EDGE_LOCK_MARGIN)
from plugin.annealer import run_sa, SAConfig, _scatter_positions, _perturb_positions

MM = 1_000_000
OX, OY = 50 * MM, 20 * MM
BW, BH = 60 * MM, 40 * MM


def P(x, y):
    return (OX + int(x * MM), OY + int(y * MM))


L_POLY = [P(0, 10), P(20, 10), P(20, 0), P(60, 0), P(60, 30), P(50, 40),
          P(32, 40), P(32, 35), P(30, 35), P(30, 40), P(0, 40)]
RECT_POLY = [P(0, 0), P(60, 0), P(60, 40), P(0, 40)]


def fpx(ref, i, x, y, w=6.0, h=4.0):
    """Footprint centred at board-relative (x, y) mm."""
    return Footprint(reference=ref, index=i, x=OX + int(x * MM), y=OY + int(y * MM),
                     angle_deg=0.0, width=int(w * MM), height=int(h * MM), locked=False,
                     pads=[], net_codes=set())


def make_model(poly, n=8, seed=3):
    """n footprints on the board, nets chaining them; R0 is edge-locked top,
    R1 left, R2 bottom, R3 right."""
    rnd = random.Random(seed)
    fps = [fpx("R%d" % i, i, rnd.randint(25, 45), rnd.randint(15, 30)) for i in range(n)]
    nets = {}
    for i in range(1, n):
        net = Net(net_code=i, net_name="N%d" % i)
        for fi in (i - 1, i):
            f = fps[fi]
            f.pads.append(Pad(net_code=i, net_name=net.net_name,
                              offset_x=MM if fi == i - 1 else -MM, offset_y=0))
            f.net_codes.add(i)
            net.pad_refs.append((fi, len(f.pads) - 1))
        nets[i] = net
    # pull R0 towards the notch: a net to a locked part sitting at the notch corner
    anchor = fpx("A", n, 10, 15, 2, 2)
    anchor.locked = True
    fps.append(anchor)
    net = Net(net_code=n, net_name="PULL")
    for fi in (0, n):
        fps[fi].pads.append(Pad(net_code=n, net_name="PULL", offset_x=0, offset_y=0))
        fps[fi].net_codes.add(n)
        net.pad_refs.append((fi, len(fps[fi].pads) - 1))
    nets[n] = net
    m = BoardModel(footprints=fps, nets=nets,
                   outline_xmin=OX, outline_ymin=OY,
                   outline_xmax=OX + BW, outline_ymax=OY + BH,
                   moveable_indices=list(range(n)),
                   outline_polygon=list(poly) if poly is not None else None)
    for i, e in enumerate(("top", "left", "bottom", "right")):
        fps[i].edge_lock = e
    m.edge_locked_indices = {0, 1, 2, 3}
    for i in range(4):
        resnap_edges(m, [i])
    return m


def rel(v, o):
    return (v - o) / MM


class TestEdgeProfile(unittest.TestCase):
    def lim(self, edge, lo, hi):
        o = OX if edge in ("top", "bottom") else OY
        v = polygon_edge_limit(L_POLY, edge, o + int(lo * MM), o + int(hi * MM))
        if v is None:
            return None
        return rel(v, OY if edge in ("top", "bottom") else OX)

    def test_top_over_notch_and_beside_it(self):
        self.assertAlmostEqual(self.lim("top", 5, 15), 10)
        self.assertAlmostEqual(self.lim("top", 25, 35), 0)
        # straddling the notch step: the innermost point decides
        self.assertAlmostEqual(self.lim("top", 15, 25), 10)
        # span ending exactly at the step: left-hand limit is the notch floor
        self.assertAlmostEqual(self.lim("top", 10, 20), 10)
        self.assertAlmostEqual(self.lim("top", 20, 30), 0)

    def test_left_over_notch(self):
        self.assertAlmostEqual(self.lim("left", 2, 8), 20)
        self.assertAlmostEqual(self.lim("left", 15, 25), 0)
        self.assertAlmostEqual(self.lim("left", 5, 15), 20)

    def test_diagonal_chamfer(self):
        self.assertAlmostEqual(self.lim("bottom", 40, 45), 40)
        # bottom boundary on the chamfer: y = 40 - (x - 50); innermost at x = 58
        self.assertAlmostEqual(self.lim("bottom", 52, 58), 32)
        self.assertAlmostEqual(self.lim("bottom", 45, 55), 35)
        # right boundary: x = 50 + (40 - y); innermost at y = 38
        self.assertAlmostEqual(self.lim("right", 32, 38), 52)
        self.assertAlmostEqual(self.lim("right", 10, 20), 60)

    def test_bottom_slot(self):
        self.assertAlmostEqual(self.lim("bottom", 28, 34), 35)
        self.assertAlmostEqual(self.lim("bottom", 33, 40), 40)

    def test_span_outside_polygon(self):
        self.assertIsNone(self.lim("top", -20, -10))
        self.assertAlmostEqual(self.lim("top", -5, 5), 10)  # clipped to the polygon

    def test_rectangle_detection(self):
        m = make_model(L_POLY)
        self.assertTrue(outline_is_polygonal(m))
        self.assertFalse(outline_is_polygonal(make_model(RECT_POLY)))
        self.assertFalse(outline_is_polygonal(make_model(None)))


class TestBboxOutsidePolygon(unittest.TestCase):
    def bb(self, x1, y1, x2, y2):
        return (OX + int(x1 * MM), OY + int(y1 * MM), OX + int(x2 * MM), OY + int(y2 * MM))

    def test_inside(self):
        self.assertEqual(bbox_outside_polygon(self.bb(25, 5, 35, 15), L_POLY), 0.0)

    def test_corner_in_notch(self):
        # top-left corner at (5, 2) is in the notch, 8 mm below its floor... distance to
        # the notch boundary is min(10-2, 20-5) = 8 mm
        e = bbox_outside_polygon(self.bb(5, 2, 25, 18), L_POLY)
        self.assertAlmostEqual(e / MM, 8.0, places=3)

    def test_vertex_strictly_inside_with_all_corners_on_board(self):
        # covers the bottom slot: all four corners on the board, slot vertices inside
        e = bbox_outside_polygon(self.bb(28, 30, 34, 38), L_POLY)
        self.assertGreater(e, 0)

    def test_corner_on_outline_counts_as_inside(self):
        self.assertEqual(bbox_outside_polygon(self.bb(20, 0, 30, 10), L_POLY), 0.0)


class TestResnapPolygon(unittest.TestCase):
    def test_top_part_drops_below_notch(self):
        m = make_model(L_POLY)
        f = m.footprints[0]
        for x, want in ((10, 10), (40, 0), (19, 10), (23, 0)):
            f.x = OX + int(x * MM)  # bbox 6 mm wide: at x=19 it straddles the step
            f.y = OY + 25 * MM
            resnap_edges(m, [0])
            self.assertEqual(f.bbox[1], OY + int(want * MM) + EDGE_LOCK_MARGIN, msg=x)
            self.assertEqual(bbox_outside_polygon(f.bbox, L_POLY), 0.0)

    def test_left_part_moves_right_of_notch(self):
        m = make_model(L_POLY)
        f = m.footprints[1]
        f.x, f.y = OX + 30 * MM, OY + 5 * MM
        resnap_edges(m, [1])
        self.assertEqual(f.bbox[0], OX + 20 * MM + EDGE_LOCK_MARGIN)
        f.y = OY + 25 * MM
        resnap_edges(m, [1])
        self.assertEqual(f.bbox[0], OX + EDGE_LOCK_MARGIN)

    def test_bottom_and_right_on_chamfer(self):
        m = make_model(L_POLY)
        b, r = m.footprints[2], m.footprints[3]
        b.x, b.y = OX + 55 * MM, OY + 10 * MM  # bbox x 52..58 -> boundary 32
        resnap_edges(m, [2])
        self.assertEqual(b.bbox[3], OY + 32 * MM - EDGE_LOCK_MARGIN)
        r.x, r.y = OX + 10 * MM, OY + 36 * MM  # bbox y 34..38 -> boundary 52
        resnap_edges(m, [3])
        self.assertEqual(r.bbox[2], OX + 52 * MM - EDGE_LOCK_MARGIN)
        self.assertEqual(bbox_outside_polygon(b.bbox, L_POLY), 0.0)
        self.assertEqual(bbox_outside_polygon(r.bbox, L_POLY), 0.0)

    def test_rigid_group_moves_as_whole(self):
        m = make_model(L_POLY)
        m.footprints[4].x = m.footprints[0].x + 3 * MM
        m.footprints[4].y = m.footprints[0].y + 5 * MM
        m.footprints[0].x = OX + 10 * MM
        m.footprints[4].x = OX + 13 * MM
        off = (m.footprints[4].x - m.footprints[0].x, m.footprints[4].y - m.footprints[0].y)
        resnap_edges(m, [0, 4])
        self.assertEqual(m.footprints[0].bbox[1], OY + 10 * MM + EDGE_LOCK_MARGIN)
        self.assertEqual((m.footprints[4].x - m.footprints[0].x,
                          m.footprints[4].y - m.footprints[0].y), off)

    def test_translate_moves_stay_on_polygon(self):
        m = make_model(L_POLY)
        random.seed(11)
        xs = set()
        for _ in range(300):
            do_translate(m, 1.0, 20 * MM)
            do_median(m, 0.5, 2 * MM)
            f = m.footprints[0]
            lim = polygon_edge_limit(L_POLY, "top", f.bbox[0], f.bbox[2])
            if lim is not None and f.bbox[0] >= OX and f.bbox[2] <= OX + BW:
                self.assertEqual(f.bbox[1], round(lim) + EDGE_LOCK_MARGIN)
            xs.add(f.x)
        self.assertGreater(len(xs), 20)

    def test_scatter_and_perturb_snap_to_polygon(self):
        m = make_model(L_POLY)
        random.seed(2)
        for fn in (_scatter_positions, _perturb_positions):
            for _ in range(20):
                fn(m)
                f = m.footprints[0]
                lim = polygon_edge_limit(L_POLY, "top", f.bbox[0], f.bbox[2])
                self.assertEqual(f.bbox[1], round(lim) + EDGE_LOCK_MARGIN)

    def test_sa_leaves_top_locked_part_on_the_board(self):
        """The PULL net drags R0 towards the notch; it must end on the board."""
        m = make_model(L_POLY)
        cfg = SAConfig(max_iterations=40, reheat_count=1, scatter_first_start=True, seed=5)
        run_sa(m, config=cfg)
        for i in range(4):
            f = m.footprints[i]
            self.assertEqual(bbox_outside_polygon(f.bbox, L_POLY), 0.0, msg=f.reference)
        f = m.footprints[0]
        lim = polygon_edge_limit(L_POLY, "top", f.bbox[0], f.bbox[2])
        self.assertEqual(f.bbox[1], round(lim) + EDGE_LOCK_MARGIN)


class TestRectangularUnchanged(unittest.TestCase):
    """A rectangular polygon (or none) snaps to the bbox edge, bit for bit."""

    def test_rect_polygon_equals_no_polygon(self):
        a, b = make_model(RECT_POLY), make_model(None)
        for m in (a, b):
            random.seed(17)
            for _ in range(200):
                do_translate(m, 1.0, 15 * MM)
                do_median(m, 0.5, 2 * MM)
        self.assertEqual([(f.x, f.y) for f in a.footprints],
                         [(f.x, f.y) for f in b.footprints])
        f = a.footprints[0]
        self.assertEqual(f.bbox[1], OY + EDGE_LOCK_MARGIN)

    def test_sa_identical_with_rect_polygon(self):
        a, b = make_model(RECT_POLY), make_model(None)
        cfg = SAConfig(max_iterations=20, reheat_count=1, scatter_first_start=True, seed=9)
        ra = run_sa(a, config=copy.deepcopy(cfg))
        rb = run_sa(b, config=copy.deepcopy(cfg))
        self.assertEqual([(f.x, f.y, f.angle_deg) for f in a.footprints],
                         [(f.x, f.y, f.angle_deg) for f in b.footprints])
        self.assertEqual(ra.final_cost, rb.final_cost)


if __name__ == "__main__":
    unittest.main()
