"""Unit tests for the antenna keep-out: resolution (plugin/constraints.py),
attached keep-out rects (Footprint.keepout_rects) in the cost function, and
the schema reading in apply_constraints. No KiCad dependency.

Run with: python3 tests/test_antenna_keepout.py -v
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugin.board_model import BoardModel, Footprint, Pad, Net, ComponentGroup
from plugin.cost_function import CostState, OVERLAP_WEIGHT
from plugin.constraints import (AntennaError, antenna_keepout_rect, apply_constraints,
                                drawing_crosscheck, pad_free_sides, parse_constraints,
                                resolve_antenna, ConstraintError)
from plugin.moves import do_translate, do_median, do_swap, revert_move, affected_indices

MM = 1_000_000


def module_fp(ref="U3", x=0, y=0, angle=0.0, mirrored=False, is_front=True):
    """ESP32-WROOM-like module: pad columns at x = +-8.6 (y -5.31 .. 11.2),
    a bottom row at y = 12.3, centre GND pads; bbox (-9.8, -6.51, 9.8, 13.5).
    The antenna (no pads) is at -y. mirrored=True mirrors y (as a back-side
    extraction can), so the antenna is at +y."""
    pads = []
    for k in range(14):
        yy = -5.31 + 1.27 * k
        pads.append((-8.6, yy))
        pads.append((8.6, yy))
    for k in range(10):
        pads.append((-5.7 + 1.27 * k, 12.3))
    pads += [(-1.0, 2.2), (0.8, 0.4)]
    cy = (13.5 + -6.51) / 2
    if mirrored:
        pads = [(px, -py) for px, py in pads]
        cy = -cy
    fp = Footprint(reference=ref, index=0, x=x, y=y, angle_deg=angle,
                   width=int(19.6 * MM), height=int(20.01 * MM), locked=False,
                   pads=[Pad(net_code=0, net_name="", offset_x=int(round(px * MM)),
                             offset_y=int(round(py * MM))) for px, py in pads],
                   cx_offset=0, cy_offset=int(round(cy * MM)), is_front=is_front)
    return fp


def mm(r):
    return tuple(round(v / MM, 3) for v in r)


class TestSideResolution(unittest.TestCase):
    def test_module_pad_free_is_minus_y(self):
        self.assertEqual(pad_free_sides(module_fp()), ["-y"])

    def test_mirrored_module_pad_free_is_plus_y(self):
        self.assertEqual(pad_free_sides(module_fp(mirrored=True)), ["+y"])

    def test_two_row_part_is_ambiguous(self):
        f = Footprint(reference="U9", index=0, x=0, y=0, angle_deg=0.0,
                      width=6 * MM, height=5 * MM, locked=False,
                      pads=[Pad(0, "", sx * 2700000, (k - 1) * 1270000)
                            for sx in (-1, 1) for k in range(4)])
        self.assertEqual(sorted(pad_free_sides(f)), ["+y", "-y"])
        with self.assertRaises(AntennaError):
            antenna_keepout_rect(f, "pad_free", 5 * MM)

    def test_small_parts(self):
        def part(pads):
            return Footprint(reference="Q", index=0, x=0, y=0, angle_deg=0.0,
                             width=3 * MM, height=3 * MM, locked=False,
                             pads=[Pad(0, "", int(px * MM), int(py * MM)) for px, py in pads])
        # resistor: pads on +-x
        self.assertEqual(pad_free_sides(part([(-0.5, 0), (0.5, 0)])), ["+y", "-y"])
        # SOT-23: two pads on +y, one on -y; the outer pins are corners that
        # belong to the two-pin row
        self.assertEqual(pad_free_sides(part([(-0.95, 1), (0.95, 1), (0, -1)])),
                         ["+x", "-x"])
        # square corner pads with equal rows go to both sides: none free
        self.assertEqual(pad_free_sides(part([(-1, -1), (1, -1), (1, 1), (-1, 1)])), [])

    def test_courtyard_size_does_not_change_the_side(self):
        f = module_fp()
        f.height = int(41.5 * MM)
        f.width = int(48.5 * MM)
        f.cy_offset = int(-7.25 * MM)
        self.assertEqual(pad_free_sides(f), ["-y"])

    def test_no_pads_is_an_error(self):
        f = Footprint(reference="X", index=0, x=0, y=0, angle_deg=0.0,
                      width=MM, height=MM, locked=False)
        with self.assertRaises(AntennaError):
            antenna_keepout_rect(f, "pad_free", MM)

    def test_explicit_side_is_used_as_given(self):
        r = antenna_keepout_rect(module_fp(), "+x", 0, 2 * MM)
        # pad edge 8.6 + 2 = 10.6 beyond; bbox edge 9.8
        self.assertEqual(mm(r), (9.8, -5.31, 10.6, 12.3))


class TestRectResolution(unittest.TestCase):
    def test_olimex_u3_local_rect(self):
        res = resolve_antenna(module_fp(), "pad_free", 15 * MM, int(7.49 * MM), 6 * MM)
        self.assertEqual(res.side, "-y")
        self.assertEqual(mm(res.rect), (-23.6, -27.8, 23.6, -6.51))
        self.assertEqual(mm(res.strip), (-8.6, -12.8, 8.6, -6.8))
        self.assertEqual(res.module_end_source, "datasheet")
        self.assertFalse(res.zero_area)

    def test_fallback_to_courtyard_edge(self):
        res = resolve_antenna(module_fp(), "pad_free", 15 * MM)
        self.assertEqual(res.module_end_source, "courtyard")
        self.assertEqual(mm(res.rect), (-23.6, -21.51, 23.6, -6.51))
        self.assertTrue(res.notes)

    def test_courtyard_already_encodes_it(self):
        f = module_fp()
        # courtyard drawn 15 mm beyond the module end (as on the Opsteker U1)
        f.height = int(41.5 * MM)
        f.cy_offset = int(((13.5) + (-28.0)) / 2 * MM)
        res = resolve_antenna(f, "pad_free", 15 * MM, int(7.49 * MM), 6 * MM)
        self.assertTrue(res.zero_area)

    def test_rotations(self):
        local = antenna_keepout_rect(module_fp(), "pad_free", 15 * MM, int(7.49 * MM))
        depth = 27.8 - 6.51
        for angle, side in ((0.0, "ymin"), (90.0, "xmin"), (180.0, "ymax"),
                            (270.0, "xmax")):
            f = module_fp(x=100 * MM, y=50 * MM, angle=angle)
            r = antenna_keepout_rect(f, "pad_free", 15 * MM, int(7.49 * MM))
            self.assertEqual(r, local)          # local frame does not rotate
            f.keepout_rects = [r]
            w = f.keepout_bboxes[0]
            b = f.bbox
            if side == "ymin":
                self.assertLessEqual(abs(w[3] - b[1]), 2)
                self.assertAlmostEqual((w[3] - w[1]) / MM, depth, places=2)
            elif side == "ymax":
                self.assertLessEqual(abs(w[1] - b[3]), 2)
                self.assertAlmostEqual((w[3] - w[1]) / MM, depth, places=2)
            elif side == "xmin":
                self.assertLessEqual(abs(w[2] - b[0]), 2)
                self.assertAlmostEqual((w[2] - w[0]) / MM, depth, places=2)
            else:
                self.assertLessEqual(abs(w[0] - b[2]), 2)
                self.assertAlmostEqual((w[2] - w[0]) / MM, depth, places=2)

    def test_mirrored(self):
        f = module_fp(mirrored=True, is_front=False)
        r = antenna_keepout_rect(f, "pad_free", 15 * MM, int(7.49 * MM))
        self.assertEqual(mm(r), (-23.6, 6.51, 23.6, 27.8))
        f.angle_deg = 90.0
        f._update_trig()
        f.keepout_rects = [r]
        w = f.keepout_bboxes[0]
        self.assertLessEqual(abs(w[0] - f.bbox[2]), 2)   # local +y -> world +x at 90

    def test_drawing_crosscheck(self):
        f = module_fp()
        res = resolve_antenna(f, "pad_free", 15 * MM, int(7.49 * MM), 6 * MM)
        self.assertIsNone(drawing_crosscheck(f, res))
        # silk at the module end, Fab drawn as large as a 15 mm courtyard
        f.drawing_rects = {"F.SilkS": (-9 * MM, int(-12.75 * MM), 9 * MM, int(12.75 * MM)),
                           "F.Fab": (-24 * MM, -28 * MM, 24 * MM, 14 * MM)}
        f.height = int(42 * MM)
        f.cy_offset = int(-7 * MM)
        res = resolve_antenna(f, "pad_free", 15 * MM, int(7.49 * MM), 6 * MM)
        xc = drawing_crosscheck(f, res)
        self.assertEqual(xc["layer"], "F.SilkS")
        self.assertTrue(xc["ok"])
        f.drawing_rects = {"F.SilkS": (-9 * MM, -9 * MM, 9 * MM, int(12.75 * MM))}
        self.assertFalse(drawing_crosscheck(f, res)["ok"])
        # Olimex case: the drawing extends beyond the courtyard and counts
        g = module_fp()
        g.drawing_rects = {"Dwgs.User": (-9 * MM, int(-12.75 * MM), 9 * MM, int(12.75 * MM))}
        res = resolve_antenna(g, "pad_free", 15 * MM, int(7.49 * MM), 6 * MM)
        xc = drawing_crosscheck(g, res)
        self.assertEqual(xc["drawn_end_beyond_pads_mm"], 7.44)
        self.assertTrue(xc["ok"])


def board_with_module(locked_u3=False):
    """75 x 75 mm board at (0, 0), U3 at (20, 30) mm with its antenna area
    (local -y) attached, plus R1 and R2 (2 x 1 mm)."""
    u3 = module_fp(x=20 * MM, y=30 * MM)
    u3.locked = locked_u3
    u3.keepout_rects = [antenna_keepout_rect(u3, "pad_free", 15 * MM, int(7.49 * MM))]
    fps = [u3]
    for k, (x, y) in enumerate(((60, 60), (65, 65))):
        fps.append(Footprint(reference="R%d" % (k + 1), index=k + 1, x=x * MM, y=y * MM,
                             angle_deg=0.0, width=2 * MM, height=MM, locked=False,
                             pads=[Pad(1, "N1", -MM // 2, 0), Pad(1, "N1", MM // 2, 0)],
                             net_codes={1}))
    for i, f in enumerate(fps):
        f.index = i
    nets = {1: Net(1, "N1", pad_refs=[(1, 0), (2, 0)])}
    return BoardModel(footprints=fps, nets=nets, outline_xmin=0, outline_ymin=0,
                      outline_xmax=75 * MM, outline_ymax=75 * MM,
                      moveable_indices=[i for i, f in enumerate(fps) if not f.locked])


class TestCost(unittest.TestCase):
    def test_inside_rect_counts_outside_does_not(self):
        m = board_with_module()
        w = m.footprints[0].keepout_bboxes[0]
        r1 = m.footprints[1]
        # inside the rect, far from the U3 body: 0.5 mm below the rect top edge
        r1.x, r1.y = 20 * MM, w[1] + MM
        cs = CostState(m, quiet=True)
        self.assertAlmostEqual(cs._overlap_penalty, (2 * MM + MM) * OVERLAP_WEIGHT)
        # just outside (above) the rect: zero
        r1.y = w[1] - MM // 2 - 1
        self.assertEqual(CostState(m, quiet=True)._overlap_penalty, 0.0)
        # beside the module body (not beyond its antenna end): zero
        r1.x, r1.y = 20 * MM + int(12 * MM), 30 * MM
        self.assertEqual(CostState(m, quiet=True)._overlap_penalty, 0.0)

    def test_boundary_unaffected(self):
        m = board_with_module()
        u3 = m.footprints[0]
        u3.y = 18 * MM   # antenna area now hangs over the top edge (bbox stays on)
        with_rect = CostState(m, quiet=True)._boundary_penalty
        self.assertLess(u3.keepout_bboxes[0][1], 0)
        u3.keepout_rects = ()
        self.assertEqual(CostState(m, quiet=True)._boundary_penalty, with_rect)
        self.assertEqual(with_rect, 0.0)

    def test_hpwl_unaffected_and_zero_overhead_without_rects(self):
        m = board_with_module()
        h = CostState(m, quiet=True).hpwl
        m.footprints[0].keepout_rects = ()
        cs = CostState(m, quiet=True)
        self.assertEqual(cs.hpwl, h)
        self.assertNotIn("_compute_pair_overlap", cs.__dict__)

    def test_same_group_exempt(self):
        m = board_with_module()
        w = m.footprints[0].keepout_bboxes[0]
        m.footprints[1].x, m.footprints[1].y = 20 * MM, w[1] + MM
        m.component_groups = [ComponentGroup(member_indices=[0, 1])]
        m.fp_to_group = {0: 0, 1: 0}
        self.assertEqual(CostState(m, quiet=True)._overlap_penalty, 0.0)

    def test_side_rules(self):
        m = board_with_module()
        w = m.footprints[0].keepout_bboxes[0]
        r1 = m.footprints[1]
        r1.x, r1.y = 20 * MM, w[1] + MM
        r1.is_front = False
        # both sides by default: a back-side body in the antenna area counts
        self.assertGreater(CostState(m, quiet=True)._overlap_penalty, 0.0)
        m.footprints[0].clearance_side = True     # front only
        self.assertEqual(CostState(m, quiet=True)._overlap_penalty, 0.0)

    def test_old_pickle_without_field(self):
        m = board_with_module()
        f = m.footprints[1]
        f.__dict__.pop("keepout_rects", None)
        self.assertEqual(f.keepout_bboxes, [])
        CostState(m, quiet=True)

    def test_neighbour_scan_finds_rect_only_contact(self):
        m = board_with_module()
        cs = CostState(m, quiet=True)
        self.assertEqual(cs._overlap_penalty, 0.0)
        w = m.footprints[0].keepout_bboxes[0]
        # R1 moves to a spot that touches only the attached rect (far from
        # every bbox): the incremental update must find the pair
        m.footprints[1].x, m.footprints[1].y = w[0] + MM, w[1] + MM
        self.assertFalse(m.footprints[0].bbox[0] < m.footprints[1].bbox[2]
                         and m.footprints[1].bbox[0] < m.footprints[0].bbox[2]
                         and m.footprints[0].bbox[1] < m.footprints[1].bbox[3]
                         and m.footprints[1].bbox[1] < m.footprints[0].bbox[3])
        new = cs.incremental_update({1})
        self.assertGreater(cs._overlap_penalty, 0.0)
        self.assertAlmostEqual(new, CostState(m, quiet=True).total_cost, delta=1.0)
        # and the rect-holder moving onto a part counts too
        m.footprints[1].x, m.footprints[1].y = 60 * MM, 60 * MM
        cs.incremental_update({1})
        self.assertEqual(cs._overlap_penalty, 0.0)
        m.footprints[0].x, m.footprints[0].y = 60 * MM, 75 * MM
        cs.incremental_update({0})
        self.assertGreater(cs._overlap_penalty, 0.0)
        self.assertAlmostEqual(cs.total_cost, CostState(m, quiet=True).total_cost, delta=1.0)

    def test_random_moves_match_fresh_state(self):
        m = board_with_module()
        rnd = random.Random(5)
        for k in range(10):
            m.footprints.append(Footprint(
                reference="C%d" % k, index=len(m.footprints),
                x=rnd.randint(5, 70) * MM, y=rnd.randint(5, 70) * MM, angle_deg=0.0,
                width=2 * MM, height=MM, locked=False,
                pads=[Pad(1, "N1", 0, 0)], net_codes={1}))
            m.nets[1].pad_refs.append((len(m.footprints) - 1, 0))
        m.moveable_indices = list(range(len(m.footprints)))
        cs = CostState(m, quiet=True)
        random.seed(99)
        for step in range(300):
            mt = random.choice("tms")
            if mt == "t":
                undo = do_translate(m, 1.0, 30 * MM)
            elif mt == "m":
                undo = do_median(m, 0.5, 3 * MM)
            else:
                undo = do_swap(m)
                if undo is None:
                    continue
            ai = affected_indices(undo)
            snap = cs.snapshot(ai)
            cost = cs.incremental_update(ai)
            self.assertAlmostEqual(cost, CostState(m, quiet=True).total_cost, delta=2.0,
                                   msg="step %d" % step)
            if random.random() < 0.5:
                revert_move(m, undo)
                cs.restore(snap)
                self.assertAlmostEqual(cs.total_cost, CostState(m, quiet=True).total_cost,
                                       delta=2.0, msg="restore %d" % step)


def cset(items):
    return parse_constraints({
        "meta": {"board_name": "t", "board_dimensions_mm": [75, 75], "layer_count": 2},
        "component_groups": [], "net_priorities": {},
        "placement_constraints": {"keep_out_zones": items}}, "test")


class TestApply(unittest.TestCase):
    def test_antenna_item_attaches_rect_not_ring(self):
        m = board_with_module()
        m.footprints[0].keepout_rects = ()
        rep = apply_constraints(m, cset([
            {"ref": "U3", "clearance_mm": 15, "layers": ["F.Cu", "B.Cu"],
             "antenna": {"side": "pad_free", "module_end_beyond_pads_mm": 7.49,
                         "depth_mm": 6}}]), log=None)
        u3 = m.footprints[0]
        self.assertEqual(u3.clearance, 0)
        self.assertEqual([mm(r) for r in u3.keepout_rects], [(-23.6, -27.8, 23.6, -6.51)])
        self.assertIsNone(u3.clearance_side)
        self.assertEqual(rep.counts()["applied"].get("keep_out_zones", 0), 1)

    def test_ring_item_unchanged(self):
        m = board_with_module()
        m.footprints[0].keepout_rects = ()
        apply_constraints(m, cset([{"ref": "U3", "clearance_mm": 4}]), log=None)
        self.assertEqual(m.footprints[0].clearance, 4 * MM)
        self.assertEqual(tuple(m.footprints[0].keepout_rects), ())

    def test_ambiguous_side_skipped(self):
        m = board_with_module()
        rep = apply_constraints(m, cset([{"ref": "R1", "clearance_mm": 5,
                                          "antenna": {"side": "pad_free"}}]), log=None)
        self.assertEqual(tuple(m.footprints[1].keepout_rects), ())
        self.assertEqual(rep.counts()["applied"].get("keep_out_zones", 0), 0)
        self.assertEqual(rep.counts()["skipped"]["keep_out_zones"], 1)

    def test_fallback_warns(self):
        m = board_with_module()
        m.footprints[0].keepout_rects = ()
        rep = apply_constraints(m, cset([{"ref": "U3", "clearance_mm": 15,
                                          "antenna": {"side": "-y"}}]), log=None)
        self.assertTrue(any("courtyard" in w for w in rep.warnings))
        self.assertEqual(len(m.footprints[0].keepout_rects), 1)

    def test_schema_errors(self):
        for bad in ({"side": "up"}, {"side": "-y", "depth_mm": -1}, "x",
                    {"side": "-y", "extra": 1}):
            with self.assertRaises(ConstraintError):
                cset([{"ref": "U3", "clearance_mm": 15, "antenna": bad}])


if __name__ == "__main__":
    unittest.main()
