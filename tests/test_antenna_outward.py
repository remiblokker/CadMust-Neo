"""antenna_outward on an edge lock (CadMust-Autopilot design note
2026-11-06): the schema flag, the rotation at apply time (four edges, both
board sides), the module-end snap anchor, and the parts that stay as they
were. No KiCad dependency.

Run with: python3 tests/test_antenna_outward.py -v
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugin.board_model import BoardModel, Footprint, Pad, Net, ComponentGroup
from plugin.constraints import (apply_constraints, parse_constraints, ConstraintError,
                                antenna_outward_angle, facing_edge, side_world_direction,
                                resolve_antenna)
from plugin.moves import resnap_edges, do_translate, do_rotate, revert_move, EDGE_LOCK_MARGIN

from test_antenna_keepout import module_fp

MM = 1_000_000
BW = BH = 75 * MM
END = 7.49   # datasheet module end beyond the outermost pad centre (mm)


def board(u3, extra=()):
    fps = [u3]
    for k, (x, y) in enumerate(((60, 60), (65, 65))):
        fps.append(Footprint(reference="R%d" % (k + 1), index=k + 1, x=x * MM, y=y * MM,
                             angle_deg=0.0, width=2 * MM, height=MM, locked=False,
                             pads=[Pad(1, "N1", -MM // 2, 0), Pad(1, "N1", MM // 2, 0)],
                             net_codes={1}))
    fps.extend(extra)
    for i, f in enumerate(fps):
        f.index = i
    nets = {1: Net(1, "N1", pad_refs=[(1, 0), (2, 0)])}
    return BoardModel(footprints=fps, nets=nets, outline_xmin=0, outline_ymin=0,
                      outline_xmax=BW, outline_ymax=BH,
                      moveable_indices=list(range(len(fps))))


def cset(edge, outward=True, antenna=True, overhang=None, side="pad_free"):
    lock = {"ref": "U3", "edge": edge}
    if outward is not None:
        lock["antenna_outward"] = outward
    if overhang is not None:
        lock["overhang_ok"] = overhang
    ko = []
    if antenna:
        ko.append({"ref": "U3", "clearance_mm": 15,
                   "antenna": {"side": side, "module_end_beyond_pads_mm": END}})
    return parse_constraints(doc({"edge_locked": [lock], "keep_out_zones": ko}))


def doc(pc):
    return {"meta": {}, "component_groups": [], "net_priorities": {},
            "placement_constraints": pc}


def antenna_end_world(f, side):
    """World coordinate (along the facing axis) of the module end."""
    res = resolve_antenna(f, side, 0, int(END * MM))
    return f.edge_anchor_world(), res


EDGE_COORD = {"top": (1, 0), "bottom": (1, BH), "left": (0, 0), "right": (0, BW)}


class TestAngle(unittest.TestCase):
    def test_local_minus_y_faces_top_at_zero(self):
        self.assertEqual(side_world_direction("-y", 0.0), (0, -1))
        self.assertEqual(antenna_outward_angle("-y", "top"), 0.0)

    def test_every_side_and_edge_round_trips(self):
        for side in ("+x", "-x", "+y", "-y"):
            for edge in ("top", "bottom", "left", "right"):
                a = antenna_outward_angle(side, edge)
                self.assertIn(a, (0.0, 90.0, 180.0, 270.0))
                self.assertEqual(facing_edge(side, a), edge)


class TestApplyEightCases(unittest.TestCase):
    """Four edges x both board sides: after apply the pad-free side points at
    the locked edge (checked on the pads in world space, not on the angle
    math), the module end lies on the edge and every pad is on the board."""

    def _case(self, edge, back):
        u3 = module_fp(x=37 * MM, y=37 * MM, angle=180.0 if back else 90.0)
        if back:
            u3.flip_side()          # the as-placed (mirrored) frame of a back part
        m = board(u3)
        rep = apply_constraints(m, cset(edge), log=None)
        self.assertEqual(rep.applied.get("antenna_outward"), 1)
        f = m.footprints[0]
        self.assertEqual(f.antenna_side, "+y" if back else "-y")
        self.assertTrue(f.overhang_ok)
        pads = [p.abs_position(f.x, f.y, f.angle_deg) for p in f.pads]
        axis, coord = EDGE_COORD[edge]
        anc = f.edge_anchor_world()
        # the module end is on the edge (whole nm rounding)
        self.assertLessEqual(abs(anc[axis] - coord), 2, (edge, back, anc))
        # every pad lies on the board side of the edge, and the pad nearest to
        # the edge is exactly END mm inside it (pad-free side faces the edge)
        dists = [abs(p[axis] - coord) for p in pads]
        self.assertAlmostEqual(min(dists) / MM, END, places=3)
        for p in pads:
            self.assertTrue(0 <= p[0] <= BW and 0 <= p[1] <= BH, (edge, back, p))
        # the opposite (padded) end is further from the edge than the pad-free end
        self.assertGreater(max(dists) / MM, END + 15)
        return rep, f

    def test_eight(self):
        for back in (False, True):
            for edge in ("top", "bottom", "left", "right"):
                with self.subTest(edge=edge, back=back):
                    self._case(edge, back)

    def test_rotation_is_reported(self):
        rep, f = self._case("top", False)    # 90 -> 0
        self.assertEqual(rep.rotated, [("U3", 90.0, 0.0)])
        self.assertEqual(rep.counts()["rotated"],
                         [{"ref": "U3", "from_deg": 90.0, "to_deg": 0.0}])
        self.assertIn("Rotated U3", rep.summary())

    def test_no_rotation_when_already_facing(self):
        u3 = module_fp(x=37 * MM, y=37 * MM, angle=0.0)
        m = board(u3)
        rep = apply_constraints(m, cset("top"), log=None)
        self.assertEqual(rep.rotated, [])
        self.assertEqual(m.footprints[0].angle_deg, 0.0)

    def test_explicit_side(self):
        u3 = module_fp(x=37 * MM, y=37 * MM)
        m = board(u3)
        apply_constraints(m, cset("left", side="+x"), log=None)
        f = m.footprints[0]
        self.assertEqual(facing_edge("+x", f.angle_deg), "left")


class TestWithoutTheFlag(unittest.TestCase):
    def test_plain_lock_keeps_angle_and_bbox_snap(self):
        u3 = module_fp(x=37 * MM, y=37 * MM, angle=90.0)
        m = board(u3)
        rep = apply_constraints(m, cset("top", outward=None), log=None)
        f = m.footprints[0]
        self.assertEqual(f.angle_deg, 90.0)
        self.assertEqual(f.edge_anchor_offset_nm, 0)
        self.assertIsNone(f.antenna_side)
        self.assertFalse(f.overhang_ok)
        self.assertEqual(rep.rotated, [])
        self.assertEqual(f.bbox[1], EDGE_LOCK_MARGIN)

    def test_outward_without_antenna_item_is_a_plain_lock(self):
        # broken case: the flag without the keep-out item it refers to
        u3 = module_fp(x=37 * MM, y=37 * MM, angle=90.0)
        m = board(u3)
        rep = apply_constraints(m, cset("top", antenna=False), log=None)
        f = m.footprints[0]
        self.assertEqual(f.angle_deg, 90.0)
        self.assertEqual(f.edge_anchor_offset_nm, 0)
        self.assertTrue(any("without an antenna keep-out" in w for w in rep.warnings))
        self.assertNotIn("antenna_outward", rep.applied)

    def test_old_pickle_defaults(self):
        f = module_fp()
        for k in ("overhang_ok", "antenna_side", "edge_anchor_offset_nm"):
            f.__dict__.pop(k, None)
        self.assertFalse(f.overhang_ok)
        self.assertIsNone(f.antenna_side)
        self.assertEqual(f.edge_anchor_offset_nm, 0)
        self.assertIsNone(f.edge_anchor_world())


class TestSnapAndMoves(unittest.TestCase):
    def _model(self, edge):
        u3 = module_fp(x=37 * MM, y=37 * MM, angle=90.0)
        m = board(u3)
        apply_constraints(m, cset(edge), log=None)
        return m

    def test_anchor_snap_differs_from_bbox_snap(self):
        m = self._model("top")
        f = m.footprints[0]
        # module end 12.8 mm out vs courtyard edge 6.51 mm: the bbox top is
        # 6.29 mm inside the edge, not the 1.0 mm a bbox snap gives
        self.assertEqual(f.edge_anchor_world()[1], 0)
        self.assertAlmostEqual(f.bbox[1] / MM, 12.8 - 6.51, places=3)

    def test_translates_keep_anchor_on_edge_and_angle(self):
        random.seed(5)
        for edge in ("top", "left", "bottom", "right"):
            m = self._model(edge)
            f = m.footprints[0]
            a0 = f.angle_deg
            axis, coord = EDGE_COORD[edge]
            for _ in range(300):
                undo = do_translate(m, 1.0, 20 * MM)
                self.assertLessEqual(abs(f.edge_anchor_world()[axis] - coord), 2)
                if random.random() < 0.5:
                    revert_move(m, undo)
                rot = do_rotate(m)
                self.assertEqual(f.angle_deg, a0)


class TestGreedyKeepsAngle(unittest.TestCase):
    def test_greedy_refine_does_not_rotate(self):
        from plugin.annealer import _greedy_refine
        from plugin.cost_function import CostState
        u3 = module_fp(x=37 * MM, y=37 * MM, angle=90.0)
        m = board(u3)
        apply_constraints(m, cset("top"), log=None)
        cs = CostState(m, quiet=True)
        _greedy_refine(m, cs, [0, 1, 2])
        self.assertEqual(m.footprints[0].angle_deg, 0.0)
        self.assertEqual(m.footprints[0].edge_anchor_world()[1], 0)


if __name__ == "__main__":
    unittest.main()
