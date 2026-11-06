"""Tests for the 2026-11-06 engine backlog (CadMust-Autopilot step 5(e)):
start HPWL in the batch JSON, back_side flips, keep-out cost scaling and
pad-level clearance. No KiCad dependency.

Run with: python3 -m unittest tests.test_engine_backlog -v
"""
import copy
import csv
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plugin.annealer import run_sa, SAConfig, _scatter_positions
from plugin.cost_function import CostState

from test_constraints import make_model, base_constraints, MM  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'batch_runner.py')


def _batch_runner():
    import importlib
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    return importlib.import_module("batch_runner")


# ---------------------------------------------------------------------------
# 1. Scattered-start HPWL
# ---------------------------------------------------------------------------

class TestStartHpwl(unittest.TestCase):
    CFG = dict(max_iterations=5, moves_per_temp=20, reheat_count=0)

    def test_scatter_start_records_scattered_state(self):
        m = make_model(n=10)
        designer = CostState(copy.deepcopy(m), quiet=True).hpwl
        # Reproduce the scatter run_sa performs first after seeding
        ref = copy.deepcopy(m)
        random.seed(7)
        _scatter_positions(ref)
        scattered = CostState(ref, quiet=True).hpwl
        res = run_sa(m, SAConfig(seed=7, scatter_first_start=True, num_starts=2,
                                 **self.CFG))
        self.assertEqual(len(res.start_hpwls), 2)
        self.assertEqual(res.start_hpwls[0], scattered)
        self.assertNotEqual(res.start_hpwls[0], designer)

    def test_no_scatter_start_is_designer(self):
        m = make_model(n=10)
        designer = CostState(copy.deepcopy(m), quiet=True).hpwl
        res = run_sa(m, SAConfig(seed=3, num_starts=3, **self.CFG))
        self.assertEqual(len(res.start_hpwls), 3)
        self.assertEqual(res.start_hpwls[0], designer)
        # starts 1.. are scattered
        self.assertNotEqual(res.start_hpwls[1], designer)

    def _cli(self, *extra):
        tmp = tempfile.mkdtemp()
        pkl = os.path.join(tmp, "b.pkl")
        with open(pkl, "wb") as f:
            pickle.dump(make_model(n=10), f)
        rj = os.path.join(tmp, "r.json")
        out = os.path.join(tmp, "r.csv")
        p = subprocess.run([sys.executable, SCRIPT, "--model", pkl, "--profile", "quick",
                            "--seed", "2", "--result-json", rj, "--out", out]
                           + list(extra), capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(rj) as f:
            res = json.load(f)
        with open(out) as f:
            rows = list(csv.DictReader(f))
        return res, rows

    def test_cli_scatter_keys(self):
        br = _batch_runner()
        designer = round(br.compute_hpwl_mm(make_model(n=10)), 1)
        res, rows = self._cli("--scatter-start")
        self.assertEqual(res["hpwl_designer_mm"], designer)
        self.assertEqual(res["hpwl_before_mm"], designer)
        self.assertIsNotNone(res["hpwl_start_mm"])
        self.assertGreater(abs(res["hpwl_start_mm"] - designer), 1.0)
        self.assertEqual(res["hpwl_scatter_start_mm"], res["hpwl_start_mm"])
        self.assertEqual(res["hpwl_starts_mm"], [res["hpwl_start_mm"]])
        # the CSV row (--out) carries the start value and the offboard policy
        self.assertEqual(float(rows[0]["hpwl_start_mm"]), res["hpwl_start_mm"])
        self.assertEqual(rows[0]["offboard"], "reject")

    def test_cli_no_scatter_start_equals_designer(self):
        res, _rows = self._cli()
        self.assertAlmostEqual(res["hpwl_start_mm"], res["hpwl_designer_mm"], delta=0.2)
        self.assertIsNone(res["hpwl_scatter_start_mm"])


# ---------------------------------------------------------------------------
# 2. back_side on a part that starts on the other side
# ---------------------------------------------------------------------------

from plugin.board_model import Footprint, Pad  # noqa: E402
from plugin.constraints import parse_constraints, apply_constraints  # noqa: E402

# One ESP32-EVB part as extracted by KiCad 10 (Q2, front, 90 degrees) and
# the same part re-extracted after KiCad's own FOOTPRINT::Flip(pos,
# FLIP_DIRECTION_TOP_BOTTOM) on the board (scratch check 2026-11-06: all
# 158 front parts of the board agree with flip_side within 2 nm).
ESP_Q2 = ({'x': 108966000, 'y': 92202000, 'angle': 270.0, 'w': 4109800, 'h': 3802459,
           'cx': 2540, 'cy': -1271,
           'pads': [(1107440, 949959), (1107439, -952500), (-1102360, 2540)]},
          {'x': 108966000, 'y': 92202000, 'angle': 90.0, 'w': 4109800, 'h': 3802459,
           'cx': 2540, 'cy': 1269,
           'pads': [(1107440, -949960), (1107440, 952499), (-1102360, -2539)]})


def _asym_fp(angle):
    fp = Footprint(reference="U1", index=0, x=40 * MM, y=30 * MM, angle_deg=angle,
                   width=6 * MM, height=4 * MM, locked=False,
                   pads=[Pad(1, "A", -2 * MM, -1 * MM), Pad(2, "B", 2 * MM, 1500000),
                         Pad(3, "C", 500000, -1700000)],
                   net_codes={1, 2, 3}, cx_offset=300000, cy_offset=-700000)
    fp.th_rects = [(-3 * MM, -2 * MM, -1 * MM, 0)]
    fp.keepout_rects = ((-3 * MM, -6 * MM, 3 * MM, -2 * MM),)
    fp.drawing_rect = (-3 * MM, -2500000, 3 * MM, 1 * MM)
    fp.drawing_rects = {"F.Fab": (-3 * MM, -2500000, 3 * MM, 1 * MM)}
    return fp


def _mirror_y(r, fy):
    return (r[0], 2 * fy - r[3], r[2], 2 * fy - r[1])


class TestFlipSide(unittest.TestCase):
    def test_flip_mirrors_world_geometry_about_origin_row(self):
        for angle in (0.0, 90.0, 180.0, 270.0, 30.0):
            fp = _asym_fp(angle)
            before_pads = [p.abs_position(fp.x, fp.y, fp.angle_deg) for p in fp.pads]
            before_bbox = fp.bbox
            before_th = fp.th_bboxes
            before_ko = fp.keepout_bboxes
            fp.flip_side()
            self.assertFalse(fp.is_front)
            self.assertTrue(fp.side_flipped)
            self.assertAlmostEqual(fp.angle_deg, (-angle) % 360.0)
            for (bx, by), p in zip(before_pads, fp.pads):
                ax, ay = p.abs_position(fp.x, fp.y, fp.angle_deg)
                self.assertLessEqual(abs(ax - bx), 2, angle)
                self.assertLessEqual(abs(ay - (2 * fp.y - by)), 2, angle)
            if angle % 90 == 0:
                for got, want in ((fp.bbox, _mirror_y(before_bbox, fp.y)),
                                  (fp.th_bboxes[0], _mirror_y(before_th[0], fp.y)),
                                  (fp.keepout_bboxes[0], _mirror_y(before_ko[0], fp.y))):
                    for g, w in zip(got, want):
                        self.assertLessEqual(abs(g - w), 2, (angle, got, want))
            self.assertIn("B.Fab", fp.drawing_rects)

    def test_double_flip_is_identity(self):
        fp = _asym_fp(90.0)
        ref = copy.deepcopy(fp)
        fp.flip_side()
        fp.flip_side()
        self.assertTrue(fp.is_front)
        self.assertFalse(fp.side_flipped)
        self.assertEqual([(p.offset_x, p.offset_y) for p in fp.pads],
                         [(p.offset_x, p.offset_y) for p in ref.pads])
        self.assertEqual(fp.bbox, ref.bbox)
        self.assertEqual(fp.th_rects, ref.th_rects)
        self.assertEqual(tuple(fp.keepout_rects), tuple(ref.keepout_rects))
        self.assertEqual(fp.angle_deg, ref.angle_deg)

    def test_esp32_evb_part_matches_kicad_flip(self):
        if ESP_Q2 is None:
            self.skipTest("no captured ESP32-EVB part")
        front, flipped = ESP_Q2
        fp = Footprint(reference="Q2", index=0, x=front["x"], y=front["y"],
                       angle_deg=front["angle"], width=front["w"], height=front["h"],
                       locked=False, pads=[Pad(0, "", ox, oy) for ox, oy in front["pads"]],
                       cx_offset=front["cx"], cy_offset=front["cy"])
        fp.flip_side()
        self.assertAlmostEqual(fp.angle_deg % 360.0, flipped["angle"] % 360.0)
        for p, (ox, oy) in zip(fp.pads, flipped["pads"]):
            self.assertLessEqual(abs(p.offset_x - ox), 2)
            self.assertLessEqual(abs(p.offset_y - oy), 2)
        self.assertLessEqual(abs(fp.cy_offset - flipped["cy"]), 2)
        self.assertLessEqual(abs(fp.cx_offset - flipped["cx"]), 2)


def _apply(m, pc):
    d = base_constraints(placement_constraints=pc)
    return apply_constraints(m, parse_constraints(d), log=None)


class TestBackSideApply(unittest.TestCase):
    def test_front_part_is_flipped_and_reported(self):
        m = make_model(n=8)
        m.footprints[4].pads[0].offset_y = 700000
        y_before = m.footprints[4].pads[0].offset_y
        m.footprints[5].is_front = False          # already on the back, listed
        m.footprints[6].is_front = False          # on the back, not listed
        rep = _apply(m, {"back_side": [{"ref": "R4"}, {"ref": "R5"}]})
        self.assertEqual(rep.flipped, ["R4"])
        self.assertEqual(rep.applied["back_side"], 2)
        self.assertFalse(m.footprints[4].is_front)
        self.assertTrue(m.footprints[4].side_flipped)
        self.assertEqual(m.footprints[4].pads[0].offset_y, -y_before)
        self.assertFalse(m.footprints[5].side_flipped)   # no flip, no mirror
        self.assertFalse(m.footprints[6].is_front)        # unlisted: stays
        self.assertTrue(any("R6" in n for n in rep.notes))
        self.assertEqual(rep.counts()["flipped"], ["R4"])
        self.assertIn("Flipped to the back side", rep.summary())

    def test_no_back_side_list_no_note(self):
        m = make_model(n=6)
        m.footprints[2].is_front = False
        rep = _apply(m, {})
        self.assertEqual(rep.flipped, [])
        self.assertEqual(rep.notes, [])
        self.assertFalse(m.footprints[2].is_front)

    def test_flipped_part_collides_as_back_side(self):
        # Two stacked front SMD parts overlap; listing one as back_side
        # removes the overlap (opposite sides, no TH pads).
        m = make_model(n=4)
        m.footprints[1].x, m.footprints[1].y = m.footprints[0].x, m.footprints[0].y
        self.assertGreater(CostState(copy.deepcopy(m), quiet=True)._overlap_penalty, 0)
        _apply(m, {"back_side": [{"ref": "R1"}]})
        cs = CostState(m, quiet=True)
        self.assertNotIn((0, 1), cs._pair_overlaps)


# ---------------------------------------------------------------------------
# 3. Keep-out cost scaling
# ---------------------------------------------------------------------------

from plugin.board_model import KeepOut  # noqa: E402
from plugin.cost_function import KEEPOUT_WEIGHT, OVERLAP_WEIGHT  # noqa: E402
from plugin.moves import do_translate, revert_move, affected_indices  # noqa: E402


class TestKeepoutScaling(unittest.TestCase):
    def _model_with_keepout(self):
        m = make_model(n=8)
        # keep-out rect 10 x 10 mm at (60, 30) mm absolute
        m.keepouts = [KeepOut(xmin=60 * MM, ymin=30 * MM, xmax=70 * MM, ymax=40 * MM)]
        return m

    def test_flat_per_part(self):
        m = self._model_with_keepout()
        many = m.footprints[4]           # R4: 4 nets (chain x2, fan-out, GND)
        few = m.footprints[3]            # R3: 3 nets
        self.assertNotEqual(len(many.net_codes), len(few.net_codes))
        for f in (many, few):
            f.x, f.y = 61 * MM, 31 * MM  # bbox 3x2 mm: corner at 59.5/30 -> intrusion
        cs = CostState(m, quiet=True)
        self.assertAlmostEqual(cs._fp_keepout[4], cs._fp_keepout[3])
        x1, y1, x2, y2 = many.bbox
        ox = min(x2, 70 * MM) - max(x1, 60 * MM)
        oy = min(y2, 40 * MM) - max(y1, 30 * MM)
        # decision 12 (2026-11-06 fix round): plus the intrusion depth
        depth = min(x2 - 60 * MM, 70 * MM - x1, y2 - 30 * MM, 40 * MM - y1)
        self.assertAlmostEqual(cs._fp_keepout[4], (ox + oy + depth) * KEEPOUT_WEIGHT)
        self.assertGreaterEqual(KEEPOUT_WEIGHT, OVERLAP_WEIGHT)

    def test_penalty_scale_still_applies(self):
        m = self._model_with_keepout()
        m.footprints[0].x, m.footprints[0].y = 65 * MM, 35 * MM
        cs = CostState(m, quiet=True)
        ko = cs._keepout_penalty
        self.assertGreater(ko, 0)
        full = cs.total_cost
        cs.update_penalty_scale(0.1)
        self.assertAlmostEqual(full - cs.total_cost,
                               0.9 * (ko + cs._boundary_penalty + cs._soft_penalty
                                      + cs._congestion_penalty), places=3)
        self.assertAlmostEqual(cs.normalized_cost, full)

    def test_incremental_matches_full(self):
        m = self._model_with_keepout()
        cs = CostState(m, quiet=True)
        rnd = random.Random(5)
        random.seed(5)
        for _ in range(300):
            undo = do_translate(m, rnd.random(), 20 * MM)
            cs.incremental_update(affected_indices(undo))
            if rnd.random() < 0.5:
                revert_move(m, undo)
                cs.incremental_update(affected_indices(undo))
        inc = cs._keepout_penalty
        cs._compute_all()
        self.assertAlmostEqual(inc, cs._keepout_penalty, places=3)


# ---------------------------------------------------------------------------
# 4. Pad-level clearance
# ---------------------------------------------------------------------------

from plugin.board_model import BoardModel, Net  # noqa: E402
from plugin.moves import do_rotate  # noqa: E402


def _pad_model(clearance=2_540_000, side_b="F", net_b=2, gap=1_000_000):
    """Two 4 x 4 mm parts side by side, courtyards gap apart. A has one
    2 x 2 mm pad at its right edge (clearance on it), B one at its left edge."""
    def part(ref, idx, x, pad_x, side, clr, net):
        pad = Pad(net_code=net, net_name="N%d" % net, offset_x=pad_x, offset_y=0)
        pad.rect = (pad_x - MM, -MM, pad_x + MM, MM)
        pad.side = side
        pad.clearance = clr
        return Footprint(reference=ref, index=idx, x=x, y=20 * MM, angle_deg=0.0,
                         width=4 * MM, height=4 * MM, locked=False, pads=[pad],
                         net_codes={net})
    a = part("RLY", 0, 20 * MM, MM, "TH", clearance, 1)
    b = part("R1", 1, 24 * MM + gap, -MM, side_b, 0, net_b)
    nets = {1: Net(1, "N1", [(0, 0)]), net_b: Net(net_b, "N%d" % net_b, [(1, 0)])}
    return BoardModel(footprints=[a, b], nets=nets, outline_xmin=0, outline_ymin=0,
                      outline_xmax=60 * MM, outline_ymax=40 * MM, moveable_indices=[0, 1])


class TestPadClearance(unittest.TestCase):
    def test_off_by_default_and_without_data(self):
        m = _pad_model()
        self.assertEqual(CostState(copy.deepcopy(m), quiet=True)._overlap_penalty, 0.0)
        self.assertFalse(SAConfig().pad_clearance)
        # pads without copper data (older pickles): no term even when on
        for f in m.footprints:
            for p in f.pads:
                p.rect = None
        self.assertEqual(CostState(m, quiet=True, pad_clearance=True)._overlap_penalty, 0.0)

    def test_clearance_violation_beyond_courtyard(self):
        m = _pad_model()
        cs = CostState(m, quiet=True, pad_clearance=True)
        # pad gap = 1 mm + 0 + 0 = courtyard gap 1 mm (pads at the edges);
        # required 2.54 -> ox = 1.54 mm, oy = 2 mm (pad heights, clearance
        # widens A to 7.08 mm tall, B is 2 mm)
        self.assertAlmostEqual(cs._overlap_penalty, (1_540_000 + 2 * MM) * 50.0)
        self.assertIn(0, cs._ext_fps)        # reach beyond the bbox

    def test_same_net_and_opposite_side_skipped(self):
        m = _pad_model(net_b=1)
        self.assertEqual(CostState(m, quiet=True, pad_clearance=True)._overlap_penalty, 0.0)
        # TH pad against a back-side SMD pad still counts; front SMD vs back SMD not
        m = _pad_model(side_b="B")
        self.assertGreater(CostState(m, quiet=True, pad_clearance=True)._overlap_penalty, 0)
        m = _pad_model(side_b="B")
        m.footprints[0].pads[0].side = "F"
        self.assertEqual(CostState(m, quiet=True, pad_clearance=True)._overlap_penalty, 0.0)

    def test_no_double_count_with_courtyard_overlap(self):
        m = _pad_model(clearance=0, gap=-1_500_000)   # courtyards overlap 1.5 mm
        off = CostState(copy.deepcopy(m), quiet=True)._overlap_penalty
        on = CostState(m, quiet=True, pad_clearance=True)._overlap_penalty
        self.assertGreater(off, 0)
        # pads overlap 1.5 mm in x and 2 mm in y < courtyard 1.5 + 4 mm: max keeps it
        self.assertEqual(on, off)

    def test_same_group_exempt(self):
        from plugin.board_model import ComponentGroup
        m = _pad_model()
        m.component_groups = [ComponentGroup(member_indices=[0, 1])]
        m.fp_to_group = {0: 0, 1: 0}
        self.assertEqual(CostState(m, quiet=True, pad_clearance=True)._overlap_penalty, 0.0)

    def test_rotation_and_flip_keep_geometry(self):
        m = _pad_model()
        fp = m.footprints[0]
        fp.set_angle(90.0)
        w = CostState(m, quiet=True, pad_clearance=True)._pads_world(0)[0]
        # local pad rect x in [0, 2] mm, y in [-1, 1]; at 90 deg (KiCad CW,
        # y down) local +x maps to world -y
        self.assertEqual((w[0], w[1], w[2], w[3]),
                         (fp.x - MM, fp.y - 2 * MM, fp.x + MM, fp.y))
        b = m.footprints[1]
        b.pads[0].rect = (-2 * MM, -MM, 0, 500_000)
        b.flip_side()
        self.assertEqual(b.pads[0].rect, (-2 * MM, -500_000, 0, MM))
        self.assertEqual(b.pads[0].side, "B")

    def test_incremental_matches_full(self):
        rnd = random.Random(11)
        m = make_model(n=10)
        for f in m.footprints:
            for p in f.pads:
                p.rect = (p.offset_x - 600_000, -600_000, p.offset_x + 600_000, 600_000)
                p.side = rnd.choice(["F", "TH"])
                p.clearance = rnd.choice([0, 0, 2_540_000])
        cs = CostState(m, quiet=True, pad_clearance=True)
        random.seed(11)
        for k in range(400):
            undo = do_translate(m, 0.5, 15 * MM) if k % 4 else do_rotate(m)
            cs.incremental_update(affected_indices(undo))
            if rnd.random() < 0.5:
                revert_move(m, undo)
                cs.incremental_update(affected_indices(undo))
        inc = cs._overlap_penalty
        cs._compute_all()
        self.assertAlmostEqual(inc, cs._overlap_penalty, places=2)

    def test_cli_flag(self):
        tmp = tempfile.mkdtemp()
        pkl = os.path.join(tmp, "b.pkl")
        with open(pkl, "wb") as f:
            pickle.dump(_pad_model(), f)
        for flag, want in (("on", True), ("off", False)):
            rj = os.path.join(tmp, "r%s.json" % flag)
            p = subprocess.run([sys.executable, SCRIPT, "--model", pkl, "--profile", "quick",
                                "--seed", "1", "--pad-clearance", flag, "--result-json", rj],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            with open(rj) as f:
                self.assertEqual(json.load(f)["pad_clearance"], want)


if __name__ == "__main__":
    unittest.main()
