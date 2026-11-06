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


if __name__ == "__main__":
    unittest.main()
