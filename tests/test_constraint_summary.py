"""Tests for plugin/constraint_summary.py (results-dialog satisfaction summary)
and the shared unlock function constraints.unlock_constrained. No KiCad.

The ESP32-EVB case needs a BoardModel pickle and the measured constraint file
from CadMust-Autopilot (private sibling project). It looks in $CADMUST_AUTOPILOT
and then ../CadMust-Autopilot next to this repository, and is skipped when
neither exists (the pickle needs KiCad's pcbnew to produce).
"""
import copy
import math
import os
import pickle
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from test_constraints import (make_model, base_constraints, cset_of, apply,  # noqa: E402
                              _full_constraints, _locked_model, OX, OY, BW, BH, MM)
from plugin.board_model import ComponentGroup  # noqa: E402
from plugin.constraints import (unlock_constrained, unlock_refs, unlock_indices,  # noqa: E402
                                constrained_refs, load_constraints, apply_constraints)
from plugin.constraint_summary import (summarize, format_text, edge_distance_nm,  # noqa: E402
                                       ROUNDING_TOL_NM, FULL_CHECK, measure_edge_locks)
from plugin.cost_function import CostState  # noqa: E402
from plugin.moves import EDGE_LOCK_MARGIN, resnap_edges  # noqa: E402
from plugin.annealer import run_sa, SAConfig  # noqa: E402


def _autopilot_dir():
    cands = [os.environ.get("CADMUST_AUTOPILOT"),
             os.path.join(os.path.dirname(ROOT), "CadMust-Autopilot")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "autopilot", "models", "esp32-evb.pkl")) \
                and os.path.isfile(os.path.join(c, "autopilot", "constraints",
                                                "esp32-evb-measured.json")):
            return c
    return None


AUTOPILOT = _autopilot_dir()


def by_kind(summary, kind):
    return {c.item: c for c in summary.checks if c.kind == kind}


# ---------------------------------------------------------------------------
# Shared unlock function
# ---------------------------------------------------------------------------

class TestUnlockConstrained(unittest.TestCase):
    def test_same_result_as_unlock_refs(self):
        d = _full_constraints()
        m1, m2 = _locked_model(), _locked_model()
        named, refs, idx = unlock_constrained(m1, cset_of(d))
        self.assertEqual(named, constrained_refs(cset_of(d)))
        self.assertEqual(refs, unlock_refs(m2, named))
        self.assertEqual(m1.moveable_indices, m2.moveable_indices)
        self.assertEqual([m1.footprints[i].reference for i in idx], refs)
        self.assertTrue(all(not m1.footprints[i].locked for i in idx))

    def test_allowed_restricts_the_unlock(self):
        m = _locked_model()
        named, refs, _ = unlock_constrained(m, cset_of(_full_constraints()),
                                            allowed={"R1", "R0"})
        self.assertIn("R5", named)
        self.assertEqual(refs, ["R1"])           # R0 is allowed but not named
        self.assertTrue(m.footprints[5].locked)
        self.assertTrue(m.footprints[0].locked)

    def test_indices_keep_duplicate_references_apart(self):
        m = _locked_model()
        m.footprints[10].reference = "R5"         # a second R5 inside R5's group
        m.footprints[10].locked = True
        _named, refs, idx = unlock_constrained(m, cset_of(_full_constraints()))
        self.assertIn(5, idx)
        self.assertIn(10, idx)
        self.assertEqual(refs.count("R5"), 2)

    def test_unlock_indices_is_the_building_block(self):
        m = _locked_model()
        self.assertEqual(unlock_indices(m, {"R1", "NOPE"}), [1])

    def test_batch_runner_uses_the_same_function(self):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        br = importlib.import_module("batch_runner")
        self.assertIs(br.unlock_constrained, unlock_constrained)
        d = _full_constraints()
        m1, m2 = _locked_model(), _locked_model()     # no J/P parts: no auto-lock
        _n, refs = br.prepare_locks(m1, "constrained", cset_of(d))
        _named, refs2, _idx = unlock_constrained(m2, cset_of(d))
        self.assertEqual(refs, refs2)
        self.assertEqual(m1.moveable_indices, m2.moveable_indices)


# ---------------------------------------------------------------------------
# Summary semantics on a synthetic model
# ---------------------------------------------------------------------------

def _two_part_model():
    m = make_model(n=4)
    m.footprints[0].x, m.footprints[0].y = OX + 20 * MM, OY + 20 * MM
    m.footprints[1].x, m.footprints[1].y = OX + 30 * MM, OY + 25 * MM
    return m


class TestGroupSpan(unittest.TestCase):
    def _diag(self, m):
        a, b = m.footprints[0].bbox, m.footprints[1].bbox
        return math.hypot(max(a[2], b[2]) - min(a[0], b[0]),
                          max(a[3], b[3]) - min(a[1], b[1])) / MM

    def test_pass_at_the_limit_and_fail_just_over(self):
        m = _two_part_model()
        diag = self._diag(m)                       # 3x2 mm parts: 13 x 7 mm union
        self.assertAlmostEqual(diag, math.hypot(13, 7), places=6)
        d = base_constraints(component_groups=[
            {"id": "ok", "members": ["R0", "R1"], "max_span_mm": 15},
            {"id": "tight", "members": ["R0", "R1"], "max_span_mm": 14}])
        apply(m, d)
        s = by_kind(summarize(m, cset_of(d)), "group_span")
        self.assertTrue(s["ok"].passed)
        self.assertFalse(s["tight"].passed)
        self.assertAlmostEqual(s["tight"].measured_mm, diag, places=6)
        self.assertEqual(s["tight"].limit_mm, 14)

    def test_agrees_with_the_cost_term(self):
        m = _two_part_model()
        d = base_constraints(component_groups=[
            {"id": "ok", "members": ["R0", "R1"], "max_span_mm": 15},
            {"id": "tight", "members": ["R0", "R1"], "max_span_mm": 14}])
        apply(m, d)
        cs = CostState(m, quiet=True)
        checks = summarize(m).checks
        for gi, (gid, _idx, _span) in enumerate(m.soft_groups):
            c = [k for k in checks if k.item == gid][0]
            self.assertEqual(c.passed, cs._group_cost[gi] == 0, gid)


class TestZone(unittest.TestCase):
    def test_inside_on_the_border_and_outside(self):
        m = make_model(n=4)
        d = base_constraints(placement_constraints={"zone_preferences": [
            {"ref": "R0", "zone": "top-left"}, {"ref": "R1", "zone": "top-left"},
            {"ref": "R2", "zone": "top-left"}]})
        f = m.footprints
        f[0].x, f[0].y = OX + 10 * MM, OY + 10 * MM          # inside
        f[1].x, f[1].y = OX + 30 * MM, OY + 10 * MM          # on the column border
        f[2].x, f[2].y = OX + 32 * MM, OY + 10 * MM          # 2 mm outside
        apply(m, d)
        s = by_kind(summarize(m, cset_of(d)), "zone")
        self.assertTrue(s["R0"].passed)
        self.assertTrue(s["R1"].passed)
        self.assertFalse(s["R2"].passed)
        self.assertAlmostEqual(s["R2"].measured_mm, 2.0, places=6)
        self.assertIn("top-left", s["R2"].detail)
        cs = CostState(m, quiet=True)
        for i, ref in enumerate(("R0", "R1", "R2")):
            self.assertEqual(s[ref].passed, i not in cs._fp_zone, ref)

    def test_zone_name_without_the_file(self):
        m = make_model(n=4)
        d = base_constraints(placement_constraints={"zone_preferences": [
            {"ref": "R0", "zone": "bottom-right"}]})
        apply(m, d)
        self.assertIn("bottom-right", summarize(m).checks[0].detail)


class TestEdgeLock(unittest.TestCase):
    def _model(self):
        m = make_model(n=4)
        d = base_constraints(placement_constraints={"edge_locked": [
            {"ref": "R0", "edge": "top"}, {"ref": "R1", "edge": "right"}]})
        apply(m, d)
        return m, d

    def test_snapped_parts_pass_at_the_snap_distance(self):
        m, d = self._model()
        s = by_kind(summarize(m, cset_of(d)), "edge_lock")
        for ref in ("R0", "R1"):
            self.assertTrue(s[ref].passed)
            self.assertAlmostEqual(s[ref].measured_mm, EDGE_LOCK_MARGIN / MM, places=3)

    def test_farther_in_fails_past_the_edge_passes(self):
        m, d = self._model()
        m.footprints[0].y += 2 * MM                      # 3 mm from the top edge
        m.footprints[1].x += 3 * MM                      # 2 mm past the right edge
        s = by_kind(summarize(m, cset_of(d)), "edge_lock")
        self.assertFalse(s["R0"].passed)
        self.assertAlmostEqual(s["R0"].measured_mm, 3.0, places=3)
        self.assertTrue(s["R1"].passed)
        self.assertIn("past the edge", s["R1"].detail)

    def test_tolerance_is_rounding_only(self):
        m, d = self._model()
        m.footprints[0].y += ROUNDING_TOL_NM
        self.assertTrue(by_kind(summarize(m), "edge_lock")["R0"].passed)
        m.footprints[0].y += 1
        self.assertFalse(by_kind(summarize(m), "edge_lock")["R0"].passed)

    def test_polygon_notch_uses_the_snap_definition(self):
        m = make_model(n=4)
        # L-shape: 20 x 10 mm notch cut from the top-left corner
        m.outline_polygon = [(OX + 20 * MM, OY), (OX + BW, OY), (OX + BW, OY + BH),
                             (OX, OY + BH), (OX, OY + 10 * MM), (OX + 20 * MM, OY + 10 * MM)]
        m.footprints[0].x, m.footprints[0].y = OX + 8 * MM, OY + 30 * MM
        d = base_constraints(placement_constraints={"edge_locked": [
            {"ref": "R0", "edge": "top"}]})
        apply(m, d)
        f = m.footprints[0]
        self.assertAlmostEqual(f.bbox[1], OY + 11 * MM, delta=1)   # under the notch
        self.assertAlmostEqual(edge_distance_nm(m, 0), EDGE_LOCK_MARGIN, delta=1)
        self.assertTrue(summarize(m).checks[0].passed)
        # measured against the bbox top instead, it would read 11 mm
        f.y -= 10 * MM                                             # into the notch
        self.assertAlmostEqual(edge_distance_nm(m, 0), -9 * MM, delta=1)


class TestFixed(unittest.TestCase):
    def test_exact_then_moved(self):
        m = make_model(n=4)
        d = base_constraints(placement_constraints={"fixed_positions": [
            {"ref": "R0", "x_mm": 12.5, "y_mm": 7.25}, {"ref": "NOPE", "x_mm": 1, "y_mm": 1}]})
        apply(m, d)
        s = by_kind(summarize(m, cset_of(d)), "fixed")
        self.assertEqual(list(s), ["R0"])           # unknown refs are the report's job
        self.assertTrue(s["R0"].passed)
        m.footprints[0].x += 3 * MM
        m.footprints[0].y += 4 * MM
        c = by_kind(summarize(m, cset_of(d)), "fixed")["R0"]
        self.assertFalse(c.passed)
        self.assertAlmostEqual(c.measured_mm, 5.0, places=6)

    def test_without_the_file_nothing_is_measured(self):
        m = make_model(n=4)
        d = base_constraints(placement_constraints={"fixed_positions": [
            {"ref": "R0", "x_mm": 1, "y_mm": 1}]})
        apply(m, d)
        self.assertEqual(summarize(m).checks, [])


class TestAntennaOutward(unittest.TestCase):
    """antenna_outward locks: module end on the edge, antenna side faces it."""

    def _model(self):
        sys.path.insert(0, os.path.dirname(__file__))
        from test_antenna_outward import board, cset
        from test_antenna_keepout import module_fp
        m = board(module_fp(x=37 * MM, y=37 * MM, angle=90.0))
        apply_constraints(m, cset("top"), log=None)
        return m

    def _checks(self, m):
        return {c.item: c for c in measure_edge_locks(m)}

    def test_pass_after_apply(self):
        c = self._checks(self._model())
        self.assertTrue(c["U3"].passed)
        self.assertTrue(c["U3 antenna"].passed)

    def test_wrong_direction_fails(self):
        # broken case: the part turned 180 degrees after the run
        m = self._model()
        m.footprints[0].set_angle(180.0)
        c = self._checks(m)
        self.assertFalse(c["U3 antenna"].passed)
        self.assertIn("faces the bottom edge", c["U3 antenna"].detail)

    def test_module_end_off_the_edge_fails(self):
        # broken case: the part 2 mm inside, the module end no longer on the edge
        m = self._model()
        m.footprints[0].y += 2 * MM
        self.assertFalse(self._checks(m)["U3"].passed)


class TestText(unittest.TestCase):
    def test_fail_lines_totals_and_pointer_to_the_checker(self):
        m = make_model(n=4)
        d = base_constraints(
            component_groups=[{"id": "g", "members": ["R0", "R9"], "max_span_mm": 5}],
            placement_constraints={"zone_preferences": [{"ref": "R0", "zone": "top-left"}]})
        m.footprints[0].x, m.footprints[0].y = OX + 80 * MM, OY + 50 * MM
        rep = apply(m, d)
        txt = format_text(summarize(m, cset_of(d)), rep, path="b.json",
                          weights=(100.0, 100.0), unlocked=["R0"])
        self.assertIn("FAIL:", txt)
        self.assertIn("zone       R0", txt)
        self.assertIn("Skipped (2):", txt)          # R9 unknown, group < 2 members
        self.assertIn("Weights: cohesion 100, zone 100", txt)
        self.assertIn("Board locks disregarded for this run (1), lock flags kept: R0", txt)
        self.assertNotIn("lock cleared", txt)
        self.assertIn(FULL_CHECK, txt)
        self.assertIn("0 pass, 1 FAIL", txt)

    def test_no_fail_section_when_everything_passes(self):
        m = make_model(n=4)
        d = base_constraints(placement_constraints={"edge_locked": [
            {"ref": "R0", "edge": "top"}]})
        rep = apply(m, d)
        txt = format_text(summarize(m, cset_of(d)), rep)
        self.assertNotIn("FAIL:", txt)
        self.assertIn("1 pass, 0 FAIL", txt)


class TestReviewFollowupText(unittest.TestCase):
    """2026-11-06 external review, section 2.6: lock wording, flipped and
    rotated parts with before -> after, unknown references and unused
    sections itemised, and what "Move selected only" does with a part the
    file names but the user did not select."""

    def test_no_disregarded_locks_wording(self):
        m = make_model(n=4)
        d = base_constraints()
        txt = format_text(summarize(m, cset_of(d)), apply(m, d), unlocked=[])
        self.assertIn("Board locks disregarded for this run: none", txt)
        self.assertNotIn("Move selected only", txt)     # option off: no line

    def test_flipped_part_listed_with_angles(self):
        m = make_model(n=4)
        m.footprints[1].set_angle(90.0)
        d = base_constraints(placement_constraints={"back_side": [{"ref": "R1"}]})
        rep = apply(m, d)
        self.assertEqual(rep.flipped_angles, [("R1", 90.0, 270.0)])
        txt = format_text(summarize(m, cset_of(d)), rep)
        self.assertIn("Side flipped by back_side (1):", txt)
        self.assertIn("R1 front -> back, angle 90 -> 270 deg", txt)
        self.assertIn("Rotated by antenna_outward: none", txt)

    def test_part_already_on_the_back_is_not_listed(self):
        # broken case for the list: only parts the file actually flipped
        m = make_model(n=4)
        m.footprints[1].flip_side()
        d = base_constraints(placement_constraints={"back_side": [{"ref": "R1"}]})
        txt = format_text(summarize(m, cset_of(d)), apply(m, d))
        self.assertIn("Side flipped by back_side: none", txt)

    def test_rotated_part_listed(self):
        from test_antenna_outward import board, cset
        from test_antenna_keepout import module_fp
        m = board(module_fp(x=37 * MM, y=37 * MM, angle=90.0))
        c = cset("top")
        rep = apply_constraints(m, c, log=None)
        txt = format_text(summarize(m, c), rep)
        self.assertIn("Rotated by antenna_outward (1):", txt)
        self.assertIn("  U3 90 -> 0 deg", txt)

    def test_unknown_refs_and_unused_sections(self):
        m = make_model(n=4)
        d = base_constraints(
            placement_constraints={"zone_preferences": [{"ref": "Q7", "zone": "top-left"}]},
            signal_classes=[{"type": "max_length", "nets": ["N1"], "max_length_mm": 10}],
            design_notes=["note"])
        rep = apply(m, d)
        txt = format_text(summarize(m, cset_of(d)), rep)
        self.assertIn("Unknown references (1): Q7", txt)
        self.assertIn("zone_preferences Q7: reference not on the board", txt)
        self.assertIn("Not used by the engine: signal_classes 1, design_notes 1", txt)

    def test_unselected_named_parts_line(self):
        m = make_model(n=4)
        d = base_constraints()
        txt = format_text(summarize(m, cset_of(d)), apply(m, d), unlocked=[],
                          unselected_named=["R2", "R3"])
        self.assertIn("Move selected only: 2 part(s) the file names were not selected "
                      "and stay where they are", txt)
        self.assertIn(": R2, R3", txt)
        txt = format_text(summarize(m, cset_of(d)), apply(m, d), unselected_named=[])
        self.assertIn("Move selected only: every part the file names was selected", txt)

    def test_antenna_window_block(self):
        from test_antenna_window import place, top_notch
        from plugin.constraint_summary import measure_antenna_windows
        from test_antenna_outward import cset
        m, _f, w = place(40, top_notch(30, 50, 8))
        c = cset("top")
        c.placement["keep_out_zones"][0]["antenna"]["depth_mm"] = 6.0
        s = summarize(m, c)
        self.assertEqual([x.classification for x in s.windows], ["window"])
        n_checks = len(s.checks)
        txt = format_text(s)
        self.assertIn("Antenna window (geometry only, not in the totals):", txt)
        self.assertIn("ok   U3 (top edge): antenna window", txt)
        self.assertIn("RF clearance not assessed", txt)
        self.assertEqual(len(s.checks), n_checks)      # not counted
        # straddling the notch wall: a warning line
        m2, _f2, _w2 = place(32, top_notch(30, 50, 8))
        self.assertIn("WARN U3 (top edge): strip partly over board material",
                      format_text(summarize(m2, c)))
        self.assertEqual(measure_antenna_windows(m, None), [])


class TestRestrictToSelection(unittest.TestCase):
    """Move selected only: extract_board_model locks unselected parts, but
    apply_constraints set fixed positions and back_side flips regardless of
    the lock, and apply_model_to_board wrote both to the board. The GUI now
    drops those items for unselected parts (constraints.restrict_to_selection)."""

    def _doc(self):
        return base_constraints(placement_constraints={
            "fixed_positions": [{"ref": "R0", "x_mm": 5, "y_mm": 5},
                                {"ref": "R1", "x_mm": 9, "y_mm": 9}],
            "back_side": [{"ref": "R2"}, {"ref": "R3"}],
            "edge_locked": [{"ref": "R2", "edge": "top"}],
            "zone_preferences": [{"ref": "R3", "zone": "top-left"}],
            "keep_out_zones": [{"ref": "R2", "clearance_mm": 2}]})

    def _model(self):
        m = make_model(n=4)
        for i in (1, 2):                       # unselected: locked at extraction
            m.footprints[i].locked = True
        m.moveable_indices = [0, 3]
        return m

    def test_unselected_parts_stay(self):
        from plugin.constraints import restrict_to_selection
        m = self._model()
        before = [(f.x, f.y, f.is_front) for f in m.footprints]
        c, skipped = restrict_to_selection(cset_of(self._doc()), {"R0", "R3"})
        rep = apply_constraints(m, c, log=None)
        self.assertEqual((m.footprints[1].x, m.footprints[1].y, m.footprints[1].is_front),
                         before[1])
        self.assertEqual((m.footprints[2].x, m.footprints[2].y, m.footprints[2].is_front),
                         before[2])
        self.assertEqual(m.footprints[0].x, OX + 5 * MM)      # selected: applied
        self.assertFalse(m.footprints[3].is_front)
        self.assertEqual(sorted((k, r) for k, r, _ in skipped),
                         [("back_side", "R2"), ("edge_locked", "R2"), ("fixed_positions", "R1")])
        self.assertEqual(rep.flipped, ["R3"])
        self.assertEqual(rep.applied.get("keep_out_zones"), 1)   # keep-out kept
        # the original set is untouched
        self.assertEqual(len(cset_of(self._doc()).placement["fixed_positions"]), 2)

    def test_without_the_restriction_unselected_parts_move(self):
        # broken case = the behaviour before the fix
        m = self._model()
        before = (m.footprints[1].x, m.footprints[1].y)
        rep = apply_constraints(m, cset_of(self._doc()), log=None)
        self.assertNotEqual((m.footprints[1].x, m.footprints[1].y), before)
        self.assertIn("R2", rep.flipped)


class TestNoGuiImports(unittest.TestCase):
    """constraint_summary must import without wx and pcbnew.

    Checking sys.modules after a plain import is not enough under KiCad's
    Python: there pcbnew is importable, so plugin/__init__.py (the action
    plugin registration) imports pcbnew and wx before constraint_summary is
    even read, and the old check failed on every commit. The child process
    (-I: no user site, no PYTHONPATH, no cwd on sys.path) blocks both names
    with a meta-path finder instead, so the package init falls back to its
    ImportError branch. The finder records every attempt; the record is
    cleared after the package itself is imported, and any wx/pcbnew attempt
    by constraint_summary or what it pulls in (also a guarded try/except
    import) fails the test. Same result on both interpreters."""

    CODE = (
        "import sys, importlib\n"
        "tried = []\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('wx', 'pcbnew'):\n"
        "            tried.append(name)\n"
        "            raise ImportError('blocked: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "sys.path.insert(0, %r)\n"
        "mod = %r\n"
        "importlib.import_module(mod.split('.')[0])\n"
        "del tried[:]\n"
        "try:\n"
        "    importlib.import_module(mod)\n"
        "except ImportError as e:\n"
        "    print(e); sys.exit(2)\n"
        "bad = tried + [m for m in ('wx', 'pcbnew') if m in sys.modules]\n"
        "sys.exit(1 if bad else 0)\n")

    def _run(self, root, module):
        return subprocess.call([sys.executable, "-I", "-c", self.CODE % (root, module)],
                               stdout=subprocess.DEVNULL)

    def test_module_imports_neither_wx_nor_pcbnew(self):
        self.assertEqual(self._run(ROOT, "plugin.constraint_summary"), 0)

    def test_check_catches_a_gui_import(self):
        # deliberately broken cases: a plain and a guarded GUI import must fail
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "pkg"))
            open(os.path.join(d, "pkg", "__init__.py"), "w").close()
            with open(os.path.join(d, "pkg", "gui.py"), "w") as fh:
                fh.write("import pcbnew\n")
            with open(os.path.join(d, "pkg", "guarded.py"), "w") as fh:
                fh.write("try:\n    import wx\nexcept ImportError:\n    wx = None\n")
            self.assertNotEqual(self._run(d, "pkg.gui"), 0)
            self.assertNotEqual(self._run(d, "pkg.guarded"), 0)


class TestAfterAnnealing(unittest.TestCase):
    def test_summary_matches_the_cost_terms_after_a_run(self):
        m = make_model(n=12)
        d = base_constraints(
            component_groups=[{"id": "a", "members": ["R0", "R1", "R2"], "max_span_mm": 8},
                              {"id": "b", "members": ["R5", "R9"], "max_span_mm": 4}],
            placement_constraints={
                "zone_preferences": [{"ref": "R3", "zone": "bottom-left"},
                                     {"ref": "R4", "zone": "center"}],
                "edge_locked": [{"ref": "R6", "edge": "left"}],
                "fixed_positions": [{"ref": "R7", "x_mm": 45, "y_mm": 30}]})
        cset = cset_of(d)
        apply_constraints(m, cset, log=None)
        run_sa(m, config=SAConfig(max_iterations=30, seed=1))
        s = summarize(m, cset)
        self.assertEqual(len(s.checks), 6)
        cs = CostState(m, quiet=True)
        for gi, (gid, _i, _s) in enumerate(m.soft_groups):
            self.assertEqual(by_kind(s, "group_span")[gid].passed,
                             cs._group_cost[gi] == 0, gid)
        for fi in m.zone_prefs:
            ref = m.footprints[fi].reference
            self.assertEqual(by_kind(s, "zone")[ref].passed, fi not in cs._fp_zone, ref)
        self.assertTrue(by_kind(s, "edge_lock")["R6"].passed)
        self.assertTrue(by_kind(s, "fixed")["R7"].passed)


@unittest.skipUnless(AUTOPILOT, "needs the ESP32-EVB pickle and esp32-evb-measured.json "
                     "from CadMust-Autopilot ($CADMUST_AUTOPILOT or ../CadMust-Autopilot); "
                     "the pickle is made with KiCad's pcbnew")
class TestEsp32Evb(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(AUTOPILOT, "autopilot", "models", "esp32-evb.pkl"), "rb") as f:
            self.model = pickle.load(f)
        self.cset = load_constraints(os.path.join(AUTOPILOT, "autopilot", "constraints",
                                                  "esp32-evb-measured.json"))

    def test_unlock_apply_and_summary(self):
        m = self.model
        locked_before = {f.reference for f in m.footprints if f.locked}
        named, refs, idx = unlock_constrained(m, self.cset)
        self.assertTrue(refs)
        self.assertTrue(set(refs) <= locked_before)
        self.assertTrue(set(refs) <= named)
        # parts the file does not name keep their board lock
        for f in m.footprints:
            if f.reference in locked_before and f.reference not in named:
                self.assertTrue(f.locked, f.reference)
        rep = apply_constraints(m, self.cset, log=None)
        self.assertEqual(rep.skipped, [])
        s = summarize(m, self.cset)
        tot = s.totals()
        a = rep.counts()["applied"]
        self.assertEqual(sum(tot["group_span"]), a["soft_groups"])
        self.assertEqual(sum(tot["zone"]), a["zone_preferences"])
        self.assertEqual(sum(tot["edge_lock"]), a["edge_locked"])
        self.assertEqual(sum(tot["fixed"]), a["fixed_positions"])
        # the file is measured from this board: all pass before any move
        self.assertEqual(s.n_fail, 0, [c.detail for c in s.failures()])

    def test_short_run_agrees_with_the_cost_terms(self):
        m = self.model
        unlock_constrained(m, self.cset)
        apply_constraints(m, self.cset, log=None)
        run_sa(m, config=SAConfig(max_iterations=15, seed=1,
                                  cohesion_weight=100.0, zone_weight=100.0))
        s = summarize(m, self.cset)
        cs = CostState(m, quiet=True, cohesion_weight=100.0, zone_weight=100.0)
        for gi, (gid, _i, _s) in enumerate(m.soft_groups):
            self.assertEqual(by_kind(s, "group_span")[gid].passed,
                             cs._group_cost[gi] == 0, gid)
        for fi in m.zone_prefs:
            ref = m.footprints[fi].reference
            self.assertEqual(by_kind(s, "zone")[ref].passed, fi not in cs._fp_zone, ref)
        for ref, c in by_kind(s, "edge_lock").items():
            self.assertTrue(c.passed, c.detail)
        for ref, c in by_kind(s, "fixed").items():
            self.assertTrue(c.passed, c.detail)


if __name__ == "__main__":
    unittest.main()
