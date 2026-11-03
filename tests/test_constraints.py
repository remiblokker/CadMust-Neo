"""Unit tests for the constraint-JSON reader (plugin/constraints.py) and the
constraint-driven cost terms and move rules. No KiCad dependency.

Run with: python3 tests/test_constraints.py -v
"""
import copy
import json
import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugin.board_model import BoardModel, Footprint, Pad, Net
from plugin.cost_function import CostState, OVERLAP_WEIGHT
from plugin.constraints import (anchor_frame_offset,
                                load_constraints, parse_constraints,
                                apply_constraints, ConstraintError, zone_rect_nm)
from plugin.moves import (do_translate, do_median, do_swap, do_rotate,
                          revert_move, affected_indices, EDGE_LOCK_MARGIN)
from plugin import annealer
from plugin.annealer import run_sa, SAConfig, _scatter_positions

MM = 1_000_000

# Board outline deliberately NOT at the origin, to exercise board-relative mm
OX, OY = 50 * MM, 20 * MM
BW, BH = 90 * MM, 60 * MM


def make_model(n=12, seed=3) -> BoardModel:
    """n footprints R0..R(n-1) on a 90 x 60 mm board at (50, 20) mm, each 3x2 mm,
    two pads, chained nets N1..N(n-1) plus a fan-out net NFAN and a power net."""
    rnd = random.Random(seed)
    fps = []
    for i in range(n):
        x = OX + rnd.randint(10, 80) * MM
        y = OY + rnd.randint(10, 50) * MM
        fps.append(Footprint(reference="R%d" % i, index=i, x=x, y=y, angle_deg=0.0,
                             width=3 * MM, height=2 * MM, locked=False,
                             pads=[], net_codes=set()))
    nets = {}

    def connect(nc, name, members):
        net = Net(net_code=nc, net_name=name)
        for fi in members:
            fp = fps[fi]
            off = -MM if len(fp.pads) % 2 == 0 else MM
            fp.pads.append(Pad(net_code=nc, net_name=name, offset_x=off, offset_y=0))
            fp.net_codes.add(nc)
            net.pad_refs.append((fi, len(fp.pads) - 1))
        nets[nc] = net

    for i in range(1, n):
        connect(i, "/N%d" % i, [i - 1, i])
    connect(n, "/BUS{slash}CLK", [0, n // 2, n - 1])
    connect(n + 1, "GND", list(range(n)))
    nets[n + 1].is_excluded = True
    return BoardModel(footprints=fps, nets=nets,
                      outline_xmin=OX, outline_ymin=OY,
                      outline_xmax=OX + BW, outline_ymax=OY + BH,
                      moveable_indices=list(range(n)))


def base_constraints(**overrides) -> dict:
    d = {
        "meta": {"board_name": "test", "board_dimensions_mm": [90, 60], "layer_count": 2},
        "component_groups": [],
        "net_priorities": {},
        "placement_constraints": {},
    }
    d.update(overrides)
    return d


def cset_of(d):
    return parse_constraints(d, "test")


def apply(model, d):
    return apply_constraints(model, cset_of(d), log=None)


def fresh_cost(model):
    return CostState(model, quiet=True).normalized_cost


# ---------------------------------------------------------------------------

class TestLoading(unittest.TestCase):
    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_load_valid_file(self):
        path = self._write(json.dumps(base_constraints()))
        cs = load_constraints(path)
        self.assertEqual(cs.meta["board_name"], "test")

    def test_bad_json_reports_line(self):
        path = self._write('{"meta": {,}')
        with self.assertRaises(ConstraintError) as cm:
            load_constraints(path)
        self.assertIn("line 1", str(cm.exception))

    def test_missing_file(self):
        with self.assertRaises(ConstraintError):
            load_constraints("/nonexistent/x.json")

    def test_structural_errors_listed(self):
        d = base_constraints(placement_constraints={
            "edge_locked": [{"ref": "R1", "edge": "north"}],
            "zone_preferences": [{"ref": "R1", "zone": "middle"}],
        })
        del d["net_priorities"]
        with self.assertRaises(ConstraintError) as cm:
            parse_constraints(d)
        msg = str(cm.exception)
        self.assertIn("net_priorities", msg)
        self.assertIn("edge must be one of", msg)
        self.assertIn("zone must be one of", msg)

    def test_unknown_top_key_is_warning(self):
        d = base_constraints(extra_stuff=1)
        cs = parse_constraints(d)
        self.assertTrue(any("extra_stuff" in w for w in cs.warnings))


class TestMapping(unittest.TestCase):
    def test_fixed_position_board_relative(self):
        m = make_model()
        rep = apply(m, base_constraints(placement_constraints={
            "fixed_positions": [{"ref": "R2", "x_mm": 3.81, "y_mm": 7.5}]}))
        f = m.footprints[2]
        self.assertEqual(f.x, OX + 3_810_000)
        self.assertEqual(f.y, OY + 7_500_000)
        self.assertTrue(f.locked)
        self.assertNotIn(2, m.moveable_indices)
        self.assertEqual(rep.applied["fixed_positions"], 1)

    def test_back_side(self):
        m = make_model()
        apply(m, base_constraints(placement_constraints={"back_side": [{"ref": "R4"}]}))
        self.assertFalse(m.footprints[4].is_front)
        self.assertTrue(m.footprints[3].is_front)

    def test_unknown_refs_and_nets_reported(self):
        m = make_model()
        rep = apply(m, base_constraints(
            net_priorities={"critical": {"nets": ["NOPE", "N1"], "weight": 3.0}},
            placement_constraints={"back_side": [{"ref": "X99"}],
                                   "zone_preferences": [{"ref": "Q7", "zone": "center"}]}))
        self.assertEqual(rep.unknown_refs, ["X99", "Q7"])
        self.assertEqual(rep.unknown_nets, ["NOPE"])
        self.assertEqual(rep.applied.get("net_weights"), 1)
        cats = {(c, i) for c, i, _ in rep.skipped}
        self.assertIn(("back_side", "X99"), cats)
        self.assertIn(("net_priorities", "NOPE"), cats)

    def test_net_name_matching(self):
        m = make_model()
        # "N1" matches "/N1" (leading slash dropped); "BUS/CLK" matches "/BUS{slash}CLK"
        apply(m, base_constraints(net_priorities={
            "high": {"nets": ["N1", "BUS/CLK"], "weight": 2.0},
            "exclude": {"nets": ["/N2"]}}))
        self.assertEqual(m.nets[1].weight, 2.0)
        self.assertEqual(m.nets[12].weight, 2.0)
        self.assertTrue(m.nets[2].is_excluded)
        self.assertEqual(m.nets[3].weight, 1.0)

    def test_net_weight_in_total_cost_not_in_hpwl(self):
        m = make_model()
        cs0 = CostState(m, quiet=True)
        raw = cs0.hpwl
        h1 = cs0.net_hpwl[1]
        apply(m, base_constraints(net_priorities={"critical": {"nets": ["N1"], "weight": 5.0}}))
        cs = CostState(m, quiet=True)
        self.assertEqual(cs.hpwl, raw)
        self.assertAlmostEqual(cs.normalized_cost - cs0.normalized_cost, 4.0 * h1, delta=1.0)
        self.assertAlmostEqual(cs.weighted_hpwl, raw + 4.0 * h1, delta=1.0)

    def test_meta_dimension_warning(self):
        m = make_model()
        rep = apply(m, base_constraints(meta={"board_name": "t", "layer_count": 2,
                                              "board_dimensions_mm": [80, 60]}))
        self.assertTrue(any("differs" in w for w in rep.warnings))
        rep = apply(make_model(), base_constraints())
        self.assertFalse(rep.warnings)

    def test_ignored_sections_counted(self):
        m = make_model()
        rep = apply(m, base_constraints(
            signal_classes=[{"id": "s", "kind": "max_length", "nets": ["N1"],
                             "max_length_mm": 5}],
            design_notes=["a", "b"],
            placement_constraints={"back_side": [{"ref": "R1", "req_ids": ["REQ-1"],
                                                  "source": "x"}]}))
        self.assertEqual(rep.ignored["signal_classes"], 1)
        self.assertEqual(rep.ignored["design_notes"], 2)
        self.assertEqual(rep.ignored["req_ids/source"], 1)


class TestTemplateGroups(unittest.TestCase):
    def test_template_places_and_registers_rigid_group(self):
        m = make_model()
        anchor = m.footprints[0]
        ax, ay = anchor.x, anchor.y
        m.footprints[3].locked = True
        m.moveable_indices.remove(3)
        rep = apply(m, base_constraints(component_groups=[{
            "id": "g", "name": "G", "members": ["R0", "R1", "R2", "R3", "R5"],
            "template": {"anchor": "R0", "relative_positions": {
                "R1": {"dx_mm": 4, "dy_mm": 0},
                "R2": {"dx_mm": -3, "dy_mm": 2.5},
                "R3": {"dx_mm": 0, "dy_mm": 6}}}}]))
        self.assertEqual((m.footprints[1].x, m.footprints[1].y), (ax + 4 * MM, ay))
        self.assertEqual((m.footprints[2].x, m.footprints[2].y),
                         (ax - 3 * MM, ay + 2_500_000))
        self.assertEqual(len(m.component_groups), 1)
        self.assertEqual(m.component_groups[0].member_indices, [0, 1, 2])
        self.assertIn(0, m.moveable_indices)
        self.assertNotIn(1, m.moveable_indices)
        self.assertNotIn(2, m.moveable_indices)
        # R3 is locked: not moved, not in the rigid body, reported
        self.assertNotIn(3, m.fp_to_group)
        self.assertTrue(any(i == "g.R3" for _, i, _ in rep.skipped))
        # R5 is a member but not in the template: noted
        self.assertTrue(any("R5" in n for n in rep.notes))
        # Translate moves the rigid group as a whole
        random.seed(1)
        m.moveable_indices = [0]
        undo = do_translate(m, 1.0, 5 * MM)
        self.assertEqual(affected_indices(undo), {0, 1, 2})
        self.assertEqual(m.footprints[1].x - m.footprints[0].x, 4 * MM)


class TestTemplateFrame(unittest.TestCase):
    def _model(self, angle):
        m = make_model()
        a = m.footprints[0]
        a.x, a.y = OX + 40 * MM, OY + 30 * MM
        a.set_angle(angle)
        apply(m, base_constraints(component_groups=[{
            "id": "g", "name": "G", "members": ["R0", "R1", "R2"],
            "template": {"anchor": "R0", "relative_positions": {
                "R1": {"dx_mm": 4, "dy_mm": 0},
                "R2": {"dx_mm": 1, "dy_mm": 2}}}}]))
        return m

    def test_anchor_at_90_degrees(self):
        m = self._model(90.0)
        a, b = m.footprints[0], m.footprints[1]
        # Pad.abs_position convention: (4, 0) at 90 degrees -> (0, -4), y down
        self.assertAlmostEqual(b.x - a.x, 0, delta=2)
        self.assertAlmostEqual(b.y - a.y, -4 * MM, delta=2)
        self.assertNotAlmostEqual(b.y - a.y, 4 * MM, delta=MM)
        # same transform as a pad at that offset
        px, py = Pad(0, "", 4 * MM, 0).abs_position(a.x, a.y, a.angle_deg)
        self.assertEqual((b.x, b.y), (px, py))
        dx, dy = anchor_frame_offset(a, b)
        self.assertAlmostEqual(dx, 4.0, places=5)
        self.assertAlmostEqual(dy, 0.0, places=5)

    def test_rotate_move_keeps_anchor_frame_offsets(self):
        for start in (0.0, 90.0, 270.0):
            m = self._model(start)
            m.moveable_indices = [0]
            random.seed(int(start) + 1)
            for _ in range(25):
                do_rotate(m)
                a = m.footprints[0]
                for idx, want in ((1, (4.0, 0.0)), (2, (1.0, 2.0))):
                    dx, dy = anchor_frame_offset(a, m.footprints[idx])
                    self.assertAlmostEqual(dx, want[0], delta=1e-4)
                    self.assertAlmostEqual(dy, want[1], delta=1e-4)
                    # members turn with the body
                    self.assertEqual(m.footprints[idx].angle_deg % 360.0,
                                     (a.angle_deg - start + 0.0) % 360.0)

    def test_template_group_gets_cohesion_over_all_members(self):
        m = make_model()
        a = m.footprints[0]
        a.x, a.y = OX + 10 * MM, OY + 10 * MM
        m.footprints[5].x, m.footprints[5].y = OX + 80 * MM, OY + 50 * MM  # loose member
        rep = apply(m, base_constraints(component_groups=[{
            "id": "g", "name": "G", "members": ["R0", "R1", "R5"], "max_span_mm": 10,
            "template": {"anchor": "R0", "relative_positions": {
                "R1": {"dx_mm": 4, "dy_mm": 0}}}}]))
        self.assertEqual(rep.applied["rigid_groups"], 1)
        self.assertEqual(rep.applied["soft_groups"], 1)
        self.assertEqual(m.soft_groups[0][1], [0, 1, 5])
        self.assertGreater(CostState(m, quiet=True)._cohesion_penalty, 60 * MM)


class TestSoftGroups(unittest.TestCase):
    def test_cohesion_cost(self):
        m = make_model()
        for i, (x, y) in enumerate([(10, 10), (40, 10), (40, 50)]):
            m.footprints[i].x = OX + x * MM
            m.footprints[i].y = OY + y * MM
        cs0 = CostState(m, quiet=True)
        rep = apply(m, base_constraints(component_groups=[
            {"id": "soft", "name": "S", "members": ["R0", "R1", "R2"], "max_span_mm": 20},
            {"id": "nospan", "name": "N", "members": ["R3", "R4"]}]))
        self.assertEqual(rep.applied["soft_groups"], 1)
        self.assertTrue(any(i == "nospan" for _, i, _ in rep.skipped))
        cs = CostState(m, quiet=True)
        # union bbox: x 8.5..41.5, y 9..51 mm
        diag = math.hypot(33 * MM, 42 * MM)
        self.assertAlmostEqual(cs._cohesion_penalty, diag - 20 * MM, delta=2)
        self.assertAlmostEqual(cs.normalized_cost - cs0.normalized_cost,
                               diag - 20 * MM, delta=2)
        # penalty_scale applies to it
        cs.update_penalty_scale(0.5)
        self.assertAlmostEqual(cs.total_cost - cs0.normalized_cost,
                               0.5 * (diag - 20 * MM), delta=2)

    def test_within_span_is_free(self):
        m = make_model()
        for i in range(3):
            m.footprints[i].x = OX + (10 + 4 * i) * MM
            m.footprints[i].y = OY + 10 * MM
        apply(m, base_constraints(component_groups=[
            {"id": "s", "name": "S", "members": ["R0", "R1", "R2"], "max_span_mm": 20}]))
        self.assertEqual(CostState(m, quiet=True)._cohesion_penalty, 0.0)


class TestZones(unittest.TestCase):
    def test_zone_rect_and_cost(self):
        m = make_model()
        r = zone_rect_nm(m, "top-left")
        self.assertEqual(r, (OX, OY, OX + BW / 3.0, OY + BH / 3.0))
        f = m.footprints[0]
        f.x = OX + 60 * MM   # zone x range 0..30 mm, so 30 mm outside
        f.y = OY + 10 * MM   # inside in y
        cs0 = CostState(m, quiet=True)
        apply(m, base_constraints(placement_constraints={
            "zone_preferences": [{"ref": "R0", "zone": "top-left"}]}))
        cs = CostState(m, quiet=True)
        self.assertAlmostEqual(cs._zone_penalty, 30 * MM, delta=1)
        self.assertAlmostEqual(cs.normalized_cost - cs0.normalized_cost, 30 * MM, delta=1)
        f.x = OX + 15 * MM
        self.assertEqual(CostState(m, quiet=True)._zone_penalty, 0.0)

    def test_center_zone(self):
        m = make_model()
        r = zone_rect_nm(m, "center")
        self.assertEqual(r, (OX + 30 * MM, OY + 20 * MM, OX + 60 * MM, OY + 40 * MM))


class TestEdgeLock(unittest.TestCase):
    def setUp(self):
        self.m = make_model()
        self.rep = apply(self.m, base_constraints(placement_constraints={
            "edge_locked": [{"ref": "R0", "edge": "top"},
                            {"ref": "R1", "edge": "right"}]}))

    def _on_edge(self):
        m = self.m
        self.assertEqual(m.footprints[0].bbox[1], m.outline_ymin + EDGE_LOCK_MARGIN)
        self.assertEqual(m.footprints[1].bbox[2], m.outline_xmax - EDGE_LOCK_MARGIN)

    def test_snapped_at_apply_and_still_moveable(self):
        self._on_edge()
        self.assertIn(0, self.m.moveable_indices)
        self.assertEqual(self.rep.applied["edge_locked"], 2)

    def test_resnap_after_translate_median_rotate(self):
        m = self.m
        m.moveable_indices = [0, 1]
        random.seed(5)
        xs = set()
        for _ in range(50):
            do_translate(m, 1.0, 20 * MM)
            self._on_edge()
            xs.add(m.footprints[0].x)
            do_median(m, 0.5, 2 * MM)
            self._on_edge()
            do_rotate(m)
            self._on_edge()
        self.assertGreater(len(xs), 10)  # it does slide along the edge

    def test_swap_rejected(self):
        m = self.m
        m.moveable_indices = [0, 1, 2]
        random.seed(2)
        for _ in range(30):
            undo = do_swap(m)
            if undo is not None:
                self.assertFalse(affected_indices(undo) & {0, 1})
                revert_move(m, undo)
        m.moveable_indices = [0, 1]
        self.assertIsNone(do_swap(m))

    def test_resnap_after_scatter(self):
        random.seed(9)
        _scatter_positions(self.m)
        self._on_edge()

    def test_locked_edge_part_is_skipped(self):
        m = make_model()
        m.footprints[0].locked = True
        before = (m.footprints[0].x, m.footprints[0].y)
        rep = apply(m, base_constraints(placement_constraints={
            "edge_locked": [{"ref": "R0", "edge": "left"}]}))
        self.assertEqual((m.footprints[0].x, m.footprints[0].y), before)
        self.assertTrue(any(i == "R0" for _, i, _ in rep.skipped))
        self.assertIsNone(m.edge_locked_indices)

    def test_rigid_group_snaps_as_a_whole(self):
        m = make_model()
        apply(m, base_constraints(
            component_groups=[{"id": "g", "name": "G", "members": ["R0", "R1"],
                               "template": {"anchor": "R0", "relative_positions": {
                                   "R1": {"dx_mm": 0, "dy_mm": 5}}}}],
            placement_constraints={"edge_locked": [{"ref": "R1", "edge": "bottom"}]}))
        self.assertEqual(m.footprints[1].bbox[3], m.outline_ymax - EDGE_LOCK_MARGIN)
        self.assertEqual(m.footprints[1].y - m.footprints[0].y, 5 * MM)
        random.seed(4)
        m.moveable_indices = [0]
        for _ in range(20):
            do_translate(m, 1.0, 10 * MM)
            self.assertEqual(m.footprints[1].bbox[3], m.outline_ymax - EDGE_LOCK_MARGIN)
            self.assertEqual(m.footprints[1].y - m.footprints[0].y, 5 * MM)


class TestClearance(unittest.TestCase):
    def _pair_model(self, gap_mm, back=False, th=False):
        m = make_model(n=2)
        a, b = m.footprints
        a.x, a.y = OX + 30 * MM, OY + 30 * MM
        b.x, b.y = a.x + 3 * MM + int(gap_mm * MM), a.y   # widths 3 mm
        if back:
            b.is_front = False
        if th:
            b.th_rects = [(-1_500_000, -MM, 1_500_000, MM)]
        return m

    def test_same_side_inflation(self):
        m = self._pair_model(2.0)
        self.assertEqual(CostState(m, quiet=True)._overlap_penalty, 0.0)
        apply(m, base_constraints(placement_constraints={
            "keep_out_zones": [{"ref": "R0", "clearance_mm": 5}]}))
        cs = CostState(m, quiet=True)
        # inflated R0 bbox overlaps R1 by 3 mm in x and the full R1 height (2 mm) in y
        self.assertAlmostEqual(cs._overlap_penalty, (3 * MM + 2 * MM) * OVERLAP_WEIGHT,
                               delta=OVERLAP_WEIGHT * 2)

    def test_layer_restriction(self):
        m = self._pair_model(2.0, back=True)
        apply(m, base_constraints(placement_constraints={
            "keep_out_zones": [{"ref": "R0", "clearance_mm": 5, "layers": ["F.Cu"]}]}))
        # other part is on the back: a front-only ring does not apply
        self.assertEqual(CostState(m, quiet=True)._overlap_penalty, 0.0)

    def test_both_sides_ring_hits_opposite_side_body(self):
        # back-side SMD part (no TH): a both-sides ring (no layers, or both
        # copper layers) counts its body, like the checker does
        for layers in (None, [], ["F.Cu", "B.Cu"], ["F.Cu", "In1.Cu", "B.Cu"]):
            m = self._pair_model(2.0, back=True)
            self.assertEqual(CostState(m, quiet=True)._overlap_penalty, 0.0)
            ko = {"ref": "R0", "clearance_mm": 5}
            if layers is not None:
                ko["layers"] = layers
            apply(m, base_constraints(placement_constraints={"keep_out_zones": [ko]}))
            cs = CostState(m, quiet=True)
            self.assertAlmostEqual(cs._overlap_penalty, (3 * MM + 2 * MM) * OVERLAP_WEIGHT,
                                   delta=OVERLAP_WEIGHT * 2, msg=str(layers))

    def test_single_side_ring(self):
        # front part with a B.Cu-only ring: counts back-side parts, not front ones
        m = self._pair_model(2.0, back=True)
        apply(m, base_constraints(placement_constraints={
            "keep_out_zones": [{"ref": "R0", "clearance_mm": 5, "layers": ["B.Cu"]}]}))
        self.assertGreater(CostState(m, quiet=True)._overlap_penalty, 0.0)
        m = self._pair_model(2.0, back=False)
        apply(m, base_constraints(placement_constraints={
            "keep_out_zones": [{"ref": "R0", "clearance_mm": 5, "layers": ["B.Cu"]}]}))
        self.assertEqual(CostState(m, quiet=True)._overlap_penalty, 0.0)

    def test_plain_th_overlap_kept_under_ring(self):
        # opposite-side TH overlap without any ring intrusion still counts
        m = self._pair_model(-2.0, back=True, th=True)   # bodies overlap 2 mm in x
        plain = CostState(m, quiet=True)._overlap_penalty
        self.assertGreater(plain, 0.0)
        apply(m, base_constraints(placement_constraints={
            "keep_out_zones": [{"ref": "R0", "clearance_mm": 1, "layers": ["F.Cu"]}]}))
        self.assertEqual(CostState(m, quiet=True)._overlap_penalty, plain)

    def test_incremental_scan_finds_ring_pairs(self):
        m = self._pair_model(10.0)
        apply(m, base_constraints(placement_constraints={
            "keep_out_zones": [{"ref": "R1", "clearance_mm": 4}]}))
        cs = CostState(m, quiet=True)
        self.assertEqual(cs._overlap_penalty, 0.0)
        # move R0 (no clearance of its own) into R1's ring but not its body
        m.footprints[0].x += 8 * MM
        new = cs.incremental_update({0})
        self.assertGreater(cs._overlap_penalty, 0.0)
        self.assertAlmostEqual(new, CostState(m, quiet=True).total_cost, delta=1.0)


class TestIncrementalCorrectness(unittest.TestCase):
    def _constrained_model(self):
        m = make_model(n=16, seed=11)
        apply(m, base_constraints(
            component_groups=[
                {"id": "s1", "name": "S1", "members": ["R0", "R3", "R5", "R7"],
                 "max_span_mm": 10},
                {"id": "s2", "name": "S2", "members": ["R5", "R8", "R9"], "max_span_mm": 6},
                {"id": "t", "name": "T", "members": ["R10", "R11"],
                 "template": {"anchor": "R10",
                              "relative_positions": {"R11": {"dx_mm": 4, "dy_mm": 0}}}}],
            net_priorities={"critical": {"nets": ["N1", "N4"], "weight": 5.0},
                            "high": {"nets": ["N6", "BUS/CLK"], "weight": 2.0}},
            placement_constraints={
                "edge_locked": [{"ref": "R2", "edge": "left"}],
                "zone_preferences": [{"ref": "R0", "zone": "top-left"},
                                     {"ref": "R6", "zone": "bottom-right"},
                                     {"ref": "R9", "zone": "center"}],
                "keep_out_zones": [{"ref": "R4", "clearance_mm": 3},
                                   {"ref": "R12", "clearance_mm": 2, "layers": ["F.Cu"]}],
                "back_side": [{"ref": "R13"}]}))
        m.footprints[13].th_rects = [(-MM, -MM, MM, MM)]
        return m

    def test_random_moves_match_fresh_state(self):
        m = self._constrained_model()
        cs = CostState(m, quiet=True)
        self.assertTrue(cs._extra_terms)
        self.assertGreater(cs._max_clear, 0)
        random.seed(123)
        for step in range(400):
            mt = random.choice(["t", "m", "s", "r"])
            if mt == "t":
                undo = do_translate(m, 1.0, 15 * MM)
            elif mt == "m":
                undo = do_median(m, 0.5, 2 * MM)
            elif mt == "s":
                undo = do_swap(m)
                if undo is None:
                    continue
            else:
                undo = do_rotate(m)
            ai = affected_indices(undo)
            snap = cs.snapshot(ai)
            cost = cs.incremental_update(ai)
            ref = CostState(m, quiet=True)
            self.assertAlmostEqual(cost, ref.total_cost, delta=2.0, msg="step %d" % step)
            self.assertAlmostEqual(cs._cohesion_penalty, ref._cohesion_penalty, delta=2.0)
            self.assertAlmostEqual(cs._zone_penalty, ref._zone_penalty, delta=2.0)
            self.assertAlmostEqual(cs._hpwl_extra, ref._hpwl_extra, delta=2.0)
            self.assertEqual(cs.hpwl, ref.hpwl)
            if random.random() < 0.5:
                revert_move(m, undo)
                cs.restore(snap)
                ref = CostState(m, quiet=True)
                self.assertAlmostEqual(cs.total_cost, ref.total_cost, delta=2.0,
                                       msg="restore step %d" % step)
                self.assertAlmostEqual(cs._cohesion_penalty, ref._cohesion_penalty, delta=2.0)
                self.assertAlmostEqual(cs._zone_penalty, ref._zone_penalty, delta=2.0)

    def test_sa_runs_with_constraints(self):
        m = self._constrained_model()
        random.seed(7)
        cfg = SAConfig(max_iterations=15, reheat_count=1, scatter_first_start=True)
        result = run_sa(m, config=cfg)
        self.assertGreater(result.total_moves, 0)
        f = m.footprints[2]
        self.assertEqual(f.bbox[0], m.outline_xmin + EDGE_LOCK_MARGIN)
        # rigid template: distance anchor to member preserved (rotation may turn it)
        d = math.hypot(m.footprints[11].x - m.footprints[10].x,
                       m.footprints[11].y - m.footprints[10].y)
        self.assertAlmostEqual(d, 4 * MM, delta=10)
        self.assertTrue(m.footprints[13].is_front is False)


class TestZeroOverhead(unittest.TestCase):
    def test_plain_model_has_no_extra_state(self):
        m = make_model()
        cs = CostState(m, quiet=True)
        self.assertFalse(cs._extra_terms)
        self.assertFalse(cs._net_w)
        self.assertEqual(cs._max_clear, 0)
        self.assertNotIn("_compute_pair_overlap", cs.__dict__)
        self.assertIsNone(m.edge_locked_indices)
        # total cost is exactly the pre-constraint formula
        expected = (float(cs.hpwl) + cs._overlap_penalty + cs._boundary_penalty
                    + cs._keepout_penalty + cs._congestion_penalty)
        self.assertEqual(cs.normalized_cost, expected)
        snap = cs.snapshot({0})
        self.assertNotIn("extra", snap)
        self.assertNotIn("hpwl_extra", snap)

    def test_empty_constraint_file_changes_nothing(self):
        m = make_model()
        before = copy.deepcopy(m)
        apply(m, base_constraints())
        self.assertEqual([(f.x, f.y, f.locked) for f in m.footprints],
                         [(f.x, f.y, f.locked) for f in before.footprints])
        self.assertEqual(m.moveable_indices, before.moveable_indices)
        self.assertEqual(fresh_cost(m), fresh_cost(before))

    def test_old_pickle_without_new_attributes(self):
        # Simulate a pickle from before the constraint fields: drop instance attrs
        m = make_model()
        for f in m.footprints:
            for k in ("edge_lock", "clearance", "clearance_side"):
                f.__dict__.pop(k, None)
        for n in m.nets.values():
            n.__dict__.pop("weight", None)
        for k in ("edge_locked_indices", "soft_groups", "zone_prefs"):
            m.__dict__.pop(k, None)
        cs = CostState(m, quiet=True)
        self.assertFalse(cs._extra_terms)
        random.seed(1)
        do_translate(m, 1.0, 5 * MM)
        self.assertIsNone(do_swap(make_model(n=1)))


class TestScatterFirstStart(unittest.TestCase):
    def test_flag_scatters_start_zero(self):
        calls = []
        orig = annealer._scatter_positions

        def spy(model):
            calls.append(1)
            orig(model)
        annealer._scatter_positions = spy
        try:
            for flag, expected in ((False, 0), (True, 1)):
                calls.clear()
                m = make_model()
                random.seed(1)
                run_sa(m, config=SAConfig(max_iterations=3, reheat_count=0,
                                          num_starts=1, scatter_first_start=flag))
                self.assertEqual(len(calls), expected)
        finally:
            annealer._scatter_positions = orig


if __name__ == '__main__':
    unittest.main()
