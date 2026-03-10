"""Unit tests for the optimizer core — no KiCad dependency."""
import sys
import os
import unittest
import random

# Add project root to path so we can import the plugin package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugin.board_model import BoardModel, Footprint, Pad, Net, KeepOut, ComponentGroup
from plugin.cost_function import CostState, point_in_polygon
from plugin.moves import do_translate, do_swap, do_rotate, revert_move, affected_indices
from plugin.annealer import run_sa, SAConfig
from plugin.silkscreen import TextRect, SilkscreenModel, place_silkscreen, SILK_CLEARANCE

MM = 1_000_000  # nanometers per millimeter


def make_test_model() -> BoardModel:
    """
    Create a minimal 4-component test model.

    U1 at (10mm, 10mm) — pads on NET1 and NET2
    U2 at (40mm, 10mm) — pads on NET1 and NET3
    R1 at (10mm, 30mm) — pads on NET2 and NET3
    R2 at (40mm, 30mm) — pad on NET2 (other pad unconnected)

    Board outline: 0 to 50mm x 0 to 40mm
    """
    fp0 = Footprint(
        reference="U1", index=0, x=10*MM, y=10*MM,
        angle_deg=0.0, width=4*MM, height=5*MM, locked=False,
        pads=[
            Pad(net_code=1, net_name="NET1", offset_x=-1*MM, offset_y=0),
            Pad(net_code=2, net_name="NET2", offset_x=1*MM, offset_y=0),
        ],
        net_codes={1, 2},
    )
    fp1 = Footprint(
        reference="U2", index=1, x=40*MM, y=10*MM,
        angle_deg=0.0, width=4*MM, height=5*MM, locked=False,
        pads=[
            Pad(net_code=1, net_name="NET1", offset_x=-1*MM, offset_y=0),
            Pad(net_code=3, net_name="NET3", offset_x=1*MM, offset_y=0),
        ],
        net_codes={1, 3},
    )
    fp2 = Footprint(
        reference="R1", index=2, x=10*MM, y=30*MM,
        angle_deg=0.0, width=2*MM, height=1*MM, locked=False,
        pads=[
            Pad(net_code=2, net_name="NET2", offset_x=-500_000, offset_y=0),
            Pad(net_code=3, net_name="NET3", offset_x=500_000, offset_y=0),
        ],
        net_codes={2, 3},
    )
    fp3 = Footprint(
        reference="R2", index=3, x=40*MM, y=30*MM,
        angle_deg=0.0, width=2*MM, height=1*MM, locked=False,
        pads=[
            Pad(net_code=2, net_name="NET2", offset_x=-500_000, offset_y=0),
            Pad(net_code=0, net_name="", offset_x=500_000, offset_y=0),
        ],
        net_codes={2},
    )

    nets = {
        1: Net(net_code=1, net_name="NET1", pad_refs=[(0, 0), (1, 0)]),
        2: Net(net_code=2, net_name="NET2", pad_refs=[(0, 1), (2, 0), (3, 0)]),
        3: Net(net_code=3, net_name="NET3", pad_refs=[(1, 1), (2, 1)]),
    }

    return BoardModel(
        footprints=[fp0, fp1, fp2, fp3],
        nets=nets,
        outline_xmin=0, outline_ymin=0,
        outline_xmax=50*MM, outline_ymax=40*MM,
        moveable_indices=[0, 1, 2, 3],
    )


class TestPad(unittest.TestCase):
    def test_abs_position_no_rotation(self):
        pad = Pad(net_code=1, net_name="N", offset_x=1*MM, offset_y=2*MM)
        ax, ay = pad.abs_position(10*MM, 20*MM, 0.0)
        self.assertEqual(ax, 11*MM)
        self.assertEqual(ay, 22*MM)

    def test_abs_position_90_degrees(self):
        pad = Pad(net_code=1, net_name="N", offset_x=1*MM, offset_y=0)
        ax, ay = pad.abs_position(10*MM, 20*MM, 90.0)
        # KiCad uses CW rotation (Y-down): (1, 0) at 90° → (0, -1)
        self.assertAlmostEqual(ax, 10*MM, delta=100)
        self.assertAlmostEqual(ay, 19*MM, delta=100)


class TestCostFunction(unittest.TestCase):
    def test_hpwl_computation(self):
        model = make_test_model()
        cs = CostState(model)
        # NET1: pads at (9mm,10mm) and (39mm,10mm) → HPWL = 30mm
        # NET2: pads at (11mm,10mm), (9.5mm,30mm), (39.5mm,30mm)
        #       → x: 39.5-9.5=30mm, y: 30-10=20mm → HPWL = 50mm
        # NET3: pads at (41mm,10mm), (10.5mm,30mm)
        #       → x: 41-10.5=30.5mm, y: 30-10=20mm → HPWL = 50.5mm
        # Total ≈ 130.5mm
        expected_nm = 130_500_000
        self.assertAlmostEqual(cs.hpwl, expected_nm, delta=1_000_000)

    def test_no_penalties_when_well_separated(self):
        model = make_test_model()
        cs = CostState(model)
        # Components are well-separated and inside boundary
        self.assertAlmostEqual(cs.total_cost, float(cs.hpwl), delta=1.0)

    def test_boundary_penalty_when_outside(self):
        model = make_test_model()
        # Move a footprint outside the board
        model.footprints[0].x = -10 * MM
        cs = CostState(model)
        self.assertGreater(cs.total_cost, float(cs.hpwl))

    def test_overlap_penalty(self):
        model = make_test_model()
        # Stack two footprints on top of each other
        model.footprints[1].x = model.footprints[0].x
        model.footprints[1].y = model.footprints[0].y
        cs = CostState(model)
        self.assertGreater(cs.total_cost, float(cs.hpwl))

    def test_incremental_matches_full(self):
        model = make_test_model()
        cs = CostState(model)
        # Move fp0 and do incremental update
        model.footprints[0].x += 5 * MM
        new_cost = cs.incremental_update({0})
        # Verify against full recompute
        cs_full = CostState(model)
        self.assertAlmostEqual(new_cost, cs_full.total_cost, delta=1.0)


class TestMoves(unittest.TestCase):
    def test_translate_and_revert(self):
        model = make_test_model()
        random.seed(42)
        fp = model.footprints[0]
        orig_x, orig_y = fp.x, fp.y
        undo = do_translate(model, 0.5, 5_000_000)
        # Position should have changed (with high probability)
        revert_move(model, undo)
        self.assertEqual(fp.x, orig_x)
        self.assertEqual(fp.y, orig_y)

    def test_swap_and_revert(self):
        model = make_test_model()
        random.seed(42)
        positions_before = [(fp.x, fp.y) for fp in model.footprints]
        undo = do_swap(model)
        self.assertIsNotNone(undo)
        revert_move(model, undo)
        positions_after = [(fp.x, fp.y) for fp in model.footprints]
        self.assertEqual(positions_before, positions_after)

    def test_rotate_and_revert(self):
        model = make_test_model()
        random.seed(42)
        fp = model.footprints[0]
        orig_angle = fp.angle_deg
        undo = do_rotate(model)
        self.assertNotEqual(fp.angle_deg, orig_angle)
        revert_move(model, undo)
        self.assertEqual(fp.angle_deg, orig_angle)

    def test_affected_indices_translate(self):
        model = make_test_model()
        random.seed(42)
        undo = do_translate(model, 0.5, 5_000_000)
        indices = affected_indices(undo)
        self.assertEqual(len(indices), 1)
        self.assertIn(undo.fp_index, indices)

    def test_affected_indices_swap(self):
        model = make_test_model()
        random.seed(42)
        undo = do_swap(model)
        indices = affected_indices(undo)
        self.assertEqual(len(indices), 2)


    def test_keepout_penalty(self):
        model = make_test_model()
        # Add a keep-out zone that overlaps with fp0 (at 10mm, 10mm)
        model.keepouts = [KeepOut(
            xmin=8*MM, ymin=8*MM, xmax=12*MM, ymax=12*MM
        )]
        cs = CostState(model)
        self.assertGreater(cs._keepout_penalty, 0.0)
        self.assertGreater(cs.total_cost, float(cs.hpwl))

    def test_no_keepout_penalty_when_clear(self):
        model = make_test_model()
        # Keep-out zone far from all footprints
        model.keepouts = [KeepOut(
            xmin=0, ymin=0, xmax=1*MM, ymax=1*MM
        )]
        cs = CostState(model)
        self.assertAlmostEqual(cs._keepout_penalty, 0.0, delta=1.0)


class TestAnnealer(unittest.TestCase):
    def test_sa_reduces_cost(self):
        """SA should reduce HPWL on a deliberately bad initial placement."""
        model = make_test_model()
        # Scramble positions to create a clearly suboptimal layout
        model.footprints[0].x = 45 * MM
        model.footprints[0].y = 35 * MM
        model.footprints[1].x = 5 * MM
        model.footprints[1].y = 5 * MM
        model.footprints[2].x = 45 * MM
        model.footprints[2].y = 5 * MM
        model.footprints[3].x = 5 * MM
        model.footprints[3].y = 35 * MM

        initial_cs = CostState(model)
        initial_cost = initial_cs.total_cost

        config = SAConfig(
            calibration_samples=50,
            max_iterations=50,
            moves_per_temp=40,
        )
        result = run_sa(model, config)

        self.assertLess(result.best_cost, initial_cost,
                        "SA should reduce cost from a scrambled placement")
        self.assertGreater(result.improvement_pct, 0)

    def test_sa_returns_valid_result(self):
        model = make_test_model()
        config = SAConfig(max_iterations=10, moves_per_temp=20)
        result = run_sa(model, config)
        self.assertGreater(result.total_moves, 0)
        self.assertGreaterEqual(result.accepted_moves, 0)
        self.assertGreater(len(result.cost_history), 0)


def make_radar_afe_model() -> BoardModel:
    """
    Simplified model based on the radar-afe board:
    - 12 representative components (mix of 0603, 0805, SOIC-8, SOT-23-6, D_SMA, inductor, pot)
    - Deliberately placed in suboptimal positions to test optimizer
    - 1 keep-out zone at top-center (radar module exclusion)
    - Board outline: 95-145.5mm x 95-145.8mm
    """
    # Component bbox sizes (from pad extents + margin, in nm)
    # Using sizes that match what extract_board_model would compute with pad physical sizes:
    # 0603:   pads at ±0.825mm, pad size ~0.95mm → pad extent 2.6mm x 0.95mm + 1mm margin ≈ 3.6mm x 2.0mm
    # 0805:   pads at ±0.95mm, pad size ~1.25mm → pad extent 3.15mm x 1.25mm + 1mm margin ≈ 4.2mm x 2.3mm
    # SOIC-8: pads at x±2.475mm, y±1.905mm, pad 1.95x0.6mm → 6.9mm x 4.4mm + 1mm ≈ 7.9mm x 5.4mm
    # SOT-23-6: pads span 2.275mm x 1.9mm, pad 1.325x0.6mm → 3.6mm x 2.5mm + 1mm ≈ 4.6mm x 3.5mm

    W_0603 = int(3.6 * MM)
    H_0603 = int(2.0 * MM)
    W_0805 = int(4.2 * MM)
    H_0805 = int(2.3 * MM)
    W_SOIC8 = int(7.9 * MM)
    H_SOIC8 = int(5.4 * MM)
    W_SOT23 = int(4.6 * MM)
    H_SOT23 = int(3.5 * MM)

    # Board outline (nm)
    BX0 = int(95.0 * MM)
    BY0 = int(95.0 * MM)
    BX1 = int(145.5 * MM)
    BY1 = int(145.8 * MM)

    # Keep-out zone (radar module area)
    KO_X0 = int(109.9 * MM)
    KO_Y0 = int(95.6 * MM)
    KO_X1 = int(135.0 * MM)
    KO_Y1 = int(121.0 * MM)

    # Create footprints - deliberately scattered in suboptimal positions
    # Some placed inside the keep-out zone to test penalty
    footprints = []
    all_nets = {}
    net_code = 1

    def make_fp(ref, idx, x_mm, y_mm, w, h, locked=False, n_pads=2, angle=0.0):
        nonlocal net_code
        pads = []
        nc_set = set()
        # Create pads spread along x-axis
        pad_spacing = (w - MM) // max(n_pads - 1, 1) if n_pads > 1 else 0
        for pi in range(n_pads):
            nc = net_code
            net_code += 1
            if net_code > 10:
                net_code = 1  # reuse nets to create connections
            ox = -w // 4 + pi * pad_spacing // 2 if n_pads > 1 else 0
            p = Pad(net_code=nc, net_name=f"NET{nc}", offset_x=ox, offset_y=0)
            pads.append(p)
            nc_set.add(nc)
            if nc not in all_nets:
                all_nets[nc] = Net(net_code=nc, net_name=f"NET{nc}")
            all_nets[nc].pad_refs.append((idx, pi))
        return Footprint(
            reference=ref, index=idx, x=int(x_mm * MM), y=int(y_mm * MM),
            angle_deg=angle, width=w, height=h, locked=locked,
            pads=pads, net_codes=nc_set,
        )

    # 4 locked connectors (at edges)
    footprints.append(make_fp("J1", 0, 117.5, 98.0, 14*MM, 3*MM, locked=True))
    footprints.append(make_fp("J2", 1, 139.8, 134.5, 6*MM, 4*MM, locked=True))
    footprints.append(make_fp("J3", 2, 116.5, 133.1, 3*MM, 5*MM, locked=True))
    footprints.append(make_fp("J4", 3, 116.5, 128.1, 3*MM, 5*MM, locked=True))

    # 8 moveable components - some deliberately inside keep-out zone
    footprints.append(make_fp("U1", 4, 115.0, 110.0, W_SOT23, H_SOT23))  # INSIDE keepout!
    footprints.append(make_fp("U2", 5, 125.0, 108.0, W_SOIC8, H_SOIC8))  # INSIDE keepout!
    footprints.append(make_fp("R1", 6, 120.0, 115.0, W_0603, H_0603))     # INSIDE keepout!
    footprints.append(make_fp("R2", 7, 130.0, 140.0, W_0603, H_0603))
    footprints.append(make_fp("C1", 8, 100.0, 130.0, W_0805, H_0805))
    footprints.append(make_fp("C2", 9, 140.0, 125.0, W_0805, H_0805))
    footprints.append(make_fp("R3", 10, 105.0, 140.0, W_0603, H_0603))
    footprints.append(make_fp("U3", 11, 127.0, 140.0, W_SOIC8, H_SOIC8))

    moveable = [i for i, fp in enumerate(footprints) if not fp.locked]

    return BoardModel(
        footprints=footprints,
        nets=all_nets,
        outline_xmin=BX0, outline_ymin=BY0,
        outline_xmax=BX1, outline_ymax=BY1,
        keepouts=[KeepOut(xmin=KO_X0, ymin=KO_Y0, xmax=KO_X1, ymax=KO_Y1)],
        moveable_indices=moveable,
    )


class TestRadarAfe(unittest.TestCase):
    """Tests with a radar-afe-like board model."""

    def test_keepout_violation_detected(self):
        """Initial placement has components in keep-out zone — penalty should be nonzero."""
        model = make_radar_afe_model()
        cs = CostState(model)
        self.assertGreater(cs._keepout_penalty, 0.0,
                           "Components inside keep-out should produce penalty")

    def test_sa_clears_keepout(self):
        """SA should move components out of the keep-out zone."""
        model = make_radar_afe_model()
        config = SAConfig(
            calibration_samples=100,
            max_iterations=100,
            moves_per_temp=80,
        )
        result = run_sa(model, config)

        # After optimization, check for keep-out violations
        final_cs = CostState(model)
        self.assertAlmostEqual(final_cs._keepout_penalty, 0.0, delta=1.0,
                               msg="SA should clear keep-out violations")

    def test_sa_no_overlaps(self):
        """SA should produce a placement with no overlapping footprints."""
        model = make_radar_afe_model()
        config = SAConfig(
            calibration_samples=100,
            max_iterations=100,
            moves_per_temp=80,
        )
        run_sa(model, config)

        # Check for overlaps
        final_cs = CostState(model)
        self.assertAlmostEqual(final_cs._overlap_penalty, 0.0, delta=1.0,
                               msg="SA should eliminate overlaps")

    def test_sa_all_inside_boundary(self):
        """SA should keep all components inside the board boundary."""
        model = make_radar_afe_model()
        config = SAConfig(
            calibration_samples=100,
            max_iterations=150,
            moves_per_temp=120,
        )
        run_sa(model, config)

        final_cs = CostState(model)
        self.assertAlmostEqual(final_cs._boundary_penalty, 0.0, delta=1.0,
                               msg="SA should keep components inside boundary")


def make_silk_model(texts, fp_bboxes, keepouts=None, board_bbox=None):
    """Helper to build a SilkscreenModel for testing."""
    if keepouts is None:
        keepouts = []
    if board_bbox is None:
        board_bbox = (0, 0, 100*MM, 100*MM)
    return SilkscreenModel(
        texts=texts, fp_bboxes=fp_bboxes,
        keepouts=keepouts, board_bbox=board_bbox,
    )


class TestSilkscreen(unittest.TestCase):

    def test_empty_model(self):
        model = make_silk_model(texts=[], fp_bboxes=[])
        result = place_silkscreen(model)
        self.assertEqual(result, [])

    def test_no_collision_keeps_position(self):
        """Text already in a clear area — should stay at original position."""
        # Footprint at center, text above it, nothing else nearby
        fp_bbox = (40*MM, 40*MM, 50*MM, 50*MM)
        text = TextRect(fp_index=0, cx=45*MM, cy=38*MM,
                        width=4*MM, height=1*MM)
        model = make_silk_model(texts=[text], fp_bboxes=[fp_bbox])
        result = place_silkscreen(model)
        # The text was already clear, but the algorithm always picks
        # the best candidate. The top candidate would be very close to
        # the original. Either way, no collision.
        cx, cy = result[0]
        # Verify the result doesn't collide with the footprint
        from plugin.silkscreen import _overlaps, _candidate_bbox
        cbox = _candidate_bbox(cx, cy, text.width, text.height)
        self.assertFalse(_overlaps(cbox, fp_bbox))

    def test_collision_moves_to_candidate(self):
        """Text overlapping an obstacle gets moved to a free candidate."""
        # Two footprints side by side: fp0 at left, fp1 at right
        fp0_bbox = (10*MM, 10*MM, 20*MM, 20*MM)
        fp1_bbox = (22*MM, 10*MM, 32*MM, 20*MM)
        # Text for fp0, currently overlapping fp1
        text = TextRect(fp_index=0, cx=21*MM, cy=15*MM,
                        width=4*MM, height=1*MM)
        model = make_silk_model(texts=[text], fp_bboxes=[fp0_bbox, fp1_bbox])
        result = place_silkscreen(model)
        cx, cy = result[0]
        # Should NOT overlap fp1 anymore
        from plugin.silkscreen import _overlaps, _candidate_bbox
        cbox = _candidate_bbox(cx, cy, text.width, text.height)
        self.assertFalse(_overlaps(cbox, fp1_bbox))

    def test_all_candidates_blocked_uses_fallback(self):
        """No free candidate — uses least-overlap fallback."""
        # Footprint surrounded by obstacles on all 8 sides + far distances
        center_bbox = (50*MM, 50*MM, 55*MM, 55*MM)
        obstacles = [
            center_bbox,
            (46*MM, 40*MM, 59*MM, 49*MM),  # top (wide+tall)
            (56*MM, 40*MM, 66*MM, 65*MM),  # right (covers NE and SE)
            (46*MM, 56*MM, 59*MM, 65*MM),  # bottom (wide+tall)
            (39*MM, 40*MM, 49*MM, 65*MM),  # left (covers NW and SW)
        ]
        text = TextRect(fp_index=0, cx=52*MM, cy=52*MM,
                        width=4*MM, height=1*MM)
        model = make_silk_model(texts=[text], fp_bboxes=obstacles)
        result = place_silkscreen(model)
        # All candidates blocked — should still return a position
        # (the least-overlap fallback, not necessarily original)
        self.assertIsNotNone(result[0])
        self.assertEqual(len(result), 1)

    def test_respects_board_boundary(self):
        """Candidates outside board boundary are skipped."""
        # Footprint near top edge — top candidate would be outside board
        fp_bbox = (45*MM, 0, 55*MM, 5*MM)
        text = TextRect(fp_index=0, cx=50*MM, cy=2*MM,
                        width=4*MM, height=1*MM)
        model = make_silk_model(
            texts=[text], fp_bboxes=[fp_bbox],
            board_bbox=(0, 0, 100*MM, 100*MM),
        )
        result = place_silkscreen(model)
        cx, cy = result[0]
        # Result must be inside board
        from plugin.silkscreen import _inside, _candidate_bbox
        cbox = _candidate_bbox(cx, cy, text.width, text.height)
        self.assertTrue(_inside(cbox, model.board_bbox))

    def test_respects_keepout(self):
        """Candidates overlapping keep-out zones are skipped."""
        fp_bbox = (50*MM, 50*MM, 55*MM, 55*MM)
        # Keep-out zone above the footprint — blocks the top candidate
        keepout = (48*MM, 44*MM, 57*MM, 50*MM)
        text = TextRect(fp_index=0, cx=52*MM, cy=52*MM,
                        width=4*MM, height=1*MM)
        model = make_silk_model(
            texts=[text], fp_bboxes=[fp_bbox], keepouts=[keepout],
        )
        result = place_silkscreen(model)
        cx, cy = result[0]
        # Result must not overlap the keepout
        from plugin.silkscreen import _overlaps, _candidate_bbox
        cbox = _candidate_bbox(cx, cy, text.width, text.height)
        self.assertFalse(_overlaps(cbox, keepout))

    def test_placed_texts_become_obstacles(self):
        """A text placed first blocks candidates for later texts."""
        # Two footprints stacked vertically, sharing the same right-side space
        fp0_bbox = (10*MM, 10*MM, 20*MM, 15*MM)
        fp1_bbox = (10*MM, 16*MM, 20*MM, 21*MM)
        text0 = TextRect(fp_index=0, cx=15*MM, cy=12*MM,
                         width=6*MM, height=1*MM)
        text1 = TextRect(fp_index=1, cx=15*MM, cy=18*MM,
                         width=6*MM, height=1*MM)
        model = make_silk_model(texts=[text0, text1],
                                fp_bboxes=[fp0_bbox, fp1_bbox])
        result = place_silkscreen(model)
        # Both texts should have been placed without overlapping each other
        from plugin.silkscreen import _overlaps, _candidate_bbox
        box0 = _candidate_bbox(result[0][0], result[0][1],
                               text0.width, text0.height)
        box1 = _candidate_bbox(result[1][0], result[1][1],
                               text1.width, text1.height)
        self.assertFalse(_overlaps(box0, box1))

    def test_density_ordering(self):
        """Most constrained refs are processed first (get priority)."""
        # fp0 is surrounded by many obstacles (high density)
        # fp1 is isolated (low density)
        fp0_bbox = (50*MM, 50*MM, 55*MM, 55*MM)
        fp1_bbox = (10*MM, 10*MM, 15*MM, 15*MM)
        nearby = [(56*MM, 50*MM, 61*MM, 55*MM),
                  (44*MM, 50*MM, 49*MM, 55*MM)]
        all_bboxes = [fp0_bbox, fp1_bbox] + nearby
        text0 = TextRect(fp_index=0, cx=52*MM, cy=52*MM,
                         width=3*MM, height=1*MM)
        text1 = TextRect(fp_index=1, cx=12*MM, cy=12*MM,
                         width=3*MM, height=1*MM)
        model = make_silk_model(texts=[text0, text1], fp_bboxes=all_bboxes)
        result = place_silkscreen(model)
        # Both should get valid positions (test doesn't crash,
        # and constrained fp0 still finds a spot)
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])

    def test_diagonal_candidate_used(self):
        """When cardinal directions are blocked, diagonal candidates are used."""
        # Footprint with obstacles blocking 4 cardinal directions but
        # leaving diagonal corners clear. Obstacles are narrow bands
        # centered on each side.
        center = (50*MM, 50*MM, 55*MM, 55*MM)
        obstacles = [
            center,
            (51*MM, 44*MM, 54*MM, 49*MM),  # top — narrow, blocks top center
            (56*MM, 51*MM, 61*MM, 54*MM),  # right — narrow, blocks right center
            (51*MM, 56*MM, 54*MM, 61*MM),  # bottom — narrow, blocks bottom center
            (44*MM, 51*MM, 49*MM, 54*MM),  # left — narrow, blocks left center
        ]
        text = TextRect(fp_index=0, cx=52*MM, cy=52*MM,
                        width=3*MM, height=1*MM)
        model = make_silk_model(texts=[text], fp_bboxes=obstacles)
        result = place_silkscreen(model)
        cx, cy = result[0]
        # Should have found a diagonal position (not original position)
        self.assertNotEqual((cx, cy), (text.cx, text.cy),
                           "Should find a diagonal candidate")
        # Verify no collision with any obstacle
        from plugin.silkscreen import _overlaps, _candidate_bbox
        cbox = _candidate_bbox(cx, cy, text.width, text.height)
        for i, ob in enumerate(obstacles):
            if i == 0:
                continue  # skip parent
            self.assertFalse(_overlaps(cbox, ob),
                           f"Should not overlap obstacle {i}")

    def test_text_to_text_clearance(self):
        """Placed texts have clearance buffer — nearby texts must keep distance."""
        # Two footprints close together horizontally
        fp0_bbox = (10*MM, 50*MM, 15*MM, 55*MM)
        fp1_bbox = (18*MM, 50*MM, 23*MM, 55*MM)
        text0 = TextRect(fp_index=0, cx=12*MM, cy=52*MM,
                         width=4*MM, height=1*MM)
        text1 = TextRect(fp_index=1, cx=20*MM, cy=52*MM,
                         width=4*MM, height=1*MM)
        model = make_silk_model(texts=[text0, text1],
                                fp_bboxes=[fp0_bbox, fp1_bbox])
        result = place_silkscreen(model)
        # Both texts must not overlap (even with clearance)
        from plugin.silkscreen import _overlaps, _candidate_bbox, _expand_bbox
        box0 = _candidate_bbox(result[0][0], result[0][1],
                               text0.width, text0.height)
        box1 = _candidate_bbox(result[1][0], result[1][1],
                               text1.width, text1.height)
        self.assertFalse(_overlaps(box0, box1),
                        "Text bboxes should not overlap")
        # Also check that there's clearance between them
        expanded0 = _expand_bbox(box0, SILK_CLEARANCE)
        expanded1 = _expand_bbox(box1, SILK_CLEARANCE)
        # At least one pair shouldn't overlap with clearance
        # (they should be placed at different sides)
        if _overlaps(expanded0, box1):
            # If box1 is within clearance of box0, they must be on different sides
            self.assertNotEqual(result[0][1], result[1][1],
                              "Texts at same Y should have clearance")


class TestPolygonBoundary(unittest.TestCase):
    """Tests for non-rectangular board outline support."""

    def test_point_in_polygon_rectangle(self):
        """Rectangle polygon — basic containment."""
        poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.assertTrue(point_in_polygon(50, 50, poly))
        self.assertFalse(point_in_polygon(150, 50, poly))
        self.assertFalse(point_in_polygon(-10, 50, poly))

    def test_point_in_polygon_corner_cut(self):
        """L-shaped polygon — point in cutaway returns False."""
        # Rectangle with top-left corner cut: board from (0,0) to (100,100)
        # but top-left corner (0,0)-(40,40) is removed.
        # Polygon vertices (clockwise):
        poly = [(40, 0), (100, 0), (100, 100), (0, 100), (0, 40), (40, 40)]
        # Point inside the main area
        self.assertTrue(point_in_polygon(50, 50, poly))
        # Point in the cutaway corner
        self.assertFalse(point_in_polygon(20, 20, poly))
        # Point near but inside the remaining top area
        self.assertTrue(point_in_polygon(60, 10, poly))
        # Point outside entirely
        self.assertFalse(point_in_polygon(150, 50, poly))

    def test_boundary_penalty_with_polygon(self):
        """Footprint inside bbox but outside polygon gets penalized."""
        # L-shaped board: rectangle with top-left corner cut
        poly_pts = [
            (40*MM, 0), (100*MM, 0), (100*MM, 100*MM),
            (0, 100*MM), (0, 40*MM), (40*MM, 40*MM),
        ]
        fp_in_cutaway = Footprint(
            reference="R1", index=0,
            x=20*MM, y=20*MM,  # in the cutaway
            angle_deg=0, width=3*MM, height=2*MM,
            locked=False, pads=[], net_codes=set(),
        )
        fp_inside = Footprint(
            reference="R2", index=1,
            x=60*MM, y=60*MM,  # clearly inside
            angle_deg=0, width=3*MM, height=2*MM,
            locked=False, pads=[], net_codes=set(),
        )
        model = BoardModel(
            footprints=[fp_in_cutaway, fp_inside],
            nets={},
            outline_xmin=0, outline_ymin=0,
            outline_xmax=100*MM, outline_ymax=100*MM,
            moveable_indices=[0, 1],
            outline_polygon=poly_pts,
        )
        cs = CostState(model)
        # Footprint in cutaway should produce penalty
        self.assertGreater(cs._boundary_penalty, 0.0,
                          "Footprint in cutaway should be penalized")

        # Move fp_in_cutaway to a valid position
        fp_in_cutaway.x = 60*MM
        fp_in_cutaway.y = 60*MM
        cs2 = CostState(model)
        # Both inside — no boundary penalty
        self.assertAlmostEqual(cs2._boundary_penalty, 0.0, delta=1.0)


def make_grouped_model() -> BoardModel:
    """Create a model with two grouped footprints (U1+C1) and two ungrouped (R1, R2)."""
    fp0 = Footprint(
        reference="U1", index=0, x=20*MM, y=20*MM,
        angle_deg=0.0, width=5*MM, height=5*MM, locked=False,
        pads=[Pad(net_code=1, net_name="NET1", offset_x=-2*MM, offset_y=0)],
        net_codes={1},
    )
    fp1 = Footprint(
        reference="C1", index=1, x=20*MM, y=15*MM,  # 5mm above U1
        angle_deg=0.0, width=2*MM, height=1*MM, locked=False,
        pads=[Pad(net_code=1, net_name="NET1", offset_x=0, offset_y=0)],
        net_codes={1},
    )
    fp2 = Footprint(
        reference="R1", index=2, x=40*MM, y=20*MM,
        angle_deg=0.0, width=2*MM, height=1*MM, locked=False,
        pads=[Pad(net_code=2, net_name="NET2", offset_x=0, offset_y=0)],
        net_codes={2},
    )
    fp3 = Footprint(
        reference="R2", index=3, x=40*MM, y=30*MM,
        angle_deg=0.0, width=2*MM, height=1*MM, locked=False,
        pads=[Pad(net_code=2, net_name="NET2", offset_x=0, offset_y=0)],
        net_codes={2},
    )

    nets = {
        1: Net(net_code=1, net_name="NET1", pad_refs=[(0, 0), (1, 0)]),
        2: Net(net_code=2, net_name="NET2", pad_refs=[(2, 0), (3, 0)]),
    }

    group = ComponentGroup(member_indices=[0, 1], locked=False)

    # moveable: group rep (0) + ungrouped (2, 3). Index 1 excluded (group non-rep)
    return BoardModel(
        footprints=[fp0, fp1, fp2, fp3],
        nets=nets,
        outline_xmin=0, outline_ymin=0,
        outline_xmax=60*MM, outline_ymax=50*MM,
        moveable_indices=[0, 2, 3],
        component_groups=[group],
        fp_to_group={0: 0, 1: 0},
    )


class TestGroupMoves(unittest.TestCase):
    """Tests for group-aware move operators."""

    def test_translate_group_moves_all_members(self):
        """Translating a grouped footprint moves all group members."""
        model = make_grouped_model()
        random.seed(42)
        fp0, fp1 = model.footprints[0], model.footprints[1]
        orig_dx = fp1.x - fp0.x
        orig_dy = fp1.y - fp0.y

        undo = do_translate(model, 0.5, 5_000_000)
        if undo.fp_index in (0, 1):
            # Both members should have moved by the same delta
            new_dx = fp1.x - fp0.x
            new_dy = fp1.y - fp0.y
            self.assertEqual(new_dx, orig_dx, "Group relative X offset preserved")
            self.assertEqual(new_dy, orig_dy, "Group relative Y offset preserved")
            # Both members should be in affected indices
            indices = affected_indices(undo)
            self.assertIn(0, indices)
            self.assertIn(1, indices)

    def test_translate_group_revert(self):
        """Reverting a group translate restores all member positions."""
        model = make_grouped_model()
        random.seed(42)
        orig_positions = [(fp.x, fp.y) for fp in model.footprints]

        # Force selecting a group member
        for _ in range(50):
            random.seed(random.randint(0, 10000))
            undo = do_translate(model, 0.5, 5_000_000)
            revert_move(model, undo)
            for i, fp in enumerate(model.footprints):
                self.assertEqual((fp.x, fp.y), orig_positions[i],
                               f"fp{i} position restored after revert")

    def test_rotate_group_preserves_relative_positions(self):
        """Rotating a group rotates all members around the group centroid."""
        model = make_grouped_model()
        fp0, fp1 = model.footprints[0], model.footprints[1]
        orig_dist = ((fp1.x - fp0.x)**2 + (fp1.y - fp0.y)**2) ** 0.5

        # Keep trying until we pick a group member
        for seed in range(100):
            random.seed(seed)
            # Reset positions
            fp0.x, fp0.y, fp0.angle_deg = 20*MM, 20*MM, 0.0
            fp1.x, fp1.y, fp1.angle_deg = 20*MM, 15*MM, 0.0
            undo = do_rotate(model)
            idx = undo.fp_index
            if idx in (0, 1):
                # Both should have been rotated
                new_dist = ((fp1.x - fp0.x)**2 + (fp1.y - fp0.y)**2) ** 0.5
                self.assertAlmostEqual(new_dist, orig_dist, delta=100,
                                      msg="Distance between group members preserved")
                # Both should have the same angle change
                self.assertEqual(fp0.angle_deg, fp1.angle_deg,
                               "Both group members rotated equally")
                revert_move(model, undo)
                break

    def test_rotate_group_revert(self):
        """Reverting a group rotation restores all members."""
        model = make_grouped_model()
        random.seed(7)
        orig = [(fp.x, fp.y, fp.angle_deg) for fp in model.footprints]

        undo = do_rotate(model)
        revert_move(model, undo)
        for i, fp in enumerate(model.footprints):
            self.assertEqual((fp.x, fp.y, fp.angle_deg), orig[i],
                           f"fp{i} restored after rotate revert")

    def test_swap_preserves_group_offsets(self):
        """Swap should preserve relative positions within a group."""
        model = make_grouped_model()
        fp0, fp1 = model.footprints[0], model.footprints[1]
        orig_dx = fp1.x - fp0.x
        orig_dy = fp1.y - fp0.y

        for seed in range(100):
            random.seed(seed)
            # Reset positions
            fp0.x, fp0.y = 20*MM, 20*MM
            fp1.x, fp1.y = 20*MM, 15*MM
            model.footprints[2].x, model.footprints[2].y = 40*MM, 20*MM
            model.footprints[3].x, model.footprints[3].y = 40*MM, 30*MM

            undo = do_swap(model)
            if undo is None:
                continue
            # Group members should maintain their relative offset
            new_dx = fp1.x - fp0.x
            new_dy = fp1.y - fp0.y
            self.assertEqual(new_dx, orig_dx,
                           f"Group X offset preserved (seed={seed})")
            self.assertEqual(new_dy, orig_dy,
                           f"Group Y offset preserved (seed={seed})")
            revert_move(model, undo)

    def test_group_excluded_from_moveable(self):
        """Non-representative group members should not be in moveable_indices."""
        model = make_grouped_model()
        # Index 0 is the group rep, index 1 is excluded
        self.assertIn(0, model.moveable_indices)
        self.assertNotIn(1, model.moveable_indices)
        # Ungrouped footprints still present
        self.assertIn(2, model.moveable_indices)
        self.assertIn(3, model.moveable_indices)

    def test_sa_preserves_group_offsets(self):
        """SA optimization should preserve relative positions within groups."""
        model = make_grouped_model()
        fp0, fp1 = model.footprints[0], model.footprints[1]
        orig_dx = fp1.x - fp0.x
        orig_dy = fp1.y - fp0.y

        config = SAConfig(
            calibration_samples=50,
            max_iterations=20,
            moves_per_temp=30,
        )
        run_sa(model, config)

        # After SA, the relative offset within the group should be preserved
        # (allowing for rotation — distance should be the same)
        orig_dist = (orig_dx**2 + orig_dy**2) ** 0.5
        new_dist = ((fp1.x - fp0.x)**2 + (fp1.y - fp0.y)**2) ** 0.5
        self.assertAlmostEqual(new_dist, orig_dist, delta=100,
                              msg="Group member distance preserved after SA")


if __name__ == '__main__':
    unittest.main()
