"""Antenna window record (plugin/antenna_window.py): where the antenna strip
of an antenna_outward part lies relative to the real outline polygon after
the module end has been snapped to the outer envelope of the locked edge.
A measurement only; the engine is not changed by it.

Synthetic 75 x 75 mm boards, ESP32-WROOM-like module (pad-centre width
17.2 mm, module end 7.49 mm beyond the pads, strip depth 6 mm). Each case
states the expected classification; the broken case shows that the bbox
(instead of the polygon) would call an antenna window "material".

Run with: python3 tests/test_antenna_window.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugin.constraints import apply_constraints, resolve_antenna_item  # noqa: E402
from plugin.moves import resnap_edges, _edge_snap_polygon  # noqa: E402
from plugin.antenna_window import antenna_window, material_fraction, summary_line  # noqa: E402

from test_antenna_keepout import module_fp  # noqa: E402
from test_antenna_outward import board, cset, BW, BH  # noqa: E402

MM = 1_000_000


def top_notch(a, b, d):
    """Top edge with one notch from x=a to x=b, d mm deep."""
    return [(0, 0), (a * MM, 0), (a * MM, d * MM), (b * MM, d * MM), (b * MM, 0),
            (BW, 0), (BW, BH), (0, BH)]


def stepped_top():
    """Top edge at y=0 for x<10, y=8 for 10<x<30, y=10 beyond x=30."""
    return [(0, 0), (10 * MM, 0), (10 * MM, 8 * MM), (30 * MM, 8 * MM),
            (30 * MM, 10 * MM), (BW, 10 * MM), (BW, BH), (0, BH)]


def place(x, poly, depth=6.0, back=False, edge="top"):
    u = module_fp(x=x * MM, y=37 * MM, angle=180.0 if back else 0.0)
    if back:
        u.flip_side()
    m = board(u)
    m.outline_polygon = poly
    c = cset(edge)
    ko = c.placement["keep_out_zones"][0]
    if depth is None:
        ko["antenna"].pop("depth_mm", None)
    else:
        ko["antenna"]["depth_mm"] = depth
    apply_constraints(m, c, log=None)
    f = m.footprints[0]
    if edge in ("top", "bottom"):
        f.x = x * MM
    else:
        f.y = x * MM
    resnap_edges(m, [0])
    res = resolve_antenna_item(f, ko)
    outline = (m.outline_xmin, m.outline_ymin, m.outline_xmax, m.outline_ymax)
    w = antenna_window(f, res, edge, _edge_snap_polygon(m), outline)
    return m, f, w


class TestClassification(unittest.TestCase):
    def test_window_notch_wider_than_module(self):
        # Opsteker U1 case: notch 30..50 (20 mm, 8 deep), module strip 31.4..48.6
        _m, _f, w = place(40, top_notch(30, 50, 8))
        self.assertEqual(w.classification, "window")
        self.assertEqual(w.status, "pass")
        self.assertAlmostEqual(w.material_fraction, 0.0, places=6)
        self.assertFalse(w.stepped)
        self.assertTrue(w.notch)
        self.assertAlmostEqual(w.module_end_nm, w.envelope_nm, delta=2)
        self.assertAlmostEqual(w.strip_area_mm2, 17.2 * 6, places=3)
        self.assertIn("RF clearance not assessed", summary_line(w))

    def test_plain_straight_edge_is_material(self):
        _m, _f, w = place(40, None)
        self.assertEqual(w.classification, "material")
        self.assertEqual(w.status, "pass")
        self.assertAlmostEqual(w.material_fraction, 1.0, places=6)

    def test_notch_for_another_part_elsewhere_on_the_edge(self):
        # notch 5..15 is not under the module: the strip is over material
        _m, _f, w = place(50, top_notch(5, 15, 8))
        self.assertEqual(w.classification, "material")

    def test_off_centre_notch_module_straddles_the_wall(self):
        # strip 23.4..40.6 over a notch 30..50: 10.6 of 17.2 mm over void
        _m, _f, w = place(32, top_notch(30, 50, 8))
        self.assertEqual(w.classification, "mixed")
        self.assertEqual(w.status, "warn")
        self.assertAlmostEqual(w.material_fraction, 6.6 / 17.2, places=3)
        self.assertTrue(w.stepped)

    def test_notch_narrower_than_module(self):
        _m, _f, w = place(40, top_notch(34, 46, 8))
        self.assertEqual(w.classification, "mixed")
        self.assertAlmostEqual(w.material_fraction, (17.2 - 12) / 17.2, places=3)

    def test_shallow_notch_is_mixed_in_depth(self):
        # notch only 3 mm deep: the inner 3 mm of the 6 mm strip is over material
        _m, _f, w = place(40, top_notch(30, 50, 3))
        self.assertEqual(w.classification, "mixed")
        self.assertAlmostEqual(w.material_fraction, 0.5, places=3)
        self.assertFalse(w.stepped)

    def test_stepped_edge_warns(self):
        # module x 21.4..38.6 straddles the 8 / 10 mm step; the anchor is on
        # the outermost level (y 0, from x < 10), so the strip is all void
        _m, _f, w = place(30, stepped_top())
        self.assertEqual(w.classification, "stepped")
        self.assertEqual(w.status, "warn")
        self.assertTrue(w.stepped)
        self.assertEqual(len(w.levels_nm), 2)
        self.assertAlmostEqual(w.material_fraction, 0.0, places=6)

    def test_overhang_past_a_lower_step_is_not_a_window(self):
        # module x 41.4..58.6 over the y=10 level only; the edge comes back out
        # on the left (8, 0) but not on the right: a step, not a notch
        _m, _f, w = place(50, stepped_top())
        self.assertFalse(w.stepped)
        self.assertFalse(w.notch)
        self.assertEqual(w.classification, "overhang")
        self.assertEqual(w.status, "warn")

    def test_back_side_mirrored_module(self):
        _m, f, w = place(40, top_notch(30, 50, 8), back=True)
        self.assertFalse(f.is_front)
        self.assertEqual(f.antenna_side, "+y")
        self.assertEqual(w.classification, "window")
        self.assertAlmostEqual(w.module_end_nm, w.envelope_nm, delta=2)

    def test_left_edge_window(self):
        poly = [(0, 0), (BW, 0), (BW, BH), (0, BH), (0, 50 * MM), (8 * MM, 50 * MM),
                (8 * MM, 30 * MM), (0, 30 * MM)]
        _m, _f, w = place(40, poly, edge="left")
        self.assertEqual(w.classification, "window")

    def test_no_depth_takes_pads_to_module_end(self):
        _m, _f, w = place(40, top_notch(30, 50, 8), depth=None)
        self.assertEqual(w.strip_source, "pad_edge_to_module_end")
        self.assertAlmostEqual(w.strip_area_mm2, 17.2 * 7.49, places=2)
        self.assertEqual(w.classification, "window")


class TestBrokenCase(unittest.TestCase):
    def test_bbox_instead_of_polygon_would_call_the_window_material(self):
        _m, _f, w = place(40, top_notch(30, 50, 8))
        bbox = [(0, 0), (BW, 0), (BW, BH), (0, BH)]
        self.assertAlmostEqual(material_fraction(w.strip_world, bbox), 1.0, places=6)
        self.assertEqual(w.classification, "window")

    def test_material_fraction_of_a_concave_polygon(self):
        # a U-shaped outline clipped by a rect across both arms and the gap
        poly = [(0, 0), (10, 0), (10, 10), (20, 10), (20, 0), (30, 0), (30, 30), (0, 30)]
        self.assertAlmostEqual(material_fraction((0, 0, 30, 10), poly), 2.0 / 3.0, places=9)
        self.assertAlmostEqual(material_fraction((12, 2, 18, 8), poly), 0.0, places=9)


if __name__ == '__main__':
    unittest.main()
