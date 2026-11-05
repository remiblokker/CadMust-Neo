"""Courtyard extraction in extract_board_model with a mocked pcbnew: a
polygon courtyard (fp_poly) returns (0, 0) from GetStart()/GetEnd() in
world coordinates and must contribute its outline vertices only.

Run with: python3 tests/test_extract_courtyard.py -v
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugin import board_model
from plugin.board_model import extract_board_model, _courtyard_points_world

MM = 1_000_000
PAD_MARGIN = 250_000


class P:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


def fake_pcbnew():
    m = types.ModuleType("pcbnew")
    for name, v in (("SHAPE_T_SEGMENT", 0), ("SHAPE_T_RECTANGLE", 1), ("SHAPE_T_ARC", 2),
                    ("SHAPE_T_CIRCLE", 3), ("SHAPE_T_POLY", 4), ("SHAPE_T_BEZIER", 5)):
        setattr(m, name, v)
    m.F_CrtYd, m.B_CrtYd, m.F_Fab, m.B_Fab = 100, 101, 102, 103
    return m


class Outline:
    def __init__(self, pts):
        self.pts = pts

    def PointCount(self):
        return len(self.pts)

    def CPoint(self, i):
        return self.pts[i]


class PolySet:
    def __init__(self, pts):
        self.ol = Outline(pts)

    def OutlineCount(self):
        return 1

    def Outline(self, i):
        return self.ol


class PolyItem:
    """fp_poly on F.CrtYd: GetStart/GetEnd give (0, 0) world, as KiCad does."""
    def __init__(self, pts, layer=100):
        self.pts = pts
        self.layer = layer

    def GetLayer(self):
        return self.layer

    def GetShape(self):
        return 4

    def GetStart(self):
        return P(0, 0)

    def GetEnd(self):
        return P(0, 0)

    def GetPolyShape(self):
        return PolySet(self.pts)


class SegItem:
    def __init__(self, a, b, layer=100, shape=0, mid=None):
        self.a, self.b, self.layer, self.shape, self.mid = a, b, layer, shape, mid

    def GetLayer(self):
        return self.layer

    def GetShape(self):
        return self.shape

    def GetStart(self):
        return self.a

    def GetEnd(self):
        return self.b

    def GetArcMid(self):
        return self.mid


class FakePad:
    def __init__(self, pos):
        self.pos = pos

    def GetPosition(self):
        return self.pos

    def GetNetCode(self):
        return 0

    def GetNetname(self):
        return ""

    def GetSize(self):
        return P(0.5 * MM, 0.5 * MM)

    def HasHole(self):
        return False


class FakeFp:
    def __init__(self, ref, pos, items):
        self.ref, self.pos, self.items = ref, pos, items

    def GetPosition(self):
        return self.pos

    def GetOrientationDegrees(self):
        return 0.0

    def Pads(self):
        return [FakePad(self.pos)]

    def GraphicalItems(self):
        return self.items

    def GetReference(self):
        return self.ref

    def IsLocked(self):
        return False

    def IsFlipped(self):
        return False


class BBox:
    def GetX(self):
        return 100 * MM

    def GetY(self):
        return 40 * MM

    def GetWidth(self):
        return 50 * MM

    def GetHeight(self):
        return 30 * MM


class FakeBoard:
    def __init__(self, fps):
        self.fps = fps

    def GetFootprints(self):
        return self.fps

    def GetBoardEdgesBoundingBox(self):
        return BBox()

    def Zones(self):
        return []

    def Groups(self):
        return []


class CourtyardPolygonTest(unittest.TestCase):
    def setUp(self):
        self._saved = sys.modules.get("pcbnew")
        sys.modules["pcbnew"] = fake_pcbnew()

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("pcbnew", None)
        else:
            sys.modules["pcbnew"] = self._saved

    def test_polygon_courtyard_ignores_start_end(self):
        # U.FL-like part at (133.5, 55.8) mm, courtyard 4.35 x 5 mm polygon.
        cx, cy = 133.5 * MM, 55.8 * MM
        hw, hh = 2.175 * MM, 2.5 * MM
        pts = [P(cx - hw, cy - hh), P(cx + hw, cy - hh), P(cx + hw, cy + hh), P(cx - hw, cy + hh)]
        board = FakeBoard([FakeFp("J7", P(cx, cy), [PolyItem(pts)])])
        model = extract_board_model(board)
        fp = model.footprints[0]
        self.assertAlmostEqual(fp.width, 4.35 * MM + 2 * PAD_MARGIN, delta=2)
        self.assertAlmostEqual(fp.height, 5.0 * MM + 2 * PAD_MARGIN, delta=2)
        self.assertLessEqual(abs(fp.cx_offset), 1)
        self.assertLessEqual(abs(fp.cy_offset), 1)

    def test_segment_courtyard_still_uses_start_end(self):
        cx, cy = 120 * MM, 50 * MM
        items = [SegItem(P(cx - 1 * MM, cy - 2 * MM), P(cx + 1 * MM, cy - 2 * MM)),
                 SegItem(P(cx + 1 * MM, cy - 2 * MM), P(cx + 1 * MM, cy + 2 * MM)),
                 SegItem(P(cx + 1 * MM, cy + 2 * MM), P(cx - 1 * MM, cy + 2 * MM)),
                 SegItem(P(cx - 1 * MM, cy + 2 * MM), P(cx - 1 * MM, cy - 2 * MM))]
        model = extract_board_model(FakeBoard([FakeFp("U1", P(cx, cy), items)]))
        fp = model.footprints[0]
        self.assertAlmostEqual(fp.width, 2 * MM + 2 * PAD_MARGIN, delta=2)
        self.assertAlmostEqual(fp.height, 4 * MM + 2 * PAD_MARGIN, delta=2)

    def test_arc_includes_mid(self):
        m = sys.modules["pcbnew"]
        item = SegItem(P(-1 * MM, 0), P(1 * MM, 0), shape=2, mid=P(0, -1 * MM))
        pts = _courtyard_points_world(item, m, P(0, 0), 1.0, 0.0)
        self.assertIn((0, -1 * MM), pts)
        self.assertEqual(len(pts), 3)

    def test_polygon_helper_returns_outline_only(self):
        m = sys.modules["pcbnew"]
        item = PolyItem([P(10, 20), P(30, 20), P(30, 40)])
        pts = _courtyard_points_world(item, m, P(0, 0), 1.0, 0.0)
        self.assertEqual(pts, [(10, 20), (30, 20), (30, 40)])


if __name__ == "__main__":
    unittest.main()
