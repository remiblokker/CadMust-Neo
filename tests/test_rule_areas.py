"""Rule areas -> footprint keep-outs (board_model.extract_keepouts).

Only rule areas with "footprints not allowed" become KeepOut entries; a
pour-only rule area ("(copperpour not_allowed) (footprints allowed)", all 8
on the Olimex ESP32-EVB) is ignored. Stub pcbnew objects, no KiCad needed;
the last test runs against the real sample boards when pcbnew is importable.

Run with: python3 -m unittest discover -s tests
"""
import os
import pickle
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugin.board_model import KeepOut, extract_keepouts

LAYER_NAMES = {0: "F.Cu", 2: "B.Cu", 4: "In1.Cu", 40: "F.SilkS"}


class _Box:
    def __init__(self, x, y, w, h):
        self._v = (x, y, w, h)

    def GetX(self): return self._v[0]
    def GetY(self): return self._v[1]
    def GetWidth(self): return self._v[2]
    def GetHeight(self): return self._v[3]


class _Outline:
    def __init__(self, box):
        self._box = box

    def __bool__(self): return True
    def BBox(self): return self._box


class _LayerSet:
    def __init__(self, ids):
        self._ids = ids

    def Seq(self): return list(self._ids)


class _Zone:
    def __init__(self, rule_area=True, footprints_forbidden=True, layers=(0, 2),
                 box=(1000, 2000, 3000, 4000), has_flag=True):
        self._rule = rule_area
        self._fp = footprints_forbidden
        self._layers = layers
        self._box = _Box(*box)
        if not has_flag:
            self.GetDoNotAllowFootprints = None  # shadow the method

    def GetIsRuleArea(self): return self._rule
    def GetDoNotAllowFootprints(self): return self._fp
    def GetLayerSet(self): return _LayerSet(self._layers)
    def Outline(self): return _Outline(self._box)


class _Board:
    def __init__(self, zones):
        self._zones = zones

    def Zones(self): return self._zones
    def GetLayerName(self, lid): return LAYER_NAMES[lid]


class TestExtractKeepouts(unittest.TestCase):
    def test_pour_only_rule_area_is_not_a_keepout(self):
        # Broken case the fix is about: before, this rule area became a keep-out.
        b = _Board([_Zone(footprints_forbidden=False)])
        self.assertEqual(extract_keepouts(b), [])

    def test_footprint_rule_area_is_kept_with_bbox(self):
        b = _Board([_Zone(box=(1000, 2000, 3000, 4000))])
        ko = extract_keepouts(b)
        self.assertEqual(ko, [KeepOut(xmin=1000, ymin=2000, xmax=4000, ymax=6000,
                                      layers="all")])

    def test_mixed_board_keeps_only_footprint_areas(self):
        b = _Board([_Zone(footprints_forbidden=False), _Zone(),
                    _Zone(rule_area=False), _Zone(footprints_forbidden=False)])
        self.assertEqual(len(extract_keepouts(b)), 1)

    def test_layers_recorded(self):
        b = _Board([_Zone(layers=(0,)), _Zone(layers=(2,)), _Zone(layers=(0, 2)),
                    _Zone(layers=(0, 4)), _Zone(layers=(40,))])
        self.assertEqual([k.layers for k in extract_keepouts(b)],
                         ["F.Cu", "B.Cu", "all", "F.Cu,In1.Cu", "all"])

    def test_zone_without_flag_api_is_a_keepout(self):
        b = _Board([_Zone(footprints_forbidden=False, has_flag=False)])
        self.assertEqual(len(extract_keepouts(b)), 1)

    def test_old_pickle_keepout_reads_default_layers(self):
        ko = KeepOut(1, 2, 3, 4, layers="F.Cu")
        del ko.__dict__["layers"]
        ko2 = pickle.loads(pickle.dumps(ko))
        self.assertEqual(ko2.layers, "all")


try:
    import pcbnew  # noqa: F401
    HAVE_PCBNEW = True
except ImportError:
    HAVE_PCBNEW = False

SAMPLES = os.path.join(os.path.dirname(__file__), "sample_boards")


@unittest.skipUnless(HAVE_PCBNEW, "needs KiCad's pcbnew")
class TestSampleBoards(unittest.TestCase):
    def test_esp32_evb_pour_areas_and_radar_afe_keepout(self):
        import pcbnew
        evb = pcbnew.LoadBoard(os.path.join(SAMPLES, "esp32-evb", "esp32-evb_unrouted.kicad_pcb"))
        self.assertEqual(sum(1 for z in evb.Zones() if z.GetIsRuleArea()), 8)
        self.assertEqual(extract_keepouts(evb), [])
        radar = pcbnew.LoadBoard(os.path.join(SAMPLES, "radar-afe", "radar-afe.kicad_pcb"))
        self.assertEqual(len(extract_keepouts(radar)), 1)


if __name__ == "__main__":
    unittest.main()
