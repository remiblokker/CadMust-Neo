"""Unit tests for scripts/route_completion.py: JSON output shape and the
completion arithmetic, against a fake KiCadRoutingTools tree (no KRT, no
KiCad, well under a second).

Run with: python3 tests/test_route_completion.py -v
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "route_completion.py"

# Fake KRT: the unrouted board has 4 open nets / 10 open links; the "router"
# writes a board carrying the marker ROUTED, on which 1 net with 3 components
# (2 links) stays open. check_drc reports 3 violations.
FAKE_CHECK_CONNECTED = '''
def run_connectivity_check(pcb_file, quiet=False, **kw):
    text = open(pcb_file).read()
    if "ROUTED" in text:
        return [{"net_id": 7, "net_name": "GND", "num_components": 3}]
    return [{"net_id": 1, "net_name": "A", "num_components": 2},
            {"net_id": 2, "net_name": "B", "num_components": 3},
            {"net_id": 3, "net_name": "C", "num_components": 2},
            {"net_id": 7, "net_name": "GND", "num_components": 6}]
'''
FAKE_PARSER = '''
class _D:
    pads_by_net = {0: [1, 2], 1: [1, 2], 2: [1, 2, 3], 3: [1, 2], 7: list(range(6)), 9: [1]}
def parse_kicad_pcb(path, *a, **kw):
    return _D()
'''
FAKE_ROUTE = '''
import json, sys
args = sys.argv[1:]
src, dst = args[0], args[1]
summary = args[args.index("--json-out") + 1]
open(dst, "w").write(open(src).read() + "\\nROUTED\\n")
json.dump({"successful": 3, "failed": 1, "total_vias": 12, "total_time": 0.1}, open(summary, "w"))
'''
FAKE_DRC = '''
import json, sys
out = sys.argv[sys.argv.index("--json") + 1]
json.dump({"violations": 3, "by_type": {"clearance": [1, 2, 3]}}, open(out, "w"))
'''


class RouteCompletionJSON(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        pr = t / "krt" / "py_router"
        pr.mkdir(parents=True)
        (t / "krt" / "VERSION").write_text("0.0.test\n")
        for name, body in (("check_connected.py", FAKE_CHECK_CONNECTED),
                           ("kicad_parser.py", FAKE_PARSER),
                           ("route.py", FAKE_ROUTE),
                           ("check_drc.py", FAKE_DRC)):
            (pr / name).write_text(textwrap.dedent(body))
        self.board = t / "b.kicad_pcb"
        self.board.write_text("(kicad_pcb (version 20240108))\n")
        self.krt = t / "krt"

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *extra):
        r = subprocess.run([sys.executable, str(SCRIPT), str(self.board),
                            "--krt-dir", str(self.krt), *extra],
                           capture_output=True, text=True, timeout=60)
        return r.returncode, json.loads(r.stdout)

    def test_shape_and_numbers(self):
        rc, d = self.run_tool()
        self.assertEqual(rc, 0)
        self.assertEqual(d["status"], "ok")
        for key in ("board", "krt", "options", "stripped", "nets_total",
                    "nets_complete", "links_total", "links_unrouted",
                    "completion_links", "completion_nets", "unrouted_nets",
                    "router_seconds", "total_seconds", "router_summary", "drc",
                    "command"):
            self.assertIn(key, d)
        self.assertFalse(d["stripped"])          # no routing in the input
        self.assertEqual(d["krt"]["version"], "0.0.test")
        self.assertEqual(d["nets_with_2plus_pads"], 4)  # net 0 and 1-pad net skipped
        self.assertEqual(d["nets_total"], 4)
        self.assertEqual(d["nets_complete"], 3)
        self.assertEqual(d["links_total"], 1 + 2 + 1 + 5)
        self.assertEqual(d["links_unrouted"], 2)
        self.assertAlmostEqual(d["completion_links"], round(1 - 2 / 9, 4))
        self.assertAlmostEqual(d["completion_nets"], 0.75)
        self.assertEqual(d["unrouted_nets"], ["GND"])
        self.assertEqual(d["drc"]["violations"], 3)
        self.assertEqual(d["drc"]["by_type"], {"clearance": 3})
        self.assertEqual(d["router_summary"]["failed"], 1)
        self.assertEqual(d["options"]["escalation"], "off")
        self.assertEqual(d["options"]["quick"], 1)

    def test_options_reach_router_and_no_drc(self):
        rc, d = self.run_tool("--layers", "F.Cu", "B.Cu", "--grid-step", "0.05",
                              "--quick", "0", "--no-drc")
        self.assertEqual(rc, 0)
        self.assertNotIn("drc", d)
        self.assertIn("--layers F.Cu B.Cu", d["command"])
        self.assertIn("--grid-step 0.05", d["command"])
        self.assertNotIn("--quick", d["command"])

    def test_router_failure_is_nonzero(self):
        (self.krt / "py_router" / "route.py").write_text("import sys; sys.exit(1)\n")
        rc, d = self.run_tool()
        self.assertEqual(rc, 3)
        self.assertEqual(d["status"], "router_failed")
        self.assertNotIn("completion_links", d)

    def test_missing_krt_is_exit_2(self):
        r = subprocess.run([sys.executable, str(SCRIPT), str(self.board),
                            "--krt-dir", str(self.krt / "nope")],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertEqual(json.loads(r.stdout)["status"], "error")

    def test_routing_detection(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import route_completion as rcm
        text = textwrap.dedent("""\
            (kicad_pcb
            \t(segment (start 0 0) (end 1 1))
            \t(arc (start 0 0) (mid 1 0) (end 1 1))
            \t(via (at 0 0))
            \t(zone
            \t\t(net 1)
            \t)
            \t(zone
            \t\t(net 0)
            \t\t(keepout (tracks not_allowed))
            \t)
            )""")
        self.assertEqual(rcm._routing_present(text),
                         {"segments": 2, "vias": 1, "copper_zones": 1})


if __name__ == "__main__":
    unittest.main()
