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


if __name__ == "__main__":
    unittest.main()
