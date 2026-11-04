#!/usr/bin/env python3
"""
batch_runner.py — Run the CadMust-Neo optimizer on all extracted board models.

Run with regular python3 (no KiCad needed — uses pickled BoardModel objects):

    python3 scripts/batch_runner.py                        # all boards, default profile
    python3 scripts/batch_runner.py --profile thorough     # 3-start run
    python3 scripts/batch_runner.py --profile all          # run all profiles
    python3 scripts/batch_runner.py --board myboard.kicad_pcb  # single board
    python3 scripts/batch_runner.py --model board.pkl --profile quick \
        --constraints board.json --scatter-start \
        --save-model out.pkl --result-json out.json      # one pickle, constraints

Prerequisite: run extract_models.py first (with KiCad Python) to produce the
pickled models in <corpus>/models/.

Results are written to <corpus>/results_<profile>_<timestamp>.csv.
Existing results are skipped unless --rerun is given.

Profiles
--------
quick     : 1 start, 100 iterations  — fast sanity check (~5s per board)
default   : 1 start, 200 iterations  — standard run (~30s per board)
thorough  : 3 starts, 200 iterations — best quality (~90s per board)
"""

import argparse
import copy
import csv
import json
import pickle
import sys
import time
from pathlib import Path

# Add project root to path so plugin package is importable
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from plugin.annealer import run_sa, SAConfig
from plugin.cost_function import CostState
from plugin.constraints import load_constraints, apply_constraints, ConstraintError


# ---------------------------------------------------------------------------
# Benchmark profiles
# ---------------------------------------------------------------------------

PROFILES = {
    "quick": SAConfig(
        num_starts=1,
        max_iterations=100,
    ),
    "default": SAConfig(
        num_starts=1,
        max_iterations=200,
    ),
    "thorough": SAConfig(
        num_starts=3,
        max_iterations=200,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MM = 1_000_000  # nm per mm

# Footprints with J/P references within this distance of a board edge are
# auto-locked. Edge connectors are almost always placed at the board edge;
# scraped boards don't preserve KiCad's locked flag, so they'd get moved.
EDGE_CONNECTOR_MARGIN = 5 * MM  # 5 mm in nm


def lock_edge_connectors(model) -> int:
    """Lock footprints that look like edge-mounted connectors.

    Heuristic: reference starts with J or P AND the footprint origin is
    within EDGE_CONNECTOR_MARGIN of any board edge.  Returns the number of
    footprints newly locked.
    """
    newly_locked = 0
    for fp in model.footprints:
        if fp.locked:
            continue
        ref = fp.reference.upper()
        if not (ref.startswith("J") or ref.startswith("P")):
            continue
        near = (
            fp.x - model.outline_xmin < EDGE_CONNECTOR_MARGIN
            or model.outline_xmax - fp.x < EDGE_CONNECTOR_MARGIN
            or fp.y - model.outline_ymin < EDGE_CONNECTOR_MARGIN
            or model.outline_ymax - fp.y < EDGE_CONNECTOR_MARGIN
        )
        if near:
            fp.locked = True
            newly_locked += 1
    if newly_locked:
        model.moveable_indices = [
            i for i in model.moveable_indices if not model.footprints[i].locked
        ]
    return newly_locked


def unlock_all(model) -> int:
    """Clear every footprint's and group's locked flag and rebuild
    moveable_indices (one representative, the first member, per group).

    For fair experiments: the vendor's KiCad locks would otherwise leak the
    existing placement. Fixed positions and edge locks then come only from
    the constraint file. Returns the number of footprints unlocked.
    """
    n = 0
    for fp in model.footprints:
        if fp.locked:
            fp.locked = False
            n += 1
    in_group = set()
    for grp in model.component_groups:
        grp.locked = False
        in_group.update(grp.member_indices[1:])
    model.moveable_indices = [i for i in range(len(model.footprints)) if i not in in_group]
    return n


def compute_hpwl_mm(model) -> float:
    """Compute HPWL in mm from current footprint positions."""
    total = 0
    for net in model.nets.values():
        if net.is_excluded or len(net.pad_refs) < 2:
            continue
        fps = model.footprints
        xs = []
        ys = []
        for fi, pi in net.pad_refs:
            fp = fps[fi]
            px, py = fp.pads[pi].abs_position(fp.x, fp.y, fp.angle_deg)
            xs.append(px)
            ys.append(py)
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total / MM


def snapshot_positions(model):
    """Save all footprint positions."""
    return [(fp.x, fp.y, fp.angle_deg) for fp in model.footprints]


def restore_positions(model, positions):
    """Restore all footprint positions (and trig cache)."""
    for fp, (x, y, a) in zip(model.footprints, positions):
        fp.x = x
        fp.y = y
        fp.set_angle(a)


def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def count_overlaps(model) -> int:
    """Count footprint pairs with bounding-box overlap."""
    fps = model.footprints
    count = 0
    for i in range(len(fps)):
        x1a, y1a, x2a, y2a = fps[i].bbox
        for j in range(i + 1, len(fps)):
            x1b, y1b, x2b, y2b = fps[j].bbox
            if x1a < x2b and x2a > x1b and y1a < y2b and y2a > y1b:
                count += 1
    return count


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "board", "profile",
    "moveable", "signal_nets",
    "hpwl_before_mm", "hpwl_after_mm", "hpwl_improvement_pct",
    "overlaps_before", "overlaps_after",
    "time_s", "temperature_steps", "total_moves", "accepted_moves",
    "accepted_pct",
]


# ---------------------------------------------------------------------------
# Single-model mode (--model)
# ---------------------------------------------------------------------------

def run_single_model(args, cset) -> int:
    """Run one pickle with --board semantics (edge-connector lock, one result
    row) but without the corpus/manifest machinery. Returns an exit code."""
    pkl_path = Path(args.model)
    if args.profile == "all":
        print("--model runs one profile; pick quick, default or thorough")
        return 2
    try:
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"  FAILED  {pkl_path} (load): {e}")
        return 1
    profile_name = args.profile
    config = copy.copy(PROFILES[profile_name])
    if args.scatter_start:
        config.scatter_first_start = True
    board_name = pkl_path.stem + ".kicad_pcb"

    n_unlocked = 0
    if args.unlock:
        n_unlocked = unlock_all(model)
        print(f"  Unlocked {n_unlocked} footprint(s); edge-connector auto-lock skipped")
        n_locked = 0
    else:
        n_locked = lock_edge_connectors(model)
    report = apply_constraints(model, cset) if cset is not None else None

    n_move = len(model.moveable_indices)
    n_nets = sum(1 for n in model.nets.values() if not n.is_excluded)
    hpwl_before = compute_hpwl_mm(model)
    overlaps_before = count_overlaps(model)
    locked_str = f", {n_locked} edge-locked" if n_locked else ""
    print(f"  Running {board_name} ({profile_name}"
          f"{', constraints' if cset is not None else ''}"
          f"{', scatter start' if args.scatter_start else ''}"
          f"{', unlocked' if args.unlock else ''}): {n_move} moveable{locked_str}, "
          f"{n_nets} nets, {hpwl_before:.0f} mm HPWL ...", flush=True)

    t_start = time.time()
    result = run_sa(model, config=config)
    elapsed = time.time() - t_start

    hpwl_after = compute_hpwl_mm(model)
    overlaps_after = count_overlaps(model)
    improvement = (hpwl_before - hpwl_after) / hpwl_before * 100 if hpwl_before > 0 else 0.0
    accepted_pct = result.accepted_moves / result.total_moves * 100 if result.total_moves > 0 else 0.0
    print(f"  {board_name}: {hpwl_before:.0f} -> {hpwl_after:.0f} mm ({improvement:+.1f}%), "
          f"overlaps {overlaps_before} -> {overlaps_after}, {elapsed:.1f}s")

    row = {
        "board": board_name,
        "profile": profile_name,
        "moveable": n_move,
        "signal_nets": n_nets,
        "hpwl_before_mm": round(hpwl_before, 1),
        "hpwl_after_mm": round(hpwl_after, 1),
        "hpwl_improvement_pct": round(improvement, 2),
        "overlaps_before": overlaps_before,
        "overlaps_after": overlaps_after,
        "time_s": round(elapsed, 1),
        "temperature_steps": result.temperature_steps,
        "total_moves": result.total_moves,
        "accepted_moves": result.accepted_moves,
        "accepted_pct": round(accepted_pct, 1),
    }
    if args.out:
        out = Path(args.out)
        write_header = not out.exists()
        with open(out, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    if args.save_model:
        with open(args.save_model, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved optimised model: {args.save_model}")

    if args.result_json:
        res = dict(row)
        res["model"] = str(pkl_path)
        res["constraints"] = args.constraints
        res["scatter_start"] = bool(args.scatter_start)
        res["edge_connectors_locked"] = n_locked
        res["unlock"] = bool(args.unlock)
        res["unlocked_footprints"] = n_unlocked
        # SAResult.total_moves over wall time (incl. greedy); the profile log's
        # Moves/sec sums cumulative per-phase counts and reads higher
        res["moves_per_s_wall"] = round(result.total_moves / elapsed) if elapsed > 0 else None
        res["constraint_report"] = report.counts() if report is not None else None
        with open(args.result_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Result JSON: {args.result_json}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch optimizer benchmark runner")
    parser.add_argument("--corpus", default=str(Path.home() / "cadmust_testboards"),
                        help="Corpus root (default: ~/cadmust_testboards)")
    parser.add_argument("--profile", default="default",
                        choices=list(PROFILES.keys()) + ["all"],
                        help="Settings profile (default: default)")
    parser.add_argument("--board", default=None,
                        help="Run on a single board filename (e.g. foo.kicad_pcb)")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run even if result already in CSV")
    parser.add_argument("--save-positions", action="store_true",
                        help="Save optimized positions to <corpus>/positions/<board>.json for later DRC")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: <corpus>/results_<profile>.csv)")
    parser.add_argument("--constraints", default=None,
                        help="Constraint JSON (CadMust-Autopilot schema v1/v2), applied "
                             "after loading each model and after edge-connector locking")
    parser.add_argument("--scatter-start", action="store_true",
                        help="Scatter start 0 too (do not seed SA with the existing placement)")
    parser.add_argument("--unlock", action="store_true",
                        help="Clear all KiCad lock flags from the pickle and skip the "
                             "edge-connector auto-lock (fair experiment: only the "
                             "constraint file fixes parts)")
    parser.add_argument("--model", default=None,
                        help="Run one pickled BoardModel from any path (ignores the corpus "
                             "and manifest; no CSV unless --out is given)")
    parser.add_argument("--save-model", default=None,
                        help="With --model: pickle the optimised BoardModel to this path")
    parser.add_argument("--result-json", default=None,
                        help="With --model: write a one-object JSON result to this path")
    args = parser.parse_args()

    cset = None
    if args.constraints:
        try:
            cset = load_constraints(args.constraints)
        except ConstraintError as e:
            print("Constraint file error: %s" % e)
            sys.exit(2)

    if args.model:
        sys.exit(run_single_model(args, cset))
    if args.save_model or args.result_json:
        print("--save-model and --result-json require --model")
        sys.exit(2)

    corpus_dir = Path(args.corpus)
    models_dir = corpus_dir / "models"
    manifest_path = corpus_dir / "manifest.json"

    if args.save_positions:
        positions_dir = corpus_dir / "positions"
        positions_dir.mkdir(exist_ok=True)
    else:
        positions_dir = None

    manifest = load_manifest(manifest_path)

    # Determine which profiles to run
    profiles_to_run = list(PROFILES.keys()) if args.profile == "all" else [args.profile]

    for profile_name in profiles_to_run:
        config = copy.copy(PROFILES[profile_name])
        if args.scatter_start:
            config.scatter_first_start = True

        # Output CSV — fixed name per profile (no timestamp) so runs are resumable
        if args.out and len(profiles_to_run) == 1:
            csv_path = Path(args.out)
        else:
            csv_path = corpus_dir / f"results_{profile_name}.csv"

        manifest_key = f"batch_{profile_name}"

        # Find boards to process
        if args.board:
            pkl_files = [models_dir / (Path(args.board).stem + ".pkl")]
            pkl_files = [p for p in pkl_files if p.exists()]
        else:
            pkl_files = sorted(models_dir.glob("*.pkl"))

        if not pkl_files:
            print(f"No model files found in {models_dir}")
            print("Run extract_models.py first.")
            sys.exit(1)

        # Done boards are tracked in the manifest (not the CSV) so resume works
        # regardless of whether the CSV still exists.
        done_boards: set = set()
        if not args.rerun:
            for key, entry in manifest.items():
                if manifest_key in entry and entry[manifest_key].get("status") == "ok":
                    done_boards.add(key)

        print(f"Profile      : {profile_name}")
        print(f"Starts       : {config.num_starts}")
        print(f"Max iters    : {config.max_iterations}")
        print(f"Boards       : {len(pkl_files)}")
        print(f"Already done : {len(done_boards)}")
        print(f"Output CSV   : {csv_path}")
        print()

        # Write/append CSV
        write_header = not csv_path.exists() or args.rerun
        csv_file = open(csv_path, "w" if args.rerun else "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        ok = skipped = failed = 0

        for pkl_path in pkl_files:
            board_name = pkl_path.stem + ".kicad_pcb"

            if board_name in done_boards:
                print(f"  CACHED  {board_name}")
                skipped += 1
                continue

            try:
                with open(pkl_path, "rb") as f:
                    model = pickle.load(f)
            except Exception as e:
                print(f"  FAILED  {board_name} (load): {e}")
                failed += 1
                continue

            # Auto-lock edge connectors (scraped boards don't preserve KiCad locks)
            if args.unlock:
                n_unlocked = unlock_all(model)
                print(f"  {board_name}: unlocked {n_unlocked} footprint(s)")
                n_locked = 0
            else:
                n_locked = lock_edge_connectors(model)
            if cset is not None:
                apply_constraints(model, cset)

            # Snapshot original positions so we can restore after run
            original_positions = snapshot_positions(model)

            n_move = len(model.moveable_indices)
            n_nets = sum(1 for n in model.nets.values() if not n.is_excluded)
            hpwl_before = compute_hpwl_mm(model)
            overlaps_before = count_overlaps(model)

            locked_str = f", {n_locked} edge-locked" if n_locked else ""
            print(f"  Running {board_name}: {n_move} moveable{locked_str}, {n_nets} nets, "
                  f"{hpwl_before:.0f} mm HPWL ...", end="", flush=True)

            t_start = time.time()
            try:
                result = run_sa(model, config=copy.copy(config))
            except Exception as e:
                print(f"\n  FAILED  {board_name} (SA): {e}")
                restore_positions(model, original_positions)
                failed += 1
                continue
            elapsed = time.time() - t_start

            hpwl_after = compute_hpwl_mm(model)
            overlaps_after = count_overlaps(model)
            improvement = (hpwl_before - hpwl_after) / hpwl_before * 100 if hpwl_before > 0 else 0.0
            accepted_pct = result.accepted_moves / result.total_moves * 100 if result.total_moves > 0 else 0.0

            print(f" {hpwl_after:.0f} mm ({improvement:+.1f}%), "
                  f"{overlaps_after} overlaps, {elapsed:.1f}s")

            row = {
                "board": board_name,
                "profile": profile_name,
                "moveable": n_move,
                "signal_nets": n_nets,
                "hpwl_before_mm": round(hpwl_before, 1),
                "hpwl_after_mm": round(hpwl_after, 1),
                "hpwl_improvement_pct": round(improvement, 2),
                "overlaps_before": overlaps_before,
                "overlaps_after": overlaps_after,
                "time_s": round(elapsed, 1),
                "temperature_steps": result.temperature_steps,
                "total_moves": result.total_moves,
                "accepted_moves": result.accepted_moves,
                "accepted_pct": round(accepted_pct, 1),
            }
            writer.writerow(row)
            csv_file.flush()

            # Save optimized positions for later apply_positions.py + DRC run
            if positions_dir is not None:
                pos_data = {
                    "board": board_name,
                    "profile": profile_name,
                    "footprints": [
                        {"reference": fp.reference, "x": fp.x, "y": fp.y, "angle_deg": fp.angle_deg}
                        for fp in model.footprints
                    ],
                }
                pos_path = positions_dir / (pkl_path.stem + f"_{profile_name}.json")
                with open(pos_path, "w") as f:
                    json.dump(pos_data, f)

            # Write result to manifest (primary done-tracking + future analysis)
            entry = manifest.get(board_name, {})
            entry[manifest_key] = {
                "status": "ok",
                "hpwl_before_mm": round(hpwl_before, 1),
                "hpwl_after_mm": round(hpwl_after, 1),
                "hpwl_improvement_pct": round(improvement, 2),
                "overlaps_before": overlaps_before,
                "overlaps_after": overlaps_after,
                "time_s": round(elapsed, 1),
                "temperature_steps": result.temperature_steps,
                "total_moves": result.total_moves,
            }
            manifest[board_name] = entry
            save_manifest(manifest_path, manifest)

            # Restore original positions (don't modify the pickled model's placement)
            restore_positions(model, original_positions)
            ok += 1

        csv_file.close()

        print()
        print(f"Results: {ok} ran, {skipped} cached, {failed} failed")
        print(f"CSV: {csv_path}")
        print()


if __name__ == "__main__":
    main()
