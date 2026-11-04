#!/usr/bin/env python3
"""
extract_models.py — Extract BoardModel objects from unrouted KiCad boards and pickle them.

MUST be run with KiCad's Python (pcbnew required for board loading):

    /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/\
Versions/Current/bin/python3 scripts/extract_models.py

Pickled models are saved to <corpus>/models/<boardname>.pkl.

Boards with invalid_outline DRC errors are automatically skipped — the optimizer
drifts all components to (0,0) without a board outline, producing meaningless results.

The extracted BoardModel is pure Python (no pcbnew dependencies), so the pickled
files can be loaded and optimized with regular Python (no KiCad needed).
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# Suppress KiCad's verbose debug/warning output to stderr
_devnull = open(os.devnull, "w")
os.dup2(_devnull.fileno(), 2)

try:
    import pcbnew
except ImportError:
    sys.exit(
        "ERROR: pcbnew not found.\n"
        "Run this script with KiCad's Python:\n"
        "  /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework"
        "/Versions/Current/bin/python3 scripts/extract_models.py"
    )

# Add project root to path so plugin package is importable
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from plugin.board_model import extract_board_model


def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def compute_hpwl_mm(model) -> float:
    """Compute initial HPWL in mm from a freshly extracted model."""
    MM = 1_000_000
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract BoardModel objects from unrouted KiCad boards")
    parser.add_argument("--corpus", default=str(Path.home() / "cadmust_testboards"),
                        help="Corpus root (default: ~/cadmust_testboards)")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-extract even if already done")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    unrouted_dir = corpus_dir / "unrouted"
    models_dir = corpus_dir / "models"
    manifest_path = corpus_dir / "manifest.json"

    models_dir.mkdir(exist_ok=True)
    manifest = load_manifest(manifest_path)

    pcb_files = sorted(unrouted_dir.glob("*.kicad_pcb"))
    if not pcb_files:
        sys.exit(f"No .kicad_pcb files found in {unrouted_dir}")

    print(f"Unrouted dir : {unrouted_dir}")
    print(f"Models dir   : {models_dir}")
    print(f"Boards       : {len(pcb_files)}")
    print()

    ok = skipped = failed = 0

    for board_path in pcb_files:
        key = board_path.name
        entry = manifest.get(key, {})
        pkl_path = models_dir / (board_path.stem + ".pkl")

        # Skip boards with invalid outline — optimizer drifts to (0,0) without one
        drc = entry.get("drc_baseline", {})
        if "invalid_outline" in drc.get("error_types", []):
            print(f"  SKIP (no outline) : {key}")
            skipped += 1
            continue

        # Skip if already extracted (unless --rerun)
        if not args.rerun and entry.get("model_status") == "ok" and pkl_path.exists():
            print(f"  CACHED            : {key}")
            skipped += 1
            continue

        try:
            board = pcbnew.LoadBoard(str(board_path))
            model = extract_board_model(board)
        except Exception as e:
            print(f"  FAILED (load)     : {key}: {e}")
            entry["model_status"] = "failed"
            entry["model_error"] = str(e)
            manifest[key] = entry
            save_manifest(manifest_path, manifest)
            failed += 1
            continue

        n_move = len(model.moveable_indices)
        n_nets = sum(1 for n in model.nets.values() if not n.is_excluded)
        hpwl_mm = compute_hpwl_mm(model)

        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"  FAILED (pickle)   : {key}: {e}")
            entry["model_status"] = "failed"
            entry["model_error"] = str(e)
            manifest[key] = entry
            save_manifest(manifest_path, manifest)
            failed += 1
            continue

        print(f"  OK  {key}: {n_move} moveable, {n_nets} signal nets, {hpwl_mm:.0f} mm HPWL")
        entry["model_status"] = "ok"
        entry["model_moveable"] = n_move
        entry["model_signal_nets"] = n_nets
        entry["model_initial_hpwl_mm"] = round(hpwl_mm, 1)
        manifest[key] = entry
        save_manifest(manifest_path, manifest)
        ok += 1

    print()
    print(f"Results: {ok} extracted, {skipped} skipped/cached, {failed} failed")
    print(f"Models in: {models_dir}")


if __name__ == "__main__":
    main()
