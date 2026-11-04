#!/usr/bin/env python3
"""
save_optimized.py — Apply optimized positions to unrouted boards and save as .kicad_pcb.

MUST be run with KiCad's Python (pcbnew required for board writing):

    /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/\
Versions/Current/bin/python3 scripts/save_optimized.py

Prerequisite: run batch_runner.py with --save-positions to produce the
position JSON files in <corpus>/positions/.

Reads:
  <corpus>/positions/<board>_<profile>.json — optimized footprint positions
  <corpus>/unrouted/<board>.kicad_pcb       — source board (unchanged)

Writes:
  <corpus>/optimized_<profile>/<board>.kicad_pcb — board with optimized placement

After this, run DRC comparison:
  python3 scripts/run_drc.py --dir <corpus>/optimized_<profile> --tag <profile>
"""

import argparse
import json
import os
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
        "/Versions/Current/bin/python3 scripts/save_optimized.py"
    )


def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def apply_positions(board, positions: list) -> int:
    """Apply footprint positions from JSON list to a pcbnew board. Returns count applied."""
    by_ref = {fp.GetReference(): fp for fp in board.GetFootprints()}
    applied = 0
    for entry in positions:
        ref = entry["reference"]
        fp = by_ref.get(ref)
        if fp is None or fp.IsLocked():
            continue
        fp.SetPosition(pcbnew.VECTOR2I(int(entry["x"]), int(entry["y"])))
        fp.SetOrientation(pcbnew.EDA_ANGLE(float(entry["angle_deg"]), pcbnew.DEGREES_T))
        applied += 1
    board.GetConnectivity().RecalculateRatsnest()
    return applied


def main():
    parser = argparse.ArgumentParser(
        description="Apply optimized positions to unrouted boards and save .kicad_pcb files")
    parser.add_argument("--corpus", default=str(Path.home() / "cadmust_testboards"),
                        help="Corpus root (default: ~/cadmust_testboards)")
    parser.add_argument("--profile", default="default",
                        help="Profile name (default: default) — matches positions/<board>_<profile>.json")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-apply even if already done")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    unrouted_dir = corpus_dir / "unrouted"
    positions_dir = corpus_dir / "positions"
    optimized_dir = corpus_dir / f"optimized_{args.profile}"
    manifest_path = corpus_dir / "manifest.json"

    optimized_dir.mkdir(exist_ok=True)
    manifest = load_manifest(manifest_path)

    pos_files = sorted(positions_dir.glob(f"*_{args.profile}.json"))
    if not pos_files:
        sys.exit(f"No position files found in {positions_dir} for profile '{args.profile}'.\n"
                 f"Run: python3 scripts/batch_runner.py --profile {args.profile} --save-positions")

    print(f"Profile      : {args.profile}")
    print(f"Positions    : {positions_dir}")
    print(f"Output dir   : {optimized_dir}")
    print(f"Boards       : {len(pos_files)}")
    print()

    ok = skipped = failed = 0

    for pos_path in pos_files:
        # Strip _<profile> suffix to recover board name
        board_name = pos_path.stem[: -len(f"_{args.profile}")] + ".kicad_pcb"
        key = board_name
        entry = manifest.get(key, {})
        save_key = f"save_{args.profile}"
        dst_path = optimized_dir / board_name

        if not args.rerun and entry.get(save_key) == "ok" and dst_path.exists():
            print(f"  CACHED  {board_name}")
            skipped += 1
            continue

        src_path = unrouted_dir / board_name
        if not src_path.exists():
            print(f"  SKIP (no source): {board_name}")
            skipped += 1
            continue

        with open(pos_path) as f:
            pos_data = json.load(f)

        try:
            board = pcbnew.LoadBoard(str(src_path))
            n_applied = apply_positions(board, pos_data["footprints"])
            board.Save(str(dst_path))
            # Delete pcbnew sidecar files
            for ext in (".kicad_pro", ".kicad_prl"):
                sidecar = dst_path.with_suffix(ext)
                if sidecar.exists():
                    sidecar.unlink()
        except Exception as e:
            print(f"  FAILED  {board_name}: {e}")
            entry[save_key] = "failed"
            manifest[key] = entry
            save_manifest(manifest_path, manifest)
            failed += 1
            continue

        print(f"  OK      {board_name} ({n_applied} footprints placed)")
        entry[save_key] = "ok"
        manifest[key] = entry
        save_manifest(manifest_path, manifest)
        ok += 1

    print()
    print(f"Results: {ok} saved, {skipped} skipped/cached, {failed} failed")
    print(f"Optimized boards in: {optimized_dir}")
    print()
    print(f"Next step — run DRC on optimized boards:")
    print(f"  python3 scripts/run_drc.py --dir {optimized_dir} --tag {args.profile}")


if __name__ == "__main__":
    main()
