#!/usr/bin/env python3
"""
strip_routing.py — Remove routing from downloaded KiCad PCB files.

Produces unrouted boards suitable for optimizer benchmarking. Stripped boards
are saved to <corpus_dir>/unrouted/ alongside a manifest update.

MUST be run with KiCad's Python (pcbnew is not available otherwise):

    /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/\
Versions/Current/bin/python3 scripts/strip_routing.py

What is removed:
    - Tracks (copper traces)
    - Vias
    - Copper zones / fills (pours)  — but keepout/rule-area zones are KEPT

What is kept:
    - Footprints (components + pads)
    - Ratsnest / net assignments
    - Board outline (Edge.Cuts)
    - Keepout zones and rule areas
    - Silkscreen, fab, courtyard layers
    - Design rules and board setup

Boards are skipped if they have fewer than MIN_FOOTPRINTS moveable footprints
or fewer than MIN_NETS signal nets, since they'd be useless for testing.
"""

import json
from pathlib import Path

MIN_FOOTPRINTS = 35  # skip small boards (radar-afe with 39 is near the lower bound)
MIN_NETS = 3         # skip boards with almost no connectivity

import os
import sys

# Suppress KiCad's verbose debug/warning output to stderr
# (duplicate image handler warnings, legacy zone fill warnings, etc.)
_devnull = open(os.devnull, "w")
os.dup2(_devnull.fileno(), 2)

try:
    import pcbnew
except ImportError:
    sys.exit(
        "ERROR: pcbnew not found.\n"
        "Run this script with KiCad's Python:\n"
        "  /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework"
        "/Versions/Current/bin/python3 scripts/strip_routing.py"
    )


def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def strip_board(src_path: Path, dst_path: Path) -> dict:
    """
    Load src_path, strip routing, save to dst_path.
    Returns a dict with stats, or raises on failure.
    """
    board = pcbnew.LoadBoard(str(src_path))

    # --- Remove tracks and vias ---
    tracks = list(board.GetTracks())
    n_tracks = 0
    n_vias = 0
    for t in tracks:
        if t.GetClass() == "PCB_VIA":
            n_vias += 1
        else:
            n_tracks += 1
        board.RemoveNative(t)

    # --- Remove copper zones (fills/pours), keep keepouts/rule areas ---
    zones = list(board.Zones())
    n_zones = 0
    for z in zones:
        if z.GetIsRuleArea():
            continue  # keepout zone — keep it
        if z.IsOnCopperLayer():
            n_zones += 1
            board.RemoveNative(z)

    # --- Count moveable footprints and signal nets ---
    footprints = list(board.GetFootprints())
    moveable = [fp for fp in footprints if not fp.IsLocked()]
    n_moveable = len(moveable)
    n_total_fps = len(footprints)

    # Count signal nets (exclude unconnected net 0 and power nets heuristically)
    netinfo = board.GetNetInfo()
    nets = [netinfo.GetNetItem(i) for i in range(netinfo.GetNetCount())]
    signal_nets = [
        n for n in nets
        if n.GetNetCode() > 0 and not _is_power_net(n.GetNetname())
    ]
    n_signal_nets = len(signal_nets)

    # --- Save (only the .kicad_pcb — delete the .kicad_pro/.kicad_prl that pcbnew creates) ---
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    board.Save(str(dst_path))
    for ext in (".kicad_pro", ".kicad_prl"):
        sidecar = dst_path.with_suffix(ext)
        if sidecar.exists():
            sidecar.unlink()

    return {
        "tracks_removed": n_tracks,
        "vias_removed": n_vias,
        "copper_zones_removed": n_zones,
        "footprints_total": n_total_fps,
        "footprints_moveable": n_moveable,
        "signal_nets": n_signal_nets,
    }


def _is_power_net(name: str) -> bool:
    import re
    name = name.lstrip("/")
    patterns = [
        r"^GND", r"^AGND$", r"^DGND$", r"^PGND$", r"^SGND$",
        r"^VSS", r"^VCC", r"^VDD", r"^VEE$", r"^VBAT$", r"^VBUS$",
        r"^\+\d+V", r"^-\d+V",
    ]
    return any(re.match(p, name, re.IGNORECASE) for p in patterns)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Strip routing from KiCad PCB files")
    parser.add_argument("--corpus", default=str(Path.home() / "cadmust_testboards"),
                        help="Corpus directory (default: ~/cadmust_testboards)")
    parser.add_argument("--min-fps", type=int, default=MIN_FOOTPRINTS,
                        help=f"Min moveable footprints (default: {MIN_FOOTPRINTS})")
    parser.add_argument("--min-nets", type=int, default=MIN_NETS,
                        help=f"Min signal nets (default: {MIN_NETS})")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    unrouted_dir = corpus_dir / "unrouted"
    manifest_path = corpus_dir / "manifest.json"

    manifest = load_manifest(manifest_path)
    if not manifest:
        sys.exit(f"No manifest.json found in {corpus_dir}. Run scrape_boards.py first.")

    pcb_files = sorted(corpus_dir.glob("*.kicad_pcb"))
    print(f"Found {len(pcb_files)} boards in {corpus_dir}")
    print(f"Output: {unrouted_dir}\n")

    ok = skipped = failed = 0

    for src_path in pcb_files:
        key = src_path.name
        entry = manifest.get(key, {})

        # Skip if already stripped AND file actually exists
        dst_path = unrouted_dir / src_path.name
        if entry.get("strip_status") == "ok" and dst_path.exists():
            print(f"  ALREADY DONE: {key}")
            ok += 1
            continue

        print(f"  Processing: {key}")
        try:
            stats = strip_board(src_path, dst_path)
        except Exception as e:
            print(f"    FAILED: {e}")
            entry["strip_status"] = "failed"
            entry["strip_error"] = str(e)
            manifest[key] = entry
            save_manifest(manifest_path, manifest)
            failed += 1
            continue

        # Quality filter
        if stats["footprints_moveable"] < args.min_fps:
            print(f"    SKIP (only {stats['footprints_moveable']} moveable footprints)")
            dst_path.unlink(missing_ok=True)
            entry["strip_status"] = "skipped_too_small"
            manifest[key] = entry
            save_manifest(manifest_path, manifest)
            skipped += 1
            continue

        if stats["signal_nets"] < args.min_nets:
            print(f"    SKIP (only {stats['signal_nets']} signal nets)")
            dst_path.unlink(missing_ok=True)
            entry["strip_status"] = "skipped_no_nets"
            manifest[key] = entry
            save_manifest(manifest_path, manifest)
            skipped += 1
            continue

        was_routed = stats["tracks_removed"] > 0 or stats["vias_removed"] > 0
        print(
            f"    OK — {stats['footprints_moveable']} fps, "
            f"{stats['signal_nets']} nets, "
            f"removed {stats['tracks_removed']} tracks / "
            f"{stats['vias_removed']} vias / "
            f"{stats['copper_zones_removed']} zones"
            + (" (was already unrouted)" if not was_routed else "")
        )

        entry["strip_status"] = "ok"
        entry["strip_stats"] = stats
        manifest[key] = entry
        save_manifest(manifest_path, manifest)
        ok += 1

    print(f"\nDone. {ok} ok, {skipped} skipped, {failed} failed.")
    print(f"Unrouted boards in: {unrouted_dir}")


if __name__ == "__main__":
    main()
