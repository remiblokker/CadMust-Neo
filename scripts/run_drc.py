#!/usr/bin/env python3
"""
run_drc.py — Run KiCad DRC on all unrouted boards and record baseline results.

Run with regular python3 (no KiCad Python needed — uses kicad-cli):

    python3 scripts/run_drc.py                        # baseline on unrouted/
    python3 scripts/run_drc.py --dir path/to/boards   # any board directory

DRC reports (JSON) are saved to <dir>/drc_reports/.
A summary is written to manifest.json and printed to stdout.

The baseline pass should be run BEFORE optimizing. After optimization, run
again with --dir pointing to the optimized boards directory, then compare.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

KICAD_CLI = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"


def run_drc(board_path: Path, report_path: Path) -> dict:
    """
    Run kicad-cli pcb drc on board_path, save JSON to report_path.
    Returns parsed report dict, or raises on failure.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        KICAD_CLI, "pcb", "drc",
        "--output", str(report_path),
        "--format", "json",
        "--severity-error",
        "--severity-warning",
        str(board_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    # kicad-cli returns non-zero when there are DRC violations — that's expected
    if not report_path.exists():
        raise RuntimeError(
            f"kicad-cli produced no output.\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    with open(report_path) as f:
        return json.load(f)


# Violation types that are intrinsic to footprint/pad design and are NOT
# affected by component placement. Excluded from placement-relevant counts.
PLACEMENT_IRRELEVANT = {
    "soldermask_bridge",   # pad-to-pad soldermask overlap — footprint property
    "copper_sliver",       # thin copper artifact — footprint geometry
    "footprint_type_mismatch",  # SMD/THT mismatch — footprint property
    "lib_footprint_issues",     # library mismatch — not placement
    "clearance",           # on stripped boards: pad-to-pad within same footprint (footprint design issue)
    "solder_mask_bridge",  # alternate spelling used by some KiCad versions
}


def summarize_report(report: dict) -> dict:
    """Extract counts and violation types from a DRC JSON report."""
    violations = report.get("violations", [])
    unconnected = report.get("unconnected_items", [])

    errors = [v for v in violations if v.get("severity") == "error"]
    warnings = [v for v in violations if v.get("severity") == "warning"]

    # Placement-relevant = excludes intrinsic footprint violations
    placement_errors = [v for v in errors if v.get("type") not in PLACEMENT_IRRELEVANT]
    placement_warnings = [v for v in warnings if v.get("type") not in PLACEMENT_IRRELEVANT]

    # Collect unique violation type names
    error_types = sorted(set(v.get("type", "?") for v in errors))
    warning_types = sorted(set(v.get("type", "?") for v in warnings))

    return {
        "drc_errors": len(errors),
        "drc_warnings": len(warnings),
        "drc_errors_placement": len(placement_errors),
        "drc_warnings_placement": len(placement_warnings),
        "unconnected": len(unconnected),
        "error_types": error_types,
        "warning_types": warning_types,
    }


def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run DRC on KiCad boards")
    parser.add_argument("--corpus", default=str(Path.home() / "cadmust_testboards"),
                        help="Corpus root (default: ~/cadmust_testboards)")
    parser.add_argument("--dir", default=None,
                        help="Board directory to scan (default: <corpus>/unrouted)")
    parser.add_argument("--tag", default="baseline",
                        help="Tag for manifest key, e.g. 'baseline' or 'optimized' (default: baseline)")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run DRC even if already done")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    board_dir = Path(args.dir) if args.dir else corpus_dir / "unrouted"
    drc_dir = board_dir / "drc_reports"
    manifest_path = corpus_dir / "manifest.json"

    manifest = load_manifest(manifest_path)

    pcb_files = sorted(board_dir.glob("*.kicad_pcb"))
    if not pcb_files:
        sys.exit(f"No .kicad_pcb files found in {board_dir}")

    print(f"Board dir : {board_dir}")
    print(f"DRC tag   : {args.tag}")
    print(f"Boards    : {len(pcb_files)}")
    print()

    ok = skipped = failed = 0
    total_errors = total_warnings = total_unconnected = 0
    clean_boards = []

    for board_path in pcb_files:
        key = board_path.name
        entry = manifest.get(key, {})
        drc_key = f"drc_{args.tag}"

        if not args.rerun and drc_key in entry:
            s = entry[drc_key]
            print(f"  CACHED  {key}: {s['drc_errors']}E {s['drc_warnings']}W {s['unconnected']}U")
            total_errors += s["drc_errors"]
            total_warnings += s["drc_warnings"]
            total_unconnected += s["unconnected"]
            if s["drc_errors"] == 0 and s["unconnected"] == 0:
                clean_boards.append(key)
            skipped += 1
            continue

        report_path = drc_dir / (board_path.stem + ".json")
        try:
            report = run_drc(board_path, report_path)
            summary = summarize_report(report)
        except Exception as e:
            print(f"  FAILED  {key}: {e}")
            entry[drc_key] = {"drc_status": "failed", "error": str(e)}
            manifest[key] = entry
            save_manifest(manifest_path, manifest)
            failed += 1
            continue

        e_count = summary["drc_errors"]
        w_count = summary["drc_warnings"]
        u_count = summary["unconnected"]
        total_errors += e_count
        total_warnings += w_count
        total_unconnected += u_count

        pe_count = summary["drc_errors_placement"]
        status = "CLEAN " if pe_count == 0 and u_count == 0 else "ISSUES"
        print(f"  {status} {key}: {e_count}E ({pe_count} placement) {w_count}W {u_count}U")
        if summary["error_types"]:
            print(f"         errors: {', '.join(summary['error_types'][:5])}")

        if e_count == 0 and u_count == 0:
            clean_boards.append(key)

        entry[drc_key] = {"drc_status": "ok", **summary}
        manifest[key] = entry
        save_manifest(manifest_path, manifest)
        ok += 1

    print()
    print(f"Results: {ok} ran, {skipped} cached, {failed} failed")
    print(f"Total violations: {total_errors} errors, {total_warnings} warnings, {total_unconnected} unconnected")
    print(f"Clean boards (0 errors, 0 unconnected): {len(clean_boards)}/{ok+skipped}")
    print(f"DRC reports in: {drc_dir}")


if __name__ == "__main__":
    main()
