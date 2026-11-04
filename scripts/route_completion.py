#!/usr/bin/env python3
"""
route_completion.py: route a placed board with KiCadRoutingTools (KRT) and
report how much of it routes. A routability measurement for a placement, not
a production route.

Run with the KRT venv (no pcbnew needed for routing and checking):

    ~/.venvs/krt/bin/python scripts/route_completion.py BOARD.kicad_pcb [options]

Install (once, see CadMust-Autopilot docs/research/KRT_METRIC_NOTES.md):

    git clone https://github.com/drandyhaas/KiCadRoutingTools ~/tools/KiCadRoutingTools
    /opt/homebrew/bin/python3.13 -m venv ~/.venvs/krt
    ~/.venvs/krt/bin/pip install -r ~/tools/KiCadRoutingTools/requirements.txt
    cd ~/tools/KiCadRoutingTools && ~/.venvs/krt/bin/python build_router.py   # prebuilt Rust .so, no cargo

KRT location: --krt-dir, else $KRT_DIR, else ~/tools/KiCadRoutingTools.

Typical chain for an SA result (batch_runner --save-model post_sa.pkl):

    <KiCad python> scripts/apply_model.py --model post_sa.pkl \
        --board unrouted.kicad_pcb --out placed.kicad_pcb
    ~/.venvs/krt/bin/python scripts/route_completion.py placed.kicad_pcb --out rc.json

Read completion together with the overlap count and the DRC number: parts
stacked on top of each other short pads of different nets, and the shorted
copper counts as connected (validation: all parts piled in one spot gave
completion_links 0.49 with 26908 DRC violations, completion_nets 0.13).

What it does:
  1. Copies the board (and its sibling .kicad_pro) into a work directory.
  2. If the board carries routing (tracks, vias, copper pours), strips it with
     scripts/strip_routing.strip_board(), which needs KiCad's Python; that is
     run as a subprocess (--kicad-python). --no-strip skips this.
  3. Counts open connections on the unrouted board (KRT check_connected).
  4. Runs KRT py_router/route.py on it (subprocess, wall-clock limited).
  5. Counts open connections on the routed board (KRT check_connected) and,
     unless --no-drc, runs KRT check_drc.py.
  6. Prints one JSON object on stdout. All KRT chatter goes to the log file
     in the work directory (kept with --keep DIR).

Metric definitions (the primary number is completion_links):
  links_total     sum over nets with >= 2 pads of (connected components - 1)
                  on the unrouted board, i.e. the ratsnest edges to route
  links_unrouted  the same sum on the routed board
  completion_links = 1 - links_unrouted / links_total
  nets_total / nets_complete / completion_nets: per net, all pads joined

Exit code: 0 whenever a measurement was made, however low the completion.
Non-zero only when the tool chain failed: 2 = bad input / KRT not found,
3 = router failed or timed out (JSON still printed, with "status").
"""

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_KRT_DIR = Path.home() / "tools" / "KiCadRoutingTools"
DEFAULT_KICAD_PYTHON = ("/Applications/KiCad/KiCad.app/Contents/Frameworks/"
                        "Python.framework/Versions/Current/bin/python3")

# Defaults, chosen to measure rather than to ship. --quick 1 is KRT's own
# "route to judge, not to ship" mode (main pass only, no rescue/escalation
# laps), built for ranking placements. Escalation "off" keeps sizes and
# clearances exact, so a hard placement cannot score well by shrinking
# tracks below the board's rules.
DEFAULTS = {
    "grid_step": 0.1,
    "quick": 1,
    "escalation": "off",
    "time_limit_s": 900,
}


def _routing_present(pcb_text: str) -> dict:
    """Cheap S-expression count of copper routing items in a board file."""
    segs = len(re.findall(r"^\s*\((?:segment|arc)\b", pcb_text, re.M))
    vias = len(re.findall(r"^\s*\(via\b", pcb_text, re.M))
    # Copper pours: zones that are not rule areas (no keepout clause before
    # the polygon data). The head is cut at the next zone so a following
    # rule area cannot mask a pour.
    zones = 0
    starts = [m.start() for m in re.finditer(r"^\s*\(zone\b", pcb_text, re.M)]
    for i, st in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(pcb_text)
        head = pcb_text[st:min(end, st + 1500)]
        head = head.split("(polygon", 1)[0]
        if "(keepout" not in head:
            zones += 1
    return {"segments": segs, "vias": vias, "copper_zones": zones}


def _krt_version(krt_dir: Path) -> dict:
    info = {"dir": str(krt_dir)}
    try:
        info["version"] = (krt_dir / "VERSION").read_text().strip()
    except OSError:
        info["version"] = None
    try:
        info["commit"] = subprocess.run(
            ["git", "-C", str(krt_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["commit"] = None
    try:
        sys.path.insert(0, str(krt_dir / "rust_router"))
        import grid_router  # noqa: F401
        info["rust_router"] = getattr(grid_router, "__version__", "present")
    except Exception as e:  # pragma: no cover - environment dependent
        info["rust_router"] = f"missing ({e.__class__.__name__})"
    return info


def connectivity(krt_dir: Path, pcb_file: Path) -> dict:
    """Open-connection counts via KRT check_connected.run_connectivity_check."""
    py_router = str(krt_dir / "py_router")
    if py_router not in sys.path:
        sys.path.insert(0, py_router)
    from check_connected import run_connectivity_check  # type: ignore

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        issues = run_connectivity_check(str(pcb_file), quiet=True)
    issues = [i for i in issues if "net_id" in i]
    links_open = sum(max(int(i.get("num_components", 1)) - 1, 0) for i in issues)
    return {
        "nets_open": len(issues),
        "links_open": links_open,
        "open_nets": sorted(i["net_name"] for i in issues),
    }


def net_count(krt_dir: Path, pcb_file: Path) -> int:
    """Nets with at least two pads (the ones that need routing at all)."""
    py_router = str(krt_dir / "py_router")
    if py_router not in sys.path:
        sys.path.insert(0, py_router)
    from kicad_parser import parse_kicad_pcb  # type: ignore

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        data = parse_kicad_pcb(str(pcb_file))
    return sum(1 for nid, pads in data.pads_by_net.items()
               if nid and len(pads) >= 2)


def strip(src: Path, dst: Path, kicad_python: str) -> None:
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from strip_routing import strip_board; "
            "from pathlib import Path; "
            "strip_board(Path(sys.argv[2]), Path(sys.argv[3]))")
    r = subprocess.run([kicad_python, "-c", code, str(SCRIPTS_DIR), str(src), str(dst)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not dst.exists():
        raise RuntimeError(f"strip_board failed (exit {r.returncode}): {r.stdout[-500:]}")


def build_route_cmd(krt_dir: Path, src: Path, dst: Path, summary: Path,
                    args) -> list:
    cmd = [sys.executable, str(krt_dir / "py_router" / "route.py"),
           str(src), str(dst), "--json-out", str(summary),
           "--grid-step", str(args.grid_step),
           "--escalation", args.escalation]
    if args.quick:
        cmd += ["--quick", str(args.quick)]
    if args.layers:
        cmd += ["--layers", *args.layers]
    if args.nets:
        cmd += ["--nets", *args.nets]
    if args.krt_arg:
        cmd += args.krt_arg
    return cmd


def run_drc(krt_dir: Path, pcb: Path, work: Path, log) -> dict:
    out = work / "drc.json"
    r = subprocess.run([sys.executable, str(krt_dir / "py_router" / "check_drc.py"),
                        str(pcb), "--quiet", "--json", str(out)],
                       stdout=log, stderr=subprocess.STDOUT, timeout=600)
    try:
        d = json.loads(out.read_text())
        return {"violations": d.get("violations"),
                "by_type": {k: (len(v) if isinstance(v, list) else v)
                            for k, v in (d.get("by_type") or {}).items()},
                "exit": r.returncode}
    except (OSError, ValueError):
        return {"violations": None, "exit": r.returncode}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("board", help="placed .kicad_pcb (routed or not)")
    p.add_argument("--krt-dir", default=os.environ.get("KRT_DIR", str(DEFAULT_KRT_DIR)))
    p.add_argument("--kicad-python", default=DEFAULT_KICAD_PYTHON,
                   help="KiCad's Python, used only to strip existing routing")
    p.add_argument("--layers", nargs="+", help="routing layers (default: all copper layers)")
    p.add_argument("--nets", nargs="+", help="KRT net patterns, e.g. '*' '!GND' (default: all)")
    p.add_argument("--grid-step", type=float, default=DEFAULTS["grid_step"])
    p.add_argument("--quick", type=int, default=DEFAULTS["quick"],
                   help="KRT --quick level, 0 = full run with rescue laps (default 1)")
    p.add_argument("--escalation", choices=["off", "board", "fab"],
                   default=DEFAULTS["escalation"])
    p.add_argument("--time-limit", type=float, default=DEFAULTS["time_limit_s"],
                   help="router wall-clock limit in seconds")
    p.add_argument("--krt-arg", action="append", default=[],
                   help="extra raw argument for route.py (repeatable)")
    p.add_argument("--project", metavar="KICAD_PRO",
                   help="project file with the net classes / design rules "
                        "(default: the board's sibling .kicad_pro). Without one "
                        "KRT falls back to a stock 0.25 mm clearance, which boxes "
                        "in fine-pitch pads")
    p.add_argument("--no-strip", action="store_true", help="do not strip existing routing")
    p.add_argument("--no-drc", action="store_true")
    p.add_argument("--keep", metavar="DIR", help="keep the work directory (routed board, log) here")
    p.add_argument("--out", metavar="FILE", help="also write the JSON to FILE")
    args = p.parse_args(argv)

    board = Path(args.board).resolve()
    krt_dir = Path(args.krt_dir).expanduser()
    if not board.exists():
        print(json.dumps({"status": "error", "error": f"board not found: {board}"}))
        return 2
    if not (krt_dir / "py_router" / "route.py").exists():
        print(json.dumps({"status": "error", "error": f"KRT not found in {krt_dir}"}))
        return 2

    if args.keep:
        work = Path(args.keep).resolve()
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix="route_completion_"))
        cleanup = True

    result = {
        "tool": "route_completion.py",
        "board": str(board),
        "status": "ok",
        "krt": _krt_version(krt_dir),
        "options": {"layers": args.layers, "nets": args.nets,
                    "grid_step": args.grid_step, "quick": args.quick,
                    "escalation": args.escalation, "time_limit_s": args.time_limit,
                    "krt_args": args.krt_arg},
    }
    rc = 0
    log_path = work / "route.log"
    t_start = time.time()
    try:
        src = work / "input.kicad_pcb"
        pro = Path(args.project).resolve() if args.project else board.with_suffix(".kicad_pro")
        present = _routing_present(board.read_text(errors="replace"))
        result["input_routing"] = present
        if any(present.values()) and not args.no_strip:
            strip(board, src, args.kicad_python)
            result["stripped"] = True
        else:
            shutil.copyfile(board, src)
            result["stripped"] = False
        if pro.exists():
            shutil.copyfile(pro, src.with_suffix(".kicad_pro"))
            result["project"] = str(pro)
        else:
            result["project"] = None   # KRT uses stock net-class defaults

        before = connectivity(krt_dir, src)
        nets_total = before["nets_open"]
        links_total = before["links_open"]
        result["nets_with_2plus_pads"] = net_count(krt_dir, src)

        routed = work / "routed.kicad_pcb"
        summary = work / "route_summary.json"
        cmd = build_route_cmd(krt_dir, src, routed, summary, args)
        result["command"] = " ".join(cmd[1:])
        t0 = time.time()
        with open(log_path, "w") as log:
            try:
                r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=str(krt_dir), timeout=args.time_limit)
                result["router_exit"] = r.returncode
            except subprocess.TimeoutExpired:
                result["status"] = "timeout"
            result["router_seconds"] = round(time.time() - t0, 2)

            if result["status"] == "ok" and not routed.exists():
                result["status"] = "router_failed"
            if result["status"] != "ok":
                rc = 3
            else:
                after = connectivity(krt_dir, routed)
                result.update({
                    "nets_total": nets_total,
                    "nets_complete": nets_total - after["nets_open"],
                    "links_total": links_total,
                    "links_unrouted": after["links_open"],
                    "completion_links": (round(1 - after["links_open"] / links_total, 4)
                                         if links_total else 1.0),
                    "completion_nets": (round(1 - after["nets_open"] / nets_total, 4)
                                        if nets_total else 1.0),
                    "unrouted_nets": after["open_nets"],
                })
                try:
                    s = json.loads(summary.read_text())
                    result["router_summary"] = {
                        k: s.get(k) for k in ("successful", "failed", "total_vias",
                                              "total_time", "min_clearance_used",
                                              "multipoint_pads_connected",
                                              "multipoint_pads_total")}
                except (OSError, ValueError):
                    result["router_summary"] = None
                if not args.no_drc:
                    result["drc"] = run_drc(krt_dir, routed, work, log)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{e.__class__.__name__}: {e}"
        rc = 3
    result["total_seconds"] = round(time.time() - t_start, 2)
    result["work_dir"] = None if cleanup else str(work)
    if args.keep:
        result["log"] = str(log_path)

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if cleanup:
        shutil.rmtree(work, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
