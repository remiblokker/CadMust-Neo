#!/usr/bin/env python3
"""
apply_model.py: write a pickled BoardModel (e.g. batch_runner --save-model)
back onto a .kicad_pcb and save it as a new file.

MUST be run with KiCad's Python (pcbnew):

    /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/\
Versions/Current/bin/python3 scripts/apply_model.py \
        --model post_sa.pkl --board unrouted.kicad_pcb --out placed.kicad_pcb

Positions and rotations come from plugin.placement.apply_model_to_board.
That function skips footprints locked on the board; a model run with
batch_runner --unlock moved those too, so by default every footprint the
model has as unlocked is unlocked for the apply and re-locked afterwards
(--respect-board-locks keeps the old behaviour). The SA never flips sides,
so board side is not written; a side mismatch between model and board is
counted and reported, it means the pickle belongs to another board.

save_optimized.py is the corpus-layout counterpart (position JSON files,
no pickle); it is left unchanged.
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pcbnew
except ImportError:
    sys.exit("ERROR: pcbnew not found. Run with KiCad's Python (see the docstring).")

from plugin.placement import apply_model_to_board  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Apply a BoardModel pickle to a .kicad_pcb")
    p.add_argument("--model", required=True, help="BoardModel pickle")
    p.add_argument("--board", required=True, help="source .kicad_pcb (typically unrouted)")
    p.add_argument("--out", required=True, help="output .kicad_pcb")
    p.add_argument("--respect-board-locks", action="store_true",
                   help="do not move footprints that are locked on the board")
    args = p.parse_args(argv)

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    board = pcbnew.LoadBoard(args.board)
    by_uuid, by_ref = {}, {}
    for fp in board.GetFootprints():
        by_uuid[fp.m_Uuid.AsString()] = fp
        by_ref[fp.GetReference()] = fp

    matched = unmatched = side_mismatch = 0
    unlocked_for_apply = []
    for mfp in model.footprints:
        kfp = by_uuid.get(mfp.uuid_str) if mfp.uuid_str else None
        if kfp is None:
            kfp = by_ref.get(mfp.reference)
        if kfp is None:
            unmatched += 1
            continue
        matched += 1
        if bool(getattr(mfp, "is_front", True)) == bool(kfp.IsFlipped()):
            side_mismatch += 1
        if kfp.IsLocked() and not mfp.locked and not args.respect_board_locks:
            kfp.SetLocked(False)
            unlocked_for_apply.append(kfp)

    try:
        apply_model_to_board(board, model)
    except AttributeError:
        # pcbnew.Refresh() needs a running editor; positions are already set.
        pass
    for kfp in unlocked_for_apply:
        kfp.SetLocked(True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    board.Save(str(out))
    # pcbnew writes default sidecars; carry the source project instead so
    # net classes and design rules travel with the placed board.
    for ext in (".kicad_pro", ".kicad_prl"):
        side = out.with_suffix(ext)
        if side.exists():
            side.unlink()
    src_pro = Path(args.board).with_suffix(".kicad_pro")
    if src_pro.exists():
        shutil.copyfile(src_pro, out.with_suffix(".kicad_pro"))

    report = {"model": args.model, "board": args.board, "out": str(out),
              "matched": matched, "unmatched": unmatched,
              "side_mismatch": side_mismatch,
              "relocked_after_apply": len(unlocked_for_apply)}
    sys.stdout.write(json.dumps(report) + "\n")
    return 1 if side_mismatch or matched == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
