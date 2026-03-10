"""Apply optimized positions back to the KiCad board."""
from __future__ import annotations

from typing import List, Tuple
from .board_model import BoardModel


def save_original_positions(board) -> List[Tuple[str, int, int, float]]:
    """Save original positions of all footprints for undo."""
    positions = []
    for fp in board.GetFootprints():
        pos = fp.GetPosition()
        positions.append((
            fp.GetReference(),
            pos.x,
            pos.y,
            fp.GetOrientationDegrees(),
        ))
    return positions


def restore_original_positions(board, positions: List[Tuple[str, int, int, float]]):
    """Restore footprints to their original positions (undo)."""
    import pcbnew
    for ref, x, y, angle in positions:
        fp = board.FindFootprintByReference(ref)
        if fp is not None:
            fp.SetPosition(pcbnew.VECTOR2I(x, y))
            fp.SetOrientation(pcbnew.EDA_ANGLE(angle, pcbnew.DEGREES_T))
    board.GetConnectivity().RecalculateRatsnest()
    pcbnew.Refresh()


def apply_model_to_board(board, model: BoardModel):
    """Write the optimized positions from the BoardModel back to pcbnew."""
    import pcbnew

    fps_by_ref = {}
    for fp in board.GetFootprints():
        fps_by_ref[fp.GetReference()] = fp

    for mfp in model.footprints:
        kfp = fps_by_ref.get(mfp.reference)
        if kfp is None or kfp.IsLocked():
            continue
        kfp.SetPosition(pcbnew.VECTOR2I(mfp.x, mfp.y))
        kfp.SetOrientation(pcbnew.EDA_ANGLE(mfp.angle_deg, pcbnew.DEGREES_T))

    board.GetConnectivity().RecalculateRatsnest()
    pcbnew.Refresh()
