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


def set_board_side(kfp, is_front: bool) -> bool:
    """Put a pcbnew footprint on the given side with a top-bottom flip around
    its own position (the mirror Footprint.flip_side models). Returns True
    when it flipped. KiCad 9+ takes a FLIP_DIRECTION, KiCad 7/8 a bool
    aFlipLeftRight (False = top-bottom)."""
    import pcbnew
    if bool(kfp.IsFlipped()) == (not is_front):
        return False
    pos = kfp.GetPosition()
    direction = getattr(pcbnew, "FLIP_DIRECTION_TOP_BOTTOM", None)
    if direction is not None:
        kfp.Flip(pos, direction)
    else:
        kfp.Flip(pos, False)
    return True


def apply_model_to_board(board, model: BoardModel) -> List[str]:
    """Write the optimized positions from the BoardModel back to pcbnew.

    Footprints the constraint file moved to the other side
    (Footprint.side_flipped) are flipped on the board first, also when they
    are locked there (the side is a constraint, not an SA move). Returns
    the references flipped.
    """
    import pcbnew

    fps_by_uuid = {}
    fps_by_ref = {}
    for fp in board.GetFootprints():
        try:
            fps_by_uuid[fp.m_Uuid.AsString()] = fp
        except Exception:
            pass
        fps_by_ref[fp.GetReference()] = fp

    flipped: List[str] = []
    for mfp in model.footprints:
        kfp = fps_by_uuid.get(mfp.uuid_str) if mfp.uuid_str else None
        if kfp is None:
            kfp = fps_by_ref.get(mfp.reference)
        if kfp is None:
            continue
        if getattr(mfp, "side_flipped", False):
            if set_board_side(kfp, getattr(mfp, "is_front", True)):
                flipped.append(mfp.reference)
        if kfp.IsLocked():
            continue
        kfp.SetPosition(pcbnew.VECTOR2I(mfp.x, mfp.y))
        kfp.SetOrientation(pcbnew.EDA_ANGLE(mfp.angle_deg, pcbnew.DEGREES_T))

    board.GetConnectivity().RecalculateRatsnest()
    try:
        pcbnew.Refresh()
    except AttributeError:
        pass   # no running editor (headless scripts)
    return flipped
