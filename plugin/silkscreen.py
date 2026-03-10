"""Auto-place silkscreen reference designators after SA placement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from .board_model import BoardModel


# Clearance between text rectangles and obstacles (0.2 mm)
SILK_CLEARANCE = 200_000


@dataclass
class TextRect:
    """A reference designator's bounding rectangle."""
    fp_index: int       # which footprint this belongs to
    cx: int             # current center x (nm)
    cy: int             # current center y (nm)
    width: int          # text bbox width (nm)
    height: int         # text bbox height (nm)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        hw, hh = self.width // 2, self.height // 2
        return (self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh)


@dataclass
class SilkscreenModel:
    """All data needed for silkscreen placement — pure Python."""
    texts: List[TextRect]
    fp_bboxes: List[Tuple[int, int, int, int]]   # indexed by fp_index
    keepouts: List[Tuple[int, int, int, int]]
    board_bbox: Tuple[int, int, int, int]         # (xmin, ymin, xmax, ymax)


def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """Check if two axis-aligned bounding boxes overlap."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _inside(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> bool:
    """Check if inner bbox is fully inside outer bbox."""
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _candidate_bbox(cx: int, cy: int, w: int, h: int) -> Tuple[int, int, int, int]:
    hw, hh = w // 2, h // 2
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def _obstacle_density(text: TextRect, model: SilkscreenModel) -> int:
    """Count how many obstacles are near this text's parent footprint."""
    parent = model.fp_bboxes[text.fp_index]
    # Expand parent bbox by a search radius
    radius = max(text.width, text.height) * 2
    search = (parent[0] - radius, parent[1] - radius,
              parent[2] + radius, parent[3] + radius)
    count = 0
    for i, fb in enumerate(model.fp_bboxes):
        if i != text.fp_index and _overlaps(search, fb):
            count += 1
    return count


def _expand_bbox(bbox: Tuple[int, int, int, int],
                 margin: int) -> Tuple[int, int, int, int]:
    """Expand a bbox by margin on all sides."""
    return (bbox[0] - margin, bbox[1] - margin,
            bbox[2] + margin, bbox[3] + margin)


def _overlap_area(a: Tuple[int, int, int, int],
                  b: Tuple[int, int, int, int]) -> int:
    """Compute overlap area between two bboxes (0 if no overlap)."""
    ox = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ox * oy


def _generate_candidates(parent: Tuple[int, int, int, int],
                         tw: int, th: int) -> List[Tuple[int, int]]:
    """Generate candidate positions around a parent bbox.

    Returns (cx, cy) for each candidate, ordered by preference:
    cardinal directions first (top, bottom, left, right),
    then diagonal corners, then cardinal at increasing clearance.
    """
    px_mid = (parent[0] + parent[2]) // 2
    py_mid = (parent[1] + parent[3]) // 2
    hw, hh = tw // 2, th // 2
    cl = SILK_CLEARANCE

    candidates = [
        # Cardinal: top, bottom, right, left
        (px_mid, parent[1] - hh - cl),
        (px_mid, parent[3] + hh + cl),
        (parent[2] + hw + cl, py_mid),
        (parent[0] - hw - cl, py_mid),
        # Diagonal corners (NE, SE, SW, NW)
        (parent[2] + hw + cl, parent[1] - hh - cl),
        (parent[2] + hw + cl, parent[3] + hh + cl),
        (parent[0] - hw - cl, parent[3] + hh + cl),
        (parent[0] - hw - cl, parent[1] - hh - cl),
        # Cardinal at 3x clearance
        (px_mid, parent[1] - hh - cl * 3),
        (px_mid, parent[3] + hh + cl * 3),
        (parent[2] + hw + cl * 3, py_mid),
        (parent[0] - hw - cl * 3, py_mid),
        # Diagonal at 3x clearance
        (parent[2] + hw + cl * 3, parent[1] - hh - cl * 3),
        (parent[2] + hw + cl * 3, parent[3] + hh + cl * 3),
        (parent[0] - hw - cl * 3, parent[3] + hh + cl * 3),
        (parent[0] - hw - cl * 3, parent[1] - hh - cl * 3),
        # Cardinal at 6x clearance (desperate)
        (px_mid, parent[1] - hh - cl * 6),
        (px_mid, parent[3] + hh + cl * 6),
        (parent[2] + hw + cl * 6, py_mid),
        (parent[0] - hw - cl * 6, py_mid),
    ]
    return candidates


def _score_candidate(cbox: Tuple[int, int, int, int],
                     text_fp_index: int,
                     model: SilkscreenModel,
                     placed_bboxes: List[Tuple[int, int, int, int]]) -> int:
    """Score a candidate position by total overlap area (lower is better)."""
    total = 0
    for i, fb in enumerate(model.fp_bboxes):
        if i == text_fp_index:
            continue
        total += _overlap_area(cbox, fb)
    for ko in model.keepouts:
        total += _overlap_area(cbox, ko)
    for pb in placed_bboxes:
        total += _overlap_area(cbox, _expand_bbox(pb, SILK_CLEARANCE))
    return total


def place_silkscreen(model: SilkscreenModel) -> List[Tuple[int, int]]:
    """
    Find collision-free positions for silkscreen reference designators.

    For each text, tries 20 candidate positions around the parent footprint
    (4 cardinal, 4 diagonal, at 1x/3x/6x clearance distances) and picks
    the closest collision-free one. If no collision-free position exists,
    picks the candidate with the least total overlap (instead of keeping
    the original position which may be worse).

    Returns a list of (cx, cy) positions, one per text in model.texts.
    """
    if not model.texts:
        return []

    # Sort by density — most constrained first so they get priority
    order = sorted(range(len(model.texts)),
                   key=lambda i: _obstacle_density(model.texts[i], model),
                   reverse=True)

    results: List[Optional[Tuple[int, int]]] = [None] * len(model.texts)
    placed_bboxes: List[Tuple[int, int, int, int]] = []

    for idx in order:
        text = model.texts[idx]
        parent = model.fp_bboxes[text.fp_index]
        px_mid = (parent[0] + parent[2]) // 2
        py_mid = (parent[1] + parent[3]) // 2

        candidates = _generate_candidates(parent, text.width, text.height)

        best = None
        best_dist = float('inf')
        # Fallback: track the candidate with minimum overlap
        fallback = None
        fallback_score = float('inf')

        for cx, cy in candidates:
            cbox = _candidate_bbox(cx, cy, text.width, text.height)

            # Must be inside board
            if not _inside(cbox, model.board_bbox):
                continue

            # Check collisions
            collision = False

            # Must not overlap any footprint bbox (except parent)
            for i, fb in enumerate(model.fp_bboxes):
                if i == text.fp_index:
                    continue
                if _overlaps(cbox, fb):
                    collision = True
                    break

            if not collision:
                # Must not overlap any keep-out zone
                for ko in model.keepouts:
                    if _overlaps(cbox, ko):
                        collision = True
                        break

            if not collision:
                # Must not overlap any already-placed text (with clearance)
                for pb in placed_bboxes:
                    if _overlaps(cbox, _expand_bbox(pb, SILK_CLEARANCE)):
                        collision = True
                        break

            if not collision:
                # Collision-free — score by distance to parent center
                dist = abs(cx - px_mid) + abs(cy - py_mid)
                if dist < best_dist:
                    best = (cx, cy)
                    best_dist = dist
            else:
                # Track least-overlap fallback
                score = _score_candidate(cbox, text.fp_index,
                                         model, placed_bboxes)
                if score < fallback_score:
                    fallback = (cx, cy)
                    fallback_score = score

        if best is not None:
            chosen = best
        elif fallback is not None:
            # No collision-free position — use least-overlap candidate
            chosen = fallback
        else:
            # No candidate inside board at all — keep original
            chosen = (text.cx, text.cy)

        results[idx] = chosen
        placed_bboxes.append(_candidate_bbox(chosen[0], chosen[1],
                                             text.width, text.height))

    return results


def extract_silkscreen_model(board, board_model: BoardModel) -> SilkscreenModel:
    """Read visible silkscreen reference designators from pcbnew."""
    import pcbnew

    texts = []
    fp_bboxes = [fp.bbox for fp in board_model.footprints]

    # Build a map from reference string to footprint index
    ref_to_idx = {fp.reference: fp.index for fp in board_model.footprints}

    for fp in board.GetFootprints():
        ref_text = fp.Reference()
        if not ref_text.IsVisible():
            continue
        layer = ref_text.GetLayer()
        if layer not in (pcbnew.F_SilkS, pcbnew.B_SilkS):
            continue

        ref_str = fp.GetReference()
        fp_idx = ref_to_idx.get(ref_str)
        if fp_idx is None:
            continue

        bbox = ref_text.GetBoundingBox()
        cx = bbox.GetX() + bbox.GetWidth() // 2
        cy = bbox.GetY() + bbox.GetHeight() // 2

        texts.append(TextRect(
            fp_index=fp_idx,
            cx=cx, cy=cy,
            width=bbox.GetWidth(),
            height=bbox.GetHeight(),
        ))

    keepouts = [(ko.xmin, ko.ymin, ko.xmax, ko.ymax)
                for ko in board_model.keepouts]

    board_bbox = (board_model.outline_xmin, board_model.outline_ymin,
                  board_model.outline_xmax, board_model.outline_ymax)

    return SilkscreenModel(
        texts=texts,
        fp_bboxes=fp_bboxes,
        keepouts=keepouts,
        board_bbox=board_bbox,
    )


def apply_silkscreen(board, silk_model: SilkscreenModel,
                     new_positions: List[Tuple[int, int]],
                     board_model: Optional['BoardModel'] = None) -> int:
    """Write new reference positions back to pcbnew. Returns count moved."""
    import pcbnew

    # Build map from reference string to board footprint
    ref_to_kfp = {}
    for fp in board.GetFootprints():
        ref_to_kfp[fp.GetReference()] = fp

    # Build map from fp_index to reference string
    idx_to_ref: dict = {}
    if board_model is not None:
        for fp in board_model.footprints:
            idx_to_ref[fp.index] = fp.reference
    else:
        # Fallback: use silk_model texts to find references
        for fp in board.GetFootprints():
            ref_to_kfp[fp.GetReference()] = fp

    moved = 0
    for text, (new_cx, new_cy) in zip(silk_model.texts, new_positions):
        if new_cx == text.cx and new_cy == text.cy:
            continue

        # Look up the pcbnew footprint by reference string
        ref_str = idx_to_ref.get(text.fp_index)
        if ref_str is None:
            continue
        kfp = ref_to_kfp.get(ref_str)
        if kfp is None:
            continue

        ref_text = kfp.Reference()
        ref_text.SetPosition(pcbnew.VECTOR2I(new_cx, new_cy))
        moved += 1

    if moved > 0:
        pcbnew.Refresh()

    return moved
