"""Simulated annealing engine with adaptive cooling, auto-calibrated T0,
reheating, and greedy refinement."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Tuple, Set

from .board_model import BoardModel
from .cost_function import CostState
from .moves import (
    select_move_type, do_translate, do_median, do_swap, do_rotate,
    revert_move, affected_indices, MoveUndo, _get_group_members,
    _compute_t_ratio,
)


@dataclass
class SAConfig:
    """Configuration for the SA optimizer."""
    calibration_samples: int = 200
    initial_accept_rate: float = 0.95

    moves_per_temp: int = 0             # 0 = auto (10 * num_moveable)
    cooling_fast: float = 0.80          # acceptance > 50%
    cooling_normal: float = 0.90        # acceptance 20-50%
    cooling_slow: float = 0.95          # acceptance < 20%

    penalty_scale_min: float = 0.10  # boundary+keepout scale at T=T0

    freeze_threshold: float = 0.01
    min_temperature: float = 1e-6
    max_iterations: int = 200

    reheat_count: int = 2            # number of reheats after initial SA
    reheat_ratio: float = 0.30       # reheat to this fraction of T0

    num_starts: int = 1              # multi-start: total independent runs (1 = classic single-run)

    callback_interval: float = 0.33  # seconds between UI updates (~3/sec)


@dataclass
class SAResult:
    """Result of an optimization run."""
    initial_cost: float
    final_cost: float
    best_cost: float
    improvement_pct: float
    temperature_steps: int
    total_moves: int
    accepted_moves: int
    cost_history: List[Tuple[int, float]] = field(default_factory=list)


def auto_calibrate_t0(
    model: BoardModel,
    cost_state: CostState,
    config: SAConfig,
) -> float:
    """
    Auto-calibrate starting temperature.

    Sample N random moves, collect uphill HPWL deltas (ignoring penalties),
    set T0 so that ~95% of median-sized uphill moves would be accepted.
    T0 = -median_delta / ln(initial_accept_rate)

    Using HPWL-only deltas prevents T0 inflation from penalty spikes
    (overlaps, boundary violations) which can be orders of magnitude
    larger than typical HPWL changes.
    """
    uphill_deltas: List[float] = []
    old_hpwl = float(cost_state.hpwl)

    # At calibration temperature (t_ratio=1.0, maximum exploration)
    board_w = model.outline_xmax - model.outline_xmin
    board_h = model.outline_ymax - model.outline_ymin
    max_span = max(board_w, board_h)
    calib_window = max_span // 2
    calib_noise = int(board_w * 0.02)

    for _ in range(config.calibration_samples):
        mt = select_move_type(1.0)
        if mt == "translate":
            undo = do_translate(model, 1.0, calib_window)
        elif mt == "median":
            undo = do_median(model, 1.0, calib_noise)
        elif mt == "swap":
            undo = do_swap(model)
            if undo is None:
                continue
        else:
            undo = do_rotate(model)

        cost_state.incremental_update(affected_indices(undo))
        new_hpwl = float(cost_state.hpwl)
        delta = new_hpwl - old_hpwl

        if delta > 0:
            uphill_deltas.append(delta)

        # Always revert — we're just sampling
        revert_move(model, undo)
        cost_state.incremental_update(affected_indices(undo))

    if not uphill_deltas:
        return old_hpwl * 0.1 if old_hpwl > 0 else 1000.0

    uphill_deltas.sort()
    median_delta = uphill_deltas[len(uphill_deltas) // 2]
    t0 = -median_delta / math.log(config.initial_accept_rate)
    return max(t0, 1.0)


@dataclass
class _ProfileData:
    """Timing data from one SA phase."""
    move_gen: float = 0.0
    incr_update: float = 0.0
    revert: float = 0.0
    callback: float = 0.0
    total_moves: int = 0
    n_reverts: int = 0

    def merge(self, other: '_ProfileData') -> '_ProfileData':
        return _ProfileData(
            self.move_gen + other.move_gen,
            self.incr_update + other.incr_update,
            self.revert + other.revert,
            self.callback + other.callback,
            self.total_moves + other.total_moves,
            self.n_reverts + other.n_reverts,
        )


def _sa_phase(
    model: BoardModel,
    cost_state: CostState,
    config: SAConfig,
    t0: float,
    start_temperature: float,
    max_steps: int,
    moves_per_temp: int,
    best_cost: float,
    best_hpwl: float,
    best_positions: List[Tuple[int, int, float]],
    total_moves: int,
    total_accepted: int,
    total_steps: int,
    overall_max_steps: int,
    cost_history: List[Tuple[int, float]],
    progress_callback: Optional[Callable] = None,
) -> Tuple[float, float, List[Tuple[int, int, float]],
           int, int, int, bool, '_ProfileData']:
    """
    Run one SA cooling phase. Returns updated best state and counters.

    best_cost is tracked via normalized_cost (penalties at full weight)
    so solutions are compared fairly regardless of current penalty_scale.
    Metropolis acceptance uses total_cost (with penalty_scale) so that
    boundary/keepout violations are cheap at high T (exploration) and
    expensive at low T (convergence).

    Returns: (best_cost, best_hpwl,
              best_positions,
              total_moves, total_accepted, total_steps, cancelled, profile)
    """
    temperature = start_temperature

    # Precompute board constants (fixed across the run)
    board_w = model.outline_xmax - model.outline_xmin
    board_h = model.outline_ymax - model.outline_ymin
    max_span = max(board_w, board_h)
    min_window = 100_000        # 0.1 mm
    max_window = max_span // 2

    # Update penalty_scale for starting temperature
    t_ratio = _compute_t_ratio(temperature, t0)
    scale = config.penalty_scale_min + (1.0 - config.penalty_scale_min) * (1.0 - t_ratio)
    cost_state.update_penalty_scale(scale)
    current_cost = cost_state.total_cost
    step = 0

    # Profiling accumulators
    t_move_gen = 0.0
    t_incr_update = 0.0
    t_revert = 0.0
    t_callback = 0.0
    n_reverts = 0
    last_callback_time = time.perf_counter()

    while temperature > config.min_temperature and step < max_steps:
        # Update penalty_scale for this temperature step
        t_ratio = _compute_t_ratio(temperature, t0)
        scale = config.penalty_scale_min + (1.0 - config.penalty_scale_min) * (1.0 - t_ratio)
        cost_state.update_penalty_scale(scale)
        current_cost = cost_state.total_cost  # refresh with new scale

        # Precompute move parameters for this temperature step (t_ratio fixed)
        window = int(min_window + (max_window - min_window) * t_ratio)
        noise = int(board_w * 0.02 * t_ratio)

        accepted = 0
        attempted = 0

        for _ in range(moves_per_temp):
            _t0 = time.perf_counter()
            mt = select_move_type(t_ratio)
            undo: Optional[MoveUndo] = None

            if mt == "translate":
                undo = do_translate(model, t_ratio, window)
            elif mt == "median":
                undo = do_median(model, t_ratio, noise)
            elif mt == "swap":
                undo = do_swap(model)
                if undo is None:
                    continue
            else:
                undo = do_rotate(model)
            _t1 = time.perf_counter()
            t_move_gen += _t1 - _t0

            attempted += 1
            ai = affected_indices(undo)
            snap = cost_state.snapshot(ai)
            new_cost = cost_state.incremental_update(ai)
            _t2 = time.perf_counter()
            t_incr_update += _t2 - _t1
            delta = new_cost - current_cost

            if delta <= 0:
                accept = True
            elif temperature > 0:
                accept = random.random() < math.exp(-delta / temperature)
            else:
                accept = False

            if accept:
                current_cost = new_cost
                accepted += 1
                # Best tracking uses normalized_cost (scale=1.0) so
                # solutions are compared fairly across temperature steps
                norm = cost_state.normalized_cost
                if norm < best_cost:
                    best_cost = norm
                    best_hpwl = float(cost_state.hpwl)
                    best_positions = _snapshot_positions(model)
            else:
                _t3 = time.perf_counter()
                revert_move(model, undo)
                cost_state.restore(snap)
                t_revert += time.perf_counter() - _t3
                n_reverts += 1

            total_moves += 1

            if progress_callback:
                now = time.perf_counter()
                if now - last_callback_time >= config.callback_interval:
                    last_callback_time = now
                    _t4 = time.perf_counter()
                    cont = progress_callback(
                        total_steps + step, overall_max_steps,
                        temperature, current_cost, best_cost, best_hpwl, t0,
                        float(cost_state.hpwl),
                    )
                    t_callback += time.perf_counter() - _t4
                    if not cont:
                        return (best_cost, best_hpwl,
                                best_positions, total_moves,
                                total_accepted, total_steps + step, True,
                                _ProfileData(t_move_gen, t_incr_update,
                                             t_revert, t_callback,
                                             attempted, n_reverts))

        total_accepted += accepted
        cost_history.append((total_moves, current_cost))

        # Adaptive cooling
        accept_rate = accepted / attempted if attempted > 0 else 0.0
        if accept_rate > 0.50:
            temperature *= config.cooling_fast
        elif accept_rate > 0.20:
            temperature *= config.cooling_normal
        else:
            temperature *= config.cooling_slow

        if accept_rate < config.freeze_threshold and temperature < t0 * 0.01:
            break

        step += 1

    return (best_cost, best_hpwl,
            best_positions,
            total_moves, total_accepted, total_steps + step, False,
            _ProfileData(t_move_gen, t_incr_update, t_revert, t_callback,
                         total_moves, n_reverts))


def _greedy_refine(
    model: BoardModel,
    cost_state: CostState,
    moved_indices: List[int],
    progress_callback: Optional[Callable] = None,
    overall_max_steps: int = 0,
    t0: float = 0.0,
    callback_interval: float = 0.33,
) -> int:
    """
    Greedy local search: try small perturbations on each moveable component.
    Accept only improvements. Repeat until no improvement in a full sweep.

    Returns number of accepted refinement moves.
    """
    from .moves import (_get_group_members, _clamp_group_to_board,
                        _hits_new_keepout, _hits_polygon_cutout,
                        _kickout_polygon_violations)

    # Force any components that SA left in a polygon cutout to the nearest
    # valid position before the greedy loop starts.
    _kickout_polygon_violations(model, moved_indices, cost_state)

    current_cost = cost_state.total_cost
    best_hpwl = float(cost_state.hpwl)
    total_refined = 0
    last_callback_time = time.perf_counter()

    # Nudge distances to try (nm)
    nudges = [50_000, 100_000, 200_000, 500_000, 1_000_000]  # 0.05-1.0mm
    rotations = [90.0, 180.0, 270.0]
    # Cardinal + diagonal directions
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    improved = True
    while improved:
        improved = False
        for mi in moved_indices:
            members = _get_group_members(model, mi)

            # Save state for this component/group
            old_states = [(idx, model.footprints[idx].x,
                           model.footprints[idx].y,
                           model.footprints[idx].angle_deg)
                          for idx in members]
            member_set: Set[int] = set(idx for idx in members)

            best_move_cost = current_cost
            best_move_states = None

            # Snapshot before trying candidates
            snap = cost_state.snapshot(member_set)

            # Try translation nudges in 8 directions at multiple distances
            for dist in nudges:
                for ddx, ddy in directions:
                    dx, dy = dist * ddx, dist * ddy
                    # Apply nudge
                    for idx in members:
                        model.footprints[idx].x += dx
                        model.footprints[idx].y += dy
                    _clamp_group_to_board(model, members)

                    # Skip if this enters a keepout or polygon cutout
                    if (_hits_new_keepout(model, members, old_states) or
                            _hits_polygon_cutout(model, members, old_states)):
                        for idx, ox, oy, oa in old_states:
                            model.footprints[idx].x = ox
                            model.footprints[idx].y = oy
                        continue

                    new_cost = cost_state.incremental_update(member_set)
                    if new_cost < best_move_cost - 1.0:
                        best_move_cost = new_cost
                        best_move_states = [
                            (idx, model.footprints[idx].x,
                             model.footprints[idx].y,
                             model.footprints[idx].angle_deg)
                            for idx in members
                        ]

                    # Revert position and restore cost state from snapshot
                    for idx, ox, oy, oa in old_states:
                        model.footprints[idx].x = ox
                        model.footprints[idx].y = oy
                        model.footprints[idx].set_angle(oa)
                    cost_state.restore(snap)

            # Try rotations (single fp only, skip for groups — too disruptive)
            if len(members) == 1:
                fp = model.footprints[members[0]]
                for rot in rotations:
                    fp.set_angle((old_states[0][3] + rot) % 360.0)
                    _clamp_group_to_board(model, members)
                    if (_hits_new_keepout(model, members, old_states) or
                            _hits_polygon_cutout(model, members, old_states)):
                        fp.x = old_states[0][1]
                        fp.y = old_states[0][2]
                        fp.set_angle(old_states[0][3])
                        continue
                    new_cost = cost_state.incremental_update(member_set)
                    if new_cost < best_move_cost - 1.0:
                        best_move_cost = new_cost
                        best_move_states = [
                            (members[0], fp.x, fp.y, fp.angle_deg)
                        ]
                    # Revert
                    fp.x = old_states[0][1]
                    fp.y = old_states[0][2]
                    fp.set_angle(old_states[0][3])
                    cost_state.restore(snap)

            # Apply best improvement found for this component
            if best_move_states is not None:
                for idx, nx, ny, na in best_move_states:
                    model.footprints[idx].x = nx
                    model.footprints[idx].y = ny
                    model.footprints[idx].set_angle(na)
                cost_state.incremental_update(member_set)
                current_cost = best_move_cost
                best_hpwl = float(cost_state.hpwl)
                total_refined += 1
                improved = True

            # Time-throttled progress callback
            if progress_callback:
                now = time.perf_counter()
                if now - last_callback_time >= callback_interval:
                    last_callback_time = now
                    progress_callback(
                        overall_max_steps, overall_max_steps,
                        0.0, current_cost, current_cost, best_hpwl, t0,
                        best_hpwl,
                    )

    return total_refined


def _perturb_positions(model: BoardModel, fraction: float = 0.25) -> None:
    """Randomly perturb a fraction of moveable components.

    Used before reheating to escape local minima. Each selected component
    (and its group members) is shifted by up to 5% of the board span.
    """
    board_w = model.outline_xmax - model.outline_xmin
    board_h = model.outline_ymax - model.outline_ymin
    max_dist = max(board_w, board_h) // 20  # 5% of board span

    indices = list(model.moveable_indices)
    random.shuffle(indices)
    n_perturb = max(1, int(len(indices) * fraction))
    perturbed = set()

    for mi in indices[:n_perturb]:
        if mi in perturbed:
            continue
        members = _get_group_members(model, mi)
        dx = random.randint(-max_dist, max_dist)
        dy = random.randint(-max_dist, max_dist)
        for idx in members:
            model.footprints[idx].x += dx
            model.footprints[idx].y += dy
            perturbed.add(idx)


def _scatter_positions(model: BoardModel) -> None:
    """Scatter all moveable components to random positions within the board.

    Used for multi-start SA starts > 0. Each component (and its group) is
    moved to a random position within the inner 75% of the board (1/8 margin
    on each side). Groups move as rigid bodies: compute group centroid, pick
    a random new centroid, translate all members by the same delta.
    """
    board_w = model.outline_xmax - model.outline_xmin
    board_h = model.outline_ymax - model.outline_ymin
    margin_x = board_w // 8
    margin_y = board_h // 8
    xmin = model.outline_xmin + margin_x
    xmax = model.outline_xmax - margin_x
    ymin = model.outline_ymin + margin_y
    ymax = model.outline_ymax - margin_y

    scattered: Set[int] = set()
    for mi in model.moveable_indices:
        if mi in scattered:
            continue
        members = _get_group_members(model, mi)

        # Compute centroid
        cx = sum(model.footprints[i].x for i in members) // len(members)
        cy = sum(model.footprints[i].y for i in members) // len(members)

        # Pick random new centroid within inner region
        new_cx = random.randint(xmin, xmax)
        new_cy = random.randint(ymin, ymax)

        dx = new_cx - cx
        dy = new_cy - cy
        for idx in members:
            model.footprints[idx].x += dx
            model.footprints[idx].y += dy
            scattered.add(idx)


def run_sa(
    model: BoardModel,
    config: Optional[SAConfig] = None,
    progress_callback: Optional[Callable[[int, int, float, float, float], bool]] = None,
) -> SAResult:
    """
    Run simulated annealing on the board model with reheating and
    greedy refinement.

    The algorithm runs an initial SA phase, then reheats (restarts from
    the best solution at a reduced temperature) up to reheat_count times.
    Finally, a greedy local search polishes the result.

    Args:
        model: BoardModel with current footprint positions
        config: SA parameters (uses defaults if None)
        progress_callback: Called with (step, max_steps, temperature,
                          current_cost, best_cost, best_hpwl).
                          Return True to continue, False to abort.

    Returns:
        SAResult with optimization statistics.
        The model's footprint positions are set to the best solution found.
    """
    if config is None:
        config = SAConfig()

    import os
    n_moveable = len(model.moveable_indices)
    if config.moves_per_temp == 0:
        moves_per_temp = max(10 * n_moveable, 50)
    else:
        moves_per_temp = config.moves_per_temp

    num_starts = max(1, config.num_starts)

    # Total steps budget per start (for progress bar scaling)
    total_phases = 1 + config.reheat_count
    steps_per_phase = config.max_iterations
    overall_max_steps = num_starts * total_phases * steps_per_phase

    # Snapshot of initial placement so we can restore before each scattered start
    initial_positions = _snapshot_positions(model)

    # Evaluate initial placement cost (start 0 uses this as starting point)
    cost_state = CostState(model)
    initial_cost = cost_state.total_cost

    # Global best across all starts
    global_best_cost = cost_state.normalized_cost
    global_best_hpwl = float(cost_state.hpwl)
    global_best_positions = _snapshot_positions(model)

    cost_history: List[Tuple[int, float]] = [(0, global_best_cost)]
    total_moves = 0
    total_accepted = 0
    total_steps = 0
    cancelled = False

    run_t0 = time.perf_counter()
    profile = _ProfileData()
    greedy_time = 0.0
    final_cost = initial_cost
    t0 = 1.0  # will be set during first calibration

    for start_i in range(num_starts):
        if cancelled:
            break

        # Restore to initial placement and scatter (for starts > 0)
        _restore_positions(model, initial_positions)
        if start_i > 0:
            _scatter_positions(model)

        # Fresh cost state for this start
        cost_state = CostState(model)

        # Calibrate T0 for this starting placement
        t0 = auto_calibrate_t0(model, cost_state, config)

        start_best_cost = cost_state.normalized_cost
        start_best_hpwl = float(cost_state.hpwl)
        start_best_positions = _snapshot_positions(model)

        # --- SA Phase 1: initial anneal ---
        phase_offset = start_i * total_phases * steps_per_phase
        (start_best_cost, start_best_hpwl, start_best_positions,
         total_moves, total_accepted, total_steps, cancelled, p) = _sa_phase(
            model, cost_state, config, t0,
            start_temperature=t0,
            max_steps=steps_per_phase,
            moves_per_temp=moves_per_temp,
            best_cost=start_best_cost,
            best_hpwl=start_best_hpwl,
            best_positions=start_best_positions,
            total_moves=total_moves,
            total_accepted=total_accepted,
            total_steps=phase_offset,
            overall_max_steps=overall_max_steps,
            cost_history=cost_history,
            progress_callback=progress_callback,
        )
        total_steps = phase_offset + steps_per_phase
        profile = profile.merge(p)

        # --- SA Phase 2: reheating ---
        for reheat_i in range(config.reheat_count):
            if cancelled:
                break

            _restore_positions(model, start_best_positions)
            cost_state._compute_all()

            reheat_temp = t0 * config.reheat_ratio * (0.5 ** reheat_i)
            if reheat_temp < config.min_temperature:
                break

            reheat_offset = phase_offset + (1 + reheat_i) * steps_per_phase
            (start_best_cost, start_best_hpwl, start_best_positions,
             total_moves, total_accepted, total_steps, cancelled, p) = _sa_phase(
                model, cost_state, config, t0,
                start_temperature=reheat_temp,
                max_steps=steps_per_phase // 2,
                moves_per_temp=moves_per_temp,
                best_cost=start_best_cost,
                best_hpwl=start_best_hpwl,
                best_positions=start_best_positions,
                total_moves=total_moves,
                total_accepted=total_accepted,
                total_steps=reheat_offset,
                overall_max_steps=overall_max_steps,
                cost_history=cost_history,
                progress_callback=progress_callback,
            )
            profile = profile.merge(p)

        # Update global best if this start improved on it
        if start_best_cost < global_best_cost:
            global_best_cost = start_best_cost
            global_best_hpwl = start_best_hpwl
            global_best_positions = start_best_positions

    # --- Phase 3: greedy refinement on global best ---
    _restore_positions(model, global_best_positions)
    cost_state._compute_all()
    cost_state.update_penalty_scale(1.0)

    if progress_callback and not cancelled:
        progress_callback(
            overall_max_steps, overall_max_steps,
            0.0, cost_state.total_cost, global_best_cost, global_best_hpwl, t0,
            float(cost_state.hpwl),
        )

    greedy_t0 = time.perf_counter()
    n_refined = _greedy_refine(
        model, cost_state, model.moveable_indices,
        progress_callback=progress_callback if not cancelled else None,
        overall_max_steps=overall_max_steps,
        t0=t0,
        callback_interval=config.callback_interval,
    )
    greedy_time = time.perf_counter() - greedy_t0
    total_moves += n_refined

    cost_state._compute_all()
    final_cost = cost_state.normalized_cost

    if final_cost < global_best_cost or n_refined > 0:
        global_best_cost = final_cost
        global_best_hpwl = float(cost_state.hpwl)

    # --- Write profile summary to log file ---
    total_time = time.perf_counter() - run_t0
    sa_time = total_time - greedy_time
    lines = [
        f"\n=== CadMust-Neo Profile ({n_moveable} moveable components"
        + (f", {num_starts} starts" if num_starts > 1 else "") + f") ===",
        f"  Total time:       {total_time:7.1f}s",
        f"  SA phases:        {sa_time:7.1f}s  ({sa_time/total_time*100:.0f}%)",
    ]
    if sa_time > 0:
        lines += [
            f"    Move generation:  {profile.move_gen:7.1f}s  ({profile.move_gen/sa_time*100:.0f}%)",
            f"    Incr. update:     {profile.incr_update:7.1f}s  ({profile.incr_update/sa_time*100:.0f}%)",
            f"      Snapshot:       {cost_state.t_snapshot:7.1f}s",
            f"      HPWL:           {cost_state.t_hpwl:7.1f}s",
            f"      Overlap:        {cost_state.t_overlap:7.1f}s",
            f"      Boundary:       {cost_state.t_boundary:7.1f}s",
            f"      Keepout:        {cost_state.t_keepout:7.1f}s",
            f"    Revert+update:    {profile.revert:7.1f}s  ({profile.revert/sa_time*100:.0f}%)",
            f"    UI callback:      {profile.callback:7.1f}s  ({profile.callback/sa_time*100:.0f}%)",
        ]
        other = sa_time - profile.move_gen - profile.incr_update - profile.revert - profile.callback
        lines.append(f"    Other:            {other:7.1f}s  ({other/sa_time*100:.0f}%)")
    lines += [
        f"  Greedy refine:    {greedy_time:7.1f}s  ({greedy_time/total_time*100:.0f}%)",
        f"  Moves: {profile.total_moves} total, {profile.n_reverts} reverted ({profile.n_reverts/max(1,profile.total_moves)*100:.0f}%)",
    ]
    if sa_time > 0:
        lines.append(f"  Moves/sec: {profile.total_moves/sa_time:.0f}")
    lines.append(f"=========================================\n")
    profile_text = "\n".join(lines)
    log_path = os.path.join(os.path.dirname(__file__), "profile.log")
    with open(log_path, "w") as f:
        f.write(profile_text)
    print(profile_text)

    return SAResult(
        initial_cost=initial_cost,
        final_cost=final_cost,
        best_cost=global_best_cost,
        improvement_pct=_pct(initial_cost, global_best_cost),
        temperature_steps=total_steps,
        total_moves=total_moves,
        accepted_moves=total_accepted,
        cost_history=cost_history,
    )


def _snapshot_positions(model: BoardModel) -> List[Tuple[int, int, float]]:
    return [(fp.x, fp.y, fp.angle_deg) for fp in model.footprints]


def _restore_positions(model: BoardModel, positions: List[Tuple[int, int, float]]):
    for fp, (x, y, a) in zip(model.footprints, positions):
        fp.x = x
        fp.y = y
        fp.set_angle(a)


def _pct(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return (old - new) / old * 100.0
