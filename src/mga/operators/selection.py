import numpy as np

from mga.commons.numba_overload import njit
from mga.commons.types import npintp, npfloat
import mga.commons.utils as utils


# =====================================================================
# Local Niche Selection API
# =====================================================================
@njit
def selection(
    out_indices,
    selection_criterion,
    maximize,
    champ_count,
    elite_count,
    tourn_count,
    tourn_size,
    niche_size,
    rng,
    stable,
):
    """
    Selects individuals locally within a niche using elitism and tournament selection.
    """
    if champ_count > 0:
        _select_champ(
            out_indices,
            selection_criterion,
            champ_count,
            0,
            maximize,
        )
    if elite_count > 0:
        _select_elite(
            out_indices,
            selection_criterion,
            elite_count,
            champ_count,
            maximize,
            stable,
        )
    if tourn_count > 0:
        _select_tournament(
            out_indices,
            selection_criterion,
            tourn_count,
            tourn_size,
            champ_count + elite_count,
            niche_size,
            rng,
            maximize,
        )


@njit
def selection_with_fallback(
    out_indices,
    fitness,
    noptimal_mask,
    penalized_objectives,
    maximize,
    maximize_secondary,
    epsilon_rel,
    epsilon_abs,
    champ_count,
    elite_count,
    tourn_count,
    tourn_size,
    niche_size,
    rng,
    stable,
):
    """
    Local selection based on secondary array (fitness), falling back to
    penalized_objectives if not enough noptimal individuals exist.
    """
    if champ_count > 0:
        _select_champ_with_fallback(
            out_indices,
            fitness,
            noptimal_mask,
            penalized_objectives,
            champ_count,
            0,
            maximize,
            maximize_secondary,
            epsilon_rel,
            epsilon_abs,
        )

    if elite_count > 0:
        _select_elite_with_fallback(
            out_indices,
            fitness,
            noptimal_mask,
            penalized_objectives,
            elite_count,
            champ_count,
            maximize,
            maximize_secondary,
            epsilon_rel,
            epsilon_abs,
            stable,
        )
    if tourn_count > 0:
        _select_tournament_with_fallback(
            out_indices,
            fitness,
            noptimal_mask,
            penalized_objectives,
            tourn_count,
            tourn_size,
            champ_count + elite_count,
            niche_size,
            rng,
            maximize,
            maximize_secondary,
            epsilon_rel,
            epsilon_abs,
        )


# =====================================================================
# Global Selection API (For Cost Optimum Niche 0 & Extrema Niches)
# =====================================================================
@njit
def selection_global_cost(
    out_flat_indices,
    flat_penalized_objectives,
    flat_scaled_points,
    local_offset,
    niche_size,
    maximize,
    champ_count,
    elite_count,
    tourn_count,
    tourn_size,
    rng,
    stable,
):
    """
    Selects parents for the cost optimization niche (niche 0):
    - `champ_count`: global best individual across all niches (cloned champ_count times).
    - `elite_count`: global top `elite_count` unique individuals across all niches (skipping duplicates).
    - `tourn_count`: local tournament within niche 0.
    Writes flat indices `[0, num_niches * pop_size)` into `out_flat_indices`.
    """
    if champ_count > 0:
        _select_champ(
            out_flat_indices,
            flat_penalized_objectives,
            champ_count,
            0,
            maximize,
        )

    if elite_count > 0:
        all_indices = np.arange(len(flat_penalized_objectives), dtype=npintp)
        key = -flat_penalized_objectives if maximize else flat_penalized_objectives
        sorted_order = np.argsort(key)
        if stable:
            _stabilize_sort(sorted_order, key)

        unique_elites = _select_top_k_unique(
            sorted_order, all_indices, flat_scaled_points, elite_count
        )
        _assign_multiple_indices(out_flat_indices, unique_elites, elite_count, champ_count)

    if tourn_count > 0:
        local_tourn = np.empty(tourn_count, dtype=npintp)
        local_criterion = flat_penalized_objectives[local_offset: local_offset + niche_size]
        _select_tournament(
            local_tourn,
            local_criterion,
            tourn_count,
            tourn_size,
            0,
            niche_size,
            rng,
            maximize,
        )
        for m in range(tourn_count):
            out_flat_indices[champ_count + elite_count + m] = local_offset + local_tourn[m]


@njit
def selection_global_extrema(
    out_flat_indices,
    flat_dim_vals,
    flat_noptimal_mask,
    flat_feasible_mask,
    flat_penalized_objectives,
    flat_scaled_points,
    noptimal_threshold,
    local_offset,
    niche_size,
    maximize,
    maximize_secondary,
    epsilon_rel,
    epsilon_abs,
    champ_count,
    elite_count,
    tourn_count,
    tourn_size,
    rng,
    stable,
):
    """
    Selects parents for an extrema niche:
    - `champ_count`: global lexicographic champion across all niches on `dim_k`.
    - `elite_count`: global top `elite_count` unique individuals across all niches on `dim_k`
      (using directional fallback if fewer than `elite_count` unique n-optimal points exist).
    - `tourn_count`: local tournament within the niche (with directional fallback).
    Writes flat indices `[0, num_niches * pop_size)` into `out_flat_indices`.
    """
    total_pop = len(flat_dim_vals)
    all_indices = np.arange(total_pop, dtype=npintp)

    # 1. Global Champion Selection
    if champ_count > 0:
        champ_idx = _get_best_lexicographic(
            all_indices,
            flat_dim_vals,
            flat_noptimal_mask,
            flat_penalized_objectives,
            maximize,
            maximize_secondary,
            epsilon_rel,
            epsilon_abs,
        )
        if champ_idx == -1:
            fb_scores = _calculate_directional_fallback_scores(
                flat_dim_vals,
                flat_feasible_mask,
                flat_penalized_objectives,
                noptimal_threshold,
                maximize,
                maximize_secondary,
            )
            champ_idx = utils.argm(fb_scores, False)

        _assign_single_index(out_flat_indices, champ_idx, champ_count, 0)

    # 2. Global Deduplicated Elite Selection (with Directional Fallback)
    if elite_count > 0:
        nopt_indices = np.where(flat_noptimal_mask)[0]
        n_nopt = len(nopt_indices)

        selected_elites = np.empty(elite_count, dtype=npintp)
        selected_count = 0

        if n_nopt > 0:
            nopt_vals = flat_dim_vals[nopt_indices]
            nopt_objs = flat_penalized_objectives[nopt_indices]
            target_val = np.max(nopt_vals) if maximize_secondary else np.min(nopt_vals)
            eff_epsilon = abs(target_val) * epsilon_rel + epsilon_abs

            nopt_scores = _calculate_lexicographic_scores(
                nopt_vals, nopt_objs, maximize, maximize_secondary, eff_epsilon, target_val
            )
            nopt_order = np.argsort(nopt_scores)
            if stable:
                _stabilize_sort(nopt_order, nopt_scores)

            for pos in range(n_nopt):
                if selected_count == elite_count:
                    break
                cand_idx = nopt_indices[nopt_order[pos]]
                cand_pt = flat_scaled_points[cand_idx]
                is_dup = False
                for s in range(selected_count):
                    if is_duplicate_point(cand_pt, flat_scaled_points[selected_elites[s]]):
                        is_dup = True
                        break
                if not is_dup:
                    selected_elites[selected_count] = cand_idx
                    selected_count += 1

        # Directional fallback for any remaining elite slots
        if selected_count < elite_count:
            fb_scores = _calculate_directional_fallback_scores(
                flat_dim_vals,
                flat_feasible_mask,
                flat_penalized_objectives,
                noptimal_threshold,
                maximize,
                maximize_secondary,
            )
            fb_order = np.argsort(fb_scores)
            if stable:
                _stabilize_sort(fb_order, fb_scores)

            for pos in range(total_pop):
                if selected_count == elite_count:
                    break
                cand_idx = fb_order[pos]
                cand_pt = flat_scaled_points[cand_idx]
                is_dup = False
                for s in range(selected_count):
                    if is_duplicate_point(cand_pt, flat_scaled_points[selected_elites[s]]):
                        is_dup = True
                        break
                if not is_dup:
                    selected_elites[selected_count] = cand_idx
                    selected_count += 1

            # Final wrap-around if total unique points in entire population < elite_count
            if selected_count < elite_count:
                fallback_len = selected_count if selected_count > 0 else total_pop
                for rem in range(selected_count, elite_count):
                    if selected_count > 0:
                        selected_elites[rem] = selected_elites[rem % fallback_len]
                    else:
                        selected_elites[rem] = fb_order[rem % fallback_len]

        _assign_multiple_indices(out_flat_indices, selected_elites, elite_count, champ_count)

    # 3. Local Tournament Selection (with Directional Fallback)
    if tourn_count > 0:
        local_vals = flat_dim_vals[local_offset: local_offset + niche_size]
        local_nopt = flat_noptimal_mask[local_offset: local_offset + niche_size]
        local_feas = flat_feasible_mask[local_offset: local_offset + niche_size]
        local_objs = flat_penalized_objectives[local_offset: local_offset + niche_size]

        local_fb_scores = _calculate_directional_fallback_scores(
            local_vals,
            local_feas,
            local_objs,
            noptimal_threshold,
            maximize,
            maximize_secondary,
        )

        tourn_indices = np.empty(tourn_size, npintp)
        start_tourn = champ_count + elite_count
        for m in range(tourn_count):
            _draw_tournament_indices(tourn_indices, niche_size, rng)
            winner = _get_best_lexicographic(
                tourn_indices,
                local_vals,
                local_nopt,
                local_objs,
                maximize,
                maximize_secondary,
                epsilon_rel,
                epsilon_abs,
            )
            if winner == -1:
                winner = utils.argm_with_indices(local_fb_scores, tourn_indices, False)
            out_flat_indices[start_tourn + m] = local_offset + winner


# =====================================================================
# Private Helper Functions
# =====================================================================
@njit(inline='always')
def _assign_single_index(out_indices, target_idx, n, start_idx):
    """Assigns the same niche row to n consecutive rows in selected."""
    out_indices[start_idx: start_idx + n] = target_idx


@njit(inline='always')
def _assign_multiple_indices(out_indices, target_indices, n, start_idx):
    """Assigns an array of niche rows to n consecutive rows in selected."""
    for j in range(n):
        out_indices[start_idx + j] = target_indices[j]


@njit
def _select_champ(out_indices, selection_criterion, n, start_idx, maximize):
    """Selects the champion (single best individual) and clones n times."""
    index = utils.argm(selection_criterion, maximize)
    _assign_single_index(out_indices, index, n, start_idx)


@njit
def _find_lexicographic_target(indices, fitness, noptimal_mask, maximize_secondary):
    """Ratchet target_fit strictly in the secondary direction."""
    has_nopt = False
    target_fit = -np.inf if maximize_secondary else np.inf
    for idx in indices:
        if noptimal_mask[idx]:
            fit = fitness[idx]
            if not has_nopt:
                target_fit, has_nopt = fit, True
            elif (maximize_secondary and fit > target_fit) or (not maximize_secondary and fit < target_fit):
                target_fit = fit
    return has_nopt, target_fit


@njit
def _find_best_lexicographic_candidate(
    indices, fitness, noptimal_mask, penalized_objectives,
    maximize, maximize_secondary, target_fit, eff_epsilon
):
    """Pick candidate within eff_epsilon of target_fit with the best primary objective."""
    best_idx = -1
    best_obj = -np.inf if maximize else np.inf
    best_fit = target_fit

    for idx in indices:
        if not noptimal_mask[idx]:
            continue
        fit = fitness[idx]
        dist = (target_fit - fit) if maximize_secondary else (fit - target_fit)

        if dist <= eff_epsilon:
            obj = penalized_objectives[idx]
            if best_idx == -1:
                best_idx, best_obj, best_fit = idx, obj, fit
            else:
                obj_diff = (obj - best_obj) if maximize else (best_obj - obj)
                if obj_diff > 0.0:
                    best_idx, best_obj, best_fit = idx, obj, fit
                elif obj_diff == 0.0:
                    fit_diff = (fit - best_fit) if maximize_secondary else (best_fit - fit)
                    if fit_diff > 0.0:
                        best_idx, best_fit = idx, fit
    return best_idx


@njit
def _get_best_lexicographic(
    indices, fitness, noptimal_mask, penalized_objectives,
    maximize, maximize_secondary, epsilon_rel, epsilon_abs
):
    has_nopt, target_fit = _find_lexicographic_target(indices, fitness, noptimal_mask, maximize_secondary)
    if not has_nopt:
        return -1

    eff_epsilon = abs(target_fit) * epsilon_rel + epsilon_abs
    return _find_best_lexicographic_candidate(
        indices, fitness, noptimal_mask, penalized_objectives,
        maximize, maximize_secondary, target_fit, eff_epsilon
    )


@njit
def _calculate_lexicographic_scores(
    noptimal_fitness,
    noptimal_obj,
    maximize,
    maximize_secondary,
    eff_epsilon,
    target_fit,
):
    """Calculates sorting scores (lower is better), abstracting pure lexicographic vs. tiered epsilon logic."""
    s_key = -noptimal_obj if maximize else noptimal_obj
    s_min, s_max = s_key.min(), s_key.max()
    s_norm = (s_key - s_min) / (s_max - s_min + 1e-12) if s_max > s_min else np.zeros_like(s_key)

    if eff_epsilon <= 0.0:
        return -noptimal_fitness if maximize_secondary else noptimal_fitness

    dist_from_target = (target_fit - noptimal_fitness) if maximize_secondary else (noptimal_fitness - target_fit)
    tier = np.floor(np.maximum(0.0, dist_from_target - 1e-12 * eff_epsilon) / eff_epsilon)
    return tier + (s_norm * 0.99)


@njit
def _calculate_directional_fallback_scores(
    secondary_vals,
    feasible_mask,
    penalized_objectives,
    noptimal_threshold,
    maximize,
    maximize_secondary,
):
    """
    Computes a directional fallback score (lower is better) when an extrema niche
    lacks enough n-optimal individuals:
    - Feasible points reside in [0.0, 2.0), trading off normalized progress along
      `secondary_vals` [0, 1] and normalized excess cost above `noptimal_threshold` [0, 1].
    - Infeasible points reside in [10.0, 12.0), ensuring feasible points are always preferred
      while still rewarding directional movement and lower penalty.
    """
    n = len(secondary_vals)
    scores = np.empty(n, dtype=npfloat)

    p_key = -secondary_vals if maximize_secondary else secondary_vals
    p_min, p_max = np.min(p_key), np.max(p_key)
    p_range = p_max - p_min

    excess = np.empty(n, dtype=npfloat)
    for i in range(n):
        diff = (noptimal_threshold - penalized_objectives[i]) if maximize else (
            penalized_objectives[i] - noptimal_threshold
        )
        excess[i] = max(0.0, diff)

    e_min, e_max = np.min(excess), np.max(excess)
    e_range = e_max - e_min

    for i in range(n):
        p_norm = (p_key[i] - p_min) / (p_range + 1e-12) if p_range > 0.0 else 0.0
        e_norm = (excess[i] - e_min) / (e_range + 1e-12) if e_range > 0.0 else 0.0
        infeas_offset = 0.0 if feasible_mask[i] else 10.0
        scores[i] = infeas_offset + 0.5 * p_norm + 0.5 * e_norm

    return scores


@njit
def _select_champ_with_fallback(
    out_indices,
    fitness,
    noptimal_mask,
    penalized_objectives,
    n,
    start_idx,
    maximize,
    maximize_secondary,
    epsilon_rel,
    epsilon_abs,
):
    """Selects champion from noptimal solutions on fitness. Falls back to objective."""
    indices = np.arange(len(fitness))
    index = _get_best_lexicographic(
        indices, fitness, noptimal_mask, penalized_objectives, maximize, maximize_secondary, epsilon_rel, epsilon_abs
    )

    if index == -1:
        index = utils.argm(penalized_objectives, maximize)

    _assign_single_index(out_indices, index, n, start_idx)


@njit
def _draw_tournament_indices(indices, ub, rng):
    """Draws random indices for selection tournament."""
    for i in range(indices.size):
        indices[i] = rng.integers(0, ub)


@njit
def _select_tournament(
    out_indices,
    selection_criterion,
    n,
    tourn_size,
    start_idx,
    niche_size,
    rng,
    maximize,
):
    """Selects {n} individuals from a population according to selection tournament."""
    if n == 0:
        return
    indices = np.empty(tourn_size, npintp)

    for m in range(start_idx, start_idx + n):
        _draw_tournament_indices(indices, niche_size, rng)
        out_indices[m] = utils.argm_with_indices(selection_criterion, indices, maximize)


@njit
def _select_tournament_with_fallback(
    out_indices,
    fitness,
    noptimal_mask,
    penalized_objectives,
    n,
    tourn_size,
    start_idx,
    niche_size,
    rng,
    maximize,
    maximize_secondary,
    epsilon_rel,
    epsilon_abs,
):
    """Select on secondary array preferred, max/minimized based on maximize_secondary."""
    if n == 0:
        return

    indices = np.empty(tourn_size, npintp)

    for m in range(start_idx, start_idx + n):
        _draw_tournament_indices(indices, niche_size, rng)

        _selected_idx = _get_best_lexicographic(
            indices,
            fitness,
            noptimal_mask,
            penalized_objectives,
            maximize,
            maximize_secondary,
            epsilon_rel,
            epsilon_abs,
        )

        if _selected_idx == -1:
            _selected_idx = utils.argm_with_indices(penalized_objectives, indices, maximize)

        out_indices[m] = _selected_idx


@njit
def _stabilize_sort(indices, values):
    """
    Sorts blocks of duplicate values within a list of indices
    in-place based on the index values for a stable order.
    """
    i = 1
    while i < len(indices):
        # Check if the value at the current index is the same as the previous one
        if values[indices[i]] == values[indices[i - 1]]:
            # Found the start of a block of equal-valued items
            start_block = i - 1
            # Find the end of the block
            while i < len(indices) and values[indices[i]] == values[indices[start_block]]:
                i += 1
            end_block = i
            # Sort the slice of indices corresponding to the duplicates.
            # This provides a stable, deterministic order.
            indices[start_block:end_block].sort()
        else:
            i += 1


@njit
def _select_elite(out_indices, selection_criterion, n, start_idx, maximize, stable):
    """Selects best individuals from a population."""
    if n == 0:
        return
    if n == len(selection_criterion):
        for j in range(start_idx, start_idx + n):
            out_indices[j] = j - start_idx
        return

    indices = np.empty(n, npintp)

    if stable:
        _indices = np.argsort(selection_criterion)
        _stabilize_sort(_indices, selection_criterion)
        if maximize:
            indices[:] = _indices[-n:]
        else:
            indices[:] = _indices[:n]
    else:
        # This is much faster but does not preserve order
        if maximize:
            indices[:] = np.argpartition(selection_criterion, -n)[-n:]
        else:
            indices[:] = np.argpartition(selection_criterion, n)[:n]

    _assign_multiple_indices(out_indices, indices, n, start_idx)


@njit
def _select_elite_with_fallback(
    out_indices,
    fitness,
    noptimal_mask,
    penalized_objectives,
    n,
    start_idx,
    maximize,
    maximize_secondary,
    epsilon_rel,
    epsilon_abs,
    stable,
):
    """Selects best `n` individuals based on the secondary array.
    If there are not `n` noptimal individuals, selects on penalized_objectives."""
    if n == 0:
        return
    _nopt = 0
    for i in range(len(noptimal_mask)):
        if noptimal_mask[i]:
            _nopt += 1

    if _nopt <= n:
        return _select_elite(out_indices, penalized_objectives, n, start_idx, maximize, stable)

    indices = np.empty(n, npintp)
    noptimal_indices = np.where(noptimal_mask)[0]
    noptimal_fitness = fitness[noptimal_indices]
    noptimal_obj = penalized_objectives[noptimal_indices]

    target_fit = np.max(noptimal_fitness) if maximize_secondary else np.min(noptimal_fitness)
    eff_epsilon = abs(target_fit) * epsilon_rel + epsilon_abs

    score = _calculate_lexicographic_scores(
        noptimal_fitness, noptimal_obj, maximize, maximize_secondary, eff_epsilon, target_fit
    )

    if stable:
        _indices = np.argsort(score)
        _stabilize_sort(_indices, score)
        indices[:] = noptimal_indices[_indices[:n]]
    else:
        indices[:] = noptimal_indices[np.argpartition(score, n)[:n]]

    _assign_multiple_indices(out_indices, indices, n, start_idx)


# =====================================================================
# Fast Floating-Point Deduplication Helpers
# =====================================================================
@njit(inline="always")
def is_duplicate_point(p1, p2, rtol=1e-5, atol=1e-6):
    """
    Fast early-exit check for whether two 1D points are identical
    within floating-point / near-clone tolerance.
    """
    for k in range(p1.shape[0]):
        diff = abs(p1[k] - p2[k])
        scale = max(abs(p1[k]), abs(p2[k]))
        if diff > atol + rtol * scale:
            return False
    return True


@njit
def _select_top_k_unique(
    sorted_order,
    candidate_pool_indices,
    flat_scaled_points,
    n,
):
    """
    Walks `sorted_order` (best-first) and selects `n` unique indices from
    `candidate_pool_indices`, skipping near-duplicates in `flat_scaled_points`.
    If fewer than `n` unique points exist, wraps around `sorted_order` to fill `n`.
    """
    out = np.empty(n, dtype=npintp)
    selected_count = 0
    total_candidates = len(sorted_order)

    for pos in range(total_candidates):
        if selected_count == n:
            break
        cand_idx = candidate_pool_indices[sorted_order[pos]]
        cand_pt = flat_scaled_points[cand_idx]

        is_dup = False
        for s in range(selected_count):
            if is_duplicate_point(cand_pt, flat_scaled_points[out[s]]):
                is_dup = True
                break

        if not is_dup:
            out[selected_count] = cand_idx
            selected_count += 1

    # Fallback: if total unique points in population < n, cycle through best available
    if selected_count < n:
        fallback_len = selected_count if selected_count > 0 else total_candidates
        for rem in range(selected_count, n):
            if selected_count > 0:
                out[rem] = out[rem % fallback_len]
            else:
                out[rem] = candidate_pool_indices[sorted_order[rem % fallback_len]]

    return out
