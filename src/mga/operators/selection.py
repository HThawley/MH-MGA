import numpy as np

from mga.commons.numba_overload import njit
from mga.commons.types import npintp
import mga.commons.utils as utils


# API functions
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
    Selects individuals using a combination of elitism and tournament selection.
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
    epsilon,
    champ_count,
    elite_count,
    tourn_count,
    tourn_size,
    niche_size,
    rng,
    stable,
):
    """
    Selects based on the secondary array (fitness), falling back to penalized_objectives if not
    enough noptimal individuals exist.
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
            epsilon,
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
            epsilon,
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
            epsilon,
        )


# private helper functions
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
    """ selects the champion (single best individual) and clones n times"""
    index = utils.argm(selection_criterion, maximize)
    _assign_single_index(out_indices, index, n, start_idx)


@njit
def _get_best_lexicographic(
    indices, fitness, noptimal_mask, penalized_objectives, maximize, maximize_secondary, epsilon
):
    """Core lexicographic loop evaluator for arrays."""
    best_idx = -1
    best_fit = -np.inf if maximize_secondary else np.inf
    best_obj = -np.inf if maximize else np.inf

    for idx in indices:
        if not noptimal_mask[idx]:
            continue

        fit = fitness[idx]
        obj = penalized_objectives[idx]

        if best_idx == -1:
            best_idx = idx
            best_fit = fit
            best_obj = obj
            continue

        diff = fit - best_fit
        if not maximize_secondary:
            diff = -diff

        if diff > epsilon:
            best_idx = idx
            best_fit = fit
            best_obj = obj
        elif abs(diff) <= epsilon:
            obj_diff = obj - best_obj
            if not maximize:
                obj_diff = -obj_diff
            if obj_diff > 0:
                best_idx = idx
                best_fit = fit
                best_obj = obj

    return best_idx


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
    epsilon,
):
    """ selects the champion (single best individual) and clones n times.
    Selects champion from noptimal solutions on fitness. Falls back to objective """
    indices = np.arange(len(fitness))
    index = _get_best_lexicographic(
        indices, fitness, noptimal_mask, penalized_objectives, maximize, maximize_secondary, epsilon
    )

    if index == -1:
        index = utils.argm(penalized_objectives, maximize)

    _assign_single_index(out_indices, index, n, start_idx)


@njit
def _draw_tournament_indices(indices, ub, rng):
    """
    Draws random indices for selection tournament
    """
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
    """
    Selects {n} individuals from a population according to selection tournament
    """
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
    epsilon
):
    """select on secondary array preferred, max/minimized based on maximize_secondary.
    penalized_objectives max/minimized based on value of `maximize`"""
    if n == 0:
        return

    indices = np.empty(tourn_size, npintp)

    for m in range(start_idx, start_idx + n):
        _draw_tournament_indices(indices, niche_size, rng)

        _selected_idx = _get_best_lexicographic(
            indices, fitness, noptimal_mask, penalized_objectives, maximize, maximize_secondary, epsilon
        )

        if _selected_idx == -1:
            # select on objective
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
    """
    Selects best individuals from a population
    """
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
    epsilon,
    stable,
):
    """Selects best `n` individuals based on the secondary array.
    If there are not `n` noptimal individuals, selects on penalized_objectives"""
    if n == 0:
        return
    _nopt = 0
    for i in range(len(noptimal_mask)):
        if noptimal_mask[i]:
            _nopt += 1

    if _nopt <= n:  # not enough near-optimal points
        return _select_elite(out_indices, penalized_objectives, n, start_idx, maximize, stable)

    indices = np.empty(n, npintp)
    noptimal_indices = np.where(noptimal_mask)[0]

    noptimal_fitness = fitness[noptimal_indices]
    noptimal_obj = penalized_objectives[noptimal_indices]

    # Combine metrics: Round primary key to bin by epsilon, normalize secondary to act as the decimal tie-breaker
    rounded_fit = np.round(noptimal_fitness / epsilon) * epsilon

    p_key = -rounded_fit if maximize_secondary else rounded_fit
    s_key = -noptimal_obj if maximize else noptimal_obj

    s_min, s_max = s_key.min(), s_key.max()
    s_norm = (s_key - s_min) / (s_max - s_min + 1e-12) if s_max > s_min else np.zeros_like(s_key)

    # Scaling s_norm guarantees it solely breaks ties without overriding the primary p_key bin
    score = p_key + (s_norm * epsilon * 0.99)

    if stable:
        _indices = np.argsort(score)
        _stabilize_sort(_indices, score)
        indices[:] = noptimal_indices[_indices[:n]]
    else:
        indices[:] = noptimal_indices[np.argpartition(score, n)[:n]]

    _assign_multiple_indices(out_indices, indices, n, start_idx)
