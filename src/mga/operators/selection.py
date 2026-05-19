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
    champ_count,
    elite_count,
    tourn_count,
    tourn_size,
    niche_size,
    rng,
    stable,
):
    """
    Selects based on fitness, falling back to penalized_objectives if not enough n-optimal individuals exist.
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
def _select_champ_with_fallback(
    out_indices, fitness, noptimal_mask, penalized_objectives, n, start_idx, maximize
):
    """ selects the champion (single best individual) and clones n times.
    Selects champion from noptimal solutions on fitness. Falls back to objective """
    index = utils.argmax_with_mask(fitness, noptimal_mask)

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
):
    """select on fitness preferred. fitness always maximised.
    penalized_objectives max/minimized based on value of `maximize`"""
    if n == 0:
        return

    indices = np.empty(tourn_size, npintp)

    for m in range(start_idx, start_idx + n):
        _draw_tournament_indices(indices, niche_size, rng)

        # first try select on fitness
        _selected_idx = utils.argm_with_indices_mask(fitness, indices, noptimal_mask, True)

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
    stable,
):
    """Selects best `n` individuals based on fitness.
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

    if stable:
        _indices = np.argsort(noptimal_fitness)
        _stabilize_sort(_indices, noptimal_fitness)
        indices[:] = noptimal_indices[_indices[-n:]]
    else:
        indices[:] = noptimal_indices[np.argpartition(noptimal_fitness, -n)[-n:]]

    _assign_multiple_indices(out_indices, indices, n, start_idx)
