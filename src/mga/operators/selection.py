import numpy as np

from mga.commons.numba_overload import njit
from mga.commons.types import npintp


# API functions
@njit
def selection(
    selected,
    niche,
    selection_criterion,
    maximize,
    elite_count,
    tourn_count,
    tourn_size,
    rng,
    stable,
):
    """
    Selects individuals using a combination of elitism and tournament selection.
    """
    if elite_count > 0:
        _select_best(selected[:elite_count], niche, selection_criterion, elite_count, maximize, stable)
    if tourn_count > 0:
        _select_tournament(selected[elite_count:], niche, selection_criterion, tourn_count, tourn_size, rng, maximize)


@njit
def selection_with_fallback(
    selected,
    niche,
    fitness,
    noptimal_mask,
    penalized_objectives,
    maximize,
    elite_count,
    tourn_count,
    tourn_size,
    rng,
    stable,
):
    """
    Selects based on fitness, falling back to penalized_objectives if not enough n-optimal individuals exist.
    """
    if elite_count > 0:
        _select_best_with_fallback(
            selected[:elite_count],
            niche,
            fitness,
            noptimal_mask,
            penalized_objectives,
            elite_count,
            maximize,
            stable
        )
    if tourn_count > 0:
        _select_tournament_with_fallback(
            selected[elite_count:],
            niche,
            fitness,
            noptimal_mask,
            penalized_objectives,
            tourn_count,
            tourn_size,
            rng,
            maximize
        )


@njit
def select_elite(selected, niche, selection_criterion, maximize):
    """
    Special case of `_select_best` when n = 1
    """
    if maximize:
        index = selection_criterion.argmax()
    else:
        index = selection_criterion.argmin()
    selected[:] = niche[index, :]


@njit
def select_elite_with_fallback(selected, niche, fitness, noptimal_mask, penalized_objectives, maximize):
    """
    Special case of `_select_best_with_fallback` when n = 1
    Selects best 'n' individuals based on 'fitness'.
    If there are not 'n' noptimal individuals, selects on 'penalized_objectives'
    """
    # loop through fitness and choose the best noptimal fitness
    # record succes via _nopt: bool
    best = -np.inf
    index = -1
    for j in range(niche.shape[0]):
        if not noptimal_mask[j]:
            continue
        elif fitness[j] > best:
            best = fitness[j]
            index = j

    if index == -1:
        select_elite(selected, niche, penalized_objectives, maximize)

    selected[:] = niche[index, :]


# private helper functions
@njit
def _draw_tournament_indices(indices, ub, rng):
    """
    Draws random indices for selection tournament
    """
    for i in range(indices.size):
        indices[i] = rng.integers(0, ub)


@njit
def _do_tournament(selection_criterion, maximize, indices):
    """
    Performs selection for selection tournament.
    This is functionally the same as `select_elite` but since 'indices'
        is small relative to length of population, this is substantially faster
        (avoids slicing etc.)
    """
    if maximize:
        _selected_idx = -1
        _best = -np.inf
        for idx in indices:
            if selection_criterion[idx] > _best:
                _selected_idx = idx
                _best = selection_criterion[idx]
    else:
        _selected_idx = -1
        _best = np.inf
        for idx in indices:
            if selection_criterion[idx] < _best:
                _selected_idx = idx
                _best = selection_criterion[idx]
    return _selected_idx


@njit
def _select_tournament(selected, niche, selection_criterion, n, tourn_size, rng, maximize):
    """
    Selects {n} individuals from a population according to selection tournament
    """
    if n == 0:
        return
    indices = np.empty(tourn_size, npintp)

    for m in range(n):
        _draw_tournament_indices(indices, niche.shape[0], rng)
        _selected_idx = _do_tournament(selection_criterion, maximize, indices)
        selected[m, :] = niche[_selected_idx, :]


@njit
def _select_tournament_with_fallback(
    selected,
    niche,
    fitness,
    noptimal_mask,
    penalized_objectives,
    n,
    tourn_size,
    rng,
    maximize,
):
    """select on fitness preferred. fitness always maximised.
    penalized_objectives max/minimized based on value of `maximize`"""
    if n == 0:
        return

    # _nopt = 0
    # for i in range(len(noptimal_mask)):
    #     if noptimal_mask[i]:
    #         _nopt += 1

    indices = np.empty(tourn_size, npintp)
    noptimality_threshold = tourn_size / 2
    # noptimality_threshold = len(selected) / 2

    for m in range(n):
        _draw_tournament_indices(indices, niche.shape[0], rng)

        _nopt = 0
        for idx in indices:
            if noptimal_mask[idx]:
                _nopt += 1

        if _nopt <= noptimality_threshold:  # mostly non-noptimal
            _selected_idx = _do_tournament(penalized_objectives, maximize, indices)
        else:  # mostly noptimal
            _selected_idx = _do_tournament(fitness, True, indices)

        selected[m, :] = niche[_selected_idx, :]


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
def _select_best(selected, niche, selection_criterion, n, maximize, stable):
    """
    Selects best individuals from a population
    """
    if n == 0:
        return
    if n == len(selected):
        selected[:, :] = niche[:, :]
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
            indices = np.argpartition(selection_criterion, -n)[-n:]
        else:
            indices = np.argpartition(selection_criterion, n)[:n]

    for j in range(n):
        selected[j, :] = niche[indices[j], :]


@njit
def _select_best_with_fallback(selected, niche, fitness, noptimal_mask, penalized_objectives, n, maximize, stable):
    """Selects best `n` individuals based on fitness.
    If there are not `n` noptimal individuals, selects on penalized_objectives"""
    if n == 0:
        return
    _nopt = 0
    for i in range(len(noptimal_mask)):
        if noptimal_mask[i]:
            _nopt += 1

    if _nopt <= n:  # not enough near-optimal points
        return _select_best(selected, niche, penalized_objectives, n, maximize, stable)

    indices = np.empty(n, npintp)

    noptimal_indices = np.where(noptimal_mask)[0]
    noptimal_fitness = fitness[noptimal_indices]

    if stable:
        _indices = np.argsort(noptimal_fitness)
        _stabilize_sort(_indices, noptimal_fitness)
        indices[:] = noptimal_indices[_indices[-n:]]
    else:
        indices[:] = noptimal_indices[np.argpartition(noptimal_fitness, -n)[-n:]]

    for j in range(n):
        selected[j, :] = niche[indices[j], :]
