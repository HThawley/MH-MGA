import numpy as np
from numba import njit

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS

# API functions
@njit
def selection(selected, niche, criteria, maximize, elite_count, tourn_count, tourn_size, rng, stable):
    """
    Selects individuals using a combination of elitism and tournament selection.
    """
    if elite_count > 0:
        _select_best(selected[:elite_count], niche, criteria, elite_count, maximize, stable)
    if tourn_count > 0:
        _select_tournament(selected[elite_count:], niche, criteria, tourn_count, tourn_size, rng, maximize)

@njit
def selection_with_fallback(selected, niche, fitness, is_noptimal, objective, maximize, elite_count, tourn_count, tourn_size, rng, stable):
    """
    Selects based on fitness, falling back to objective if not enough n-optimal individuals exist.
    """
    if elite_count > 0:
        _select_best_with_fallback(
            selected[:elite_count], niche, fitness, is_noptimal, objective, elite_count, maximize, stable
        )
    if tourn_count > 0:
        _select_tournament_with_fallback(
            selected[elite_count:], niche, fitness, is_noptimal, objective, tourn_count, tourn_size, rng, maximize
        )

@njit
def select_elite(selected, niche, objective, maximize):
    """ 
    Special case of `_select_best` when n = 1
    """
    if maximize:
        index = objective.argmax()
    else:
        index = objective.argmin()
    selected[:] = niche[index, :] 

@njit
def select_elite_with_fallback(selected, niche, fitness, is_noptimal, objective, maximize):
    """ 
    Special case of `_select_best_with_fallback` when n = 1
    Selects best 'n' individuals based on 'fitness'.
    If there are not 'n' noptimal individuals, selects on 'objective'
    """
    # loop through fitness and choose the best noptimal fitness
    # record succes via _nopt: bool
    best = -np.inf
    index = -1
    for j in range(niche.shape[0]):
        if not is_noptimal[j]:
            continue
        elif fitness[j] > best:
            best = fitness[j]
            index = j
            
    if index == -1:
        select_elite(selected, niche, objective, maximize)
        
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
def _do_tournament(criteria, maximize, indices):
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
            if criteria[idx] > _best:
                _selected_idx = idx
                _best = criteria[idx]
    else:
        _selected_idx = -1
        _best = np.inf
        for idx in indices:
            if criteria[idx] < _best:
                _selected_idx = idx
                _best = criteria[idx]
    return _selected_idx

@njit
def _select_tournament(selected, niche, objective, n, tournsize, rng, maximize):
    """
    Selects {n} individuals from a population according to selection tournament
    """
    if n == 0:
        return
    indices = np.empty(tournsize, INT)
    
    for m in range(n):
        _draw_tournament_indices(indices, niche.shape[0], rng)
        _selected_idx = _do_tournament(objective, maximize, indices)
        selected[m, :] = niche[_selected_idx, :]

@njit
def _select_tournament_with_fallback(selected, niche, fitness, is_noptimal, objective, n, tournsize, rng, maximize):
    """select on fitness preferred. fitness always maximised. 
    objective max/minimized based on value of `maximize`"""
    if n == 0: 
        return
    indices = np.empty(tournsize, INT)
    noptimality_threshold = tournsize / 2 
    
    for m in range(n):
        _draw_tournament_indices(indices, niche.shape[0], rng)
        
        _nopt = 0
        for idx in indices:
            if is_noptimal[idx]:
                _nopt+=1 

        if _nopt <= noptimality_threshold: # mostly non-noptimal
            _selected_idx = _do_tournament(objective, maximize, indices)
        else: # mostly noptimal
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
        if values[indices[i]] == values[indices[i-1]]:
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
def _select_best(selected, niche, objective, n, maximize, stable):
    """
    Selects best individuals from a population
    """
    if n == 0: 
        return 

    indices = np.empty(n, INT)
    
    if stable:
        _indices = np.argsort(objective).astype(INT)
        _stabilize_sort(_indices, objective)
        if maximize: 
            indices[:] = _indices[-n:]
        else: 
            indices[:] = _indices[:n]
    else: 
        # This is much faster but does not preserve order 
        if maximize:
            indices = np.argpartition(objective, -n)[-n:].astype(INT)
        else:
            indices = np.argpartition(objective, n)[:n].astype(INT)
    
    for j in range(n):
        selected[j, :] = niche[indices[j], :]
    
@njit
def _select_best_with_fallback(selected, niche, fitness, is_noptimal, objective, n, maximize, stable):
    """ Selects best `n` individuals based on fitness.
    If there are not `n` noptimal individuals, selects on objective"""
    if n == 0:
        return
    _nopt = 0 
    for i in range(len(is_noptimal)):
        if is_noptimal[i]:
            _nopt += 1 
    
    if _nopt <= n: # not enough near-optimal points
        return _select_best(selected, niche, objective, n, maximize, stable)
    
    indices = np.empty(n, INT)
    
    noptimal_indices = np.where(is_noptimal)[0]
    noptimal_fitness = fitness[noptimal_indices]
    
    if stable: 
        _indices = np.argsort(noptimal_fitness)
        _stabilize_sort(_indices, noptimal_fitness)
        indices[:] = noptimal_indices[_indices[-n:]]
    else: 
        indices[:] = noptimal_indices[np.argpartition(noptimal_fitness, -n)[-n:]]
    
    for j in range(n):
        selected[j, :] = niche[indices[j], :]

