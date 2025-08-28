import numpy as np
from numba import njit

# API functions

@njit
def selection(selected, niche, criteria, maximize, elite_count, tourn_count, tourn_size, rng, stable):
    """
    Selects individuals using a combination of elitism and tournament selection.
    """
    if elite_count > 0:
        selected[:elite_count] = _select_best(niche, criteria, elite_count, maximize, stable)
    if tourn_count > 0:
        selected[elite_count:] = _select_tournament(niche, criteria, tourn_count, tourn_size, rng, maximize)

@njit
def selection_with_fallback(selected, niche, fitness, is_noptimal, objective, maximize, elite_count, tourn_count, tourn_size, rng, stable):
    """
    Selects based on fitness, falling back to objective if not enough n-optimal individuals exist.
    """
    if elite_count > 0:
        selected[:elite_count] = _select_best_with_fallback(
            niche, fitness, is_noptimal, objective, elite_count, maximize, stable
        )
    if tourn_count > 0:
        selected[elite_count:] = _select_tournament_with_fallback(
            niche, fitness, is_noptimal, objective, tourn_count, tourn_size, rng, maximize
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
    """
    if maximize:
        _selected_idx = 0
        _best = -np.inf
        for idx in indices:
            if criteria[idx] > _best:
                _selected_idx = idx
                _best = criteria[idx]
    else:
        _selected_idx = 0
        _best = np.inf
        for idx in indices:
            if criteria[idx] < _best:
                _selected_idx = idx
                _best = criteria[idx]
    return _selected_idx 

@njit
def _select_tournament(niche, objective, n, tournsize, rng, maximize):
    """
    Selects {n} individuals from a population according to selection tournament
    """
    selected = np.empty((n, niche.shape[1]), np.float64)
    indices = np.empty(tournsize, np.int64)
    
    for m in range(n):
        _draw_tournament_indices(indices, niche.shape[0], rng)
        _selected_idx = _do_tournament(objective, maximize, indices)
        selected[m, :] = niche[_selected_idx, :]

    return selected

@njit
def _select_tournament_with_fallback(niche, fitness, noptimality, objective, n, tournsize, rng, maximize):
    """select on fitness preferred. fitness always maximised. 
    objective max/minimized based on value of `maximize`"""
    selected = np.empty((n, niche.shape[1]), np.float64)
    indices = np.empty(tournsize, np.int64)
    noptimality_threshold = tournsize / 2 
    
    for m in range(n):
        _draw_tournament_indices(indices, niche.shape[0], rng)
        
        _nopt = 0
        for idx in indices:
            if noptimality[idx]:
                _nopt+=1 
        
        if _nopt <= noptimality_threshold: # mostly non-noptimal
            _selected_idx = _do_tournament(objective, maximize, indices)
        else: # mostly noptimal
            _selected_idx = _do_tournament(fitness, True, indices)

        selected[m, :] = niche[_selected_idx, :]
    
    return selected 

@njit
def _select_best(niche, objective, n, maximize, stable):
    """
    Selects best individuals from a population
    """
    selected = np.empty((n, niche.shape[1]))
    indices = np.empty(n, np.int64)
    
    if stable:
        _indices = np.argsort(objective)
        _stabilise_sort(_indices, objective)
        if maximize: 
            indices[:] = _indices[-n:]
        else: 
            indices[:] = _indices[:n]
    else: 
        # This is much faster but does not preserve order 
        if maximize:
            indices = np.argpartition(objective, -n)[-n:]
        else:
            indices = np.argpartition(objective, n)[:n]   
        
    for j in range(n):
        selected[j, :] = niche[indices[j], :]
    
    return selected 

@njit
def _select_best_with_fallback(niche, fitness, noptimality, objective, n, maximize, stable):
    """ Selects best `n` individuals based on fitness.
    If there are not `n` noptimal individuals, selects on objective"""
    
    _nopt = 0 
    for i in range(len(noptimality)):
        if noptimality[i]:
            _nopt += 1 
    
# =============================================================================
### Alteranative code block. Much slower 
#     if _nopt < n: # mostly non-noptimal
#         return selBest(niche, objective, n, maximize, stable)
#     
#     selected = np.empty((n, niche.shape[1]))
#     indices = np.empty(n, np.int64)
#     
#     noptimal_indices = np.where(noptimality)[0]
#     noptimal_fitness = fitness[noptimal_indices]
#     
#     if _nopt == n: # edge case breaks numba but also we can skip and be more efficient anyway
#         indices[:] = noptimal_indices
#     elif stable: 
# =============================================================================

# =============================================================================
#   This code block makes the entire algorithm better than 10x faster 
#   Requires a slightly dodgy alteration of program logic with _nopt < / <= n
#   But I think this simplification is insignificant 
    if _nopt <= n: # mostly non-noptimal
        return _select_best(niche, objective, n, maximize, stable)
    
    selected = np.empty((n, niche.shape[1]))
    indices = np.empty(n, np.int64)
    
    noptimal_indices = np.where(noptimality)[0]
    noptimal_fitness = fitness[noptimal_indices]
    
    if stable: 
# =============================================================================
        _indices = np.argsort(noptimal_fitness)
        _stabilise_sort(_indices, noptimal_fitness)
        indices[:] = noptimal_indices[_indices[-n:]]
    else: 
        indices[:] = noptimal_indices[np.argpartition(noptimal_fitness, -n)[-n:]]
    
    for j in range(n):
        selected[j, :] = niche[indices[j], :]
        
    return selected 

@njit
def _stabilise_sort(indices, values):
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
    return indices