# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 19:36:57 2025

@author: u6942852
"""

from numba import njit
import numpy as np 

#%% mutation functions
    
@njit
def mutFloat(item, sigma, rng):
    """ gaussian mutation for single variable """
    return rng.normal(item, sigma)

@njit
def mutInt(item, sigma, rng):
    """Integer mutation for single variable"""
    return round(rng.normal(item, sigma))

@njit
def mutBool(item,):
    """Boolean mutation for single variable"""
    return 1.0 - item

@njit 
def mutGaussianMixedVec(population, sigma, indpb, rng, integrality, boolean_mask, startidx=0):
    for i in range(population.shape[0]):
        for j in range(startidx, population.shape[1]):
            for k in range(population.shape[2]):
                if rng.random() < indpb:
                    if boolean_mask[k]:
                        population[i, j, k] = mutBool(population[i, j, k]) 
                    elif integrality[k]:
                        population[i, j, k] = mutInt(population[i, j, k], sigma[k], rng)
                    else: 
                        population[i, j, k] = mutFloat(population[i, j, k], sigma[k], rng)

@njit
def mutGaussianFloatVec(population, sigma, indpb, rng, integrality, boolean_mask, startidx=0):
    for i in range(population.shape[0]):
        for j in range(startidx, population.shape[1]):
            for k in range(population.shape[2]):
                if rng.random() < indpb:
                    population[i, j, k] = mutFloat(population[i, j, k], sigma[k], rng)

# crossover functions

@njit
def cx_vec(population, crossoverInst, cx, rng, startidx=0):
    for i in range(population.shape[0]):
        # shuffle parents for mating
        rng.shuffle(population[i, startidx:])
        
        for ind1, ind2 in zip(population[i, startidx:][::2], population[i, startidx:][1::2]):
            # crossover probability
            if rng.random() < crossoverInst:
                # Apply crossover
                cx(ind1, ind2, rng)

@njit
def _do_cx(ind1, ind2, index1, index2):
    buffer = 0.0
    # modify in place
    for i in range(index1, index2):
        buffer = ind1[i]
        ind1[i] = ind2[i]
        ind2[i] = buffer   

@njit
def cxOnePoint(ind1, ind2, rng):
    index1 = rng.integers(0, len(ind1))
    _do_cx(ind1, ind2, index1, len(ind1))
    
@njit
def cxTwoPoint(ind1, ind2, rng):
    # +1 / -1 adjustments are made to ensure there is always a crossover 
    # only valid for ndim >= 3
    index1 = rng.integers(0, len(ind1)-1)
    index2 = rng.integers(index1+1, len(ind1))
    _do_cx(ind1, ind2, index1, index2)

# selection functions

# @keeptime("_selT_draw_indices", on_switch)
@njit 
def _selT_draw_indices(indices, ub, rng):
    for i in range(indices.size):
        indices[i] = rng.integers(0, ub)

# @keeptime("selTournament_fallback", on_switch)
@njit
def selTournament_fallback(niche, fitness, noptimality, objective, n, tournsize, rng, maximize):
    """select on fitness preferred. fitness always maximised. 
    objective max/minimized based on value of `maximize`"""
    selected = np.empty((n, niche.shape[1]), np.float64)
    indices = np.empty(tournsize, np.int64)
    noptimality_threshold = tournsize / 2 
    
    for m in range(n):
        _selT_draw_indices(indices, niche.shape[0], rng)
        
        _nopt = 0
        for idx in indices:
            if noptimality[idx]:
                _nopt+=1 
        
        if _nopt <= noptimality_threshold: # mostly non-noptimal
            _selected_idx = _do_selTournament(objective, maximize, indices)
        else: # mostly noptimal
            _selected_idx = _do_selTournament(fitness, True, indices)

        selected[m, :] = niche[_selected_idx, :]
    
    return selected 

# @keeptime("selTournament", on_switch)
@njit
def selTournament(niche, objective, n, tournsize, rng, maximize):
    """objective max/minimized based on value of `maximize`"""
    selected = np.empty((n, niche.shape[1]), np.float64)
    indices = np.empty(tournsize, np.int64)
    
    for m in range(n):
        _selT_draw_indices(indices, niche.shape[0], rng)
        _selected_idx = _do_selTournament(objective, maximize, indices)
        selected[m, :] = niche[_selected_idx, :]

    return selected # , selected_value

@njit 
def _do_selTournament(criteria, maximize, indices):
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
def _stabilise_sort(indices, values):
    """Sorts blocks of duplicate values within a list of indices
    in-place based on the index values for a stable order."""
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

# @keeptime("selBest_fallback", on_switch)
@njit
def selBest_fallback(niche, fitness, noptimality, objective, n, maximize, stable):
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
        return selBest(niche, objective, n, maximize, stable)
    
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

# @keeptime("selBest", on_switch)
@njit
def selBest(niche, objective, n, maximize, stable):
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


# @keeptime("selBest1", on_switch)
@njit
def selBest1(niche, objective, maximize):
    """Special case of selBest where n=1"""
    if maximize:
        index = objective.argmax()
    else:
        index = objective.argmin()
    return niche[index, :] 

# @keeptime("selBest_fallback", on_switch)
@njit
def selBest1_fallback(niche, fitness, noptimality, objective, maximize):
    """ Special case of selBest_fallback where n = 1
    Selects best `n` individuals based on fitness.
    If there are not `n` noptimal individuals, selects on objective"""
    
    # loop through fitness and choose the best noptimal fitness
    # record succes via _nopt: bool
    _nopt = False
    best = -np.inf
    index = -1
    for i in range(niche.shape[0]):
        if not noptimality[i]:
            continue
        elif fitness[i] > best:
            best = fitness[i]
            index = i
            _nopt = True
            
    if not _nopt:
        return selBest1(niche, objective, maximize)
        
    return niche[index, :] 
