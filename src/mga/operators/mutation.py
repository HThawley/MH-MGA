import numpy as np
from numba import njit

# API functions

@njit 
def mutate_gaussian_population_mixed(population, sigma, indpb, rng, integrality, boolean_mask, startidx=0):
    """
    Mutate individuals in a population 
    Compatible with mixed dtypes 
    """
    for i in range(population.shape[0]):
        for j in range(startidx, population.shape[1]):
            for k in range(population.shape[2]):
                if rng.random() < indpb:
                    if boolean_mask[k]:
                        population[i, j, k] = _mutate_bool(population[i, j, k], sigma[k], rng) 
                    elif integrality[k]:
                        population[i, j, k] = _mutate_int(population[i, j, k], sigma[k], rng)
                    else: 
                        population[i, j, k] = _mutate_float(population[i, j, k], sigma[k], rng)

@njit
def mutate_gaussian_population_float(population, sigma, indpb, rng, integrality, boolean_mask, startidx=0):
    """
    Mutate individuals in a population 
    Compatible only with float-only 
    """
    for i in range(population.shape[0]):
        for j in range(startidx, population.shape[1]):
            for k in range(population.shape[2]):
                if rng.random() < indpb:
                    population[i, j, k] = _mutate_float(population[i, j, k], sigma[k], rng)

@njit 
def mutate_gaussian_niche_mixed(niche, sigma, indpb, rng, integrality, boolean_mask, startidx=0):
    """
    Mutate individuals in a niche 
    Compatible with mixed dtypes 
    """
    for j in range(startidx, niche.shape[0]):
        for k in range(niche.shape[1]):
            if rng.random() < indpb:
                if boolean_mask[k]:
                    niche[j, k] = _mutate_bool(niche[j, k], sigma[k], rng) 
                elif integrality[k]:
                    niche[j, k] = _mutate_int(niche[j, k], sigma[k], rng)
                else: 
                    niche[j, k] = _mutate_float(niche[j, k], sigma[k], rng)

@njit
def mutate_gaussian_niche_float(niche, sigma, indpb, rng, integrality, boolean_mask, startidx=0):
    """
    Mutate individuals in a niche 
    Compatible only with float-only 
    """
    for j in range(startidx, niche.shape[0]):
        for k in range(niche.shape[1]):
            if rng.random() < indpb:
                niche[j, k] = _mutate_float(niche[j, k], sigma[k], rng)

# private helper functions

@njit
def _mutate_float(item, sigma, rng):
    """ gaussian mutation for single variable """
    return rng.normal(item, sigma)

@njit
def _mutate_int(item, sigma, rng):
    """Integer mutation for single variable"""
    return round(rng.normal(item, sigma))

@njit
def _mutate_bool(item, sigma, rng):
    """Boolean mutation for single variable"""
    if abs(rng.normal(0, sigma)) <= rng.random():
        return item
    else: 
        return 1.0 - item
