# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 19:50:40 2025

@author: u6942852
"""

import numpy as np 
from numba import njit, prange
from convexhullarea import convex_hull_area

@njit 
def shannonIndex(values, lb, ub, nbin, npoint, counts):
    bin_width = (ub-lb) / nbin

    for i in range(npoint):
        idx = int((values[i]-lb)/bin_width)
        if idx < 0: idx = 0
        if idx > nbin-1: idx = nbin-1
        counts[idx] += 1
    
    H = 0.0
    nonzero = 0
    for i in range(nbin):
        if counts[i] > 0:
            p = counts[i] / npoint
            H -= p * np.log(p)
            nonzero += 1

    # Miller-Madow bias correction
    H += (nonzero - 1) / (2.0 * npoint)
    return H 

@njit 
def meanOfShannon(points, lb, ub):
    """ mean of shannon index along each dimension of a set of points
    To be called on problem.noptima """
    npoint, ndim = points.shape
    nbin = max(2, int(npoint**0.5)) # sqrt of number of samples, consider updating later 
    acc = 0
    counts = np.zeros(nbin, dtype=np.int64)
    for k in range(ndim):
        counts[:] = 0
        acc += shannonIndex(points[:, k], lb[k], ub[k], nbin, npoint, counts)
    acc /= np.log(nbin) # normalize 
    acc /= ndim # take mean
    return acc 

@njit
def sumOfFitness(fitness, noptimality):
    """ sum of maximum (noptimal) fitness of each population""" 
    acc = 0
    for i in range(fitness.shape[0]):
        _max = 0
        for j in range(fitness.shape[1]):
            if noptimality[i, j]:
                if fitness[i, j] > _max:
                    _max = fitness[i, j]
        acc += _max
    return acc

@njit 
def meanOfFitness(fitness, noptimality):
    return sumOfFitness(fitness, noptimality) / fitness.shape[0]

@njit(parallel=True)
def VESA_pop(population):
    """ Measures VESA of each niche - I don't know why you would do this """ 
    vesa = np.zeros(population.shape[0], np.float64)
    for i in prange(population.shape[0]):
        for k in range(population.shape[2]):
            for _k in range(k + 1, population.shape[2]):
                points = np.stack((population[i, :, k], population[i, :, _k]), axis=-1)
                vesa[i] += convex_hull_area(points)
    return vesa
    
@njit(parallel=True)
def VESA(points):
    """ Measures VESA of a set of points 
    To be called on problem.noptima """
    vesa = 0.0
    for k in prange(points.shape[1]):
        for _k in range(k + 1, points.shape[1]):
            projection = np.stack((points[:, k], points[:, _k]), axis=-1)
            vesa += convex_hull_area(projection)
    return vesa

@njit
def std(quantity, noptimality):
    """ standard deviation. `quantity` may be problem.objective or problem.fitness"""
    stds = np.empty(quantity.shape[0])
    for i in range(quantity.shape[0]):
        _feas = False
        for j in range(noptimality.shape[1]):
            if noptimality[i, j] is True:
                _feas = True
                break
        if _feas is True:
            stds[i] = np.std(quantity[i][noptimality[i]])
        else: 
            stds[i] = np.inf
    return stds

@njit
def var(quantity, noptimality):
    """ statistical variance. `quantity` may be problem.objective or problem.fitness"""
    vars = np.empty(quantity.shape[0])
    for i in range(quantity.shape[0]):
        _feas = False
        for j in range(noptimality.shape[1]):
            if noptimality[i, j] is True:
                _feas = True
                break
        if _feas is True:
            vars[i] = np.var(quantity[i][noptimality[i]])
        else: 
            vars[i] = np.inf

    return vars


def diversity_custom_option(x):
    """ Allows qualitative things like, maximizing/minimizing specific technologies """
    
    return sum(x[:5]) + min(x[8:10])
    
    
