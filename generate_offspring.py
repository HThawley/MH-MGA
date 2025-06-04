# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:08:46 2025

@author: hmtha
"""

# =============================================================================
# Generate offspring
# =============================================================================
import numpy as np

from deap_objects import creator, toolbox
from deap import tools


def apply_bounds(ind, lb, ub):
    for i in range(len(ind)):
        ind[i] = min(ub[i], max(lb[i], ind[i]))
        
    return ind

def generate_offspring(
        parents, 
        nelite, 
        maxpop,
        mutation,
        gaussian_sigma, 
        crossover, ## Do we need this?? Depends on cx method
        lb,     # lower bounds on decision variables
        ub,     # upper bounds on decision variables
        ):

# =============================================================================
#      do crossover and mutation
# =============================================================================
    
    ## cxOnePoint used for testing 
    ## TODO: choose cx method
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=gaussian_sigma, indpb=mutation)
    toolbox.register("apply_bounds", apply_bounds, lb=lb, ub=ub)
    # clone parents as the deap functions work in-place
    offspring = [[toolbox.clone(niche[idx%nelite]) for idx in range(maxpop)] for niche in parents]
    
    
    for niche in offspring:
        # shuffle parents for mating
        np.random.shuffle(niche)
        
        for ind1, ind2 in zip(niche[::2], niche[1::2]):
            # crossover probability
            if np.random.random() < crossover:
                # Apply crossover
                toolbox.mate(ind1, ind2)
    
        for ind in niche:
            # Apply mutation
            toolbox.mutate(ind)
            toolbox.apply_bounds(ind)
            # Mark children's fitness as invalid (were retained in cloning)
            # del ind.fitness.values
            # ind.fitness.valid = False

    
    return offspring

