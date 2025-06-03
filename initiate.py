# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:08:03 2025

@author: hmtha
"""

# =============================================================================
# Create initial population
#   Either random or based off of known optimum, or another heuristic 
# =============================================================================

import random

from deap import base
from deap import creator
from deap import tools

def initiate_populations(
                    nniche, # number of desired niches
                    maxpop, # maximum population size per niche
                    ndim,   # number of decision variables per individual
                    **kwargs
                    ):
    
    # For now, we only initialise the co-evolving near-optimal niches. 
    # The optimum is given as a starting point. In principle, the optimum could be 
    # a niche in itself, with its specific fitness function, and become part
    # of the same meta-heuristic search as the one that looks for the near-optimal solutions.

    # To initialise niches, we first need to create a DEAP fitness object to assign to it
    creator.create("FitnessMaxDistance", base.Fitness, weights=(1.0,)) 
    # Hence, we can create a list object for individuals to be evaluated based on the above fitness
    creator.create("Individual", list, fitness=creator.FitnessMaxDistance) 
    
    # We create a toolbox to register attributes we want the individuals to have and
    # create the actual instances of the individuals to be packaged into niches
    toolbox = base.Toolbox()

    # The initial values of the individuals are set to random numbers between 0 and 1
    toolbox.register("attr_uniform", random.uniform, 0, 1) 
    
    # An instance of 'Individual' is an 'individual' with the 'attr_uniform' attribute
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_uniform, 1)  
    # A 'niche' is a list of the created individuals with the desired attributes
    toolbox.register("niche", tools.initRepeat, list, toolbox.individual)
    
    # Create a list of niches (as many as the desired alternatives)
    niches_list = [toolbox.niche(n=maxpop) for _ in range(nniche)]
    
    return(niches_list) 
    
