# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:08:03 2025

@author: hmtha
"""

# =============================================================================
# Create initial population
#   Either random or based off of known optimum, or another heuristic 
# =============================================================================

import numpy as np

from deap_objects import creator, toolbox
from deap import tools

def initiate_populations(
                    nniche, # number of desired niches
                    maxpop, # maximum population size per niche
                    lb,     # lower bounds on decision variables
                    ub,     # upper bounds on decision variables
                    coordinate_generator = np.random.uniform, 
                    **kwargs
                    ):
    
    toolbox.register("attr_coordinates", lambda: coordinate_generator(lb, ub).tolist())
    
    def create_deap_individual():
        ind = creator.Individual(toolbox.attr_coordinates()) # Create an instance of creator.Individual 
        return ind
    
    toolbox.register("individual", create_deap_individual)

    # A 'niche' is a list of the created individuals with the desired attributes
    toolbox.register("niche", tools.initRepeat, list, toolbox.individual)

    # Create a list of niches (as many as the desired alternatives)
    niches_list = [toolbox.niche(n=maxpop) for _ in range(nniche)]
    return niches_list
    
