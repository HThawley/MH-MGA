# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:08:20 2025

@author: hmtha
"""

# =============================================================================
# Select Elites (tournament function)
#   * hyperparameterise no. in tournament?
# =============================================================================

from deap_objects import creator, base, toolbox
# =============================================================================
# TODO: include elitism
# =============================================================================
def tournament(
        population, 
        nelite, 
        tournsize,
        ):
    
    toolbox.register("select", tools.selTournament, k=nelite, tournsize=tournsize) 

    elites = [toolbox.select(niche) for niche in niches_list]
        
    return elites
