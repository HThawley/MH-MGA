# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:08:20 2025

@author: hmtha
"""


from deap_objects import creator, base, toolbox

def select_parents(
        population, 
        elitek,
        tournk, 
        tournsize,
        ):
    
    # order irrelevant
    parents = [
        tools.selBest(niche, k=elitek) + 
        tools.selTournament(niche, k=tournk, tournsize=tournsize) 
        for niche in population]
        
    return parents
