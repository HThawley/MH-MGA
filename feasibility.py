# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:10:01 2025

@author: hmtha
"""

# =============================================================================
# Feasibility
#   * Re-crossover with elite for very infeasible solutions
# =============================================================================

from crossover_mutation import crossover, mutation

def feasibility(*args, **kwargs):
    """
    Try to make very infeasible solutions more feasible 
        (but not necessarily guarenteed to be feasible) 
    Re-crossover/mutation
    """
    pass