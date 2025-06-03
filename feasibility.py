# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:10:01 2025

@author: hmtha
"""

# =============================================================================
# Feasibility
#   * Re-crossover with elite for very infeasible solutions
# =============================================================================

import numpy as np

def feasibility(
        ndim,
        values,
        optimal_values,
        **kwargs
        ):
    """
    Try to make very infeasible solutions more feasible 
        (but not necessarily guarenteed to be feasible) 
    Re-crossover/mutation
    """
    
    def original_assessed_function(ndim, values): 
        
        # For the first test, ndim=2 and works for a function with two decision variables
        try: 
            len(range(ndim)) == len(values)
            z = sum(np.sin(19*np.pi*values[i]) + values[i]/1.7 for i in range(ndim)) + 2
        except:
            print("The dimension size does not matche the length of the values")
        
        return z
    
    
    def distance_from_optimum(ndim,values,optimal_values):
        """Measures the distance from the near-optimality region."""
        
        try: 
            len(values) == len(optimal_values)
            dist_from_optimum = (
                original_assessed_function(ndim, values)
                - original_assessed_function(ndim, optimal_values)
                )/original_assessed_function(ndim, optimal_values)
        except:
            print("The size of the values is not the same between the optimum and the assessed individual")
        
        return abs(dist_from_optimum)
    
    return