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

def original_objective(values): 
    
    # For the first test, ndim=2 and works for a function with two decision variables
    z = sum(np.sin(19*np.pi*values[i]) + values[i]/1.7 for i in range(ndim)) + 2
    
    return z


def population_feasibility(
        population,
        optimal_values,
        slack,
        original_objective
        ):

    def individual_feasibility(
            values,
            optimal_values,
            slack,
            original_objective,
            **kwargs
            ):
        """
        Measures the distance from the near-optimality region. 
        The aim is to penalise infeasibility as part of the 
        fitness function proportionally to such a distance.
        The slack is given as a fractional value.   
        """
            
        try: 
            len(values) == len(optimal_values)
            dist_from_optimum = (
                original_objective(values)
                - original_objective(optimal_values)
                )/original_objective(optimal_values)
        except:
            print("The size of the values is not the same between the optimum and the assessed individual")
        
        dist_from_slack = dist_from_optimum - slack
        
        return abs(dist_from_slack)

    pop_feasibility_score = np.empty_like(population)
    for niche_id in range(population.shape[0]):
        for indiv_id in range(population[niche_id].shape[0]):
            pop_feasibility_score[niche_id][indiv_id] = (
                individual_feasibility(
                    values=population[niche_id][indiv_id],
                    optimal_values=optimal_values,
                    slack=slack,
                    original_objective=original_objective)
            )

    return pop_feasibility_score