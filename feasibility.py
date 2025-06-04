# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:10:01 2025

@author: hmtha
"""

# =============================================================================
# Fitness & feasibility
# 
# Fitness is calculated based on distance from other niches. Feasibility embeds 
# information on both technical feasibility (hours of lost load) and 
# distance from the accepted slack. The degree of infeasibility of an individual
# becomes a proportional penality to the fitness.
# =============================================================================

import numpy as np

def original_objective(values): 
    
    # For the first test, ndim=2 and works for a function with two decision variables
    z = sum(np.sin(19*np.pi*values[i]) + values[i]/1.7 for i in range(ndim)) + 2
    
    return z

###
# --- FEASIBILITY ------
###

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

    # This for loop could be possibly made more efficient
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


###
# --- FITNESS ------
###


def find_centroids(population):
    
    centroids = []

    # Iterate over each niche and calculate the centroid
    for niche in population:
        num_indiv = len(niche)  # Number of individuals in the niche
        num_vars_indiv = len(niche[0])  # Number of decision variables per individual
        
        # Calculate the centroid for each decision variable
        # TODO: check that the formula below (taken from the Jacob's repo) 
        # is actually what we want.
        centroid = [
            sum(individual[i] for individual in niche) / 
            num_indiv for i in range(num_vars_indiv)
            ]
        centroids.append(centroid)  

    return np.array(centroids)

# WORK-IN-PROGRESS/not ready: simplify/improve this fitness function taken from Jacob's repo
def fitness(population):
    
    distances = []
    minimal_distances = []
    fitness_SP = {}

    centroids = find_centroids(population)

    # Step 1: Calculate per-variable distances for each individual in each niche
    for q, population in enumerate(population):
        
        niche_distances = []

        for indiv in population[q]:
            indiv_distances = []
            for p, centroid in enumerate(centroids):
                if p != q:  # Skip the centroid of the same subpopulation
                    per_var_distances = [abs(indiv[i] - centroid[i]) for i in range(len(indiv))]
                    indiv_distances.append(per_var_distances)
            
            niche_distances.append(indiv_distances)
        
        distances.append(niche_distances)

    # Step 2: Calculate Minimal Distances per Variable
    for subpopulation_distances in distances:
        subpopulation_minimal = []
        
        for individual_distances in subpopulation_distances:
            min_distance_per_variable = [min(distance[i] for distance in individual_distances) for i in range(len(individual_distances[0]))]
            subpopulation_minimal.append(min_distance_per_variable)
        
        minimal_distances.append(subpopulation_minimal)

    # Step 3: Calculate Fitness SP for each individual
    for sp_index, subpopulation in enumerate(minimal_distances, start=1):
        fitness_values = [(min(individual),) for individual in subpopulation]
        fitness_SP[sp_index] = fitness_values

    return fitness_SP