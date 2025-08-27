import numpy as np
from numba import njit

# API functions

@njit
def evaluate_fitness_dist_to_centroids(fitness, population, centroids):
    """
    Calculates fitness as the Euclidean distance to the nearest centroid of another niche.
    """
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            min_dist = np.inf
            for c in range(centroids.shape[0]):
                if i == c: 
                    continue
                dist = np.sum((population[i, j] - centroids[c])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
            fitness[i, j] = min_dist

# private helper functions

@njit
def _euclidean_distance(p1, p2):
    """Euclidean distance"""
    return sum((p1-p2)**2)**0.5
    
@njit
def _dimensionwise_distance(p1, p2):
    return min(np.abs(p1-p2))