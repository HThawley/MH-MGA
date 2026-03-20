import numpy as np
from numba import njit


# API functions
@njit
def evaluate_fitness_dist_to_centroids(fitness, points, centroids):
    """
    Calculates fitness as the Euclidean distance to the nearest centroid of another niche.
    """
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            min_dist = np.inf
            for c in range(centroids.shape[0]):
                if i == c:
                    continue
                dist = _euclidean_distance(points[i, j], centroids[c])
                if dist < min_dist:
                    min_dist = dist
            fitness[i, j] = min_dist


@njit
def evaluate_fitness_dist_to_centroids_ext(
    fitness, points, centroids, objective_values, objective_scaler
):
    """
    Calculates fitness including the scaled objective value as an additional dimension.

    Currently only supports euclidean distance
    """
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            min_dist = np.inf
            scaled_obj = objective_values[i, j] / objective_scaler
            for c in range(centroids.shape[0]):
                if i == c:
                    continue

                # Euclidean distance squared
                dist_sq = 0
                for k in range(points.shape[2]):
                    dist_sq += (points[i, j, k] - centroids[c, k])**2
                # Add objective dimension Euclidean distance squared
                dist_sq += (scaled_obj - centroids[c, -1])**2
                dist = dist_sq

                if dist < min_dist:
                    min_dist = dist
            fitness[i, j] = min_dist**0.5


# private helper functions
@njit
def _euclidean_distance(p1, p2):
    """Euclidean distance"""
    return np.sum((p1-p2)**2)**0.5


@njit
def _dimensionwise_distance(p1, p2):
    return np.min(np.abs(p1-p2))
