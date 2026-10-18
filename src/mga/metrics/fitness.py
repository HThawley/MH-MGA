import numpy as np

from mga.commons.numba_overload import njit
from mga.metrics.distances import (
    _minkowski_distance,
    _angular_separation,
    _angular_separation_ext,
)


# API functions
@njit
def evaluate_fitness_angular(
    fitness, points, centroids, raw_objectives, objective_scaler, include_objective
):
    """
    Calculates fitness as the angular separation to the nearest centroid of another niche

    optionally includes the scaled objective value as an additional dimension.
    """
    anchor = centroids[0]

    if include_objective:
        _angular_with_objective(anchor, fitness, points, centroids, raw_objectives, objective_scaler)
    else:
        _angular_standard(anchor, fitness, points, centroids)


@njit
def evaluate_fitness_distance(
    fitness, points, centroids, raw_objectives, objective_scaler, method, include_objective
):
    """
    Calculates fitness as the {method} distance to the nearest centroid of another niche.

    optionally includes the scaled objective value as an additional dimension.
    """
    if method == 2:
        power = 2.0
        inv_power = 0.5
    elif method == 1:
        power = 1.0
        inv_power = 1.0
    else:
        raise Exception

    if include_objective:
        _distance_with_objective(
            fitness, points, centroids, raw_objectives, objective_scaler, power, inv_power
        )
    else:
        _distance_standard(fitness, points, centroids, power, inv_power)


@njit
def evaluate_fitness_pure_optimization(fitness, points, centroids, raw_objectives, repulsion_weight, method):
    if method == 2:
        power = 2.0
        inv_power = 0.5
    elif method == 1:
        power = 1.0
        inv_power = 1.0
    else:
        raise Exception

    for i in range(1, points.shape[0]):
        for j in range(points.shape[1]):
            dist = _minkowski_distance(points[i, j], centroids[0], power, inv_power)
            fitness[i, j] = (dist * repulsion_weight) + (raw_objectives[i, j] * (1.0 - repulsion_weight))


# private helper functions
@njit
def _distance_with_objective(
    fitness, points, centroids, raw_objectives, objective_scaler, power, inv_power
):
    num_niches, pop_size, ndim = points.shape

    for i in range(num_niches):
        for j in range(pop_size):
            min_dist = np.inf
            scaled_obj = raw_objectives[i, j] * objective_scaler
            for c in range(num_niches):
                if i == c:
                    continue

                # Calculate base spatial distance (un-inversed to allow adding the objective dimension)
                dist = _minkowski_distance(points[i, j], centroids[c, :ndim], power, 1.0)

                # Add objective dimension distance
                dist += abs(scaled_obj - centroids[c, -1]) ** power

                if dist < min_dist:
                    min_dist = dist

            fitness[i, j] = min_dist ** inv_power


@njit
def _distance_standard(fitness, points, centroids, power, inv_power):
    num_niches, pop_size, ndim = points.shape

    for i in range(num_niches):
        for j in range(pop_size):
            min_dist = np.inf
            for c in range(num_niches):
                if i == c:
                    continue

                dist = _minkowski_distance(points[i, j], centroids[c], power, 1.0)

                if dist < min_dist:
                    min_dist = dist

            fitness[i, j] = min_dist ** inv_power


@njit
def _angular_standard(anchor, fitness, points, centroids):
    num_niches, pop_size, ndim = points.shape

    for i in range(num_niches):
        for j in range(pop_size):
            min_ang_sep = np.inf
            for c in range(1, num_niches):
                if i == c:
                    continue
                ang_sep = _angular_separation(points[i, j], centroids[c], anchor)
                if ang_sep < min_ang_sep:
                    min_ang_sep = ang_sep
            fitness[i, j] = min_ang_sep


@njit
def _angular_with_objective(anchor, fitness, points, centroids, raw_objectives, objective_scaler):
    num_niches, pop_size, ndim = points.shape
    anchor_obj = anchor[-1]

    for i in range(num_niches):
        for j in range(pop_size):
            min_ang_sep = np.inf
            scaled_obj = raw_objectives[i, j] * objective_scaler

            for c in range(1, num_niches):
                if i == c:
                    continue

                ang_sep = _angular_separation_ext(
                    points[i, j], centroids[c, :ndim], anchor[:ndim], scaled_obj, centroids[c, -1], anchor_obj
                )

                if ang_sep < min_ang_sep:
                    min_ang_sep = ang_sep

            fitness[i, j] = min_ang_sep
