import numpy as np

from mga.commons.numba_overload import njit


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
        power = 2
        inv_power = 0.5
    elif method == 1:
        power = 1
        inv_power = 1
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
        power = 2
        inv_power = 0.5
    elif method == 1:
        power = 1
        inv_power = 1
    else:
        raise Exception

    for i in range(1, points.shape[0]):
        for j in range(points.shape[1]):
            dist = ((points[i, j] - centroids[0]) ** power).sum() ** inv_power
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

                dist = 0
                for k in range(ndim):
                    dist += (points[i, j, k] - centroids[c, k]) ** power

                # Add objective dimension distance
                dist += (scaled_obj - centroids[c, -1]) ** power

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

                dist = 0
                for k in range(ndim):
                    dist += (points[i, j, k] - centroids[c, k]) ** power

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
                    points[i, j], centroids[c], anchor, scaled_obj, centroids[c, -1], anchor_obj
                )

                if ang_sep < min_ang_sep:
                    min_ang_sep = ang_sep

            fitness[i, j] = min_ang_sep


@njit(fastmath=True, inline="always")
def _accumulate_product_and_norms(p, c, anchor, dot_product, norm_p_sq, norm_c_sq):
    v_p = p - anchor
    v_c = c - anchor

    dot_product += v_p * v_c
    norm_p_sq += v_p * v_p
    norm_c_sq += v_c * v_c

    return dot_product, norm_p_sq, norm_c_sq


@njit(fastmath=True, inline="always")
def _get_vector_components(p, c, anchor):
    """Calculates dot product and squared norms for the decision variable arrays."""
    dot_product = 0.0
    norm_p_sq = 0.0
    norm_c_sq = 0.0

    for k in range(len(p)):
        dot_product, norm_p_sq, norm_c_sq = _accumulate_product_and_norms(
           p[k], c[k], anchor[k], dot_product, norm_p_sq, norm_c_sq
        )

    return dot_product, norm_p_sq, norm_c_sq


@njit(fastmath=True, inline="always")
def _calc_similarity(dot_product, norm_p_sq, norm_c_sq):
    """Calculates final 1 - cosine similarity from components."""
    if norm_p_sq == 0.0 or norm_c_sq == 0.0:
        return 0.0

    return 1.0 - (dot_product / ((norm_p_sq * norm_c_sq) ** 0.5))


@njit(fastmath=True, inline="always")
def _angular_separation(p, c, anchor):
    """
    Calculates 1 - cosine similarity between vector (p - anchor) and (c - anchor).
    Returns 0.0 to 2.0.
    """
    dp, np_sq, nc_sq = _get_vector_components(p, c, anchor)
    return _calc_similarity(dp, np_sq, nc_sq)


@njit(fastmath=True, inline="always")
def _angular_separation_ext(p, c, anchor, p_obj, c_obj, anchor_obj):
    """
    Calculates 1 - cosine similarity with an appended objective dimension.
    """
    dp, np_sq, nc_sq = _get_vector_components(p, c, anchor)

    # Add objective dimension
    dp, np_sq, nc_sq = _accumulate_product_and_norms(p_obj, c_obj, anchor_obj, dp, np_sq, nc_sq)

    return _calc_similarity(dp, np_sq, nc_sq)
