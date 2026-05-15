import numpy as np

from mga.commons.numba_overload import njit


# API functions
@njit
def evaluate_fitness_angular_to_centroids(fitness, points, centroids):
    """
    Calculates fitness as the minimum angular separation (1 - cosine similarity)
    to the nearest centroid of another niche.
    """
    anchor = centroids[0]

    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            min_ang_sep = np.inf
            for c in range(centroids.shape[0]):
                if i == c:
                    continue
                ang_sep = _angular_separation(points[i, j], centroids[c], anchor)
                if ang_sep < min_ang_sep:
                    min_ang_sep = ang_sep
            fitness[i, j] = min_ang_sep


@njit
def evaluate_fitness_angular_to_centroids_ext(
    fitness, points, centroids, raw_objectives, objective_scaler
):
    """
    Calculates angular fitness including the scaled objective value
    as an additional dimension.
    """
    anchor = centroids[0]
    anchor_obj = anchor[-1]

    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            min_ang_sep = np.inf
            scaled_obj = raw_objectives[i, j] / objective_scaler

            for c in range(centroids.shape[0]):
                if i == c:
                    continue

                ang_sep = _angular_separation_ext(
                    points[i, j], centroids[c], anchor, scaled_obj, centroids[c, -1], anchor_obj
                )

                if ang_sep < min_ang_sep:
                    min_ang_sep = ang_sep

            fitness[i, j] = min_ang_sep


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
def evaluate_fitness_dist_to_centroids_ext(fitness, points, centroids, raw_objectives, objective_scaler):
    """
    Calculates fitness including the scaled objective value as an additional dimension.

    Currently only supports euclidean distance
    """
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            min_dist = np.inf
            scaled_obj = raw_objectives[i, j] / objective_scaler
            for c in range(centroids.shape[0]):
                if i == c:
                    continue

                # Euclidean distance squared
                dist_sq = 0
                for k in range(points.shape[2]):
                    dist_sq += (points[i, j, k] - centroids[c, k]) ** 2
                # Add objective dimension Euclidean distance squared
                dist_sq += (scaled_obj - centroids[c, -1]) ** 2
                dist = dist_sq

                if dist < min_dist:
                    min_dist = dist
            fitness[i, j] = min_dist**0.5


@njit
def evaluate_fitness_pure_optimization(fitness, points, centroids, raw_objectives, repulsion_weight):
    for i in range(1, points.shape[0]):
        for j in range(points.shape[1]):
            dist = _euclidean_distance(points[i, j], centroids[0])
            fitness[i, j] = (dist * repulsion_weight) + (raw_objectives[i, j] * (1.0 - repulsion_weight))


# private helper functions
@njit(inline="always")
def _euclidean_distance(p1, p2):
    """Euclidean distance"""
    return np.sum((p1 - p2) ** 2) ** 0.5


@njit
def _dimensionwise_distance(p1, p2):
    return np.min(np.abs(p1 - p2))


@njit(fastmath=True, inline="always")
def _accumulate_product_and_norms(p, c, anchor, dot_product, norm_p_sq, norm_c_sq):
    v_p = p - anchor
    v_c = c - anchor

    dot_product += v_p * v_c
    norm_p_sq += v_p * v_p
    norm_c_sq += v_c * v_c


@njit(fastmath=True, inline="always")
def _get_vector_components(p, c, anchor):
    """Calculates dot product and squared norms for the decision variable arrays."""
    dot_product = 0.0
    norm_p_sq = 0.0
    norm_c_sq = 0.0

    for k in range(len(p)):
        _accumulate_product_and_norms(p[k], c[k], anchor[k], dot_product, norm_p_sq, norm_c_sq)

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
    _accumulate_product_and_norms(p_obj, c_obj, anchor_obj, dp, np_sq, nc_sq)

    return _calc_similarity(dp, np_sq, nc_sq)
