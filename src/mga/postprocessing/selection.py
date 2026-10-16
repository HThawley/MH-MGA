import numpy as np
from mga.commons.numba_overload import njit
from mga.commons.types import npfloat, npintp
from mga.metrics.diversity import volume_estimation_by_shadow_addition


def project_space(points: np.ndarray, space_scaler: np.ndarray | None = None) -> np.ndarray:
    """
    Projects raw decision vectors into a scaled evaluation space using the provided space scaler.
    Supports both 1D element-wise scaling arrays and 2D linear projection matrices, aligning
    with the MHMGA space normalization and aggregation logic.

    points is of shape (N, ndim)
    space_scaler is of shape (ndim, scaled_ndim)
    """
    if space_scaler is None:
        return points

    space_scaler = np.asarray(space_scaler, dtype=npfloat)
    if space_scaler.ndim == 1 or (space_scaler.ndim == 2 and space_scaler.shape[0] == 1):
        return points * space_scaler.reshape(1, -1)
    elif space_scaler.ndim == 2:
        return points @ space_scaler

    raise ValueError("space_scaler must be 1D or 2D")


def get_diverse_indices(
    points: np.ndarray,
    n: int,
    strategy: str = "max-min",
    fitness_method: str | int = "l2"
) -> np.ndarray:
    """
    Identifies a topologically diverse subset of near-optimal alternatives and returns their row indices.

    The 0th index is always selected first, under the assumption it represents the global optimum anchor.
    Subset selection is performed iteratively using either spatial distance (max-min) or volumetric
    expansion (VESA) heuristics.
    """
    n = min(n, points.shape[0])
    if n <= 1:
        return np.arange(n, dtype=npintp)

    if strategy == "max-min":
        method_map = {"angular": 0, "l1": 1, "l2": 2}
        if isinstance(fitness_method, str):
            fitness_method = method_map.get(fitness_method.lower(), 2)

        if fitness_method == 0:
            return _max_min_dispersion_angular(points, n)
        elif fitness_method in (1, 2):
            return _max_min_dispersion_distance(points, n, fitness_method)
        else:
            raise ValueError(f"Unknown fitness_method: {fitness_method}")

    elif strategy == "vesa":
        return _vesa_dispersion(points, n)

    raise ValueError(f"Unknown strategy: '{strategy}'")


@njit
def _max_min_dispersion_distance(
    points: np.ndarray,
    n: int,
    method: int,
) -> np.ndarray:
    """
    Iteratively builds a diverse subset by selecting the candidate point that maximizes the
    minimum L1 or L2 spatial distance to all previously selected points.
    """
    num_points, ndim = points.shape
    selected = np.zeros(n, dtype=npintp)
    selected[0] = 0

    min_dists = np.full(num_points, np.inf, dtype=npfloat)

    power = 2.0 if method == 2 else 1.0
    inv_power = 0.5 if method == 2 else 1.0

    for k in range(1, n):
        last_selected = points[selected[k - 1]]
        best_dist = -1.0
        best_idx = -1

        for i in range(num_points):
            dist = 0.0
            for d in range(ndim):
                dist += abs(points[i, d] - last_selected[d]) ** power
            dist = dist ** inv_power

            if dist < min_dists[i]:
                min_dists[i] = dist

            if min_dists[i] > best_dist:
                best_dist = min_dists[i]
                best_idx = i

        selected[k] = best_idx

    return selected


@njit
def _max_min_dispersion_angular(
    points: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Iteratively builds a diverse subset by selecting the candidate point that maximizes the
    minimum angular separation (1 - cosine similarity) to all previously selected points,
    using the 0th point as the geometric origin.
    """
    num_points, ndim = points.shape
    selected = np.zeros(n, dtype=npintp)
    selected[0] = 0
    anchor = points[0]

    min_dists = np.full(num_points, np.inf, dtype=npfloat)

    for k in range(1, n):
        last_selected = points[selected[k - 1]]
        best_dist = -1.0
        best_idx = -1

        for i in range(num_points):
            dp = 0.0
            np_sq = 0.0
            nc_sq = 0.0

            for d in range(ndim):
                v_p = points[i, d] - anchor[d]
                v_c = last_selected[d] - anchor[d]

                dp += v_p * v_c
                np_sq += v_p * v_p
                nc_sq += v_c * v_c

            if np_sq == 0.0 or nc_sq == 0.0:
                dist = 0.0
            else:
                dist = 1.0 - (dp / ((np_sq * nc_sq) ** 0.5))

            if dist < min_dists[i]:
                min_dists[i] = dist

            if min_dists[i] > best_dist:
                best_dist = min_dists[i]
                best_idx = i

        selected[k] = best_idx

    return selected


@njit
def _vesa_dispersion(
    points: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Iteratively builds a diverse subset by greedily selecting the candidate point that provides
    the largest marginal increase to the Volume Estimation by Shadow Addition (VESA) metric.
    """
    num_points, ndim = points.shape
    selected = np.zeros(n, dtype=npintp)
    selected[0] = 0

    is_selected = np.zeros(num_points, dtype=np.bool_)
    is_selected[0] = True

    for k in range(1, n):
        best_vesa = -1.0
        best_idx = -1

        subset = np.empty((k + 1, ndim), dtype=npfloat)
        subset_mask = np.ones(k + 1, dtype=np.bool_)

        for j in range(k):
            subset[j] = points[selected[j]]

        for i in range(num_points):
            if is_selected[i]:
                continue

            subset[k] = points[i]
            current_vesa = volume_estimation_by_shadow_addition(subset, subset_mask)

            if current_vesa > best_vesa:
                best_vesa = current_vesa
                best_idx = i

        selected[k] = best_idx
        is_selected[best_idx] = True

    return selected
