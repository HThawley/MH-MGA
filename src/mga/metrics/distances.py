import numpy as np
from mga.commons.numba_overload import njit
from numba.types import UniTuple


@njit(fastmath=True, inline="always")
def _minkowski_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    power: float,
    inv_power: float,
) -> float:
    """Calculates L1/L2 spatial distance between two vectors."""
    dist = 0.0
    for k in range(len(p1)):
        dist += abs(p1[k] - p2[k]) ** power
    return dist ** inv_power


@njit(fastmath=True, inline="always")
def _accumulate_product_and_norms(
    p: np.ndarray,
    c: np.ndarray,
    anchor: np.ndarray,
    dot_product: float,
    norm_p_sq: float,
    norm_c_sq: float,
) -> UniTuple:
    v_p = p - anchor
    v_c = c - anchor
    dot_product += v_p * v_c
    norm_p_sq += v_p * v_p
    norm_c_sq += v_c * v_c
    return dot_product, norm_p_sq, norm_c_sq


@njit(fastmath=True, inline="always")
def _get_vector_components(
    p: np.ndarray,
    c: np.ndarray,
    anchor: np.ndarray,
) -> UniTuple:
    dot_product = 0.0
    norm_p_sq = 0.0
    norm_c_sq = 0.0
    for k in range(len(p)):
        dot_product, norm_p_sq, norm_c_sq = _accumulate_product_and_norms(
           p[k], c[k], anchor[k], dot_product, norm_p_sq, norm_c_sq
        )
    return dot_product, norm_p_sq, norm_c_sq


@njit(fastmath=True, inline="always")
def _calc_similarity(
    dot_product: float,
    norm_p_sq: float,
    norm_c_sq: float,
) -> float:
    if norm_p_sq == 0.0 or norm_c_sq == 0.0:
        return 0.0
    return 1.0 - (dot_product / ((norm_p_sq * norm_c_sq) ** 0.5))


@njit(fastmath=True, inline="always")
def _angular_separation(
    p: np.ndarray,
    c: np.ndarray,
    anchor: np.ndarray,
) -> float:
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
