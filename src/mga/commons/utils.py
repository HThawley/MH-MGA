import numpy as np

from mga.commons.numba_overload import njit


@njit(inline='always')
def argmax(array):
    best_i = -1
    best = -np.inf
    for i in range(array.shape[0]):
        val = array[i]
        if val > best:
            best_i = i
            best = val
    return best_i


@njit(inline='always')
def argmin(array):
    best_i = -1
    best = np.inf
    for i in range(array.shape[0]):
        val = array[i]
        if val < best:
            best_i = i
            best = val
    return best_i


@njit
def argm(array, maximize):
    if maximize:
        return argmax(array)
    else:
        return argmin(array)


@njit(inline='always')
def argmax_2d(array):
    best_i = -1
    best_j = -1
    best = -np.inf
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            val = array[i, j]
            if val > best:
                best_i = i
                best_j = j
                best = val
    return best_i, best_j


@njit(inline='always')
def argmin_2d(array):
    best_i = -1
    best_j = -1
    best = np.inf
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            val = array[i, j]
            if val < best:
                best_i = i
                best_j = j
                best = val
    return best_i, best_j


@njit
def argm_2d(array, maximize):
    if maximize:
        return argmax_2d(array)
    else:
        return argmin_2d(array)


@njit(inline='always')
def argmax_with_mask(array, mask, best=-np.inf):
    best_i = -1
    for i in range(array.shape[0]):
        if mask[i]:
            val = array[i]
            if val > best:
                best_i = i
                best = val
    return best_i


@njit(inline='always')
def argmin_with_mask(array, mask, best=np.inf):
    best_i = -1
    for i in range(array.shape[0]):
        if mask[i]:
            val = array[i]
            if val < best:
                best_i = i
                best = val
    return best_i


@njit
def argm_with_mask(array, mask, best, maximize):
    if maximize:
        return argmax_with_mask(array, mask, best)
    else:
        return argmin_with_mask(array, mask, best)


@njit(inline='always')
def argmax_with_indices(array, indices, best=-np.inf):
    best_i = -1
    for idx in indices:
        val = array[idx]
        if val > best:
            best_i = idx
            best = val
    return best_i


@njit(inline='always')
def argmin_with_indices(array, indices, best=np.inf):
    best_i = -1
    for idx in indices:
        val = array[idx]
        if val < best:
            best_i = idx
            best = val
    return best_i


@njit
def argm_with_indices(array, indices, maximize):
    if maximize:
        return argmax_with_indices(array, indices)
    else:
        return argmin_with_indices(array, indices)


@njit(inline='always')
def argmax_with_indices_mask(array, indices, mask, best=-np.inf):
    best_i = -1
    for idx in indices:
        if mask[idx]:
            val = array[idx]
            if val > best:
                best_i = idx
                best = val
    return best_i


@njit(inline='always')
def argmin_with_indices_mask(array, indices, mask, best=np.inf):
    best_i = -1
    for idx in indices:
        if mask[idx]:
            val = array[idx]
            if val < best:
                best_i = idx
                best = val
    return best_i


@njit
def argm_with_indices_mask(array, indices, mask, maximize):
    if maximize:
        return argmax_with_indices_mask(array, indices, mask)
    else:
        return argmin_with_indices_mask(array, indices, mask)


@njit(inline='always')
def argmax_with_mask_2d(array, mask, best=-np.inf):
    best_i = -1
    best_j = -1
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if mask[i, j]:
                val = array[i, j]
                if val > best:
                    best_i = i
                    best_j = j
                    best = val
    return best_i, best_j


@njit(inline='always')
def argmin_with_mask_2d(array, mask, best=np.inf):
    best_i = -1
    best_j = -1
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if mask[i, j]:
                val = array[i, j]
                if val < best:
                    best_i = i
                    best_j = j
                    best = val
    return best_i, best_j


@njit
def argm_with_mask_2d(array, mask, best, maximize):
    if maximize:
        return argmax_with_mask_2d(array, mask, best)
    else:
        return argmin_with_mask_2d(array, mask, best)


@njit(inline='always')
def loguniform_dither(logbounds, rng):
    return np.exp(rng.uniform(logbounds[0], logbounds[1]))


@njit(inline='always')
def uniform_dither(bounds, rng):
    return rng.uniform(bounds[0], bounds[1])


@njit(inline='always')
def safe_divide_scalar(
    num,
    denom,
    fail=0.0,
):
    """ Zero-safe division of two scalars """
    if denom == 0.0:
        return fail
    return num / denom


@njit(inline='always')
def safe_divide_array(
    num,
    denom,
    fail=0.0,
):
    """ Zero-safe division of two arrays. """
    retarr = num.copy().ravel()
    denom_ravel = denom.ravel()
    for i in range(retarr.size):
        if denom_ravel[i] == 0.0:
            retarr[i] = fail
        else:
            retarr[i] /= denom_ravel[i]
    return retarr.reshape(num.shape)
