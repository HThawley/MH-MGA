from numba import njit


# API functions
@njit
def mutate_gaussian_population_mixed(points, sigma, indpb, rng, integrality, booleanality, startidx=0):
    """
    Mutate individuals in a population
    Compatible with mixed dtypes
    """
    for i in range(points.shape[0]):
        for j in range(startidx, points.shape[1]):
            for k in range(points.shape[2]):
                if rng.random() < indpb:
                    if booleanality[k]:
                        points[i, j, k] = _mutate_bool(points[i, j, k], sigma[k], rng)
                    elif integrality[k]:
                        points[i, j, k] = _mutate_int(points[i, j, k], sigma[k], rng)
                    else:
                        points[i, j, k] = _mutate_float(points[i, j, k], sigma[k], rng)


@njit
def mutate_gaussian_population_float(points, sigma, indpb, rng, startidx=0):
    """
    Mutate individuals in a population
    Compatible only with float-only
    """
    for i in range(points.shape[0]):
        for j in range(startidx, points.shape[1]):
            for k in range(points.shape[2]):
                if rng.random() < indpb:
                    points[i, j, k] = _mutate_float(points[i, j, k], sigma[k], rng)


@njit
def mutate_gaussian_niche_mixed(niche, sigma, indpb, rng, integrality, booleanality, startidx=0):
    """
    Mutate individuals in a niche
    Compatible with mixed dtypes
    """
    for j in range(startidx, niche.shape[0]):
        for k in range(niche.shape[1]):
            if rng.random() < indpb:
                if booleanality[k]:
                    niche[j, k] = _mutate_bool(niche[j, k], sigma[k], rng)
                elif integrality[k]:
                    niche[j, k] = _mutate_int(niche[j, k], sigma[k], rng)
                else:
                    niche[j, k] = _mutate_float(niche[j, k], sigma[k], rng)


@njit
def mutate_gaussian_niche_float(niche, sigma, indpb, rng, startidx=0):
    """
    Mutate individuals in a niche
    Compatible only with float-only
    """
    for j in range(startidx, niche.shape[0]):
        for k in range(niche.shape[1]):
            if rng.random() < indpb:
                niche[j, k] = _mutate_float(niche[j, k], sigma[k], rng)


# private helper functions


@njit
def _mutate_float(item, sigma, rng):
    """gaussian mutation for single variable"""
    return rng.normal(item, sigma)


@njit
def _mutate_int(item, sigma, rng):
    """Integer mutation for single variable. Returns integer-valued float"""
    return round(rng.normal(item, sigma))


@njit
def _mutate_bool(item, sigma, rng):
    """Boolean mutation for single variable. Returns boolean-valued float"""
    if abs(rng.normal(0, sigma)) <= rng.random():
        return item
    else:
        return 1.0 - item
