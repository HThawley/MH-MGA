from mga.commons.numba_overload import njit


# API functions
@njit(cache=True)
def mutate_gaussian_population_mixed(points, mutation_sigma, mutation_prob, rng, integrality, booleanality, startidx=0):
    """
    Mutate individuals in a population
    Compatible with mixed dtypes
    """
    for i in range(points.shape[0]):
        _mutate_gaussian_niche_mixed(points[i], mutation_sigma, mutation_prob, rng, integrality, booleanality, startidx)


@njit(cache=True)
def mutate_gaussian_population_float(points, mutation_sigma, mutation_prob, rng, startidx=0):
    """
    Mutate individuals in a population
    Compatible only with float-only
    """
    for i in range(points.shape[0]):
        _mutate_gaussian_niche_float(points[i], mutation_sigma, mutation_prob, rng, startidx)


@njit(cache=True)
def mutate_skew_population_mixed(
    points, mutation_sigma, mutation_prob, mutation_alpha, rng, integrality, booleanality, startidx=0
):
    """
    Mutate individuals in a population
    Compatible with mixed dtypes
    """
    for i in range(points.shape[0]):
        _mutate_skew_niche_mixed(
            points[i], mutation_sigma, mutation_prob, mutation_alpha, rng, integrality, booleanality, startidx
        )


@njit(cache=True)
def mutate_skew_population_float(points, mutation_sigma, mutation_prob, mutation_alpha, rng, startidx=0):
    """
    Mutate individuals in a population
    Compatible only with float-only
    """
    for i in range(points.shape[0]):
        _mutate_skew_niche_float(points[i], mutation_sigma, mutation_prob, mutation_alpha, rng, startidx)


# private helper functions
@njit
def _mutate_gaussian_niche_mixed(niche, mutation_sigma, mutation_prob, rng, integrality, booleanality, startidx=0):
    """
    Mutate individuals in a niche
    Compatible with mixed dtypes
    """
    for j in range(startidx, niche.shape[0]):
        for k in range(niche.shape[1]):
            if booleanality[k]:
                if rng.random() < mutation_prob:
                    niche[j, k] = _mutate_bool(niche[j, k], mutation_sigma[k], rng)
            elif integrality[k]:
                if rng.random() < mutation_prob:
                    niche[j, k] = _mutate_int_normal(niche[j, k], mutation_sigma[k], rng)
            else:
                if rng.random() < mutation_prob:
                    niche[j, k] = _mutate_float_normal(niche[j, k], mutation_sigma[k], rng)


@njit
def _mutate_gaussian_niche_float(niche, mutation_sigma, mutation_prob, rng, startidx=0):
    """
    Mutate individuals in a niche
    Compatible only with float-only
    """
    for j in range(startidx, niche.shape[0]):
        for k in range(niche.shape[1]):
            if rng.random() < mutation_prob:
                niche[j, k] = _mutate_float_normal(niche[j, k], mutation_sigma[k], rng)


@njit
def _mutate_skew_niche_mixed(
    niche, mutation_sigma, mutation_prob, mutation_alpha, rng, integrality, booleanality, startidx=0
):
    """
    Mutate individuals in a niche
    Compatible with mixed dtypes
    """
    for j in range(startidx, niche.shape[0]):
        for k in range(niche.shape[1]):
            if booleanality[k]:
                if rng.random() < mutation_prob:
                    niche[j, k] = _mutate_bool(niche[j, k], mutation_sigma[k], rng)
            elif integrality[k]:
                if rng.random() < mutation_prob:
                    niche[j, k] = _mutate_int_skew(niche[j, k], mutation_sigma[k], mutation_alpha, rng)
            else:
                if rng.random() < mutation_prob:
                    niche[j, k] = _mutate_float_skew(niche[j, k], mutation_sigma[k], mutation_alpha, rng)


@njit
def _mutate_skew_niche_float(niche, mutation_sigma, mutation_prob, mutation_alpha, rng, startidx=0):
    """
    Mutate individuals in a niche
    Compatible only with float-only
    """
    for j in range(startidx, niche.shape[0]):
        for k in range(niche.shape[1]):
            if rng.random() < mutation_prob:
                niche[j, k] = _mutate_float_skew(niche[j, k], mutation_sigma[k], mutation_alpha, rng)


@njit(inline='always')
def _mutate_float_normal(item, mutation_sigma, rng):
    """gaussian mutation for single variable"""
    return rng.normal(item, mutation_sigma)


@njit(inline='always')
def _mutate_int_normal(item, mutation_sigma, rng):
    """Integer mutation for single variable. Returns integer-valued float"""
    return round(_mutate_float_normal(item, mutation_sigma, rng))


@njit(inline='always')
def _mutate_bool(item, mutation_sigma, rng):
    """Boolean mutation for single variable. Returns boolean-valued float"""
    if abs(rng.normal(0, mutation_sigma)) <= rng.random():
        return item
    else:
        return 1.0 - item


@njit(inline='always')
def _mutate_float_skew(item, mutation_sigma, alpha, rng):
    """
    Skew-normal mutation for a single variable using the Azzalini method.
    alpha > 0 skews right, alpha < 0 skews left, alpha = 0 is normal.
    """
    u1 = rng.normal(0.0, 1.0)
    u2 = rng.normal(0.0, 1.0)

    if u2 < alpha * u1:
        z = u1
    else:
        z = -u1

    return item + mutation_sigma * z


@njit(inline='always')
def _mutate_int_skew(item, mutation_sigma, alpha, rng):
    """Skew-normal Integer mutation for single variable. Returns integer-valued float"""
    return round(_mutate_float_skew(item, mutation_sigma, alpha, rng))
