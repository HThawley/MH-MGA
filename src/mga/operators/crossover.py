from mga.commons.numba_overload import njit


# API functions
@njit(cache=True)
def crossover_population(points, crossover_prob, cx_func, rng, start_idx=0):
    """
    Applies crossover to an entire population, niche by niche.
    """
    for i in range(points.shape[0]):
        # Shuffle parents for mating
        _crossover_niche(points[i], crossover_prob, cx_func, rng, start_idx)


@njit(inline='always')
def cx_one_point(ind1, ind2, rng):
    """
    Single point crossover
    """
    # index1=0 leads to total swap instead of cx
    index1 = rng.integers(1, len(ind1))
    _do_cx(ind1, ind2, index1, len(ind1))


@njit(inline='always')
def cx_two_point(ind1, ind2, rng):
    """
    Double point crossover
    """
    # choose 2 with no replacement
    index1 = rng.integers(0, len(ind1)+1)
    index2 = rng.integers(0, len(ind1))
    if index2 >= index1:
        index2 += 1
    else:  # index2 < index1
        index1, index2 = index2, index1

    _do_cx(ind1, ind2, index1, index2)


# private helper functions
@njit
def _crossover_niche(niche, crossover_prob, cx_func, rng, start_idx=0):
    """
    Applies crossover to a niche
    """
    # Shuffle parents for mating
    rng.shuffle(niche[start_idx:])

    for ind1, ind2 in zip(niche[start_idx:][::2], niche[start_idx:][1::2]):
        if rng.random() < crossover_prob:
            cx_func(ind1, ind2, rng)


@njit(inline='always')
def _do_cx(ind1, ind2, index1, index2):
    """
    Perform crossover
    """
    buffer = 0.0
    # modify in place
    for i in range(index1, index2):
        buffer = ind1[i]
        ind1[i] = ind2[i]
        ind2[i] = buffer
