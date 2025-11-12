import pytest
import numpy as np
from numba import njit, int64
from numba.types import UniTuple
from numba.experimental import jitclass

from mga.operators import crossover as cx


@jitclass([("retval", int64)])
class mock_rng1:
    def __init__(self, n):
        self.retval = n

    def integers(self, lb, ub):
        return self.retval


@jitclass([("retval", UniTuple(int64, 2)), ("calls", int64)])
class mock_rng2:
    def __init__(self, n):
        self.calls = -1
        self.retval = n

    def integers(self, lb, ub):
        self.calls += 1
        return self.retval[self.calls]


# fixtures


@pytest.fixture
def point1():
    yield np.array([0.10, 0.30, 0.50, 0.70, 0.90])


@pytest.fixture
def point2():
    yield np.array([0.15, 0.35, 0.55, 0.75, 0.95])


@pytest.fixture
def large_population():
    yield np.arange(20000).reshape(1, 10000, 2)


@pytest.fixture
def points():
    yield np.array(
        [
            [
                [0.15, 0.40, 0.65],
                [0.20, 0.45, 0.70],
                [0.25, 0.50, 0.75],
                [0.30, 0.55, 0.80],
                [0.35, 0.60, 0.85],
            ],
            [
                [0.65, 0.15, 0.40],
                [0.70, 0.20, 0.45],
                [0.75, 0.25, 0.50],
                [0.80, 0.30, 0.55],
                [0.85, 0.35, 0.60],
            ],
            [
                [0.40, 0.65, 0.15],
                [0.45, 0.70, 0.20],
                [0.50, 0.75, 0.25],
                [0.55, 0.80, 0.30],
                [0.60, 0.85, 0.35],
            ],
        ]
    )


@pytest.fixture
def points_copy(points):
    yield points.copy()


@pytest.fixture
def crossed_points(points_copy, rng):
    cx.crossover_population(points_copy, 1.0, cx._cx_one_point, rng, 0)
    yield points_copy


# tests


def test_crossover_maintains_shape(points, crossed_points):
    """
    input shape matches output shape
    """
    assert points.shape == crossed_points.shape, "crossover returns incorrect shape"


def test_crossover_population_does_something(points, crossed_points):
    """
    The result of cx.crossover_population is not the input array
    Also, that crossover_population occurs in-place
    """
    assert not (crossed_points == points).all(), "crossover returns input"


def test_coords_conserved(points, crossed_points):
    """
    No coordinate is lost - same counts of each unique number before and after
    """
    org_counts = dict(zip(*np.unique(points, return_counts=True)))
    new_counts = dict(zip(*np.unique(crossed_points, return_counts=True)))

    for coord, count in org_counts.items():
        assert new_counts[coord] == count, "coordinates not conserved within pop during cx"


def test_coords_conserved_within_niche(points, crossed_points):
    """
    Within each niche, same counts of each unique number before and after
    i.e. crossover only occurs within and not between niches
    """
    for i in range(points.shape[0]):
        org_counts = dict(zip(*np.unique(points[i], return_counts=True)))
        new_counts = dict(zip(*np.unique(crossed_points[i], return_counts=True)))
        for coord, count in org_counts.items():
            assert new_counts[coord] == count, "coordinates not conserved within niche during cx"


def test_column_count(points, crossed_points):
    """
    Within each niche, the counts of unique numbers in each column should be constant
    before and after crossover
    i.e. crossover is not mixing coords between dimensions
    """
    for i in range(points.shape[0]):
        for k in range(points.shape[2]):
            org_counts = dict(zip(*np.unique(points[i, :, k], return_counts=True)))
            new_counts = dict(zip(*np.unique(crossed_points[i, :, k], return_counts=True)))
            for coord, count in org_counts.items():
                assert new_counts[coord] == count, "coordinates not conserved within columns and niche during cx"


@pytest.mark.parametrize("cx_prob", [0.0, 0.25, 0.5, 1.0])
def test_cx_prob(large_population, rng, cx_prob):
    """
    Test that approximately {cx_prob} of population are cx'ed
    Tested on `_cx_one_point`. No need to test of `_cx_two_point` since
        the functionality is in `crossover_population` not `_cx_n_point`.
        Also it is harder to test on two point.
    """
    lp = large_population.copy()
    cx.crossover_population(lp, cx_prob, cx._cx_one_point, rng, 0)
    assert (lp[:, :, 0] % 2 == 0).all(), "check column count test"
    assert (lp[:, :, 1] % 2 == 1).all(), "check column count test"
    lp = lp[:, lp[0, :, 0].argsort(), :]
    cx_rate = 1 - ((lp == large_population).all(axis=(0, 2)).sum() / large_population.shape[1])
    if cx_prob == 0.0:
        assert cx_rate == 0, "cx_prob not applied correctly"
    else:
        # 5 % point difference
        assert abs(1 - cx_rate / cx_prob) < 0.05, "cx_prob not applied correctly"


@pytest.mark.parametrize("rng", [1, 2, 3, 4])
def test_cx_one_point(point1, point2, rng):
    """
    test _cx_one_point
    """
    _rng = mock_rng1(rng)

    result1 = point1.copy()
    result2 = point2.copy()
    result1[rng:] = point2[rng:]
    result2[rng:] = point1[rng:]
    cx._cx_one_point(point1, point2, _rng)

    assert (point1 == result1).all(), "_cx_one_point not cx-ing correctly"
    assert (point2 == result2).all(), "_cx_one_point not cx-ing correctly"


@pytest.mark.parametrize("rng", [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
def test_cx_two_point(point1, point2, rng):
    """
    test _cx_two_point
    """
    _rng = mock_rng2(rng)

    result1 = point1.copy()
    result2 = point2.copy()
    result1[rng[0] : rng[1] + 1] = point2[rng[0] : rng[1] + 1]  # rng indices are inclusive
    result2[rng[0] : rng[1] + 1] = point1[rng[0] : rng[1] + 1]
    cx._cx_two_point(point1, point2, _rng)

    assert (point1 == result1).all(), "_cx_two_point not cx-ing correctly"
    assert (point2 == result2).all(), "_cx_two_point not cx-ing correctly"


@pytest.mark.parametrize("ndim", [2, 3, 4, 10])
def test_cx_1p_index_uniformity(ndim):
    """
    Test that the random indices chosen by _cx_one_point are uniformly distributed
    """
    rng = np.random.default_rng()
    _indices = np.zeros(ndim)
    n_samples = 10000
    for i in range(n_samples):
        point1 = -np.ones(ndim)
        point2 = np.arange(ndim)
        cx._cx_one_point(point1, point2, rng)
        _indices[np.where(point1 != -1)[0][0]] += 1

        p1_cx_mask = point1 != -1
        p2_cx_mask = point2 == -1
        assert (p1_cx_mask == p2_cx_mask).all(), "cx 1p not symmetric"
        # crossover occurs
        assert (p2_cx_mask).sum() > 0, "cx did not occur (check index1=0?)"
        # crossover mechanics tested elsewhere

    # uniform sampling
    # with one point this is total swap not cx
    assert _indices[0] == 0, "cx 1-point should not start at index 0"
    _indices = _indices[1:]

    mean_samples = n_samples / (ndim - 1)
    ten_percent = mean_samples * 0.1
    assert _indices.max() < mean_samples + ten_percent, "cx 1p oversampling some indices"
    assert _indices.min() > mean_samples - ten_percent, "cx 1p undersampling some indices"


@pytest.mark.parametrize("ndim", [3, 4, 5, 10])
def test_cx_2p_index_uniformity(ndim):
    """
    Test that the random indices chosen by _cx_two_point are uniformly distributed
    """
    rng = np.random.default_rng()
    _indices = np.zeros(ndim)
    n_samples = 10000
    for i in range(n_samples):
        point1 = -np.ones(ndim)
        point2 = np.arange(ndim)
        cx._cx_two_point(point1, point2, rng)
        _indices[np.where(point1 != -1)[0][0]] += 1
        _indices[np.where(point1 != -1)[0][-1]] += 1

        p1_cx_mask = point1 != -1
        p2_cx_mask = point2 == -1
        assert (p1_cx_mask == p2_cx_mask).all(), "cx 2p not symmetric"
        # crossover occurs
        assert (p2_cx_mask).sum() > 0, "cx did not occur"
        # crossover mechanics tested elsewhere

    mean_samples = 2 * n_samples / ndim
    ten_percent = mean_samples * 0.1
    assert _indices.max() < mean_samples + ten_percent, "cx 2p oversampling some indices"
    assert _indices.min() > mean_samples - ten_percent, "cx 2p undersampling some indices"
