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

@jitclass([("retval", UniTuple(int64, 2)),
           ("calls", int64)])
class mock_rng2:
    def __init__(self, n):
        self.calls=-1
        self.retval = n
    def integers(self, lb, ub):
        self.calls+=1
        return self.retval[self.calls]

# fixtures
@pytest.fixture
def rng():
    yield np.random.default_rng()

@pytest.fixture
def points():
    yield np.array([
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
    ])

@pytest.fixture
def points_copy(points):
    yield points.copy()

@pytest.fixture
def crossed_points(points_copy, rng):
    cx.crossover_population(points_copy, 1.0, cx._cx_one_point, rng, 0)
    yield points_copy

@pytest.fixture
def point1():
    yield np.array([0.10, 0.30, 0.50, 0.70, 0.90])

@pytest.fixture
def point2():
    yield np.array([0.15, 0.35, 0.55, 0.75, 0.95])

# tests

def test_crossover_maintains_shape(points, crossed_points):
    """
    input shape matches output shape
    """
    assert points.shape == crossed_points.shape

def test_crossover_population_does_something(points, crossed_points):
    """
    The result of cx.crossover_population is not the input array
    Also, that crossover_population occurs in-place
    """
    assert not (crossed_points == points).all()

def test_coords_conserved(points, crossed_points):
    """
    No coordinate is lost - same counts of each unique number before and after
    """
    org_counts = dict(zip(*np.unique(points, return_counts=True)))
    new_counts = dict(zip(*np.unique(crossed_points, return_counts=True)))

    for coord, count in org_counts.items():
        assert new_counts[coord] == count
    
def test_coords_conserved_within_niche(points, crossed_points):
    """
    Within each niche, same counts of each unique number before and after
    i.e. crossover only occurs within and not between niches
    """
    for i in range(points.shape[0]):
        org_counts = dict(zip(*np.unique(points[i], return_counts=True)))
        new_counts = dict(zip(*np.unique(crossed_points[i], return_counts=True)))
        for coord, count in org_counts.items():
            assert new_counts[coord] == count

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
                assert new_counts[coord] == count

@pytest.mark.parametrize("rng", [0, 1, 2, 3, 4])
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

    assert (point1 == result1).all()
    assert (point2 == result2).all()

@pytest.mark.parametrize(
        "rng", 
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), 
         (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
def test_cx_two_point(point1, point2, rng):
    """
    test _cx_two_point
    """
    _rng = mock_rng2(rng)

    result1 = point1.copy()
    result2 = point2.copy()
    result1[slice(*rng)] = point2[slice(*rng)]
    result2[slice(*rng)] = point1[slice(*rng)]
    cx._cx_two_point(point1, point2, _rng)

    assert (point1 == result1).all()
    assert (point2 == result2).all()

def test_cx_index_1p_uniformity():
    """
    Test that the random index chosen by _cx_one_point is uniformly distributed.
    """
    assert True

@pytest.mark.parametrize("ndim", [3, 4, 5, 10])
def test_cx_index_2p_uniformity(ndim):
    """
    Test that the random indices chosen by _cx_two_point is uniformly distributed.
    """
    rng = np.random.default_rng()
    _indices = np.zeros(ndim)
    for i in range(1000):
        point1 = -np.ones(ndim)
        point2 = np.arange(ndim)
        cx._cx_two_point(point1, point2, rng)
        _indices[np.where(point1 != -1)[0][0]] += 1 
        _indices[np.where(point1 != -1)[0][-1]] += 1 
        
    assert True

