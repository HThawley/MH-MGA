import pytest
import numpy as np

from mga.operators import crossover as cx

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
def mock_rng():
    class rng_:
        def __init__(self):
            pass
        def random(self):
            return 0.75
    yield rng_()

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

def test_cx_one_point(point1, point2, mock_rng):
    # need the integers method in the mock rng to test cx_one_point
    assert mock_rng.random() == 0.75
