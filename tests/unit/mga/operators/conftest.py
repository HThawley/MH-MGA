import pytest
import numpy as np 

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
def point1():
    yield np.array([0.10, 0.30, 0.50, 0.70, 0.90])

@pytest.fixture
def point2():
    yield np.array([0.15, 0.35, 0.55, 0.75, 0.95])

@pytest.fixture
def large_population():
    yield np.arange(20000).reshape(1, 10000, 2)

@pytest.fixture 
def integrality():
    yield np.array([False, True, True])

@pytest.fixture
def bool_mask():
    yield np.array([False, False, True])

@pytest.fixture
def mixed_dtype_points(rng):

    yield np.dstack((
        rng.uniform(0, 10, 10000), 
        rng.integers(0, 100, 10000),
        rng.uniform(0, 1, 10000) > 0.75
    ))


