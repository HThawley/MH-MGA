import pytest
import numpy as np

import mga.metrics.diversity as dv


# fixtures
@pytest.fixture
def sample_points_data():
    """Provides a sample set of points, feasibility, and bounds."""
    points = np.array(
        [
            [0.1, 0.2, 0.9],
            [0.8, 0.8, 0.1],
            [0.1, 0.8, 0.2],
            [0.9, 0.1, 0.7],
            [0.5, 0.5, 0.5],  # Infeasible point
            [0.3, 0.2, 0.9],
            [0.6, 0.8, 0.4],
            [0.1, 0.5, 0.2],
            [0.9, 0.1, 0.7],
            [0.8, 0.6, 0.7],
        ]
    )
    feasibility = np.array([True, True, True, True, False, True, True, True, True, True])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    return points, feasibility, lb, ub


@pytest.fixture
def sample_fitness_data():
    """Provides sample fitness and noptimality data."""
    fitness = np.array([[1.0, 2.0, 0.5], [3.0, 1.5, 2.5], [0.0, 0.0, 5.0]])
    # noptimality marks which individuals are considered 'optimal' or feasible
    noptimality = np.array([[True, True, False], [True, False, True], [False, False, True]])
    return fitness, noptimality


# --- Tests for API Functions ---


def test_mean_of_shannon_of_projections(sample_points_data):
    """
    Tests the shannon projection function.
    Note: This is a basic test to ensure it runs and produces a plausible value.
    The exact value is complex to calculate by hand.
    """
    points, feasibility, lb, ub = sample_points_data
    result = dv.mean_of_shannon_of_projections(points, feasibility, lb, ub)

    assert isinstance(result, float)
    assert (
        0.0 <= result <= 1.0
    ), "Shannon index should be normalized between 0 and 1. (If a bit above 1, could be MM correction)"


def test_sum_of_fitness(sample_fitness_data):
    """
    Tests the sum_of_fitness function by summing the max fitness for each population.
    """
    fitness, noptimality = sample_fitness_data
    # Expected:
    # Pop 0: max of [1.0, 2.0] -> 2.0
    # Pop 1: max of [3.0, 2.5] -> 3.0
    # Pop 2: max of [5.0]      -> 5.0
    # Total sum: 2.0 + 3.0 + 5.0 = 10.0
    expected_sum = 10.0
    result = dv.sum_of_fitness(fitness, noptimality)
    assert np.isclose(result, expected_sum), f"Expected sum of fitness {expected_sum}, but got {result}"


def test_mean_of_fitness(sample_fitness_data):
    """
    Tests the mean_of_fitness function.
    """
    fitness, noptimality = sample_fitness_data
    # Expected sum is 10.0 (from test_sum_of_fitness)
    # Number of populations is 3
    # Expected mean: 10.0 / 3
    expected_mean = 10.0 / 3.0
    result = dv.mean_of_fitness(fitness, noptimality)
    assert np.isclose(result, expected_mean), f"Expected mean fitness {expected_mean}, but got {result}"


def test_volume_estimation_by_shadow_addition():
    """
    Tests VESA with a simple 3D shape (a cube).
    The projections onto 2D planes should be squares.
    """
    # Points of a unit cube, plus one internal point
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0.5, 0.5, 0.5],  # internal point
        ]
    )
    feasibility = np.array([True] * points.shape[0])

    # Projections:
    # 1. XY plane: convex hull of points is a 1x1 square, area = 1.0
    # 2. XZ plane: convex hull of points is a 1x1 square, area = 1.0
    # 3. YZ plane: convex hull of points is a 1x1 square, area = 1.0
    # Expected VESA is the sum of these areas: 1.0 + 1.0 + 1.0 = 3.0
    expected_vesa = 3.0
    result = dv.volume_estimation_by_shadow_addition(points, feasibility)
    assert np.isclose(result, expected_vesa), f"Expected VESA {expected_vesa}, but got {result}"


# tests
def test_shannon_index():
    """
    Tests the _shannon_index helper function.
    """
    nbin = 4
    npoint = 8
    lb, ub = 0.0, 4.0
    counts = np.zeros(nbin, dtype=np.int64)

    # Case 1: Uniform distribution, should have max entropy
    values = np.array([0.5, 0.6, 1.5, 1.6, 2.5, 2.6, 3.5, 3.6])  # 2 points per bin
    # p = 2/8 = 0.25 for each bin
    # H = - (4 * (0.25 * log(0.25))) = -log(0.25) = log(4)
    expected_H = np.log(4)
    # With Miller-Madow correction
    expected_H += (nbin - 1) / (2.0 * npoint)
    result = dv._shannon_index(values, lb, ub, nbin, npoint, counts)
    assert np.isclose(result, expected_H), f"Uniform dist Shannon index expected {expected_H}, but got {result}"

    # Case 2: All points in one bin, should have zero entropy
    counts[:] = 0  # Reset counts
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # All in bin 0
    # H = -(1 * log(1)) = 0
    # With Miller-Madow correction
    expected_H = (1 - 1) / (2.0 * npoint)  # (nonzero_bins - 1) / (2N)
    result = dv._shannon_index(values, lb, ub, nbin, npoint, counts)
    assert np.isclose(result, expected_H), f"Single bin Shannon index expected {expected_H}, but got {result}"


def test_convex_hull_area():
    """
    Tests the _convex_hull_area helper.
    """
    # Case 1: A simple square
    points_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    result = dv._convex_hull_area(points_square)
    expected = 1.0
    assert np.isclose(result, expected), f"Square area expected {expected}, but got {result}"

    # Case 2: A triangle
    points_triangle = np.array([[0, 0], [2, 0], [1, 2]])
    # Area = 0.5 * base * height = 0.5 * 2 * 2 = 2.0
    result = dv._convex_hull_area(points_triangle)
    expected = 2.0
    assert np.isclose(result, expected), f"Triangle area expected {expected}, but got {result}"

    # Case 3: Collinear points
    points_collinear = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    result = dv._convex_hull_area(points_collinear)
    expected = 0.0
    assert np.isclose(result, expected), f"Collinear points area expected {expected}, but got {result}"

    # Case 4: Points forming a more complex shape (L-shape)
    points_l_shape = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
    # Convex hull should be the 3 1x1 squares + the right triangle in the upper rigth quadrant
    # plt.plot([0, 2, 2, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 0]) # L-shape
    # plt.plot([0, 2, 2, 1, 0, 0], [0, 0, 1, 2, 2, 0]) # convex hull

    result = dv._convex_hull_area(points_l_shape)
    expected = 3.5
    assert np.isclose(result, expected), f"L-shape convex hull area expected {expected}, but got {result}"


def test_cross_product():
    """
    Tests the 2D cross product helper.
    """
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3_left = np.array([1, 1])  # Left turn
    p3_right = np.array([1, -1])  # Right turn
    p3_collinear = np.array([2, 0])  # Collinear

    result = dv._cross_product(p1, p2, p3_left)
    assert result > 0, f"_cross_product(...) has wrong sign. Expected > 0, got {result}"
    result = dv._cross_product(p1, p2, p3_right)
    assert result < 0, f"_cross_product(...) has wrong sign. Expected < 0, got {result}"
    result = dv._cross_product(p1, p2, p3_collinear)
    assert np.isclose(result, 0), f"_cross_product(...) has wrong sign. Expected == 0, got {result}"


def test_shoelace_area():
    """
    Tests the shoelace formula implementation.
    """
    # A 2x2 square, ordered counter-clockwise
    points = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    num_points = 4
    result = dv._shoelace_area(points, num_points)
    expected = 4.0
    assert np.isclose(result, expected), f"Square shoelace area expected {expected}, but got {result}"

    # A right triangle
    points_tri = np.array([[0, 0], [3, 0], [0, 4]])
    num_points_tri = 3
    # Area = 0.5 * 3 * 4 = 6.0
    result = dv._shoelace_area(points_tri, num_points_tri)
    expected = 6.0
    assert np.isclose(result, expected), f"Triangle shoelace area expected {expected}, but got {result}"
