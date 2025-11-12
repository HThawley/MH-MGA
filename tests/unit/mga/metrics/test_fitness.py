import pytest
import numpy as np

from mga.commons.types import DEFAULTS

FLOAT, INT = DEFAULTS
from mga.metrics import fitness as ft


def test_evaluate_fitness_dist_to_centroids():
    """Tests the calculation of fitness based on distance to the nearest other centroid."""
    # This test directly checks the logic of the fitness function with 3 niches
    # to ensure the "nearest other centroid" logic is working correctly.

    # 3 niches, 2 individuals per niche, 2 dimensions
    points = np.array(
        [[[1.0, 0.0], [2.0, 0.0]], [[9.0, 0.0], [8.0, 0.0]], [[5.0, 9.0], [5.0, 8.0]]],  # Niche 0  # Niche 1  # Niche 2
        dtype=FLOAT,
    )

    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]], dtype=FLOAT)  # Centroid 0  # Centroid 1  # Centroid 2

    # Pre-calculated expected fitness values
    expected_fitness = np.array(
        [
            [9.0, 8.0],  # Niche 0 -> min dist to C1 or C2 is C1
            [9.0, 8.0],  # Niche 1 -> min dist to C0 or C2 is C0
            [np.sqrt(106), np.sqrt(89)],  # Niche 2 -> min dist to C0 or C1 (equidistant)
        ],
        dtype=FLOAT,
    )

    # Array to store the results
    actual_fitness = np.empty_like(expected_fitness)

    # Call the function to test
    ft.evaluate_fitness_dist_to_centroids(actual_fitness, points, centroids)

    # Assert that the calculated fitness is close to the expected values
    assert np.allclose(
        actual_fitness, expected_fitness
    ), f"Fitness calculation is incorrect. Expected: \n{expected_fitness}\n Got: \n{actual_fitness}"
