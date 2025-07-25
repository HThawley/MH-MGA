# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:12:23 2025

@author: u6942852
"""

import numpy as np 
from numba import njit

@njit
def _cross_product(p1, p2, p3):
    """
    Calculates the 2D cross product (z-component) of vectors p1p2 and p1p3.
    This determines the orientation of the triplet (p1, p2, p3).
    A positive value means a counter-clockwise turn (left turn).
    A negative value means a clockwise turn (right turn).
    A zero value means the points are collinear.
    """
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

@njit
def _shoelace_area(points_buffered, num_points):
    """
    Calculates the area of a polygon given its vertices using the Shoelace formula.
    The points must be ordered (e.g., clockwise or counter-clockwise).
    """
    area = 0.0
    for i in range(num_points):
        j = (i + 1) % num_points
        area += points_buffered[i, 0] * points_buffered[j, 1]
        area -= points_buffered[j, 0] * points_buffered[i, 1]
    return abs(area) / 2.0

@njit
def convex_hull_area(points):
    """
    Calculates the area of the convex hull of a set of 2D points.

    Uses the Monotone Chain (Andrew's) algorithm to find the convex hull vertices,
    and then the Shoelace formula to calculate the area.

    Args:
        points (np.ndarray): A 2D NumPy array of shape (N, 2) representing
                             the N points, where each row is [x, y].

    Returns:
        float: The area of the convex hull. Returns 0.0 if there are fewer than 3 unique points.
    """
    n = points.shape[0]
    
    indices = np.argsort(points[:, 0])
    points = points[indices]
    
    # Handle ties in x-coordinate by sorting the y-coordinate for those specific blocks
    i = 0
    while i < n:
        j = i
        # Find the end of the current block of points with the same x-coordinate
        while j < n and points[j, 0] == points[i, 0]:
            j += 1
        # If there's more than one point in this block, sort them by y-coordinate
        if j - i > 1:
            slice_to_sort = points[i:j]
            # Sort the slice by y-coordinate
            indices = np.argsort(slice_to_sort[:, 1])
            points[i:j] = slice_to_sort[indices]
        i = j


    hull_points = np.empty((2 * n, 2), dtype=points.dtype)
    hull_idx = 0

    for p in points:
        while hull_idx >=2 and _cross_product(
                hull_points[hull_idx-2], 
                hull_points[hull_idx-1], 
                p) <= 0:
            hull_idx -= 1 # remove point
        hull_points[hull_idx] = p # Add point
        hull_idx += 1 # Increment index

    t = hull_idx + 1
    
    for p in points[n-2::-1]:
        while hull_idx >= t and _cross_product(
                hull_points[hull_idx-2], 
                hull_points[hull_idx-1], 
                p) <= 0:
            hull_idx -= 1 # remove point
        hull_points[hull_idx] = p # Add point
        hull_idx += 1 # Increment index

    # Extract the actual hull points from the buffer
    hull_points = hull_points[:hull_idx]

    # If the hull has fewer than 3 points (e.g., all points are collinear), area is 0.
    if hull_idx < 3:
        return 0.0

    return _shoelace_area(hull_points, hull_idx)



# Example Usage:
if __name__ == "__main__":
    # Test Case 1: Simple square
    points1 = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ], dtype=np.float64)
    area1 = convex_hull_area(points1)
    print(f"Area of convex hull for square: {area1}") # Expected: 1.0

    # Test Case 2: Points forming a triangle
    points2 = np.array([
        [0, 0],
        [5, 0],
        [2, 3],
        [1, 1], # Inside the triangle
        [4, 1]  # Inside the triangle
    ], dtype=np.float64)
    area2 = convex_hull_area(points2)
    print(f"Area of convex hull for triangle: {area2}") # Expected: 7.5 (base 5, height 3)

    # Test Case 3: Points forming a pentagon
    points3 = np.array([
        [0, 0],
        [2, 0],
        [3, 2],
        [1, 3],
        [-1, 2],
        [1, 1], # Inside
        [0.5, 0.5] # Inside
    ], dtype=np.float64)
    area3 = convex_hull_area(points3)
    print(f"Area of convex hull for pentagon: {area3}") # Expected: ~8.0

    # Test Case 4: Collinear points
    points4 = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3]
    ], dtype=np.float64)
    area4 = convex_hull_area(points4)
    print(f"Area of convex hull for collinear points: {area4}") # Expected: 0.0

    # Test Case 5: Single point
    points5 = np.array([
        [0, 0]
    ], dtype=np.float64)
    area5 = convex_hull_area(points5)
    print(f"Area of convex hull for single point: {area5}") # Expected: 0.0

    # Test Case 6: Two points
    points6 = np.array([
        [0, 0],
        [1, 1]
    ], dtype=np.float64)
    area6 = convex_hull_area(points6)
    print(f"Area of convex hull for two points: {area6}") # Expected: 0.0

    # Test Case 7: Random points (larger set)
    
    np.random.seed(42)
   
    def test(n=100):
        points7 = np.random.rand(n, 2) * n
        area7 = convex_hull_area(points7)
        # print(f"Area of convex hull for 100 random points: {area7}")

    # Test Case 8: Points with negative coordinates
    points8 = np.array([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
        [0, 0] # Inside
    ], dtype=np.float64)
    area8 = convex_hull_area(points8)
    print(f"Area of convex hull for points with negative coordinates: {area8}") # Expected: 4.0
