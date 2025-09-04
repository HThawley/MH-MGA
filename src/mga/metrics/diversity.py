import numpy as np 
from numba import njit, prange

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS

# API functions

@njit 
def mean_of_shannon_of_projections(points, feasibility, lb, ub):
    """
    mean of shannon index along each dimension of a set of points
    To be called on problem.noptima 
    """
    npoint, ndim = points.shape
    nbin = max(2, int(npoint**0.5)) # sqrt of number of samples, consider updating later 
    acc = 0
    counts = np.zeros(nbin, dtype=INT)
    feasible_points = points[feasibility].T
    for k in range(ndim):
        counts[:] = 0
        acc += _shannon_index(feasible_points[k], lb[k], ub[k], nbin, npoint, counts)
    acc /= np.log(nbin) # normalize 
    acc /= ndim # take mean
    return acc 

@njit
def sum_of_fitness(fitness, noptimality):
    """ sum of maximum (noptimal) fitness of each population""" 
    acc = 0
    for i in range(fitness.shape[0]):
        _max = 0
        for j in range(fitness.shape[1]):
            if noptimality[i, j]:
                if fitness[i, j] > _max:
                    _max = fitness[i, j]
        acc += _max
    return acc

@njit 
def mean_of_fitness(fitness, noptimality):
    return sum_of_fitness(fitness, noptimality) / fitness.shape[0]

@njit
def volume_estimation_by_shadow_addition(points, feasibility):
    vesa = 0.0
    feasible_points = points[feasibility].T
    for k in prange(points.shape[1]):
        for _k in range(k + 1, points.shape[1]):
            projection = np.stack((feasible_points[k], feasible_points[_k]), axis=-1)
            vesa += _convex_hull_area(projection)
    return vesa

@njit
def std(quantity, noptimality):
    """ standard deviation. `quantity` may be problem.objective or problem.fitness"""
    return _stat_measure(quantity, noptimality, np.std)

@njit
def var(quantity, noptimality):
    """ statistical variation. `quantity` may be problem.objective or problem.fitness"""
    return _stat_measure(quantity, noptimality, np.var)

# private helper functions

@njit 
def _shannon_index(values, lb, ub, nbin, npoint, counts):
    bin_width = (ub-lb) / nbin

    for i in range(npoint):
        idx = int((values[i]-lb)/bin_width)
        if idx < 0: idx = 0
        if idx > nbin-1: idx = nbin-1
        counts[idx] += 1
    
    H = 0.0
    nonzero = 0
    for i in range(nbin):
        if counts[i] > 0:
            p = counts[i] / npoint
            H -= p * np.log(p)
            nonzero += 1

    # Miller-Madow bias correction
    H += (nonzero - 1) / (2.0 * npoint)
    return H 

@njit
def _stat_measure(quantity, noptimality, stat):
    result = np.empty(quantity.shape[0])
    for i in range(quantity.shape[0]):
        _feas = False
        for j in range(noptimality.shape[1]):
            if noptimality[i, j] is True:
                _feas = True
                break
        if _feas is True:
            result[i] = stat(quantity[i][noptimality[i]])
        else: 
            result[i] = np.inf
    return result

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
def _convex_hull_area(points):
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