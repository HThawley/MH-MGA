import numpy as np 
import pytest

import mga.moopopulation as mp

def test_simple_minimization():
    points = np.array([[0, 0], [1, 1], [2, 2]])
    objectives = np.array([[0, 0], [1, 1], [2, 2]])
    maximize = np.array([False, False])
    feas = np.ones_like(objectives, dtype=bool)

    pareto_points, pareto_objs = mp._select_pareto(points, objectives, maximize, feas)
    assert len(pareto_points) == 1
    assert (pareto_points[0] == [0, 0]).all()


def test_two_non_dominated():
    points = np.array([[0, 1], [1, 0], [2, 2]])
    objectives = points.copy()
    maximize = np.array([False, False])
    feas = np.ones_like(objectives, dtype=bool)

    pareto_points, pareto_objs = mp._select_pareto(points, objectives, maximize, feas)
    assert len(pareto_points) == 2
    assert any((pareto_points == [0, 1]).all(1))
    assert any((pareto_points == [1, 0]).all(1))


def test_maximization():
    points = np.array([[0, 0], [1, 1], [2, 2]])
    objectives = np.array([[0, 0], [1, 1], [2, 2]])
    maximize = np.array([True, True])
    feas = np.ones_like(objectives, dtype=bool)

    pareto_points, pareto_objs = mp._select_pareto(points, objectives, maximize, feas)
    assert len(pareto_points) == 1
    assert (pareto_points[0] == [2, 2]).all()


def test_infeasible_excluded():
    points = np.array([[0, 0], [1, 1]])
    objectives = np.array([[0, 0], [1, 1]])
    maximize = np.array([False, False])
    feas = np.array([[False, False], [True, True]])

    pareto_points, pareto_objs = mp._select_pareto(points, objectives, maximize, feas)
    assert len(pareto_points) == 1
    assert (pareto_points[0] == [1, 1]).all()


def test_no_feasible():
    points = np.array([[0, 0], [1, 1]])
    objectives = np.array([[0, 0], [1, 1]])
    maximize = np.array([False, False])
    feas = np.array([[False, False], [False, False]])

    pareto_points, pareto_objs = mp._select_pareto(points, objectives, maximize, feas)
    assert pareto_points.shape[0] == 0
    assert pareto_objs.shape[0] == 0