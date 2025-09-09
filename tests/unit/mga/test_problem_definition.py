import pytest
import numpy as np

from mga.problem_definition import OptimizationProblem
from mga.utils import type_asserts


def array_sum(array, *args, **kwargs):
    return np.abs(array.sum())

def array_sum_c(array, *args, **kwargs):
    return np.abs(array.sum()), array.sum()

def array_sum_v(array, *args, **kwargs):
    return np.abs(array.sum(axis=1))

def array_sum_cv(array, *args, **kwargs):
    return np.abs(array.sum(axis=1)), array.sum(axis=1)

def dummy(*args, **kwargs):
    return 0

def dummy_c(*args, **kwargs):
    return 0, 0

def dummy_v(*args, **kwargs):
    return np.array([0])

def dummy_cv(*args, **kwargs):
    return np.array([0]), np.array([0])

dummy_bounds1 = np.zeros(1), np.ones(1)
dummy_bounds2 = np.zeros(2), np.ones(2)

def test_op_init_default_success():
    """
    OptimizationProblem inits succesfully with default inputs and a scalar objective
    """
    problem = OptimizationProblem(dummy, dummy_bounds1)
    attrs = ("objective", "lower_bounds", "upper_bounds", "maximize", "vectorized", 
             "constraints", "fargs", "fkwargs", "ndim", "integrality", "boolean_mask", 
             "known_optimum_point", "known_optimum_value")
    for attr in attrs:
        assert hasattr(problem, attr)

    assert hasattr(problem, "evaluate")
    assert callable(problem.evaluate)

    assert callable(problem.objective)
    for name, dtype in zip(
            ("lower_bounds", "upper_bounds", "integrality", "boolean_mask", "known_optimum_point"), 
            ("numeric", "numeric", "boolean", "boolean", "numeric")):
        array = getattr(problem, name)
        assert isinstance(array, np.ndarray)
        assert type_asserts.array_dtype_is(array, dtype)
        assert array.ndim == 1
        assert array.size == problem.ndim
    
    assert type_asserts.is_numeric(problem.known_optimum_value)
    assert isinstance(problem.fargs, tuple)
    assert isinstance(problem.fkwargs, dict)

@pytest.mark.parametrize("func", [array_sum, dummy])
def test_op_objective_init_success(func):
    """ 
    OptimizationProblem inherits function appropriately
    """
    problem = OptimizationProblem(func, dummy_bounds1)

    assert id(problem.objective) == id(func)
    assert callable(problem.objective)
    
@pytest.mark.parametrize("func", [1, None])
def test_op_objective_init_failure(func):
    """ 
    OptimizationProblem raises exceptions when objective is not callable
    """
    with pytest.raises(Exception):
        problem = OptimizationProblem(func, dummy_bounds1)

@pytest.mark.parametrize(
        "bounds", 
        [
            (np.zeros(2, dtype=int), np.ones(2, dtype=int)), 
            (np.zeros(2, dtype=float), np.ones(2, dtype=float)), 
            (np.zeros(2, dtype=int), np.ones(2, dtype=float)), 
            (dummy_bounds1), 
        ]
)
def test_op_bounds_init_success(bounds):
    """ 
    OptimizationProblem inherits bounds appropriately
    """
    lb, ub = bounds
    problem = OptimizationProblem(dummy, bounds)

    assert isinstance(problem.lower_bounds, np.ndarray)
    assert isinstance(problem.upper_bounds, np.ndarray)
    assert (problem.lower_bounds == lb).all()
    assert (problem.upper_bounds == ub).all()

@pytest.mark.parametrize(
        "bounds", 
        [
            (np.zeros(2, dtype=int), np.ones(3, dtype=int)), # mismatched length
            (np.zeros(2, dtype=str), np.ones(2, dtype=int)), # incorrect dtype
            (np.zeros(2, dtype=bool), np.ones(2, dtype=int)), # incorrect dtype
            (np.zeros(2), np.inf*np.ones(2)), # contains inf
            (np.zeros(2), np.nan*np.ones(2)), # contains nan
            (np.zeros(0), np.ones(0)), # empty
        ]
)
def test_op_bounds_init_failure(bounds):
    """ 
    OptimizationProblem raises errors with inappropriate bounds
    """
    with pytest.raises(Exception):
        problem = OptimizationProblem(dummy, bounds)

@pytest.mark.parametrize("bool_kwarg", ["maximize", "vectorized", "constraints"])
@pytest.mark.parametrize("value", [True, False, np.True_])
def test_op_boolkwarg_sanitization_success(bool_kwarg, value):
    """
    OptimizationProblem fails to init with inappropriate kwargs'
    """
    if bool_kwarg == "vectorized" and value == True:
        func = dummy_v
    elif bool_kwarg == "constraints" and value == True:
        func = dummy_c
    else: 
        func = dummy
    OptimizationProblem(func, dummy_bounds1, **{bool_kwarg:value})

@pytest.mark.parametrize("bool_kwarg", ["maximize", "vectorized", "constraints"])
@pytest.mark.parametrize("value", [None, 1, 0.0, np.array([False])])
def test_op_boolkwarg_sanitization_failure(bool_kwarg, value):
    """
    OptimizationProblem fails to init with inappropriate 
       ( "maximize", "vectorized", "constraints") kwargs
    """
    with pytest.raises(TypeError):
        OptimizationProblem(dummy, dummy_bounds1, **{bool_kwarg:value})

@pytest.mark.parametrize("bounds", [dummy_bounds1, dummy_bounds2])
@pytest.mark.parametrize("value", [False, np.True_, None])
def test_op_integrality_scalar_bool_or_none(bounds, value):
    """
    OptimizationProblem inits with valid integrality=<scalar>
    problem.integrality is broadcast to the correction shape
    """
    problem = OptimizationProblem(dummy, bounds, integrality=value)
    assert isinstance(problem.integrality, np.ndarray)
    if value == None: 
        value = False
    assert (problem.integrality == value).all()
    assert len(problem.integrality) == problem.ndim
    assert problem.integrality.ndim == 1

@pytest.mark.parametrize("value", [np.array([False, False]), np.array([True, False]), np.array([True, True])])
def test_op_integrality_vector_bool_success(value):
    """
    OptimizationProblem inits with valid integrality=<vector>
    """
    OptimizationProblem(dummy, dummy_bounds2, integrality=value)

@pytest.mark.parametrize("value", [np.array([False, False, False]), np.array([1, 0]), np.array([True])])
def test_op_integrality_vector_bool_failure(value):
    """
    OptimizationProblem fails to init with invalid integrality=<vector>
    """
    with pytest.raises((TypeError, ValueError)):
        OptimizationProblem(dummy, dummy_bounds2, integrality=value)

@pytest.mark.parametrize("func", [dummy, array_sum])
@pytest.mark.parametrize("points", [np.array([0.5, 0.5]), np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])])
def test_op_broadcast(func, points):
    """
    problem.evaluate returns a tuple of two numbers with constraints=True, vectorized=False
    """
    problem = OptimizationProblem(func, dummy_bounds2, constraints=False, vectorized=False)
    result = problem.evaluate(points)

    points = np.atleast_2d(points)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert type_asserts.array_dtype_is(result, np.ndarray)
    for res in result:
        assert type_asserts.array_dtype_is(res, "numeric")
        assert res.shape[0] == points.shape[0]
        assert res.ndim == 1

    for i in range(len(points)): 
        assert result[0][i] == func(points[i])
        assert result[1][i] == 0.

@pytest.mark.parametrize("func", [dummy_c, array_sum_c])
@pytest.mark.parametrize("points", [np.array([0.5, 0.5]), np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])])
def test_op_constraints_broadcast(func, points):
    """
    problem.evaluate returns a tuple of two numbers with constraints=True, vectorized=False
    """
    problem = OptimizationProblem(func, dummy_bounds2, constraints=True, vectorized=False)
    result = problem.evaluate(points)

    points = np.atleast_2d(points)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert type_asserts.array_dtype_is(result, np.ndarray)
    for res in result:
        assert type_asserts.array_dtype_is(res, "numeric")
        assert res.shape[0] == points.shape[0]
        assert res.ndim == 1

    for i in range(len(points)): 
        assert result[0][i] == func(points[i])[0]
        assert result[1][i] == func(points[i])[1]


@pytest.mark.parametrize("func", [dummy_v, array_sum_v])
@pytest.mark.parametrize("points", [np.array([0.5, 0.5]), np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])])
def test_op_vectorized_broadcast(func, points):
    """
    problem.evaluate returns a single numpy array with constraints=False, vectorized=True
    """
    problem = OptimizationProblem(func, dummy_bounds2, constraints=False, vectorized=True)
    result = problem.evaluate(points)

    points = np.atleast_2d(points)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert type_asserts.array_dtype_is(result, np.ndarray)
    for res in result:
        assert type_asserts.array_dtype_is(res, "numeric")
        assert res.shape[0] == points.shape[0]
        assert res.ndim == 1

    for i in range(len(points)): 
        assert result[0][i] == func(np.atleast_2d(points[i]))
        assert result[1][i] == 0.

@pytest.mark.parametrize("func", [dummy_cv, array_sum_cv])
@pytest.mark.parametrize("points", [np.array([0.5, 0.5]), np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])])
def test_op_vectorized_constraints_broadcast(func, points):
    """
    problem.evaluate returns a tuple of two numpy arrays with constraints=True, vectorized=True
    """
    problem = OptimizationProblem(func, dummy_bounds2, constraints=True, vectorized=True)
    result = problem.evaluate(points)

    points = np.atleast_2d(points)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert type_asserts.array_dtype_is(result, np.ndarray)
    for res in result:
        assert type_asserts.array_dtype_is(res, "numeric")
        assert res.shape[0] == points.shape[0]
        assert res.ndim == 1

    for i in range(len(points)): 
        assert result[0][i] == func(np.atleast_2d(points[i]))[0]
        assert result[1][i] == func(np.atleast_2d(points[i]))[1]
