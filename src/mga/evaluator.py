from numba.core.registry import CPUDispatcher
from typing import Callable

from mga.population import Population
from mga.commons.numba_overload import njit, prange


def construct(
    numba: bool,
    vectorized: bool,
    constraints: bool,
    return_scaled: bool,
    parallelize: bool = False
):
    if numba:
        return construct_njit_eval_func(
            vectorized, constraints, return_scaled, parallelize
        )
    else:
        return construct_python_eval_func(
            vectorized, constraints, return_scaled  # parallelize silently ignored
        )


def construct_python_eval_func(  # noqa: C901
    vectorized: bool,
    constraints: bool,
    return_scaled: bool,
) -> Callable:
    """
    Constructs the appropriate evaluation function based on problem properties.
    """

    if vectorized and constraints and return_scaled:
        def _evaluator(population: Population, objective_func: Callable, fargs: tuple, fkwargs: dict):
            """
            Evaluates a vectorized objective function that returns obj_vals, violation_vals, and scaled_points.
            """
            num_niches, pop_size, ndim = population.points.shape
            N = num_niches * pop_size
            flat_points = population.points.reshape((N, ndim))

            raw_objectives, violations, scaled_points = objective_func(flat_points, *fargs, **fkwargs)

            population.raw_objectives[:] = raw_objectives.reshape((num_niches, pop_size))
            population.violations[:] = violations.reshape((num_niches, pop_size))
            population.scaled_points[:] = scaled_points.reshape((num_niches, pop_size, population.scaled_ndim))

            population.penalized_objectives[:] = (
                population.raw_objectives + population.violations * population.violation_factor
            )

    elif vectorized and not constraints and return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a vectorized objective function that returns obj_vals and scaled_points.
            """
            num_niches, pop_size, ndim = population.points.shape
            N = num_niches * pop_size
            flat_points = population.points.reshape((N, ndim))

            raw_objectives, scaled_points = objective_func(flat_points, *fargs, **fkwargs)

            population.raw_objectives[:] = raw_objectives.reshape((num_niches, pop_size))
            population.scaled_points[:] = scaled_points.reshape((num_niches, pop_size, population.scaled_ndim))

            population.penalized_objectives[:] = population.raw_objectives

    elif not vectorized and constraints and return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a vectorized objective function that returns obj_vals and scaled_points.
            """
            for i in range(population.num_niches):
                for j in range(population.pop_size):
                    raw_obj, viol, sp = objective_func(population.points[i, j], *fargs, **fkwargs)
                    population.raw_objectives[i, j] = raw_obj
                    population.violations[i, j] = viol
                    population.scaled_points[i, j, :] = sp

            population.penalized_objectives[:] = (
                population.raw_objectives + population.violations * population.violation_factor
            )

    elif not vectorized and not constraints and return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a scalar objective function that returns obj_val and scaled_points
            """
            for i in range(population.num_niches):
                for j in range(population.points[i].shape[0]):
                    raw_obj, sp = objective_func(population.points[i, j], *fargs, **fkwargs)
                    population.raw_objectives[i, j] = raw_obj
                    population.scaled_points[i, j, :] = sp

            population.penalized_objectives[:] = population.raw_objectives

    elif vectorized and constraints and not return_scaled:
        def _evaluator(population: Population, objective_func: Callable, fargs: tuple, fkwargs: dict):
            num_niches, pop_size, ndim = population.points.shape
            N = num_niches * pop_size
            flat_points = population.points.reshape((N, ndim))

            raw_objectives, violations = objective_func(flat_points, *fargs, **fkwargs)

            population.raw_objectives[:] = raw_objectives.reshape((num_niches, pop_size))
            population.violations[:] = violations.reshape((num_niches, pop_size))

            population.penalized_objectives[:] = (
                population.raw_objectives + population.violations * population.violation_factor
            )

    elif vectorized and not constraints and not return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a vectorized objective function that returns only obj_vals.
            """
            num_niches, pop_size, ndim = population.points.shape
            N = num_niches * pop_size
            flat_points = population.points.reshape((N, ndim))

            raw_objectives = objective_func(flat_points, *fargs, **fkwargs)
            population.raw_objectives[:] = raw_objectives.reshape((num_niches, pop_size))
            population.penalized_objectives[:] = population.raw_objectives

    elif not vectorized and constraints and not return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a scalar objective function that returns obj_val, violation_val.
            """
            for i in range(population.num_niches):
                for j in range(population.pop_size):
                    (
                        population.raw_objectives[i, j],
                        population.violations[i, j],
                    ) = objective_func(population.points[i, j], *fargs, **fkwargs)

            population.penalized_objectives[:] = (
                population.raw_objectives + population.violations * population.violation_factor
            )

    else:  # not vectorized and not constraints and not return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a scalar objective function that returns only obj_val.
            """
            for i in range(population.num_niches):
                for j in range(population.points[i].shape[0]):
                    population.raw_objectives[i, j] = objective_func(population.points[i, j], *fargs, **fkwargs)

            population.penalized_objectives[:] = population.raw_objectives

    return _evaluator


def construct_njit_eval_func(  # noqa: C901
    vectorized: bool,
    constraints: bool,
    return_scaled: bool,
    parallelize: bool,
) -> CPUDispatcher:
    """
    Constructs the appropriate evaluation function based on problem properties.
    """
    if vectorized and constraints and return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns obj_vals, violation_vals.
            """
            num_niches, pop_size, ndim = population.points.shape
            N = num_niches * pop_size
            flat_points = population.points.reshape((N, ndim))
            raw_objectives, violations, scaled_points = objective_func(flat_points, *fargs)

            for idx in prange(N):
                i = idx // pop_size
                j = idx % pop_size
                population.raw_objectives[i, j] = raw_objectives[idx]
                population.violations[i, j] = violations[idx]
                population.scaled_points[i, j, :] = scaled_points[idx, :]

                population.penalized_objectives[i, j] = (
                    raw_objectives[idx] + violations[idx] * population.violation_factor
                )

    elif vectorized and not constraints and return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns only obj_vals.
            """
            num_niches, pop_size, ndim = population.points.shape
            N = num_niches * pop_size
            flat_points = population.points.reshape((N, ndim))
            raw_objectives, scaled_points = objective_func(flat_points, *fargs)

            for idx in prange(N):
                i = idx // pop_size
                j = idx % pop_size
                population.raw_objectives[i, j] = raw_objectives[idx]
                population.scaled_points[i, j, :] = scaled_points[idx, :]

                population.penalized_objectives[i, j] = raw_objectives[idx]

    elif not vectorized and constraints and return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns obj_val, violation_val.
            """
            num_niches, pop_size, ndim = population.points.shape
            N = num_niches * pop_size
            for idx in prange(N):
                i = idx // pop_size
                j = idx % pop_size
                raw_obj, viol, sp = objective_func(population.points[i, j], *fargs)
                population.raw_objectives[i, j] = raw_obj
                population.violations[i, j] = viol
                population.scaled_points[i, j, :] = sp

                population.penalized_objectives[i, j] = (
                    population.raw_objectives[i, j] + population.violations[i, j] * population.violation_factor
                )

    elif not vectorized and not constraints and return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns only obj_val.
            """
            num_niches, pop_size, ndim = population.points.shape
            N = num_niches * pop_size
            for idx in prange(N):
                i = idx // pop_size
                j = idx % pop_size
                raw_obj, sp = objective_func(population.points[i, j], *fargs)
                population.raw_objectives[i, j] = raw_obj
                population.scaled_points[i, j, :] = sp

                population.penalized_objectives[i, j] = population.raw_objectives[i, j]

    elif vectorized and constraints and not return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns obj_vals, violation_vals.
            """
            num_niches, pop_size, ndim = population.points.shape
            N = num_niches * pop_size
            flat_points = population.points.reshape((N, ndim))
            raw_objectives, violations = objective_func(flat_points, *fargs)

            for idx in prange(N):
                i = idx // pop_size
                j = idx % pop_size
                population.raw_objectives[i, j] = raw_objectives[idx]
                population.violations[i, j] = violations[idx]

                population.penalized_objectives[i, j] = (
                    raw_objectives[idx] + violations[idx] * population.violation_factor
                )

    elif vectorized and not constraints and not return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns only obj_vals.
            """
            num_niches, pop_size, ndim = population.points.shape
            N = num_niches * pop_size
            flat_points = population.points.reshape((N, ndim))
            raw_objectives = objective_func(flat_points, *fargs)

            for idx in prange(N):
                i = idx // pop_size
                j = idx % pop_size
                population.raw_objectives[i, j] = raw_objectives[idx]

                population.penalized_objectives[i, j] = raw_objectives[idx]

    elif not vectorized and constraints and not return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns obj_val, violation_val.
            """
            num_niches, pop_size, ndim = population.points.shape
            N = num_niches * pop_size
            for idx in prange(N):
                i = idx // pop_size
                j = idx % pop_size

                raw_obj, viol = objective_func(population.points[i, j], *fargs)
                population.raw_objectives[i, j] = raw_obj
                population.violations[i, j] = viol

                population.penalized_objectives[i, j] = (
                    population.raw_objectives[i, j] + population.violations[i, j] * population.violation_factor
                )

    else:  # not vectorized and not constraints and not return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns only obj_val.
            """
            num_niches, pop_size, ndim = population.points.shape
            N = num_niches * pop_size
            for idx in prange(N):
                i = idx // pop_size
                j = idx % pop_size
                population.raw_objectives[i, j] = objective_func(population.points[i, j], *fargs)

                population.penalized_objectives[i, j] = population.raw_objectives[i, j]

    return _evaluator
