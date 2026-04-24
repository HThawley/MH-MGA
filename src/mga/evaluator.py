
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
            for i in range(population.num_niches):
                (
                    population.raw_objectives[i, :],
                    population.violations[i, :],
                    population.scaled_points[i, :, :],
                 ) = objective_func(population.points[i], *fargs, **fkwargs)

            population.penalized_objectives[:] = (
                population.raw_objectives + population.violations * population.violation_factor
            )

    elif vectorized and not constraints and return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a vectorized objective function that returns obj_vals and scaled_points.
            """
            for i in range(population.num_niches):
                (
                    population.raw_objectives[i, :],
                    population.scaled_points[i, :, :],
                ) = objective_func(population.points[i], *fargs, **fkwargs)

            population.penalized_objectives[:] = population.raw_objectives  # No penalty

    elif not vectorized and constraints and return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a scalar objective function that returns obj_val, violation_val, and scaled_points
            """
            for i in range(population.num_niches):
                for j in range(population.pop_size):
                    (
                        population.raw_objectives[i, j],
                        population.violations[i, j],
                        population.scaled_points[i, j, :],
                    ) = objective_func(population.points[i, j], *fargs, **fkwargs)

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
                    (
                        population.raw_objectives[i, j],
                        population.scaled_points[i, j, :],
                    ) = objective_func(population.points[i, j], *fargs, **fkwargs)

            population.penalized_objectives[:] = population.raw_objectives  # No penalty

    elif vectorized and constraints and not return_scaled:
        def _evaluator(population: Population, objective_func: Callable, fargs: tuple, fkwargs: dict):
            for i in range(population.num_niches):
                (
                    population.raw_objectives[i, :],
                    population.violations[i, :],
                ) = objective_func(population.points[i], *fargs, **fkwargs)

            population.penalized_objectives[:] = (
                population.raw_objectives + population.violations * population.violation_factor
            )

    elif vectorized and not constraints and not return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a vectorized objective function that returns only obj_vals.
            """
            for i in range(population.num_niches):
                population.raw_objectives[i, :] = objective_func(population.points[i], *fargs, **fkwargs)

            population.penalized_objectives[:] = population.raw_objectives  # No penalty

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

            population.penalized_objectives[:] = population.raw_objectives  # No penalty

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
            for i in prange(population.num_niches):
                (
                    population.raw_objectives[i, :],
                    population.violations[i, :],
                    population.scaled_points[i, :, :],
                ) = objective_func(population.points[i], *fargs)

                for j in range(population.pop_size):
                    population.penalized_objectives[i, j] = (
                        population.raw_objectives[i, j]
                        + population.violations[i, j] * population.violation_factor
                    )

    elif vectorized and not constraints and return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns only obj_vals.
            """
            for i in prange(population.num_niches):
                (
                    population.raw_objectives[i, :],
                    population.scaled_points[i, :, :]
                ) = objective_func(population.points[i], *fargs)

                for j in range(population.pop_size):
                    population.penalized_objectives[i, j] = population.raw_objectives[i, j]

    elif not vectorized and constraints and return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns obj_val, violation_val.
            """
            for i in prange(population.num_niches):
                for j in range(population.pop_size):
                    (
                        population.raw_objectives[i, j],
                        population.violations[i, j],
                        population.scaled_points[i, j, :],
                    ) = objective_func(population.points[i, j], *fargs)

                    population.penalized_objectives[i, j] = (
                        population.raw_objectives[i, j]
                        + population.violations[i, j] * population.violation_factor
                    )

    elif not vectorized and not constraints and return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns only obj_val.
            """
            for i in prange(population.num_niches):
                for j in range(population.pop_size):
                    (
                        population.raw_objectives[i, j],
                        population.scaled_points[i, j, :],
                    ) = objective_func(population.points[i, j], *fargs)

                    population.penalized_objectives[i, j] = population.raw_objectives[i, j]

    elif vectorized and constraints and not return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns obj_vals, violation_vals.
            """
            for i in prange(population.num_niches):
                (
                    population.raw_objectives[i, :],
                    population.violations[i, :]
                ) = objective_func(population.points[i], *fargs)

                for j in range(population.pop_size):
                    population.penalized_objectives[i, j] = (
                        population.raw_objectives[i, j]
                        + population.violations[i, j] * population.violation_factor
                    )

    elif vectorized and not constraints and not return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns only obj_vals.
            """
            for i in prange(population.num_niches):
                population.raw_objectives[i, :] = objective_func(population.points[i], *fargs)

                for j in range(population.pop_size):
                    population.penalized_objectives[i, j] = population.raw_objectives[i, j]

    elif not vectorized and constraints and not return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns obj_val, violation_val.
            """
            for i in prange(population.num_niches):
                for j in range(population.pop_size):
                    (
                        population.raw_objectives[i, j],
                        population.violations[i, j]
                    ) = objective_func(population.points[i, j], *fargs)

                    population.penalized_objectives[i, j] = (
                        population.raw_objectives[i, j]
                        + population.violations[i, j] * population.violation_factor
                    )

    else:  # not vectorized and not constraints and not return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns only obj_val.
            """
            for i in prange(population.num_niches):
                for j in range(population.points[i].shape[0]):
                    population.raw_objectives[i, j] = objective_func(population.points[i, j], *fargs)

                    population.penalized_objectives[i, j] = population.raw_objectives[i, j]

    return _evaluator
