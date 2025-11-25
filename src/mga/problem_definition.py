import numpy as np
from numba.core.registry import CPUDispatcher
from collections.abc import Callable

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS
from mga.utils import typing  # noqa: E402


class OptimizationProblem:
    """
    Encapsulates the definition of the optimization problem.
    This includes the objective function, variable bounds, and optimization sense.
    """
    def __init__(
        self,
        objective: Callable,
        bounds: tuple[np.ndarray, np.ndarray],
        maximize: bool = False,
        vectorized: bool = False,
        constraints: bool = False,
        integrality: bool | np.ndarray[bool] = False,
        known_optimum: np.ndarray = None,
        fargs: tuple = (),
        fkwargs: dict = None,
    ):
        """
        Initializes the optimization problem definition.
        """
        fkwargs = {} if fkwargs is None else fkwargs
        # Sanitize inputs
        typing.sanitize_type(objective, Callable, "objective")
        typing.sanitize_type(bounds, "arraylike", "bounds")
        typing.sanitize_array_type(bounds, "arraylike", "bounds")
        if len(bounds) != 2:
            raise ValueError(f"'bounds' expected length 2 i.e. (lower, upper). Received length: {len(bounds)}")
        for i, bound in enumerate(bounds):
            name = f"bounds[{i}]"
            typing.sanitize_array_type(bound, "numeric", name)
            typing.sanitize_array_type(bound, "finite", name)
            if len(bound) == 0:
                raise ValueError(f"'{name}' must have length > 0")

        if len(bounds[0]) != len(bounds[1]):
            raise ValueError("Upper and lower bound shapes must match."
                             f"Lower: {len(bounds[0])}, Upper: {len(bounds[1])}")

        # Other parameters
        typing.sanitize_type(maximize, "boolean", "maximize")
        typing.sanitize_type(vectorized, "boolean", "vectorized")
        typing.sanitize_type(constraints, "boolean", "constraints")

        if typing.is_boolean(integrality):
            integrality = integrality * np.ones(len(bounds[0]), np.bool_)
        elif integrality is None:
            integrality = np.zeros(len(bounds[0]), np.bool_)

        # Optional parameter validation
        if known_optimum is not None:
            typing.sanitize_array_type(known_optimum, "numeric", "known_optimum")

        typing.sanitize_type(fargs, tuple, "fargs")
        typing.sanitize_type(fkwargs, dict, "fkwargs")

        # instantiation
        self.objective = objective
        if isinstance(objective, CPUDispatcher):
            self.objective_jitted = True
        else:
            self.objective_jitted = False

        self.lower_bounds, self.upper_bounds = bounds
        self.lower_bounds = self.lower_bounds.astype(FLOAT)
        self.upper_bounds = self.upper_bounds.astype(FLOAT)
        assert (self.lower_bounds <= self.upper_bounds).all()
        self.ndim = len(self.lower_bounds)

        self.maximize = maximize
        self.vectorized = vectorized
        self.constraints = constraints
        self.fargs = fargs
        self.fkwargs = fkwargs

        self.integrality = integrality
        self.booleanality = (((self.upper_bounds - self.lower_bounds) == 1) & self.integrality)

        if known_optimum is not None:
            self.known_optimum_point = known_optimum.astype(FLOAT)
        else:
            self.known_optimum_point = (self.upper_bounds + self.lower_bounds)/2
        self.known_optimum_value = self.evaluate(np.atleast_2d(self.known_optimum_point))[0][0]

    def evaluate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the objective function and constraints for a set of points.
        """
        points = np.atleast_2d(points)
        obj_values = np.empty(points.shape[0], FLOAT)
        violations = np.empty(points.shape[0], FLOAT)

        if self.constraints:
            if self.vectorized:
                obj_values[:], violations[:] = self.objective(points, *self.fargs, **self.fkwargs)
            else:
                for j in range(points.shape[0]):
                    obj_values[j], violations[j] = self.objective(points[j], *self.fargs, **self.fkwargs)
        else:
            if self.vectorized:
                obj_values[:] = self.objective(points, *self.fargs, **self.fkwargs)
            else:
                for j in range(points.shape[0]):
                    obj_values[j] = self.objective(points[j], *self.fargs, **self.fkwargs)
            violations[:] = 0.0

        return obj_values, violations

    def evaluate_population(self, population):
        for i in range(population.num_niches):
            population.objective_values[i], population.violations[i] = self.evaluate(population.points[i])
        population.penalized_objectives[:] = (
            population.objective_values + population.violations * population.violation_factor
        )


class MultiObjectiveProblem:
    """
    Encapsulates the definition of a multi-objective optimization problem.
    This includes the objective functions, variable bounds, and optimization senses.
    """
    def __init__(
        self,
        objective: Callable,
        bounds: tuple[np.ndarray, np.ndarray],
        n_objs: int,
        maximize: bool | np.ndarray[bool] = False,
        vectorized: bool = False,
        feasibility: bool = False,
        integrality: bool | np.ndarray[bool] = False,
        fargs: tuple = (),
        fkwargs: dict = None,
    ):
        """
        Initializes the multi-objective optimization problem definition.
        """
        if not callable(objective):
            raise TypeError("'objective' must be callable")

        self.objective = objective
        self.lower_bounds, self.upper_bounds = bounds
        self.ndim = len(self.lower_bounds)
        self.n_objs = n_objs

        self.vectorized = vectorized
        self.feasibility = feasibility
        self.fargs = fargs
        self.fkwargs = fkwargs if fkwargs is not None else {}

        if isinstance(maximize, bool):
            self.maximize = np.array([maximize] * self.n_objs, dtype=np.bool_)
        else:
            if not typing.is_array_like(maximize) or len(maximize) != self.n_objs:
                raise ValueError(f"'maximize' must be a bool or an iterable of length n_objs ({self.n_objs})")
            self.maximize = np.array(maximize, dtype=np.bool_)

        if isinstance(integrality, bool):
            self.integrality = np.array([integrality] * self.ndim, dtype=np.bool_)
        else:
            if not typing.is_array_like(integrality) or len(integrality) != self.ndim:
                raise ValueError(f"'integrality' must be a bool or an iterable of length ndim ({self.ndim})")
            self.integrality = np.array(integrality, dtype=np.bool_)

        self.booleanality = (((self.upper_bounds - self.lower_bounds) == 1) & self.integrality)

    def evaluate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the objective functions and feasibility for a set of points.
        Returns objective values and feasibility flags.
        """
        points = np.atleast_2d(points)
        num_points = points.shape[0]
        obj_values = np.empty((num_points, self.n_objs), dtype=FLOAT)
        is_feasible = np.ones((num_points, self.n_objs), dtype=np.bool_)

        if self.feasibility:
            if self.vectorized:
                obj_values, is_feasible = self.objective(points, *self.fargs, **self.fkwargs)
            else:
                for j in range(num_points):
                    obj_values[j, :], is_feasible[j, :] = self.objective(points[j, :], *self.fargs, **self.fkwargs)

        else:
            if self.vectorized:
                obj_values[:] = self.objective(points, *self.fargs, **self.fkwargs)
            else:
                for j in range(num_points):
                    obj_values[j, :] = self.objective(points[j, :], *self.fargs, **self.fkwargs)

        return obj_values, is_feasible
