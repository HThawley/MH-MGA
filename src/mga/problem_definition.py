import numpy as np
from collections.abc import Callable

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
        integrality: np.ndarray = None,
        constraints: bool = False,
        known_optimum: np.ndarray = None,
        fargs: tuple = (),
        fkwargs: dict = {},
    ):
        """
        Initializes the optimization problem definition.
        """
        self.objective = objective
        self.lower_bounds, self.upper_bounds = bounds
        self.maximize = maximize
        self.vectorized = vectorized
        self.constraints = constraints
        self.fargs = fargs
        self.fkwargs = fkwargs
        self.ndim = len(self.lower_bounds)

        if integrality is None:
            self.integrality = np.zeros(self.ndim, dtype=np.bool_)
        else:
            self.integrality = integrality
            
        self.boolean_mask = (((self.upper_bounds - self.lower_bounds) == 1) & self.integrality)

        if known_optimum is not None:
            self.known_optimum_point = known_optimum
            self.known_optimum_value = self.evaluate(np.array([known_optimum]))[0]
        else:
            self.known_optimum_point = None
            self.known_optimum_value = -np.inf if maximize else np.inf

    def evaluate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the objective function and constraints for a set of points.
        """
        obj_values = np.empty(points.shape[0], np.float64)
        violations = np.empty(points.shape[0], np.float64)

        if self.constraints:
            if self.vectorized:
                obj_values[:], violations[:] = self.objective(np.atleast_2d(points), *self.fargs, **self.fkwargs)
            else: 
                for j in range(points.shape[0]):
                    obj_values[j], violations[j] = self.objective(points[j], *self.fargs, **self.fkwargs)
        else:
            if self.vectorized:
                obj_values[:] = self.objective(np.atleast_2d(points), *self.fargs, **self.fkwargs)[0]
            else:
                for j in range(points.shape[0]):
                    obj_values[j] = self.objective(points[j], *self.fargs, **self.fkwargs)
            violations[:] = 0.0

        return obj_values, violations