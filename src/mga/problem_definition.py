import numpy as np
from collections.abc import Callable

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS
from mga.utils import type_asserts  

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
        integrality: bool|np.ndarray[bool] = False,
        known_optimum: np.ndarray = None,
        fargs: tuple = (),
        fkwargs: dict = {},
    ):
        """
        Initializes the optimization problem definition.
        """
        # Sanitize inputs
        if not callable(objective): 
            raise TypeError("'objective' must be callable")
        if not type_asserts.is_array_like(bounds): 
            raise TypeError(f"'bounds' expected a tuple of arrays. Received: {type(bounds)}")
        if not type_asserts.array_dtype_is(bounds, "arraylike"): 
            raise TypeError(f"'bounds' expected a tuple of arrays. Received: {type(bounds)}")
        if not len(bounds) == 2: 
            raise ValueError(f"'bounds' expected length 2 i.e. (lower, upper). Received length: {len(bounds)}")
        for bound in bounds: 
            if not type_asserts.array_dtype_is(bound, "numeric"): 
                raise TypeError("Upper and lower bounds expected numeric dtype")
            if not type_asserts.array_dtype_is(bound, "finite"):
                raise TypeError("Upper and lower bounds must be finite")
            if not len(bound) > 0:
                raise ValueError("upper and lower bounds must have length > 0")
        if not len(bounds[0]) == len(bounds[1]): 
            raise ValueError(f"Upper and lower bound shapes must match. Lower: {len(bounds[0])}, Upper: {len(bounds[1])}")
        if not type_asserts.is_boolean(maximize): 
            raise TypeError(f"'maximize' expected a bool. Received: {type(maximize)}")
        if not type_asserts.is_boolean(vectorized): 
            raise TypeError(f"'vectorized' expected a bool. Received: {type(vectorized)}")
        if not type_asserts.is_boolean(constraints): 
            raise TypeError(f"'constraints' expected a bool. Received: {type(constraints)}")
        if type_asserts.is_boolean(integrality):
            integrality = integrality * np.ones(len(bounds[0]), np.bool_)
        elif integrality is None:
            integrality = np.zeros(len(bounds[0]), np.bool_)
        if not type_asserts.is_array_like(integrality): 
            raise TypeError(f"'integrality' expected an array or bool. Received: {type(integrality)}")
        if not type_asserts.array_dtype_is(integrality, "boolean"):
            raise TypeError("'integrality' expected boolean dtype")
        if not len(integrality) == len(bounds[0]):
            raise ValueError(f"'integrality' length must match bounds ({len(bounds[0])}). Received: {len(integrality)}")
        if known_optimum is not None: 
            if not type_asserts.is_array_like(known_optimum, "numeric"): 
                raise TypeError("'known_optimum' expected numeric dtype")
        if not isinstance(fargs, tuple): 
            raise TypeError(f"'fargs' expected an array or bool. Received: {type(fargs)}")
        if not isinstance(fkwargs, dict): 
            raise TypeError(f"'fkwargs' expected an array or bool. Received: {type(fkwargs)}")

        # instantiation
        self.objective = objective
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
            
        self.boolean_mask = (((self.upper_bounds - self.lower_bounds) == 1) & self.integrality)

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
                obj_values[:] = self.objective(points, *self.fargs, **self.fkwargs)[0]
            else:
                for j in range(points.shape[0]):
                    obj_values[j] = self.objective(points[j], *self.fargs, **self.fkwargs)
            violations[:] = 0.0

        return obj_values, violations