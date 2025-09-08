import pytest 
import numpy as np 

from mga.commons.types import DEFAULTS
FLOAT, INT = DEFAULTS
from mga.mga import MGAProblem
from mga.problem_definition import OptimizationProblem
from mga.population import Population

class MockOptimizationProblem(OptimizationProblem):
    def __init__(self, ndim=3, maximize=True):
        self.ndim = ndim
        self.lower_bounds = np.zeros(ndim, dtype=FLOAT)
        self.upper_bounds = np.ones(ndim, dtype=FLOAT)
        self.integrality = np.zeros(ndim, dtype=bool)
        self.boolean_mask = np.zeros(ndim, dtype=bool)
        self.known_optimum_point = np.full(ndim, 0.9, dtype=FLOAT)
        self.known_optimum_value = -0.43 if maximize else 2.43
        self.maximize = maximize

    def re__init__(self, **kwargs):
        if 'ndim' in kwargs.keys():
            self.ndim = kwargs['ndim']
            self.lower_bounds = np.zeros(self.ndim, dtype=FLOAT)
            self.upper_bounds = np.ones(self.ndim, dtype=FLOAT)
            self.integrality = np.zeros(self.ndim, dtype=bool)
            self.boolean_mask = np.zeros(self.ndim, dtype=bool)
            self.known_optimum_point = np.full(self.ndim, 0.9, dtype=FLOAT)
        if 'maximize' in kwargs.keys():
            self.maximize = kwargs['maximize']
            self.known_optimum_value = -0.43 if self.maximize else 2.43
            
@pytest.fixture
def mock_problem():
    yield MockOptimizationProblem(ndim=3)

def test_mga_init(mock_problem):
    problem = MGAProblem(mock_problem)
    #TODO: 
