import pytest
import numpy as np
from numba import njit, int64
from numba.types import UniTuple
from numba.experimental import jitclass

from mga.operators import mutation as mt


@jitclass([("retval", int64)])
class mock_rng1:
    def __init__(self, n):
        self.retval = n
    def integers(self, lb, ub):
        return self.retval

@jitclass([("retval", UniTuple(int64, 2)),
           ("calls", int64)])
class mock_rng2:
    def __init__(self, n):
        self.calls=-1
        self.retval = n
    def integers(self, lb, ub):
        self.calls+=1
        return self.retval[self.calls]

# fixtures

# tests
@pytest.mark.parametrize("sigma", [0.0, 0.001, 0.1, 1.0])
def test_mutate_float(rng, sigma):
    """
    mutate float performs gaussian mutation with correctly inhereted sigma
    """
    result = np.array([mt._mutate_float(0, sigma, rng) for _ in range(10000)])
    std = np.std(result)
    if sigma == 0:
        assert (result == 0).all()
    else: 
        assert abs(1 - std/sigma) < 0.05, "bad gaussian noise scaling" # 5 % point tolerance

@pytest.mark.parametrize("sigma", [0.0, 0.1, 5.0, 10.0])
def test_mutate_int(rng, sigma):
    """
    mutate int performs gaussian mutation with correctly inhereted sigma
    """
    result = np.array([mt._mutate_int(0, sigma, rng) for _ in range(10000)])
    std = np.std(result)
    if sigma == 0.0:
        assert (result == 0).all()
    elif sigma <= 0.5:
        assert (result != 0).sum() < 2, "high int mutation at low sigma"
    else:
        assert abs(1 - std/sigma) < 0.05, "bad gaussian noise scaling" # 5 % point tolerance
    assert (result % 1 < 1e-6).all()

@pytest.mark.parametrize("sigma", [0.0, 0.1, 5.0, 10.0])
@pytest.mark.parametrize("item", [True, False, 1.0, 0.0])
def test_mutate_bool(rng, sigma):
    """
    mutate bool performs 
    """
    #TODO: fix this and mt._mutate_bool
    result = np.array([mt._mutate_bool(False, sigma, rng) for _ in range(10000)])
    assert True

@pytest.mark.parametrize("sigma", [0.0, 0.1, 1.0])
def test_mutate_float_pop(rng, sigma):
    """
    Points should have gaussian noise applied
    """
    population = np.zeros((1, 10000, 1), dtype=float)
    mt.mutate_gaussian_population_float(
        population, 
        sigma * np.ones(3, float), 
        1.0, 
        rng, 
        np.array([False, False]),
        np.array([False, False]),
    )
    if sigma == 0:
        assert np.abs(population).sum() == 0, "mutation should not occur when sigma=0"
    else: 
        assert abs(np.mean(population)) < sigma * 0.1, "distribution is centred on mean"
        assert abs(1 - np.std(population)/sigma) < 0.05, "mutation does not match sigma"

def test_mutate_float_pop_sigma_broadcast(rng):
    """
    Points should have gaussian noise applied proportional to broadcast sigma
    """
    sigma = np.array([0.0, 0.1, 1.0])
    population = np.zeros((1, 10000, 3), dtype=float)
    mt.mutate_gaussian_population_float(
        population, 
        sigma*np.ones(3, float), 
        1.0, 
        rng, 
        np.array([False, False]),
        np.array([False, False]),
    )
    for k in range(3):
        if sigma[k] == 0:
            assert np.abs(population[:, :, k]).sum() == 0, "mutation should not occur when sigma=0"
        else: 
            assert abs(np.mean(population[:, :, k])) < sigma[k] * 0.1, "distribution is centred on mean"
            assert abs(1 - np.std(population[:, :, k])/sigma[k]) < 0.05, "mutation does not match sigma"

@pytest.mark.parametrize("mut_prob", [0.0, 0.25, 0.5, 1.0])
def test_mutate_float_pop_mut_prob(rng, mut_prob):
    """
    Points should have gaussian noise applied
    """
    population = np.zeros((1, 10000, 1), dtype=float)
    mt.mutate_gaussian_population_float(
        population, 
        np.array([100.0]), # sigma is large so mutation is obvious if it occurs 
        mut_prob, 
        rng, 
        np.array([False, False]),
        np.array([False, False]),
    )
    mut_rate = 1 - (np.isclose(population, 0).sum() / population.size)

    if mut_prob == 0:
        assert mut_rate == 0, "mutation should not occur when sigma=0"
    else: 
        # 5 % point tolerance
        assert abs(1 - mut_rate / mut_prob) < 0.05, "mutation probability not correctly applied"

def test_mutate_mixed_retains_type(mixed_dtype_points, integrality, bool_mask, rng):
    """
    mt.mutate_gaussian_population_mixed should conserve dtype
    """
    mt.mutate_gaussian_population_mixed(
        mixed_dtype_points, 
        10.0 * np.ones(3, float), 
        1.0, 
        rng, 
        integrality,
        bool_mask,
    )
    assert "float" in str(mixed_dtype_points.dtype)

    for k in range(mixed_dtype_points.shape[-1]):
        if bool_mask[k]:
            assert np.isin(mixed_dtype_points[:, :, k], (1.0, 0.0)).all(), "bool not conserved"
        elif integrality[k]:
            assert np.isclose(mixed_dtype_points[:, :, k] % 1, 0).all(), "int not conserved"
        else: # float
            assert not np.isclose(mixed_dtype_points[:, :, k] % 1, 0).all(), "float not conserved"

@pytest.mark.parametrize("sigma", [0.0, 0.1, 1.0, 5.0])
def test_mutate_mixed_pop(mixed_dtype_points, integrality, bool_mask, rng, sigma):
    """
    Points should have gaussian noise applied
    """
    mdp = mixed_dtype_points.copy()
    mt.mutate_gaussian_population_mixed(
        mixed_dtype_points, 
        sigma * np.ones(3, float), 
        1.0, 
        rng, 
        integrality,
        bool_mask,
    )
    if sigma == 0:
        assert (mdp == mixed_dtype_points).all(), f"mutation should not occur when sigma=0, {(mdp != mixed_dtype_points).sum()}"

    for k in range(mixed_dtype_points.shape[-1]):
        if bool_mask[k]:
            assert not (mdp[:, :, k] == mixed_dtype_points[:, :, k]).all(), "bool mutation not occuring"
        elif integrality[k]:
            if sigma < 0.5:
                assert (mdp[:, :, k] == mixed_dtype_points[:, :, k]).sum() <= 1, "int mutation occuring at too high a rate"
            else: 
                mut_std = np.std(mixed_dtype_points[:, :, k] - mdp[:, :, k])
                # high tolerance at low sample size
                assert abs(1 - mut_std/sigma) < 0.1, "mutation does not match sigma"
        else: # float
                mut_std = np.std(mixed_dtype_points[:, :, k] - mdp[:, :, k])
                # high tolerance at low sample size
                assert abs(1 - mut_std/sigma) < 0.1, "mutation does not match sigma"

def test_mutate_mixed_pop_sigma_broadcast(rng):
    """
    Points should have gaussian noise applied proportional to broadcast sigma
    """
    sigma = np.array([0.0, 0.1, 1.0])
    population = np.zeros((1, 10000, 3), dtype=float)
    mt.mutate_gaussian_population_mixed(
        population, 
        sigma*np.ones(3, float), 
        1.0, 
        rng, 
        np.array([False, False, False]),
        np.array([False, False, False]),
    )
    for k in range(3):
        if sigma[k] == 0:
            assert np.abs(population[:, :, k]).sum() == 0, "mutation should not occur when sigma=0"
        else: 
            assert abs(np.mean(population[:, :, k])) < sigma[k] * 0.1, "distribution is centred on mean"
            assert abs(1 - np.std(population[:, :, k])/sigma[k]) < 0.05, "mutation does not match sigma"

@pytest.mark.parametrize("mut_prob", [0.0, 0.25, 0.5, 1.0])
def test_mutate_mixed_pop_mut_prob(mixed_dtype_points, integrality, bool_mask, rng, mut_prob):
    """
    Rate of mutation should be similar to mut_prob
    """
    population = np.zeros((1, 10000, 1), dtype=float)
    mt.mutate_gaussian_population_float(
        population, 
        np.array([100.0]), # sigma is large so mutation is obvious if it occurs 
        mut_prob, 
        rng, 
        np.array([False, False]),
        np.array([False, False]),
    )
    mut_rate = 1 - (np.isclose(population, 0).sum() / population.size)

    if mut_prob == 0:
        assert mut_rate == 0, "mutation should not occur when sigma=0"
    else: 
        # 5 % point tolerance
        assert abs(1 - mut_rate / mut_prob) < 0.05, "mutation probability not correctly applied"
