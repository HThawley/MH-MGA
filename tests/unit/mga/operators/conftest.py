import pytest
import numpy as np 

# fixtures

@pytest.fixture
def rng():
    yield np.random.default_rng()


