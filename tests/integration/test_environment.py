import pytest 
import numpy as np
import subprocess
import pkgutil
import importlib
import mga

mga_modules = [name for _, name, _ in pkgutil.walk_packages(mga.__path__, mga.__name__ + '.')]

@pytest.mark.parametrize("precision", [32, 64])
def test_precision_update_good(precision:int):
    """
    test float/int byte length setting. 
    Launched as a subprocess to ensure isolated environment
    """
    try: 
        subprocess.run(
            ["python", "tests/integration/_test_precision_update_good.py", "-p", str(int(precision))],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise e

@pytest.mark.parametrize("precision", [1, 32.1])
def test_precision_update_bad(precision:int):
    """
    test float/int byte length setting. 
    Launched as a subprocess to ensure isolated environment
    """
    try: 
        subprocess.run(
            ["python", "tests/integration/_test_precision_update_bad.py", "-p", str(int(precision))],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
            raise e

@pytest.mark.parametrize("module", mga_modules)
def test_precision_update_warning(module:str):
    """
    test float/int byte length setting. 
    Launched as a subprocess to ensure isolated environment
    """
    result = subprocess.run(
        ["python", "tests/integration/_test_precision_update_warning.py", "-m", module],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Error in execution: {result.stderr}"
    mod = importlib.import_module(module)
    if hasattr(mod, "INT") or hasattr(mod, "FLOAT"):
        assert "UserWarning" in result.stderr, "UserWarning not raised when updating precision after import"
    else:
        assert "UserWarning" not in result.stderr, "UserWarning raised inappropiately. Check settability?"

@pytest.mark.parametrize("module", mga_modules)
def test_precision_settability(module:str):
    """
    test float/int byte length setting. 
    Launched as a subprocess to ensure isolated environment
    """
    try: 
        subprocess.run(
            ["python", "tests/integration/_test_precision_settability.py", "-m", module],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise e
