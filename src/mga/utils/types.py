import numpy as np
import numbers
from typing import Type

def is_numeric(obj):
    """
    check whether object is numeric
    """
    return isinstance(obj, numbers.Number) and not is_boolean(obj)

def is_integer(obj):
    """
    check whether object is an integer
    """
    return isinstance(obj, numbers.Integral) and not is_boolean(obj)

def is_float(obj):
    """
    check whether object is a float
    """
    return is_numeric(obj) and not is_integer(obj)

def is_boolean(obj):
    """
    check whether object is boolean
    """ 
    return isinstance(obj, (bool, np.bool_))

def is_finite(obj):
    """
    check whether numeric object is finite
    """
    if not (is_numeric(obj) or is_boolean(obj)):
        return False
    return bool(not np.isinf(obj) and not np.isnan(obj)) # return as python bool

def is_array_like(obj):
    """
    check whether object is array-like. 
    """
    if isinstance(obj, (str, dict)):
        return False
    if not hasattr(obj, "__len__"):
        return False
    if not hasattr(obj, "__iter__"):
        return False
    if not hasattr(obj, "__getitem__"):
        return False
    return True
    
def array_dtype_is(obj, dtype: str|Type):
    """
    check dtype of an array-like object
    """
    if not is_array_like(obj): 
        raise TypeError(f"'obj' must be array-like. Supplied a {type(obj)}")
    if not isinstance(dtype, (str, Type)): 
        raise TypeError(f"'dtype' must be a string or a type. Supplied a {type(dtype)}.")
    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype not in ("numeric", "boolean", "arraylike", "float", "integer", "finite"):
            raise ValueError(f"'dtype' expects a type or a string. Supported string dtypes: "\
                             f"('numeric', 'boolean', 'arraylike', 'float', 'integer', 'finite'). Supplied: '{dtype}'")
    
    if len(obj) == 0: 
        # empty array has compliant dtype 
        return True

    if dtype == "numeric":
        return all(is_numeric(element) for element in obj)
    elif dtype == "boolean":
        return all(is_boolean(element) for element in obj)
    elif dtype == "arraylike":
        return all(is_array_like(element) for element in obj)
    elif dtype == "float":
        return all(is_float(element) for element in obj)
    elif dtype == "integer":
        return all(is_integer(element) for element in obj)
    elif dtype == "finite":
        return all(is_finite(element) for element in obj)
    else: 
        return all(isinstance(element, dtype) for element in obj)

def flatten(obj, type=list):
    retobj = []
    for item in obj:
        if is_array_like(item):
            retobj.extend(flatten(item))
        else:
            retobj.append(item)
    return type(retobj)