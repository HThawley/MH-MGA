import numpy as np
import numbers
from typing import Type


NoneType = type(None)


def sanitize_array_type(obj, dtype, name):
    if not is_array_like(obj):
        raise TypeError(f"'{name}' expected array-like. Got: {type(obj)}")
    if not array_dtype_is(obj, dtype):
        if hasattr(obj, "dtype"):
            raise TypeError(f"Array-like '{name}' expected dtype: {dtype}. Got: {obj.dtype}")
        else:
            raise TypeError(f"Array-like '{name}' expected dtype: {dtype}. Inferred: {type(obj[0])}")
    return True


def sanitize_type(obj, dtype, name):
    if isinstance(dtype, (tuple, list)):
        check = max((is_dtype(obj, dt) for dt in dtype))
        if not check:
            raise TypeError(f"'{name}' expected type one of {dtype}. Got: '{type(obj)}'")
    else:
        if not is_dtype(obj, dtype):
            raise TypeError(f"{name} expected type: '{dtype}'. Got: '{type(obj)}'")
    return True


def is_dtype(obj, dtype):
    if isinstance(dtype, type):
        return isinstance(obj, dtype)
    if dtype not in func_dict.keys():
        raise NotImplementedError(f"string dtype: {dtype} not Implemented. Supported: {func_dict.keys()}")
    return func_dict[dtype](obj)


def format_and_sanitize_ditherer(obj, name, strict_type, ge=None, le=None):
    sanitize_type(obj, ("float", "arraylike"), name)
    if not is_array_like(obj):
        sanitize_range(obj, name, ge=ge, le=le)
        return np.array([obj, obj], strict_type)
    if is_array_like(obj):
        for i, elem in enumerate(obj):
            sanitize_range(elem, f"{obj}[{i}]", ge=ge, le=le)
        if len(obj) == 1:
            return strict_type(obj[0])
        elif len(obj) != 2:
            raise ValueError(f"'{name}' should be scalar or of length 2. Got length: {len(obj)}")
        return np.array(obj, strict_type)


# def sanitize_type_wfunc(obj, dtype_func, name, dtype_name):
#     if not dtype_func(obj):
#         raise TypeError(f"'{name}' expected type: '{dtype_name}'. Got: '{type(obj)}'")
#     return True


# def sanitize_type_wfunc_mult(obj, dtype_funcs, name, dtype_names):
#     check = max((dtype_func(obj) for dtype_func in dtype_funcs))
#     if not check:
#         raise TypeError(f"'{name}' expected one of {dtype_names}'. Got: '{type(obj)}'")
#     return True


def sanitize_value(obj, values):
    return obj in values


def sanitize_range(obj, name, lt=None, le=None, ge=None, gt=None):
    if lt is not None:
        if not obj < lt:
            raise ValueError(f"'{name}' should be less than {lt}. Got: {obj}")

    if le is not None:
        if not obj <= le:
            raise ValueError(f"'{name}' should be less than or equal to {le}. Got: {obj}")

    if ge is not None:
        if not obj >= ge:
            raise ValueError(f"'{name}' should be greater than or equal to {ge}. Got: {obj}")

    if gt is not None:
        if not obj > gt:
            raise ValueError(f"'{name}' should be greater than {gt}. Got: {obj}")
    return True


def is_none(obj):
    """
    check whether object is None
    """
    return obj is None


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
    return bool(not np.isinf(obj) and not np.isnan(obj))  # return as python bool


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


def array_dtype_is(obj, dtype: str | Type):
    """
    check dtype of an array-like object
    """
    if not is_array_like(obj):
        raise TypeError(f"'obj' must be array-like. Supplied a {type(obj)}")
    if not isinstance(dtype, (str, Type)):
        raise TypeError(f"'dtype' must be a string or a type. Supplied a {type(dtype)}.")
    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype not in func_dict.keys():
            raise ValueError(
                f"'dtype' expects a type or a string. Supported string dtypes: "
                f"{tuple(func_dict.keys())}. Supplied: '{dtype}'"
            )
        func = func_dict[dtype]

    if len(obj) == 0:
        # empty array has compliant dtype
        return True

    if isinstance(dtype, str):
        return all(func(element) for element in obj)
    return all(isinstance(element, dtype) for element in obj)


def flatten(obj, type=list):
    retobj = []
    for item in obj:
        if is_array_like(item):
            retobj.extend(flatten(item))
        else:
            retobj.append(item)
    return type(retobj)


func_dict = {
    "numeric": is_numeric,
    "boolean": is_boolean,
    "arraylike": is_array_like,
    "float": is_float,
    "integer": is_integer,
    "finite": is_finite,
    "none": is_none,
}
