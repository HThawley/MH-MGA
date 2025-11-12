import pytest
import numpy as np
from numba import int32, int64, float32, float64, boolean

from mga.utils import typing

integers = [
    0,
    -1,
    np.int16(-2.0),
    np.int32(3.0),
    np.int64(4),
    np.uint16(5e0),
    np.uint32(6_000),
    np.uint64(7),
    int32(-8),
    int64(9),
]

finite_floats = [
    0.0,
    1.0,
    float(2),
    float("-3"),
    np.float16(4),
    np.float32(5.6789),
    np.float64(6.0123),
    float32(-7),
    float64(-8.8888),
]

infinite_floats = [
    np.nan,
    np.inf,
    -np.inf,
    float("inf"),
]

booleans = [
    True,
    False,
    np.True_,
    np.False_,
    boolean(True),
]

boolean_disambig = [
    np.array([True]),
    1,
    0,
    -1,
    1.0,
]

array_like_dempty = [
    (),
    [],
    np.array([], float),
    np.array([], int),
    np.array([], bool),
]

array_like_dfloat = [
    (1.0, 2.0, 3.0),
    [1.0, 2.0, 3.0],
    np.array([1, 2, 3], float),
]

array_like_dint = [
    (1, 2, 3),
    [1, 2, 3],
    np.array([1, 2, 3], int),
]

array_like_dbool = [
    (True, False, True),
    [True, False, True],
    np.array([-1, 0, 1], bool),
]

array_like_dinfinite = [
    (1.0, np.nan, np.inf),
    [1.0, np.nan, np.inf],
    np.array([1.0, np.nan, np.inf]),
]

array_like_dfinite = [
    (1.0, 2, True),
    [1.0, 2, True],
]

array_like_dnone = [
    (1, 2.0, "c"),
    [-1, 0.0, None],
    ["a", "b"],
]

array_likes = (
    array_like_dempty + array_like_dfloat + array_like_dint + array_like_dbool + array_like_dinfinite + array_like_dnone
)

other_collections = [
    {},
    {1, 2, 3},
    {"a": 1},
    "abc",
]


@pytest.mark.parametrize("obj", integers + finite_floats + infinite_floats)
def test_is_numeric_true(obj):
    assert typing.is_numeric(obj)


@pytest.mark.parametrize("obj", booleans + array_likes + other_collections)
def test_is_numeric_false(obj):
    assert not typing.is_numeric(obj)


@pytest.mark.parametrize("obj", integers)
def test_is_integer_true(obj):
    assert typing.is_integer(obj)


@pytest.mark.parametrize("obj", finite_floats + infinite_floats + booleans + array_likes + other_collections)
def test_is_integer_false(obj):
    assert not typing.is_integer(obj)


@pytest.mark.parametrize("obj", finite_floats + infinite_floats)
def test_is_float_true(obj):
    assert typing.is_float(obj)


@pytest.mark.parametrize("obj", integers + booleans + array_likes + other_collections)
def test_is_float_false(obj):
    assert not typing.is_float(obj)


@pytest.mark.parametrize("obj", booleans)
def test_is_boolean_true(obj):
    assert typing.is_boolean(obj)


@pytest.mark.parametrize(
    "obj", boolean_disambig + integers + finite_floats + infinite_floats + array_likes + other_collections
)
def test_is_boolean_false(obj):
    assert not typing.is_boolean(obj)


@pytest.mark.parametrize("obj", finite_floats + integers + booleans)
def test_is_finite_true(obj):
    assert typing.is_finite(obj)


@pytest.mark.parametrize("obj", infinite_floats + array_likes + other_collections)
def test_is_finite_false(obj):
    assert not typing.is_finite(obj)


@pytest.mark.parametrize("obj", array_likes)
def test_is_arraylike_true(obj):
    assert typing.is_array_like(obj)


@pytest.mark.parametrize("obj", infinite_floats + finite_floats + integers + booleans + other_collections)
def test_is_arraylike_false(obj):
    assert not typing.is_array_like(obj)


@pytest.mark.parametrize("obj", array_like_dempty + array_like_dfloat + array_like_dinfinite)
def test_array_dtype_is_float_true(obj):
    assert typing.array_dtype_is(obj, "float")


@pytest.mark.parametrize("obj", array_like_dint + array_like_dbool + array_like_dnone + array_like_dfinite)
def test_array_dtype_is_float_false(obj):
    assert not typing.array_dtype_is(obj, "float")


@pytest.mark.parametrize("obj", array_like_dempty + array_like_dint)
def test_array_dtype_is_int_true(obj):
    assert typing.array_dtype_is(obj, "integer")


@pytest.mark.parametrize(
    "obj", array_like_dfloat + array_like_dbool + array_like_dinfinite + array_like_dnone + array_like_dfinite
)
def test_array_dtype_is_int_false(obj):
    assert not typing.array_dtype_is(obj, "integer")


@pytest.mark.parametrize("obj", array_like_dempty + array_like_dbool)
def test_array_dtype_is_bool_true(obj):
    assert typing.array_dtype_is(obj, "boolean")


@pytest.mark.parametrize(
    "obj", array_like_dfloat + array_like_dint + array_like_dinfinite + array_like_dnone + array_like_dfinite
)
def test_array_dtype_is_bool_false(obj):
    assert not typing.array_dtype_is(obj, "boolean")


@pytest.mark.parametrize("obj", array_like_dempty + array_like_dfloat + array_like_dint + array_like_dinfinite)
def test_array_dtype_is_numeric_true(obj):
    assert typing.array_dtype_is(obj, "numeric")


@pytest.mark.parametrize("obj", array_like_dbool + array_like_dnone + array_like_dfinite)
def test_array_dtype_is_numeric_false(obj):
    assert not typing.array_dtype_is(obj, "numeric")


@pytest.mark.parametrize(
    "obj", array_like_dempty + array_like_dfloat + array_like_dbool + array_like_dint + array_like_dfinite
)
def test_array_dtype_is_finite_true(obj):
    assert typing.array_dtype_is(obj, "finite")


@pytest.mark.parametrize("obj", array_like_dinfinite + array_like_dnone)
def test_array_dtype_is_finite_false(obj):
    assert not typing.array_dtype_is(obj, "finite")
