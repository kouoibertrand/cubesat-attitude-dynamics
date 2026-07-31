import numpy as np
import pytest

from cubesat_attitude.quaternion import normalize_quaternion


def test_normalize_simple_quaternion() -> None:
    q = np.array([2.0, 0.0, 0.0, 0.0])

    result = normalize_quaternion(q)

    expected = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected)


def test_normalize_general_quaternion() -> None:
    q = np.array([1.0, 2.0, 2.0, 0.0])

    result = normalize_quaternion(q)

    expected = np.array([1 / 3, 2 / 3, 2 / 3, 0.0])
    np.testing.assert_allclose(result, expected)


def test_normalized_quaternion_has_unit_norm() -> None:
    q = np.array([1.0, -4.0, 2.0, 3.0])

    result = normalize_quaternion(q)

    np.testing.assert_allclose(np.linalg.norm(result), 1.0)


def test_zero_quaternion_is_rejected() -> None:
    q = np.zeros(4)

    with pytest.raises(ValueError):
        normalize_quaternion(q)


def test_invalid_shape_is_rejected() -> None:
    q = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        normalize_quaternion(q)


def test_input_is_not_modified() -> None:
    q = np.array([1.0, 2.0, 2.0, 0.0])
    q_before = q.copy()

    normalize_quaternion(q)

    np.testing.assert_array_equal(q, q_before)