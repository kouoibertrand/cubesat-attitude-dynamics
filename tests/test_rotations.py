import numpy as np
import pytest

from cubesat_attitude.quaternion import (
    inverse_quaternion,
    rotate_vector,
)


def test_rotate_vector_with_identity_quaternion() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    vector = np.array([1.0, 2.0, 3.0])

    result = rotate_vector(q, vector)

    np.testing.assert_allclose(result, vector)


def test_rotate_vector_ninety_degrees_about_z() -> None:
    angle = np.pi / 2
    q = np.array(
        [
            np.cos(angle / 2),
            0.0,
            0.0,
            np.sin(angle / 2),
        ]
    )
    vector = np.array([1.0, 0.0, 0.0])

    result = rotate_vector(q, vector)

    expected = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_rotate_vector_preserves_norm() -> None:
    angle = np.pi / 3
    axis = np.array([1.0, 1.0, 1.0])
    axis = axis / np.linalg.norm(axis)

    q = np.array(
        [
            np.cos(angle / 2),
            *(axis * np.sin(angle / 2)),
        ]
    )
    vector = np.array([2.0, -1.0, 4.0])

    result = rotate_vector(q, vector)

    np.testing.assert_allclose(
        np.linalg.norm(result),
        np.linalg.norm(vector),
        atol=1e-12,
    )


def test_rotate_vector_then_inverse_rotation_returns_original() -> None:
    angle = np.pi / 4
    q = np.array(
        [
            np.cos(angle / 2),
            np.sin(angle / 2),
            0.0,
            0.0,
        ]
    )
    vector = np.array([0.0, 1.0, 2.0])

    rotated = rotate_vector(q, vector)
    recovered = rotate_vector(inverse_quaternion(q), rotated)

    np.testing.assert_allclose(recovered, vector, atol=1e-12)


def test_rotate_vector_is_invariant_under_quaternion_sign_change() -> None:
    q = np.array([0.5, 0.5, 0.5, 0.5])
    vector = np.array([1.0, 2.0, 3.0])

    result_positive = rotate_vector(q, vector)
    result_negative = rotate_vector(-q, vector)

    np.testing.assert_allclose(
        result_positive,
        result_negative,
        atol=1e-12,
    )


def test_rotate_vector_is_invariant_under_quaternion_scaling() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])
    vector = np.array([1.0, -2.0, 0.5])

    result = rotate_vector(q, vector)
    scaled_result = rotate_vector(2.0 * q, vector)

    np.testing.assert_allclose(result, scaled_result, atol=1e-12)


def test_rotate_vector_rejects_zero_quaternion() -> None:
    q = np.zeros(4)
    vector = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        rotate_vector(q, vector)


def test_rotate_vector_rejects_invalid_quaternion_shape() -> None:
    q = np.array([1.0, 0.0, 0.0])
    vector = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        rotate_vector(q, vector)


def test_rotate_vector_rejects_invalid_vector_shape() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    vector = np.array([1.0, 0.0])

    with pytest.raises(ValueError):
        rotate_vector(q, vector)


def test_rotate_vector_does_not_modify_inputs() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])
    vector = np.array([1.0, -2.0, 3.0])

    q_before = q.copy()
    vector_before = vector.copy()

    rotate_vector(q, vector)

    np.testing.assert_array_equal(q, q_before)
    np.testing.assert_array_equal(vector, vector_before)