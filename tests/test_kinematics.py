import numpy as np
import pytest

from cubesat_attitude.kinematics import quaternion_derivative


def test_quaternion_derivative_known_rotation_about_z() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega_body = np.array([0.0, 0.0, 2.0])

    result = quaternion_derivative(q, omega_body)

    expected = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(result, expected)


def test_quaternion_derivative_is_zero_for_zero_angular_velocity() -> None:
    q = np.array([0.5, 0.5, 0.5, 0.5])
    omega_body = np.zeros(3)

    result = quaternion_derivative(q, omega_body)

    expected = np.zeros(4)
    np.testing.assert_array_equal(result, expected)


def test_quaternion_derivative_known_general_result() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])
    omega_body = np.array([5.0, 6.0, 7.0])

    result = quaternion_derivative(q, omega_body)

    expected = np.array([-28.0, 1.0, 6.0, 2.0])
    np.testing.assert_allclose(result, expected)


def test_quaternion_derivative_is_tangent_for_unit_quaternion() -> None:
    q = np.array([0.5, 0.5, 0.5, 0.5])
    omega_body = np.array([1.0, 2.0, 3.0])

    q_dot = quaternion_derivative(q, omega_body)

    np.testing.assert_allclose(
        np.dot(q, q_dot),
        0.0,
        atol=1e-12,
    )


def test_quaternion_derivative_raises_for_invalid_quaternion_shape() -> None:
    q = np.array([1.0, 0.0, 0.0])
    omega_body = np.array([0.0, 0.0, 1.0])

    with pytest.raises(ValueError):
        quaternion_derivative(q, omega_body)


def test_quaternion_derivative_raises_for_invalid_angular_velocity_shape() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega_body = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        quaternion_derivative(q, omega_body)


def test_quaternion_derivative_does_not_modify_inputs() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])
    omega_body = np.array([5.0, 6.0, 7.0])

    q_before = q.copy()
    omega_before = omega_body.copy()

    quaternion_derivative(q, omega_body)

    np.testing.assert_array_equal(q, q_before)
    np.testing.assert_array_equal(omega_body, omega_before)


def test_quaternion_derivative_norm_for_unit_quaternion() -> None:
    q = np.array([0.5, 0.5, 0.5, 0.5])
    omega_body = np.array([1.0, 2.0, 3.0])

    q_dot = quaternion_derivative(q, omega_body)

    np.testing.assert_allclose(
        np.linalg.norm(q_dot),
        0.5 * np.linalg.norm(omega_body),
    )
