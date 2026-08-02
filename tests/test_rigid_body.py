import numpy as np
import pytest

from cubesat_attitude.rigid_body import (
    angular_acceleration,
    angular_momentum,
    rotational_kinetic_energy,
    validate_inertia_tensor,
)


def test_validate_inertia_tensor_accepts_valid_tensor() -> None:
    inertia = np.diag([1.0, 2.0, 3.0])

    result = validate_inertia_tensor(inertia)

    np.testing.assert_array_equal(result, inertia)


def test_validate_inertia_tensor_rejects_invalid_shape() -> None:
    inertia = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
        ]
    )

    with pytest.raises(ValueError):
        validate_inertia_tensor(inertia)


def test_validate_inertia_tensor_rejects_non_symmetric_tensor() -> None:
    inertia = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )

    with pytest.raises(ValueError):
        validate_inertia_tensor(inertia)


def test_validate_inertia_tensor_rejects_non_positive_definite_tensor() -> None:
    inertia = np.diag([1.0, -2.0, 3.0])

    with pytest.raises(ValueError):
        validate_inertia_tensor(inertia)


def test_validate_inertia_tensor_rejects_singular_tensor() -> None:
    inertia = np.diag([1.0, 0.0, 3.0])

    with pytest.raises(ValueError):
        validate_inertia_tensor(inertia)


def test_validate_inertia_tensor_does_not_modify_input() -> None:
    inertia = np.diag([1.0, 2.0, 3.0])
    inertia_before = inertia.copy()

    validate_inertia_tensor(inertia)

    np.testing.assert_array_equal(inertia, inertia_before)


def test_angular_momentum_known_result() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.array([1.0, 1.0, 1.0])

    result = angular_momentum(inertia, omega_body)

    expected = np.array([2.0, 3.0, 5.0])
    np.testing.assert_allclose(result, expected)


def test_angular_momentum_rejects_invalid_angular_velocity_shape() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.array([1.0, 1.0])

    with pytest.raises(ValueError):
        angular_momentum(inertia, omega_body)


def test_angular_momentum_does_not_modify_inputs() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.array([1.0, 1.0, 1.0])

    inertia_before = inertia.copy()
    omega_before = omega_body.copy()

    angular_momentum(inertia, omega_body)

    np.testing.assert_array_equal(inertia, inertia_before)
    np.testing.assert_array_equal(omega_body, omega_before)


def test_rotational_kinetic_energy_known_result() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.array([1.0, 1.0, 1.0])

    result = rotational_kinetic_energy(inertia, omega_body)

    expected = 5.0
    np.testing.assert_allclose(result, expected)


def test_rotational_kinetic_energy_is_zero_for_zero_angular_velocity() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.zeros(3)

    result = rotational_kinetic_energy(inertia, omega_body)

    np.testing.assert_allclose(result, 0.0)


def test_rotational_kinetic_energy_rejects_invalid_angular_velocity_shape() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.array([1.0, 1.0])

    with pytest.raises(ValueError):
        rotational_kinetic_energy(inertia, omega_body)


def test_rotational_kinetic_energy_does_not_modify_inputs() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.array([1.0, 1.0, 1.0])

    inertia_before = inertia.copy()
    omega_before = omega_body.copy()

    rotational_kinetic_energy(inertia, omega_body)

    np.testing.assert_array_equal(inertia, inertia_before)
    np.testing.assert_array_equal(omega_body, omega_before)


def test_angular_acceleration_is_zero_for_zero_state_and_zero_torque() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.zeros(3)
    torque_body = np.zeros(3)

    result = angular_acceleration(inertia, omega_body, torque_body)

    np.testing.assert_allclose(result, np.zeros(3))


def test_angular_acceleration_is_zero_for_principal_axis_rotation() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.array([1.0, 0.0, 0.0])
    torque_body = np.zeros(3)

    result = angular_acceleration(inertia, omega_body, torque_body)

    np.testing.assert_allclose(result, np.zeros(3))


def test_angular_acceleration_known_torque_free_result() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.array([1.0, 1.0, 1.0])
    torque_body = np.zeros(3)

    result = angular_acceleration(inertia, omega_body, torque_body)

    expected = np.array([-1.0, 1.0, -0.2])
    np.testing.assert_allclose(result, expected)


def test_angular_acceleration_known_result_with_torque() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.zeros(3)
    torque_body = np.array([2.0, 3.0, 5.0])

    result = angular_acceleration(inertia, omega_body, torque_body)

    expected = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(result, expected)


def test_angular_acceleration_rejects_invalid_angular_velocity_shape() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.array([1.0, 1.0])
    torque_body = np.zeros(3)

    with pytest.raises(ValueError):
        angular_acceleration(inertia, omega_body, torque_body)


def test_angular_acceleration_rejects_invalid_torque_shape() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.zeros(3)
    torque_body = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        angular_acceleration(inertia, omega_body, torque_body)


def test_angular_acceleration_does_not_modify_inputs() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.array([1.0, 1.0, 1.0])
    torque_body = np.array([1.0, 2.0, 3.0])

    inertia_before = inertia.copy()
    omega_before = omega_body.copy()
    torque_before = torque_body.copy()

    angular_acceleration(inertia, omega_body, torque_body)

    np.testing.assert_array_equal(inertia, inertia_before)
    np.testing.assert_array_equal(omega_body, omega_before)
    np.testing.assert_array_equal(torque_body, torque_before)
