import numpy as np
import pytest

from cubesat_attitude.state import state_derivative


def test_state_derivative_known_principal_axis_rotation() -> None:
    state = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
        ]
    )
    inertia = np.diag([2.0, 3.0, 5.0])
    torque_body = np.zeros(3)

    result = state_derivative(
        0.0,
        state,
        inertia,
        torque_body,
    )

    expected = np.array(
        [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    np.testing.assert_allclose(result, expected)


def test_state_derivative_is_zero_for_rest_state() -> None:
    state = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    inertia = np.diag([2.0, 3.0, 5.0])
    torque_body = np.zeros(3)

    result = state_derivative(
        0.0,
        state,
        inertia,
        torque_body,
    )

    np.testing.assert_allclose(result, np.zeros(7))


def test_state_derivative_known_torque_free_result() -> None:
    state = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
        ]
    )
    inertia = np.diag([2.0, 3.0, 5.0])
    torque_body = np.zeros(3)

    result = state_derivative(
        0.0,
        state,
        inertia,
        torque_body,
    )

    expected = np.array(
        [
            0.0,
            0.5,
            0.5,
            0.5,
            -1.0,
            1.0,
            -0.2,
        ]
    )

    np.testing.assert_allclose(result, expected)


def test_state_derivative_rejects_invalid_state_shape() -> None:
    state = np.zeros(6)
    inertia = np.diag([2.0, 3.0, 5.0])
    torque_body = np.zeros(3)

    with pytest.raises(ValueError):
        state_derivative(
            0.0,
            state,
            inertia,
            torque_body,
        )


def test_state_derivative_does_not_modify_inputs() -> None:
    state = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
        ]
    )
    inertia = np.diag([2.0, 3.0, 5.0])
    torque_body = np.array([0.1, 0.2, 0.3])

    state_before = state.copy()
    inertia_before = inertia.copy()
    torque_before = torque_body.copy()

    state_derivative(
        0.0,
        state,
        inertia,
        torque_body,
    )

    np.testing.assert_array_equal(state, state_before)
    np.testing.assert_array_equal(inertia, inertia_before)
    np.testing.assert_array_equal(torque_body, torque_before)
