"""Complete rotational-state dynamics."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kinematics import quaternion_derivative
from .rigid_body import angular_acceleration


def state_derivative(
    time: float,
    state: ArrayLike,
    inertia: ArrayLike,
    torque_body: ArrayLike,
) -> NDArray[np.float64]:
    """Compute the time derivative of the rotational state.

    Parameters
    ----------
    time:
        Current time in seconds. Included for compatibility with ODE solvers.
        It is not used when the body-frame torque is constant.
    state:
        Rotational state stored as
        [q0, q1, q2, q3, omega_x, omega_y, omega_z].
        The quaternion represents the body-to-inertial rotation.
        Angular velocity is expressed in the body frame, in rad/s.
    inertia:
        Inertia tensor about the center of mass, expressed in the body frame,
        stored as a 3x3 matrix, in kg⋅m².
    torque_body:
        External torque expressed in the body frame, stored as
        [tau_x, tau_y, tau_z], in N⋅m.

    Returns
    -------
    numpy.ndarray
        State derivative stored as
        [q0_dot, q1_dot, q2_dot, q3_dot,
        omega_x_dot, omega_y_dot, omega_z_dot],
        with shape (7,).

    Raises
    ------
    ValueError
        If state does not have shape (7,). The inertia tensor and torque
        vector are validated by the underlying rigid-body functions.
    """
    state_array = np.asarray(state, dtype=float)

    if state_array.shape != (7,):
        raise ValueError("State vector must have shape (7,).")

    quaternion = state_array[:4]
    omega_body = state_array[4:]

    q_dot = quaternion_derivative(
        quaternion,
        omega_body,
    )

    omega_dot = angular_acceleration(
        inertia,
        omega_body,
        torque_body,
    )

    return np.concatenate((q_dot, omega_dot))
