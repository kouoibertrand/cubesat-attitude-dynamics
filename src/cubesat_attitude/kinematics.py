import numpy as np
from numpy.typing import ArrayLike, NDArray

from .quaternion import (
    hamilton_product,
)


def quaternion_derivative(
    q: ArrayLike,
    omega_body: ArrayLike,
) -> NDArray[np.float64]:
    """Compute the quaternion time derivative from body angular velocity.

    Parameters
    ----------
    q:
        Body-to-inertial quaternion stored as [q0, q1, q2, q3],
        with the scalar component first.
    omega_body:
        Angular velocity expressed in the body frame, stored as
        [omega_x, omega_y, omega_z], in rad/s.

    Returns
    -------
    numpy.ndarray
        Quaternion time derivative of shape (4,), in s^-1.

    Raises
    ------
    ValueError
        If q does not have shape (4,) or omega_body does not have shape (3,).

    Notes
    -----
    Uses the convention:

        q_dot = 0.5 * q ⊗ [0, omega_body]
    """
    q_array = np.asarray(q, dtype=float)
    omega_array = np.asarray(omega_body, dtype=float)

    if q_array.shape != (4,):
        raise ValueError("A quaternion must have shape (4,).")

    if omega_array.shape != (3,):
        raise ValueError("Angular velocity must have shape (3,).")

    omega_quaternion = np.array([0.0, *omega_array])

    return 0.5 * hamilton_product(q_array, omega_quaternion)
