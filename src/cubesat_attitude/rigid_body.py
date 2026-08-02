"""Rigid-body rotational dynamics."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_inertia_tensor(inertia: ArrayLike) -> NDArray[np.float64]:
    """Validate and return a rigid-body inertia tensor.

    Parameters
    ----------
    inertia:
        Inertia tensor about the center of mass, expressed in the body frame,
        stored as a 3x3 matrix, in kg⋅m².

    Returns
    -------
    numpy.ndarray
        Validated inertia tensor of shape (3, 3), in kg⋅m².

    Raises
    ------
    ValueError
        If the inertia tensor does not have shape (3, 3), is not symmetric,
        or is not positive definite.
    """
    inertia_array = np.asarray(inertia, dtype=float)

    if inertia_array.shape != (3, 3):
        raise ValueError("Inertia tensor must have shape (3, 3).")

    if not np.allclose(inertia_array, inertia_array.T):
        raise ValueError("Inertia tensor must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(inertia_array)

    if np.any(eigenvalues <= 0.0):
        raise ValueError("Inertia tensor must be positive definite.")

    return inertia_array


def validate_body_vector(
    vector: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Validate and return a three-dimensional body-frame vector.

    Parameters
    ----------
    vector:
        Vector-like object expected to have shape (3,).
    name:
        Name used in the error message.

    Returns
    -------
    numpy.ndarray
        Validated vector of shape (3,).
    """
    vector_array = np.asarray(vector, dtype=float)

    if vector_array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")

    return vector_array


def angular_momentum(
    inertia: ArrayLike,
    omega_body: ArrayLike,
) -> NDArray[np.float64]:
    """Compute angular momentum expressed in the body frame.

    Parameters
    ----------
    inertia:
        Inertia tensor about the center of mass, expressed in the body frame,
        stored as a 3x3 matrix, in kg⋅m².
    omega_body:
        Angular velocity expressed in the body frame, stored as
        [omega_x, omega_y, omega_z], in rad/s.

    Returns
    -------
    numpy.ndarray
        Angular momentum expressed in the body frame, with shape (3,),
        in kg⋅m²/s.
    """
    inertia_array = validate_inertia_tensor(inertia)
    omega_array = validate_body_vector(
        omega_body,
        name="Angular velocity",
    )

    return inertia_array @ omega_array


def rotational_kinetic_energy(
    inertia: ArrayLike,
    omega_body: ArrayLike,
) -> float:
    """Compute the rotational kinetic energy of a rigid body.

    Parameters
    ----------
    inertia:
        Inertia tensor about the center of mass, expressed in the body frame,
        stored as a 3x3 matrix, in kg⋅m².
    omega_body:
        Angular velocity expressed in the body frame, stored as
        [omega_x, omega_y, omega_z], in rad/s.

    Returns
    -------
    float
        Rotational kinetic energy in joules.
    """
    inertia_array = validate_inertia_tensor(inertia)
    omega_array = validate_body_vector(
        omega_body,
        name="Angular velocity",
    )

    energy = 0.5 * omega_array @ inertia_array @ omega_array

    return float(energy)


def angular_acceleration(
    inertia: ArrayLike,
    omega_body: ArrayLike,
    torque_body: ArrayLike,
) -> NDArray[np.float64]:
    """Compute angular acceleration from Euler's rigid-body equations.

    Parameters
    ----------
    inertia:
        Inertia tensor about the center of mass, expressed in the body frame,
        stored as a 3x3 matrix, in kg⋅m².
    omega_body:
        Angular velocity expressed in the body frame, stored as
        [omega_x, omega_y, omega_z], in rad/s.
    torque_body:
        External torque expressed in the body frame, stored as
        [tau_x, tau_y, tau_z], in N⋅m.

    Returns
    -------
    numpy.ndarray
        Angular acceleration expressed in the body frame, with shape (3,),
        in rad/s².
    """
    inertia_array = validate_inertia_tensor(inertia)
    omega_array = validate_body_vector(
        omega_body,
        name="Angular velocity",
    )
    torque_array = validate_body_vector(
        torque_body,
        name="Torque",
    )

    angular_momentum_body = inertia_array @ omega_array

    gyroscopic_term = np.cross(
        omega_array,
        angular_momentum_body,
    )

    right_hand_side = torque_array - gyroscopic_term

    return np.linalg.solve(
        inertia_array,
        right_hand_side,
    )
