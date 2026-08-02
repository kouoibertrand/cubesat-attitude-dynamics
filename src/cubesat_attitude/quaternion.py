"""Elementary quaternion operations."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize_quaternion(q: ArrayLike) -> NDArray[np.float64]:
    """Return a normalized copy of a quaternion.

    Parameters
    ----------
    q:
        Quaternion stored as [q0, q1, q2, q3], with the scalar component first.

    Returns
    -------
    numpy.ndarray
        Quaternion of shape (4,) and unit norm.

    Raises
    ------
    ValueError
        If the input does not contain exactly four components or if its norm
        is zero.
    """
    q_array = np.asarray(q, dtype=float)

    if q_array.shape != (4,):
        raise ValueError("A quaternion must have shape (4,).")

    norm = np.linalg.norm(q_array)
    if norm == 0:
        raise ValueError("Cannot normalize a quaternion with zero norm.")

    return q_array / norm


def conjugate_quaternion(q: ArrayLike) -> NDArray[np.float64]:
    """Return the conjugate of a quaternion.

    Parameters
    ----------
    q:
        Quaternion stored as [q0, q1, q2, q3], with the scalar component first.

    Returns
    -------
    numpy.ndarray
        Conjugate quaternion of shape (4,).

    Raises
    ------
    ValueError
        If the input does not contain exactly four components.
    """

    q_array = np.asarray(q, dtype=float)

    if q_array.shape != (4,):
        raise ValueError("A quaternion must have shape (4,).")

    return np.array([q_array[0], -q_array[1], -q_array[2], -q_array[3]])


def hamilton_product(
    q1: ArrayLike,
    q2: ArrayLike,
) -> NDArray[np.float64]:
    """Return the Hamilton product of two quaternions.

    Parameters
    ----------
    q1:
        First quaternion stored as [q0, q1, q2, q3], with the scalar component first.
    q2:
        Second quaternion stored as [q0, q1, q2, q3], with the scalar component first.

    Returns
    -------
    numpy.ndarray
        Hamilton product q1 ⊗ q2, with shape (4,).

    Raises
    ------
    ValueError
        If either input does not have shape (4,).
    """

    q1_array = np.asarray(q1, dtype=float)
    q2_array = np.asarray(q2, dtype=float)

    if q1_array.shape != (4,) or q2_array.shape != (4,):
        raise ValueError("Both quaternions must have shape (4,).")

    w1, x1, y1, z1 = q1_array
    w2, x2, y2, z2 = q2_array

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def inverse_quaternion(q: ArrayLike) -> NDArray[np.float64]:
    """Return the inverse of a quaternion.

    Parameters
    ----------
    q:
        Quaternion stored as [q0, q1, q2, q3], with the scalar component first.

    Returns
    -------
    numpy.ndarray
        Inverse quaternion of shape (4,).

    Raises
    ------
    ValueError
        If the input does not contain exactly four components or if its norm
        is zero.
    """
    q_array = np.asarray(q, dtype=float)

    if q_array.shape != (4,):
        raise ValueError("A quaternion must have shape (4,).")

    norm_squared = np.linalg.norm(q_array) ** 2
    if norm_squared == 0:
        raise ValueError("Cannot invert a quaternion with zero norm.")

    q_conjugate = conjugate_quaternion(q_array)
    return q_conjugate / norm_squared


def rotate_vector(
    q: ArrayLike,
    vector: ArrayLike,
) -> NDArray[np.float64]:
    """Rotate a body-frame vector into the inertial frame.

    Parameters
    ----------
    q:
        Body-to-inertial quaternion stored as [q0, q1, q2, q3],
        with the scalar component first. The quaternion does not need to be normalized; any nonzero scalar multiple
        represents the same rotation.
    vector:
        Vector expressed in the body frame, stored as [v_x, v_y, v_z].

    Returns
    -------
    numpy.ndarray
        The same physical vector expressed in the inertial frame,
        with shape (3,).

    Raises
    ------
    ValueError
        If q does not have shape (4,), if vector does not have shape (3,),
        or if q has zero norm.

    Notes
    -----
    The rotation is computed as:

        vector_inertial = q ⊗ vector_body ⊗ q_inverse
    """
    q_array = np.asarray(q, dtype=float)
    vector_array = np.asarray(vector, dtype=float)

    if q_array.shape != (4,):
        raise ValueError("A quaternion must have shape (4,).")

    if vector_array.shape != (3,):
        raise ValueError("A vector must have shape (3,).")

    vector_quaternion = np.array([0.0, *vector_array])
    q_inverse = inverse_quaternion(q_array)

    rotated_quaternion = hamilton_product(
        hamilton_product(q_array, vector_quaternion),
        q_inverse,
    )

    return rotated_quaternion[1:]
