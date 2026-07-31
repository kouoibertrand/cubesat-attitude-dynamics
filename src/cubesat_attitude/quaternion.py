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