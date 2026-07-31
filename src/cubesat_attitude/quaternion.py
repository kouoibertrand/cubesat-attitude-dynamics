"""Quaternion utilities following the project's conventions.

Quaternions are represented as numpy arrays of shape (4,) with the scalar
component first: [q0, q1, q2, q3].
"""
from __future__ import annotations

import numpy as np
from typing import Sequence


def as_array(q: Sequence[float]) -> np.ndarray:
	return np.asarray(q, dtype=float).reshape(4)


def normalize(q: Sequence[float]) -> np.ndarray:
	q = as_array(q)
	n = np.linalg.norm(q)
	if n == 0:
		raise ValueError("zero quaternion cannot be normalized")
	return q / n


def conjugate(q: Sequence[float]) -> np.ndarray:
	q = as_array(q)
	return np.array([q[0], -q[1], -q[2], -q[3]])


def hamilton_product(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
	a = as_array(a)
	b = as_array(b)
	w1, x1, y1, z1 = a
	w2, x2, y2, z2 = b
	w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
	x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
	y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
	z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
	return np.array([w, x, y, z])


def to_rotation_matrix(q: Sequence[float]) -> np.ndarray:
	q = normalize(q)
	w, x, y, z = q
	R = np.array(
		[
			[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
			[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
			[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
		]
	)
	return R


def kinematic_derivative(q: Sequence[float], omega: Sequence[float]) -> np.ndarray:
	"""Compute quaternion derivative given body-frame angular velocity.

	qdot = 0.5 * q ⊗ [0, omega]
	"""
	q = as_array(q)
	omega = np.asarray(omega, dtype=float).reshape(3)
	omega_quat = np.concatenate(([0.0], omega))
	return 0.5 * hamilton_product(q, omega_quat)


__all__ = [
	"as_array",
	"normalize",
	"conjugate",
	"hamilton_product",
	"to_rotation_matrix",
	"kinematic_derivative",
]

