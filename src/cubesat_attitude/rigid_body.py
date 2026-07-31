"""Rigid-body dynamics utilities.

Provides functions for inertia validation, angular momentum, kinetic energy
and Euler rigid-body equations for torque-free or forced motion.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence


def validate_inertia(J: Sequence[Sequence[float]]) -> np.ndarray:
	J = np.asarray(J, dtype=float)
	if J.shape != (3, 3):
		raise ValueError("inertia matrix must be 3x3")
	if not np.allclose(J, J.T, atol=1e-12):
		raise ValueError("inertia matrix must be symmetric")
	# Positive definite check (principal moments positive)
	eig = np.linalg.eigvalsh(J)
	if np.any(eig <= 0):
		raise ValueError("inertia matrix must be positive definite")
	return J


def angular_momentum(J: Sequence[Sequence[float]], omega: Sequence[float]) -> np.ndarray:
	J = validate_inertia(J)
	omega = np.asarray(omega, dtype=float).reshape(3)
	return J @ omega


def rotational_kinetic_energy(J: Sequence[Sequence[float]], omega: Sequence[float]) -> float:
	omega = np.asarray(omega, dtype=float).reshape(3)
	J = validate_inertia(J)
	return 0.5 * float(omega @ (J @ omega))


def euler_body_acceleration(J: Sequence[Sequence[float]], omega: Sequence[float], tau: Sequence[float]) -> np.ndarray:
	"""Return angular acceleration in the body frame.

	Euler's equation:
	  J * omega_dot + omega x (J * omega) = tau
	so: omega_dot = J^{-1} (tau - omega x J omega)
	"""
	J = validate_inertia(J)
	omega = np.asarray(omega, dtype=float).reshape(3)
	tau = np.asarray(tau, dtype=float).reshape(3)
	H = J @ omega
	omega_cross_H = np.cross(omega, H)
	omega_dot = np.linalg.solve(J, tau - omega_cross_H)
	return omega_dot


__all__ = [
	"validate_inertia",
	"angular_momentum",
	"rotational_kinetic_energy",
	"euler_body_acceleration",
]
