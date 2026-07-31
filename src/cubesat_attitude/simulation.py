"""Simulation utilities: state derivative and integrator.

State vector is [q0, q1, q2, q3, omega_x, omega_y, omega_z].
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Sequence, Optional
from scipy.integrate import solve_ivp

from .quaternion import kinematic_derivative, normalize


State = np.ndarray


def state_derivative(t: float, state: Sequence[float], J: Sequence[Sequence[float]], torque_func: Optional[Callable[[float, Sequence[float]], Sequence[float]]] = None) -> State:
	state = np.asarray(state, dtype=float).reshape(7)
	q = state[:4]
	omega = state[4:]
	tau = np.zeros(3)
	if torque_func is not None:
		tau = np.asarray(torque_func(t, state), dtype=float).reshape(3)
	qdot = kinematic_derivative(q, omega)
	from .rigid_body import euler_body_acceleration

	omega_dot = euler_body_acceleration(J, omega, tau)
	return np.concatenate((qdot, omega_dot))


def integrate(initial_state: Sequence[float], J: Sequence[Sequence[float]], t_span: Sequence[float], t_eval: Optional[Sequence[float]] = None, torque_func: Optional[Callable[[float, Sequence[float]], Sequence[float]]] = None, rtol: float = 1e-9, atol: float = 1e-12):
	initial_state = np.asarray(initial_state, dtype=float).reshape(7)
	# ensure quaternion normalized
	initial_state[:4] = normalize(initial_state[:4])
	sol = solve_ivp(lambda t, y: state_derivative(t, y, J, torque_func), t_span, initial_state, t_eval=t_eval, rtol=rtol, atol=atol)
	# normalize quaternions in output
	if sol.y.size:
		ys = sol.y.copy()
		for i in range(ys.shape[1]):
			ys[:4, i] = normalize(ys[:4, i])
		sol.y = ys
	return sol


__all__ = ["state_derivative", "integrate"]
