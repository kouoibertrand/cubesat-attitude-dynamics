"""Example: torque-free rotation integration and basic diagnostics."""
import numpy as np

from cubesat_attitude import integrate


def main():
	# simple diagonal inertia
	J = np.diag([0.02, 0.03, 0.04])
	# initial quaternion (identity) and angular velocity
	q0 = np.array([1.0, 0.0, 0.0, 0.0])
	omega0 = np.array([1.0, 0.5, 0.2])
	state0 = np.concatenate((q0, omega0))
	t_span = (0.0, 10.0)
	t_eval = np.linspace(0, 10, 201)
	sol = integrate(state0, J, t_span, t_eval=t_eval)
	qs = sol.y[:4].T
	omegas = sol.y[4:].T
	# print basic invariants
	from cubesat_attitude.diagnostics import conservation_errors

	d = conservation_errors(J, qs, omegas)
	print("Quaternion norm: min=", d["qnorms"].min(), "max=", d["qnorms"].max())
	print("Energy: min=", d["energies"].min(), "max=", d["energies"].max())
	print("Angular momentum norm: min=", d["Hnorm"].min(), "max=", d["Hnorm"].max())


if __name__ == "__main__":
	main()

