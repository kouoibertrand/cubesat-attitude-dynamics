import numpy as np
from cubesat_attitude import rotational_kinetic_energy, angular_momentum, euler_body_acceleration


def test_angular_momentum_and_energy_diagonal():
	J = np.diag([0.02, 0.03, 0.04])
	omega = np.array([1.0, 2.0, -0.5])
	H = angular_momentum(J, omega)
	assert np.allclose(H, J @ omega)
	E = rotational_kinetic_energy(J, omega)
	assert E > 0


def test_euler_accel_torque_free():
	J = np.diag([0.02, 0.03, 0.04])
	omega = np.array([0.1, 0.2, 0.3])
	tau = np.zeros(3)
	adot = euler_body_acceleration(J, omega, tau)
	# dimension check and finite
	assert adot.shape == (3,) and np.all(np.isfinite(adot))
