import numpy as np
from cubesat_attitude import normalize, hamilton_product, conjugate


def test_normalize_and_conjugate():
	q = np.array([2.0, 0.0, 0.0, 0.0])
	qn = normalize(q)
	assert np.allclose(np.linalg.norm(qn), 1.0)
	qc = conjugate(qn)
	assert np.allclose(qn[0], qc[0]) and np.allclose(qn[1:], -qc[1:])


def test_hamilton_identity():
	# identity quaternion
	e = np.array([1.0, 0.0, 0.0, 0.0])
	q = np.array([0.3, -0.5, 0.2, 0.1])
	r = hamilton_product(e, q)
	assert np.allclose(r, q)
