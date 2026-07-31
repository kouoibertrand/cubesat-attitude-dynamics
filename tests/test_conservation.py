import numpy as np
from cubesat_attitude import integrate
from cubesat_attitude.diagnostics import conservation_errors


def test_conservation_torque_free():
    J = np.diag([0.02, 0.03, 0.04])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    omega0 = np.array([0.7, -0.3, 0.2])
    state0 = np.concatenate((q0, omega0))
    t_span = (0.0, 5.0)
    t_eval = np.linspace(0, 5, 101)
    sol = integrate(state0, J, t_span, t_eval=t_eval)
    qs = sol.y[:4].T
    omegas = sol.y[4:].T
    d = conservation_errors(J, qs, omegas)
    energies = d["energies"]
    Hnorm = d["Hnorm"]
    # relative variation should be small for a short accurate integration
    assert (energies.max() - energies.min()) / max(abs(energies[0]), 1e-12) < 1e-6
    assert (Hnorm.max() - Hnorm.min()) / max(abs(Hnorm[0]), 1e-12) < 1e-6
