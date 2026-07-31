"""Diagnostics helpers for conservation checks and numerical metrics."""
from __future__ import annotations

import numpy as np
from typing import Sequence

from .rigid_body import rotational_kinetic_energy, angular_momentum


def quaternion_norm(q: Sequence[float]) -> float:
    q = np.asarray(q, dtype=float).reshape(4)
    return float(np.linalg.norm(q))


def energy(J: Sequence[Sequence[float]], omega: Sequence[float]) -> float:
    return rotational_kinetic_energy(J, omega)


def angular_momentum_vec(J: Sequence[Sequence[float]], omega: Sequence[float]) -> np.ndarray:
    return angular_momentum(J, omega)


def conservation_errors(J: Sequence[Sequence[float]], qs, omegas):
    """Compute time series of diagnostics for a trajectory.

    qs: array-like shape (N,4) quaternions
    omegas: array-like shape (N,3)
    returns: dict with norms and relative energy error
    """
    qs = np.asarray(qs)
    omegas = np.asarray(omegas)
    n = qs.shape[0]
    qnorms = np.linalg.norm(qs, axis=1)
    energies = np.array([energy(J, omegas[i]) for i in range(n)])
    Hs = np.array([angular_momentum_vec(J, omegas[i]) for i in range(n)])
    Hnorm = np.linalg.norm(Hs, axis=1)
    return {"qnorms": qnorms, "energies": energies, "Hnorm": Hnorm}


__all__ = ["quaternion_norm", "energy", "angular_momentum_vec", "conservation_errors"]
