import numpy as np
import pytest
from src.particles import Particle
from src.fields import TokamakField, MirrorField, ZeroElectricField
from src.integrators import boris_step, boris_integrate

PROTON_MASS = 1.6726219e-27
PROTON_CHARGE = 1.6021766e-19

def test_boris_zero_b_zero_e():
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.zeros(3), np.array([1e4, 0.0, 0.0]))
    b = lambda x, y, z: np.zeros(3)
    e = ZeroElectricField()
    p2 = boris_step(b, e, p, 1e-6)
    np.testing.assert_allclose(p2.position, np.array([1e-2, 0.0, 0.0]), rtol=1e-12)
    np.testing.assert_allclose(p2.velocity, p.velocity, rtol=1e-12)

def test_boris_uniform_b_gyration():
    b = lambda x, y, z: np.array([0.0, 0.0, 1.0])
    e = ZeroElectricField()
    v_perp = 1e5
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.zeros(3), np.array([v_perp, 0.0, 0.0]))
    dt = 1e-9
    n_steps = 1000
    result = boris_integrate(b, e, p, dt, n_steps)
    traj = result["trajectory"]
    omega_c = PROTON_CHARGE * 1.0 / PROTON_MASS
    T = 2 * np.pi / omega_c
    # After one gyro-period energy should be conserved
    idx = int(round(T / dt))
    idx = min(idx, n_steps)
    E0 = p.kinetic_energy
    E1 = 0.5 * PROTON_MASS * np.sum(result["velocity"][idx] ** 2)
    assert abs((E1 - E0) / E0) < 1e-6

def test_boris_energy_conservation_mirror():
    field = MirrorField(B0=1.0, bm=4.0, L=1.0)
    e = ZeroElectricField()
    v = np.array([1e5, 0.0, 5e4])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([0.01, 0.0, 0.0]), v)
    result = boris_integrate(field, e, p, dt=1e-9, n_steps=2000)
    energies = 0.5 * PROTON_MASS * np.sum(result["velocity"] ** 2, axis=1)
    rel_err = np.max(np.abs((energies - energies[0]) / energies[0]))
    assert rel_err < 1e-8
