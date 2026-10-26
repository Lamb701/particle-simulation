import numpy as np
import pytest
from src.particles import Particle
from src.fields import TokamakField, MirrorField, ZeroElectricField
from src.integrators import boris_step, boris_integrate, guiding_center_integrate, guiding_center_initial

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

def test_guiding_center_mirror_bounces():
    field = MirrorField(B0=1.0, bm=4.0, L=1.0)
    # Use a trapped pitch angle (v_perp > v_parallel) so the guiding center bounces.
    v = np.array([8e4, 0.0, 1e4])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([0.01, 0.0, 0.0]), v)
    result = guiding_center_integrate(field, p, dt=2e-8, n_steps=1500)
    z = result["trajectory"][:, 2]
    assert np.max(z) > 0.05
    assert np.min(z) < -0.05


def test_guiding_center_initial_offset():
    # In a uniform B field, the guiding center should be displaced by one Larmor radius.
    b = lambda x, y, z: np.array([0.0, 0.0, 1.0])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.zeros(3), np.array([1e5, 0.0, 0.0]))
    x_gc, v_parallel, mu = guiding_center_initial(b, p)
    expected_rho = PROTON_MASS * 1e5 / (PROTON_CHARGE * 1.0)
    np.testing.assert_allclose(x_gc, np.array([0.0, -expected_rho, 0.0]), rtol=1e-12)
    np.testing.assert_allclose(v_parallel, 0.0, atol=1e-12)
    np.testing.assert_allclose(mu, 0.5 * (1e5) ** 2 / 1.0, rtol=1e-12)


def test_guiding_center_rk4_convergence():
    # GC with RK4 should be stable and show error growth with increasing dt.
    # For very smooth trajectories the absolute error can be near machine
    # precision, so we check the trend rather than an exact fourth-order slope.
    field = TokamakField(B0=5.0, R0=10.0, a=1.0, q0=1.0, q1=2.0)
    v = np.array([2e5, 0.0, 5e3])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.2, 0.0, 0.0]), v)

    dt_ref = 1e-9
    n_ref = 50000
    ref = guiding_center_integrate(field, p, dt=dt_ref, n_steps=n_ref)
    ref_final = ref["trajectory"][-1]

    dts = [2e-9, 4e-9, 8e-9, 1.6e-8]
    errors = []
    for dt in dts:
        n = int(round(n_ref * dt_ref / dt))
        result = guiding_center_integrate(field, p, dt=dt, n_steps=n)
        errors.append(np.linalg.norm(result["trajectory"][-1] - ref_final))

    slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
    assert slope > 0.0
