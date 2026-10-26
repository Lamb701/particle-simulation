import numpy as np
from src.particles import Particle
from src.fields import MirrorField, TokamakField
from src.integrators import boris_integrate
from src.diagnostics import (
    kinetic_energy,
    magnetic_moment,
    classify_orbit,
    convergence_test,
)
from src.fields import ZeroElectricField

PROTON_MASS = 1.6726219e-27
PROTON_CHARGE = 1.6021766e-19


def test_kinetic_energy():
    v = np.array([3.0, 4.0, 0.0])
    np.testing.assert_allclose(kinetic_energy(v, 2.0), 25.0)


def test_magnetic_moment_uniform_b():
    b = lambda x, y, z: np.array([0.0, 0.0, 1.0])
    v = np.array([1e5, 0.0, 0.0])
    mu = magnetic_moment(v, b, 0.0, 0.0, 0.0)
    expected = 0.5 * (1e5) ** 2 / 1.0
    np.testing.assert_allclose(mu, expected, rtol=1e-12)


def test_classify_trapped_in_mirror():
    field = MirrorField(B0=1.0, bm=4.0, L=1.0)
    e = ZeroElectricField()
    v = np.array([5e4, 0.0, 1e4])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([0.01, 0.0, 0.0]), v)
    result = boris_integrate(field, e, p, dt=1e-9, n_steps=50000)
    label = classify_orbit(result["trajectory"], field)
    assert label == "trapped"


def test_classify_passing_in_tokamak():
    field = TokamakField(B0=5.0, R0=10.0, a=1.0, q0=1.0, q1=0.0)
    e = ZeroElectricField()
    v = np.array([0.0, 0.0, 5e5])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.2, 0.0, 0.0]), v)
    result = boris_integrate(field, e, p, dt=1e-10, n_steps=3000)
    label = classify_orbit(result["trajectory"], field)
    assert label == "passing"


def test_convergence_second_order():
    field = MirrorField(B0=1.0, bm=2.0, L=1.0)
    e = ZeroElectricField()
    v = np.array([1e5, 0.0, 5e4])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([0.01, 0.0, 0.0]), v)
    dt_list = [2e-10, 4e-10, 8e-10]
    result = convergence_test(field, e, p, dt_ref=1e-10, n_steps_ref=8000, dt_list=dt_list)
    slopes = np.polyfit(np.log(result["dt_list"]), np.log(result["errors"]), 1)[0]
    assert 1.8 < slopes < 2.2
