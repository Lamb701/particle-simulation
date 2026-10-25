import numpy as np
from src.particles import Particle

def test_particle_creation():
    p = Particle(
        charge=1.602e-19,
        mass=1.673e-27,
        position=np.array([1.0, 0.0, 0.0]),
        velocity=np.array([1e5, 2e5, 3e5]),
    )
    assert p.charge == 1.602e-19
    assert p.mass == 1.673e-27
    np.testing.assert_allclose(p.position, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(p.velocity, np.array([1e5, 2e5, 3e5]))

def test_particle_copy_is_independent():
    p = Particle(1.0, 2.0, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    q = p.copy()
    q.position[0] = 99.0
    assert p.position[0] == 0.0
    q.velocity[0] = 42.0
    assert p.velocity[0] == 1.0

def test_particle_kinetic_energy():
    p = Particle(1.0, 2.0, np.zeros(3), np.array([3.0, 4.0, 0.0]))
    np.testing.assert_allclose(p.speed, 5.0)
    np.testing.assert_allclose(p.kinetic_energy, 0.5 * 2.0 * 25.0)
