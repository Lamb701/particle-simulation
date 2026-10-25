import numpy as np
from src.particles import Particle


def _as_float_array(v):
    return np.asarray(v, dtype=float)


def boris_step(field_b, field_e, particle, dt):
    q = particle.charge
    m = particle.mass
    x = particle.position.copy()
    v = particle.velocity.copy()

    E = _as_float_array(field_e(*x))
    B = _as_float_array(field_b(*x))

    q_over_m = q / m
    half_dt_q_over_m = 0.5 * dt * q_over_m

    v_minus = v + half_dt_q_over_m * E

    t = half_dt_q_over_m * B
    s = 2.0 * t / (1.0 + np.dot(t, t))

    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    v_new = v_plus + half_dt_q_over_m * E
    x_new = x + dt * v_new

    return Particle(charge=q, mass=m, position=x_new, velocity=v_new)


def boris_integrate(field_b, field_e, particle, dt, n_steps):
    trajectory = np.zeros((n_steps + 1, 3))
    velocity = np.zeros((n_steps + 1, 3))
    times = np.zeros(n_steps + 1)

    p = particle.copy()
    trajectory[0] = p.position
    velocity[0] = p.velocity

    for i in range(n_steps):
        p = boris_step(field_b, field_e, p, dt)
        trajectory[i + 1] = p.position
        velocity[i + 1] = p.velocity
        times[i + 1] = times[i] + dt

    return {"trajectory": trajectory, "velocity": velocity, "times": times}
