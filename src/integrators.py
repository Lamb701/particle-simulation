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
    x_new = x + dt * 0.5 * (v + v_new)

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


def guiding_center_step(field_b, particle, dt, mu):
    x = particle.position.copy()
    v = particle.velocity.copy()
    q = particle.charge
    m = particle.mass

    B = np.asarray(field_b(*x))
    B_mag = np.linalg.norm(B)
    b_hat = B / B_mag

    v_parallel = np.dot(v, b_hat)

    # Magnetic gradient and curvature
    dx = 1e-6
    gradB = np.zeros(3)
    for i, axis in enumerate([(dx, 0, 0), (0, dx, 0), (0, 0, dx)]):
        Bp = np.asarray(field_b(x[0] + axis[0], x[1] + axis[1], x[2] + axis[2]))
        Bm = np.asarray(field_b(x[0] - axis[0], x[1] - axis[1], x[2] - axis[2]))
        gradB[i] = (np.linalg.norm(Bp) - np.linalg.norm(Bm)) / (2 * dx)

    # Curvature vector (b . grad)b
    db = np.zeros((3, 3))
    for i, axis in enumerate([(dx, 0, 0), (0, dx, 0), (0, 0, dx)]):
        Bp = np.asarray(field_b(x[0] + axis[0], x[1] + axis[1], x[2] + axis[2]))
        Bm = np.asarray(field_b(x[0] - axis[0], x[1] - axis[1], x[2] - axis[2]))
        db[:, i] = (Bp / np.linalg.norm(Bp) - Bm / np.linalg.norm(Bm)) / (2 * dx)
    curvature = db @ b_hat

    # mu is the mass-normalized magnetic moment, v_perp^2 / (2 B).
    # The corresponding physical magnetic moment is m * mu.
    a_parallel = -mu * gradB @ b_hat
    v_parallel_new = v_parallel + a_parallel * dt

    # Drifts using the physical magnetic moment m * mu.
    v_grad_curv = (m * mu / q) * np.cross(B, gradB) / (B_mag ** 2) \
                  + (m * v_parallel ** 2 / q) * np.cross(b_hat, curvature) / B_mag

    v_gc = v_parallel_new * b_hat + v_grad_curv
    x_new = x + dt * v_gc

    v_new = v_parallel_new * b_hat + np.cross(b_hat, np.cross(v, b_hat))
    return Particle(charge=q, mass=m, position=x_new, velocity=v_new)


def guiding_center_integrate(field_b, particle, dt, n_steps, mu):
    trajectory = np.zeros((n_steps + 1, 3))
    velocity = np.zeros((n_steps + 1, 3))
    times = np.zeros(n_steps + 1)

    p = particle.copy()
    trajectory[0] = p.position
    velocity[0] = p.velocity

    for i in range(n_steps):
        p = guiding_center_step(field_b, p, dt, mu)
        trajectory[i + 1] = p.position
        velocity[i + 1] = p.velocity
        times[i + 1] = times[i] + dt

    return {"trajectory": trajectory, "velocity": velocity, "times": times}
