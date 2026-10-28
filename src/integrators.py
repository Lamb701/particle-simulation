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


def guiding_center_initial(field_b, particle):
    """Convert full particle state to guiding-center state (x_gc, v_parallel, mu).

    The guiding-center position is offset from the particle position by the
    Larmor radius vector: x_gc = x + m(v × B)/(q B^2).
    """
    x = particle.position.copy()
    v = particle.velocity.copy()
    q = particle.charge
    m = particle.mass

    B = np.asarray(field_b(*x))
    B_mag = np.linalg.norm(B)
    b_hat = B / B_mag

    v_parallel = float(np.dot(v, b_hat))
    x_gc = x + (m / q) * np.cross(v, B) / (B_mag ** 2)

    v_perp_sq = float(v @ v) - v_parallel ** 2
    if v_perp_sq < 0:
        v_perp_sq = 0.0
    mu = 0.5 * v_perp_sq / B_mag  # mass-normalized magnetic moment

    return x_gc, v_parallel, mu


def guiding_center_rhs(field_b, x_gc, v_parallel, mu, charge, mass):
    """Right-hand side of the guiding-center equations.

    Returns (dx_gc/dt, dv_parallel/dt).
    """
    B = np.asarray(field_b(*x_gc))
    B_mag = np.linalg.norm(B)
    b_hat = B / B_mag

    dx = 1e-6
    gradB = np.zeros(3)
    db = np.zeros((3, 3))
    for i, axis in enumerate([(dx, 0, 0), (0, dx, 0), (0, 0, dx)]):
        Bp = np.asarray(field_b(x_gc[0] + axis[0], x_gc[1] + axis[1], x_gc[2] + axis[2]))
        Bm = np.asarray(field_b(x_gc[0] - axis[0], x_gc[1] - axis[1], x_gc[2] - axis[2]))
        gradB[i] = (np.linalg.norm(Bp) - np.linalg.norm(Bm)) / (2 * dx)
        db[:, i] = (Bp / np.linalg.norm(Bp) - Bm / np.linalg.norm(Bm)) / (2 * dx)

    curvature = db @ b_hat
    a_parallel = -mu * (gradB @ b_hat)

    v_grad_curv = (mass * mu / charge) * np.cross(B, gradB) / (B_mag ** 2) \
                  + (mass * v_parallel ** 2 / charge) * np.cross(b_hat, curvature) / B_mag

    v_gc = v_parallel * b_hat + v_grad_curv
    return v_gc, a_parallel


def guiding_center_integrate(field_b, particle, dt, n_steps):
    """Integrate a guiding-center orbit with RK4.

    State is (x_gc, v_parallel, mu); no full-velocity reconstruction is done.
    Returns trajectory, parallel velocity, times, and mu.
    """
    x_gc, v_parallel, mu = guiding_center_initial(field_b, particle)
    q = particle.charge
    m = particle.mass

    trajectory = np.zeros((n_steps + 1, 3))
    vparas = np.zeros(n_steps + 1)
    times = np.zeros(n_steps + 1)

    trajectory[0] = x_gc
    vparas[0] = v_parallel

    for i in range(n_steps):
        k1_x, k1_v = guiding_center_rhs(field_b, x_gc, v_parallel, mu, q, m)
        k2_x, k2_v = guiding_center_rhs(field_b, x_gc + 0.5 * dt * k1_x, v_parallel + 0.5 * dt * k1_v, mu, q, m)
        k3_x, k3_v = guiding_center_rhs(field_b, x_gc + 0.5 * dt * k2_x, v_parallel + 0.5 * dt * k2_v, mu, q, m)
        k4_x, k4_v = guiding_center_rhs(field_b, x_gc + dt * k3_x, v_parallel + dt * k3_v, mu, q, m)

        x_gc = x_gc + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        v_parallel = v_parallel + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        trajectory[i + 1] = x_gc
        vparas[i + 1] = v_parallel
        times[i + 1] = times[i] + dt

    return {"trajectory": trajectory, "v_parallel": vparas, "times": times, "mu": mu}
