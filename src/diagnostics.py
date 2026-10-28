import numpy as np


def kinetic_energy(velocity, mass):
    v = np.asarray(velocity, dtype=float)
    return 0.5 * mass * float(v @ v)


def magnetic_moment(velocity, b_field, x, y, z):
    v = np.asarray(velocity, dtype=float)
    B = np.asarray(b_field(x, y, z), dtype=float)
    B_mag = np.linalg.norm(B)
    if B_mag == 0:
        return 0.0
    v_perp_sq = float(v @ v) - (np.dot(v, B) / B_mag) ** 2
    if v_perp_sq < 0:
        v_perp_sq = 0.0
    return 0.5 * v_perp_sq / B_mag


def classify_orbit(trajectory, velocity, b_field):
    """Classify an orbit by counting sign changes of v_parallel along the trajectory.

    Uses the actual velocity array rather than finite-differencing positions.
    A trapped particle shows at least two sign changes (one full bounce).
    """
    vparas = []
    for pos, vel in zip(trajectory, velocity):
        B = np.asarray(b_field(*pos))
        B_mag = np.linalg.norm(B)
        if B_mag == 0:
            vparas.append(0.0)
        else:
            vparas.append(np.dot(vel, B) / B_mag)

    vparas = np.asarray(vparas)
    if len(vparas) == 0:
        return "passing"

    # Suppress small-amplitude oscillations around zero to avoid spurious counts.
    threshold = 0.01 * np.max(np.abs(vparas))
    states = np.zeros(len(vparas), dtype=int)
    states[vparas > threshold] = 1
    states[vparas < -threshold] = -1

    sign_changes = 0
    last_nonzero = 0
    for i in range(1, len(states)):
        if states[i] != 0 and last_nonzero != 0 and states[i] != last_nonzero:
            sign_changes += 1
        if states[i] != 0:
            last_nonzero = states[i]

    if sign_changes >= 2:
        return "trapped"
    elif sign_changes == 1:
        return "barely_trapped"
    else:
        return "passing"


def trapped_pitch_condition(particle, b_field, b_max):
    """Check whether a particle is trapped by the pitch-angle/loss-cone criterion.

    A particle is trapped if v_parallel^2 / v^2 < B_local / B_max,
    where B_max is the maximum field strength along its orbit.
    """
    B = np.asarray(b_field(*particle.position))
    B_mag = np.linalg.norm(B)
    v = particle.velocity
    v_total = np.linalg.norm(v)
    if v_total == 0 or B_mag == 0:
        return False
    v_parallel = np.dot(v, B) / B_mag
    return (v_parallel / v_total) ** 2 < B_mag / b_max


def convergence_test(field_b, field_e, particle, dt_ref, n_steps_ref, dt_list):
    from src.integrators import boris_integrate

    ref_result = boris_integrate(field_b, field_e, particle, dt_ref, n_steps_ref)
    ref_final = ref_result["trajectory"][-1]

    errors = []
    actual_dts = []
    for dt in dt_list:
        n_steps = int(round(n_steps_ref * dt_ref / dt))
        result = boris_integrate(field_b, field_e, particle, dt, n_steps)
        err = np.linalg.norm(result["trajectory"][-1] - ref_final)
        errors.append(err)
        actual_dts.append(dt)

    return {"dt_list": np.array(actual_dts), "errors": np.array(errors)}
