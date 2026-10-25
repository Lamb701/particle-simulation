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


def classify_orbit(trajectory, b_field):
    v_parallel_sign_changes = 0
    for i in range(1, len(trajectory)):
        v = trajectory[i] - trajectory[i - 1]
        B = np.asarray(b_field(*trajectory[i]))
        B_mag = np.linalg.norm(B)
        if B_mag == 0:
            continue
        v_parallel = np.dot(v, B) / B_mag
        if i > 1:
            prev_v = trajectory[i - 1] - trajectory[i - 2]
            prev_B = np.asarray(b_field(*trajectory[i - 1]))
            prev_v_parallel = np.dot(prev_v, prev_B) / np.linalg.norm(prev_B)
            if v_parallel * prev_v_parallel < 0:
                v_parallel_sign_changes += 1

    if v_parallel_sign_changes >= 2:
        return "trapped"
    elif v_parallel_sign_changes == 1:
        return "barely_trapped"
    else:
        return "passing"


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
