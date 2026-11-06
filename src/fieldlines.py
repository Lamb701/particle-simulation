import numpy as np


def fieldline_rhs(field_b, x):
    """Direction of the magnetic field line at position x."""
    B = np.asarray(field_b(*x))
    B_mag = np.linalg.norm(B)
    if B_mag == 0:
        return np.zeros(3)
    return B / B_mag


def integrate_fieldline(field_b, x0, ds, n_steps):
    """Integrate a magnetic field line using RK4 with fixed arc length step ds."""
    x = np.asarray(x0, dtype=float).copy()
    trajectory = np.zeros((n_steps + 1, 3))
    trajectory[0] = x

    for i in range(n_steps):
        k1 = fieldline_rhs(field_b, x)
        k2 = fieldline_rhs(field_b, x + 0.5 * ds * k1)
        k3 = fieldline_rhs(field_b, x + 0.5 * ds * k2)
        k4 = fieldline_rhs(field_b, x + ds * k3)
        x = x + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory[i + 1] = x

    return trajectory


def flux_surface_points(R0, a, n_r=5, n_theta=100):
    """Return points on nested circular flux surfaces for visualization."""
    surfaces = []
    for r in np.linspace(0.1 * a, a, n_r):
        theta = np.linspace(0, 2 * np.pi, n_theta)
        R = R0 + r * np.cos(theta)
        Z = r * np.sin(theta)
        surfaces.append((R, Z, r))
    return surfaces
