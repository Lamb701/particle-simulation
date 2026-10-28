import numpy as np
from src.fields import TokamakField
from src.gui import initial_velocity_from_pitch


def test_initial_velocity_parallel_direction():
    field = TokamakField(B0=5.0, R0=10.0, a=1.0, q0=1.0, q1=2.0)
    position = (10.2, 0.0, 0.0)
    v_para = 5e4
    v_perp = 1e5
    v = initial_velocity_from_pitch(field, position, v_para, v_perp)

    B = field(*position)
    b = B / np.linalg.norm(B)
    v_parallel_actual = np.dot(v, b)
    v_perp_actual = np.sqrt(np.dot(v, v) - v_parallel_actual ** 2)

    np.testing.assert_allclose(v_parallel_actual, v_para, rtol=1e-12)
    np.testing.assert_allclose(v_perp_actual, v_perp, rtol=1e-12)


def test_initial_velocity_not_z_axis():
    # Regression test: parallel direction should follow the actual B field,
    # not just the z axis.
    field = TokamakField(B0=5.0, R0=10.0, a=1.0, q0=1.0, q1=2.0)
    position = (10.2, 0.0, 0.0)
    v = initial_velocity_from_pitch(field, position, 5e4, 0.0)

    B = field(*position)
    b = B / np.linalg.norm(B)
    z_axis = np.array([0.0, 0.0, 1.0])
    angle = np.arccos(np.clip(np.abs(np.dot(b, z_axis)), 0.0, 1.0))
    # The actual B is mostly toroidal, so it should be far from the z axis.
    assert angle > np.radians(45)
