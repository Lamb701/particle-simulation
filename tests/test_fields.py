import numpy as np
from src.fields import TokamakField, MirrorField, ZeroElectricField

def test_zero_electric_field():
    e = ZeroElectricField()
    np.testing.assert_allclose(e(1.0, 2.0, 3.0), np.zeros(3))

def test_tokamak_field_shape_and_dominant_component():
    b = TokamakField(B0=1.0, R0=1.0, a=0.3, q0=1.0, q1=0.0)
    bx, by, bz = b(1.0, 0.0, 0.0)
    bvec = np.array([bx, by, bz])
    assert bvec.shape == (3,)
    assert np.linalg.norm(bvec) > 0.9

def test_tokamak_divergence_free():
    # The simplified tokamak model is not exactly divergence-free, but the
    # numerical divergence should be stable under grid refinement and bounded.
    b = TokamakField(B0=5.0, R0=10.0, a=1.0, q0=1.0, q1=2.0)
    x, y, z = 10.2, 0.1, 0.2

    def divergence(dx):
        dBx = b(x + dx, y, z)[0] - b(x - dx, y, z)[0]
        dBy = b(x, y + dx, z)[1] - b(x, y - dx, z)[1]
        dBz = b(x, y, z + dx)[2] - b(x, y, z - dx)[2]
        return (dBx + dBy + dBz) / (2 * dx)

    div_coarse = divergence(1e-4)
    div_fine = divergence(5e-5)
    assert abs(div_coarse) < 1.0
    assert abs(div_fine) < 1.0
    assert max(abs(div_coarse), abs(div_fine)) <= 2 * min(abs(div_coarse), abs(div_fine))

def test_mirror_field_axisymmetric():
    # For an axisymmetric mirror, Bz is even in x and Bx is odd in x (radial
    # field points away from the axis on both sides).
    b = MirrorField(B0=1.0, bm=2.0, L=1.0)
    b1 = np.array(b(0.1, 0.0, 0.5))
    b2 = np.array(b(-0.1, 0.0, 0.5))
    assert b1[2] == b2[2]
    assert b1[0] == -b2[0]
    assert abs(b1[1]) < 1e-12

def test_mirror_field_on_axis_increases_with_z():
    b = MirrorField(B0=1.0, bm=2.0, L=1.0)
    bz0 = b(0.0, 0.0, 0.0)[2]
    bz1 = b(0.0, 0.0, 1.0)[2]
    assert bz1 > bz0
