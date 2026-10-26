import numpy as np
from abc import ABC, abstractmethod


class MagneticField(ABC):
    @abstractmethod
    def __call__(self, x, y, z):
        """Return B = (Bx, By, Bz) in Tesla."""
        ...


class ElectricField(ABC):
    @abstractmethod
    def __call__(self, x, y, z):
        """Return E = (Ex, Ey, Ez) in V/m."""
        ...


class ZeroElectricField(ElectricField):
    def __call__(self, x, y, z):
        return np.zeros(3)


class TokamakField(MagneticField):
    def __init__(self, B0, R0, a, q0, q1=0.0):
        self.B0 = float(B0)
        self.R0 = float(R0)
        self.a = float(a)
        self.q0 = float(q0)
        self.q1 = float(q1)

    def __call__(self, x, y, z):
        x, y, z = float(x), float(y), float(z)
        R = np.sqrt(x * x + y * y)
        if R == 0:
            R = 1e-12
        phi = np.arctan2(y, x)

        r = np.sqrt((R - self.R0) ** 2 + z * z)
        if r > self.a:
            r = self.a  # clamp to last-closed surface for simplicity

        q = self.q0 + self.q1 * (r / self.a) ** 2

        B_phi = self.B0 * self.R0 / R
        B_theta = (r / (R * q)) * B_phi

        e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0])
        e_theta = np.array([
            -(z / r) * np.cos(phi),
            -(z / r) * np.sin(phi),
            (R - self.R0) / r,
        ]) if r > 1e-12 else np.array([0.0, 0.0, 1.0])

        return B_phi * e_phi + B_theta * e_theta


class MirrorField(MagneticField):
    def __init__(self, B0, bm, L):
        self.B0 = float(B0)
        self.bm = float(bm)
        self.L = float(L)

    def __call__(self, x, y, z):
        x, y, z = float(x), float(y), float(z)
        r_perp = np.sqrt(x * x + y * y)
        z_over_L = z / self.L
        Bz = self.B0 * (1.0 + self.bm * z_over_L * z_over_L)
        dBz_dz = 2.0 * self.B0 * self.bm * z / (self.L * self.L)
        Br = -0.5 * r_perp * dBz_dz
        if r_perp > 1e-12:
            Bx = Br * x / r_perp
            By = Br * y / r_perp
        else:
            Bx, By = 0.0, 0.0
        return np.array([Bx, By, Bz])
