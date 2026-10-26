import numpy as np
from abc import ABC, abstractmethod


class MagneticField(ABC):
    @abstractmethod
    def __call__(self, x, y, z):
        """返回磁场 B = (Bx, By, Bz)，单位特斯拉。"""
        ...


class ElectricField(ABC):
    @abstractmethod
    def __call__(self, x, y, z):
        """返回电场 E = (Ex, Ey, Ez)，单位伏特每米。"""
        ...


class ZeroElectricField(ElectricField):
    def __call__(self, x, y, z):
        return np.zeros(3)


class TokamakField(MagneticField):
    """圆截面、大环径比托卡马克的解析磁场模型。

    环向场：B_phi = B0 * R0 / R
    极向场：B_theta = r * B0 / (R * q(r))，由安全因子剖面 q(r) 给出。
    该场在几何上类似托卡马克，但不满足 Grad-Shafranov 平衡，
    仅用于单粒子轨道教学演示。
    """

    def __init__(self, B0, R0, a, q0, q1=0.0):
        self.B0 = float(B0)
        self.R0 = float(R0)
        self.a = float(a)
        self.q0 = float(q0)
        self.q1 = float(q1)

    def __call__(self, x, y, z):
        x, y, z = float(x), float(y), float(z)
        R = np.sqrt(x * x + y * y)
        if R < 1e-12:
            raise ValueError("TokamakField is undefined on the symmetry axis R=0")
        phi = np.arctan2(y, x)

        r = np.sqrt((R - self.R0) ** 2 + z * z)
        if r > self.a:
            raise ValueError(
                f"Point is outside the last closed flux surface: r={r:.4f} > a={self.a}"
            )

        q = self.q0 + self.q1 * (r / self.a) ** 2

        B_phi = self.B0 * self.R0 / R
        B_theta = (r / (R * q)) * self.B0

        e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0])
        if r > 1e-12:
            e_theta = np.array([
                -(z / r) * np.cos(phi),
                -(z / r) * np.sin(phi),
                (R - self.R0) / r,
            ])
        else:
            e_theta = np.array([0.0, 0.0, 1.0])

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
