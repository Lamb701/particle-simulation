"""Generate the three report figures used in notebooks/complete_report.ipynb.

Run from the repository root:
    python scripts/report_figures.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src.particles import Particle
from src.fields import TokamakField
from src.fieldlines import integrate_fieldline, flux_surface_points
from src.integrators import guiding_center_integrate

PROTON_MASS = 1.6726219e-27
PROTON_CHARGE = 1.6021766e-19

field = TokamakField(B0=5.0, R0=10.0, a=1.0, q0=1.0, q1=2.0)
R0 = 10.0


def ensure_dir():
    os.makedirs("figures", exist_ok=True)


def plot_fieldlines():
    x0 = np.array([10.2, 0.0, 0.0])
    traj = integrate_fieldline(field, x0, 1e-3, 100000)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.6)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("托卡马克磁力线（三维螺旋）")

    ax2 = fig.add_subplot(1, 2, 2)
    R = np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2)
    ax2.plot(R, traj[:, 2], lw=0.8)
    for Rs, Zs, r in flux_surface_points(R0, 1.0, n_r=5):
        ax2.plot(Rs, Zs, "k--", lw=0.5, alpha=0.5)
    ax2.set_xlabel("R")
    ax2.set_ylabel("Z")
    ax2.set_title("磁力线的极向面投影（虚线为磁面）")
    ax2.set_aspect("equal")

    fig.tight_layout()
    fig.savefig("figures/report_fieldlines.png", dpi=200)
    plt.close(fig)


def plot_pitch_scan():
    v_total = 1.2e5
    pitch_ratios = [0.9, 0.5, 0.35, 0.30, 0.28, 0.27, 0.25, 0.02]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    sign_counts = []
    for ax, ratio in zip(axes.flat, pitch_ratios):
        v_para = ratio * v_total
        v_perp = np.sqrt(v_total ** 2 - v_para ** 2)
        v = np.array([v_perp, v_para, 0.0])
        p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.4, 0.0, 0.0]), v)
        result = guiding_center_integrate(field, p, dt=1e-7, n_steps=400000)

        vparas = result["v_parallel"]
        sign_changes = np.where(np.diff(np.sign(vparas)) != 0)[0]
        sign_counts.append(len(sign_changes))
        if len(sign_changes) >= 2 and ratio < 0.27:
            start, end = sign_changes[0], sign_changes[1]
            R = np.sqrt(result["trajectory"][start:end, 0] ** 2 + result["trajectory"][start:end, 1] ** 2)
            Z = result["trajectory"][start:end, 2]
        else:
            R = np.sqrt(result["trajectory"][:200000, 0] ** 2 + result["trajectory"][:200000, 1] ** 2)
            Z = result["trajectory"][:200000, 2]

        ax.plot(R - R0, Z, lw=0.6)
        ax.set_xlabel("R - R0")
        ax.set_ylabel("Z")
        ax.set_title(f"ξ = {ratio}")

    for ax in axes.flat:
        ax.set_xlim(0.35, 0.42)
        ax.set_ylim(-0.2, 0.2)

    fig.tight_layout()
    fig.savefig("figures/report_pitch_scan.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pitch_ratios, sign_counts, "o-")
    ax.axvline(x=0.277, color="r", linestyle="--", label="理论临界 ξ_c ≈ 0.277")
    ax.set_xlabel("pitch angle ξ = v_parallel / v")
    ax.set_ylabel("v_parallel 变号次数")
    ax.set_title("passing 与 trapped 的分界")
    ax.legend()
    fig.savefig("figures/report_pitch_boundary.png", dpi=200)
    plt.close(fig)


def plot_energy_scan():
    ratio = 0.1
    energies = [2e4, 5e4, 1e5, 2e5]

    x0 = np.array([10.4, 0.0, 0.0])
    fieldline = integrate_fieldline(field, x0, 1e-3, 100000)
    R_fl = np.sqrt(fieldline[:, 0] ** 2 + fieldline[:, 1] ** 2)
    Z_fl = fieldline[:, 2]
    R_fl_ds = R_fl[::100]
    Z_fl_ds = Z_fl[::100]

    deviations = []
    orbits = {}
    for v_total in energies:
        v_para = ratio * v_total
        v_perp = np.sqrt(v_total ** 2 - v_para ** 2)
        v = np.array([v_perp, v_para, 0.0])
        p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.4, 0.0, 0.0]), v)
        result = guiding_center_integrate(field, p, dt=1e-7, n_steps=150000)
        R = np.sqrt(result["trajectory"][:, 0] ** 2 + result["trajectory"][:, 1] ** 2)
        Z = result["trajectory"][:, 2]
        orbits[v_total] = (R, Z)

        R_ds = R[::100]
        Z_ds = Z[::100]
        min_dists = [np.min(np.sqrt((R_fl_ds - R_ds[i]) ** 2 + (Z_fl_ds - Z_ds[i]) ** 2)) for i in range(len(R_ds))]
        deviations.append(np.max(min_dists))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(energies, deviations, "o-", lw=1.5)
    axes[0].set_xlabel("粒子速度 v [m/s]")
    axes[0].set_ylabel("导心轨道与磁力线的最大偏移 [m]")
    axes[0].set_title("导心轨道与磁力线偏移随能量的变化")

    for i, v_total in enumerate([2e4, 2e5]):
        R, Z = orbits[v_total]
        axes[1].plot(R_fl - R0, Z_fl, "k--", lw=0.8, label="磁力线" if i == 0 else None)
        axes[1].plot(R - R0, Z, lw=0.6, label=f"v={v_total:.0e} m/s")

    axes[1].set_xlabel("R - R0")
    axes[1].set_ylabel("Z")
    axes[1].set_title("低能量与高能量导心轨道对比（放大）")
    axes[1].set_xlim(0.395, 0.405)
    axes[1].set_ylim(-0.01, 0.01)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("figures/report_energy_scan.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ensure_dir()
    plot_fieldlines()
    plot_pitch_scan()
    plot_energy_scan()
    print("report figures saved to figures/")
