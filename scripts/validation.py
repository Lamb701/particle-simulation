import os
import sys
from pathlib import Path

# Allow running this script directly from the repo root: python scripts/validation.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，按常见系统字体依次尝试
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src.particles import Particle
from src.fields import TokamakField, MirrorField
from src.integrators import boris_integrate, guiding_center_integrate
from src.diagnostics import kinetic_energy, magnetic_moment, convergence_test

PROTON_MASS = 1.6726219e-27
PROTON_CHARGE = 1.6021766e-19


def ensure_figures_dir():
    os.makedirs("figures", exist_ok=True)


def plot_energy_conservation_tokamak():
    field = TokamakField(B0=5.0, R0=10.0, a=1.0, q0=1.0, q1=2.0)
    v = np.array([1e5, 0.0, 5e4])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.2, 0.0, 0.0]), v)

    # E=0: Boris conserves |v| exactly (up to roundoff), so this is a sanity check.
    result_zero_e = boris_integrate(field, lambda x, y, z: np.zeros(3), p, dt=1e-10, n_steps=5000)
    energies_zero = np.array([kinetic_energy(vv, PROTON_MASS) for vv in result_zero_e["velocity"]])

    # Nonzero E: energy should change according to the work done by the field.
    e_field = lambda x, y, z: np.array([1000.0, 0.0, 0.0])
    result_e = boris_integrate(field, e_field, p, dt=1e-10, n_steps=5000)
    energies_e = np.array([kinetic_energy(vv, PROTON_MASS) for vv in result_e["velocity"]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(result_zero_e["times"], (energies_zero - energies_zero[0]) / energies_zero[0])
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel(r"$\Delta E / E_0$")
    axes[0].set_title("E = 0：Boris 能量守恒（机器精度）")

    axes[1].plot(result_e["times"], (energies_e - energies_e[0]) / energies_e[0])
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel(r"$\Delta E / E_0$")
    axes[1].set_title("E ≠ 0：电场做功导致能量变化")

    fig.tight_layout()
    fig.savefig("figures/energy_tokamak.png", dpi=150)
    plt.close(fig)


def plot_mu_conservation_mirror():
    field = MirrorField(B0=1.0, bm=4.0, L=1.0)
    e = lambda x, y, z: np.zeros(3)
    v = np.array([1e4, 0.0, 5e4])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([0.01, 0.0, 0.0]), v)

    # Measure max |Δμ/μ| for several timesteps, so numerical drift can be
    # distinguished from the physical oscillation level.
    dt_list = [4e-9, 2e-9, 1e-9, 5e-10, 2.5e-10, 1.25e-10]
    max_errors = []
    for dt in dt_list:
        result = boris_integrate(field, e, p, dt=dt, n_steps=5000)
        traj = result["trajectory"]
        vel = result["velocity"]
        mus = np.array([magnetic_moment(vv, field, *xx) for vv, xx in zip(vel, traj)])
        max_errors.append(np.max(np.abs((mus - mus[0]) / mus[0])))

    fig, ax = plt.subplots()
    ax.loglog(dt_list, max_errors, "o-")
    ax.set_xlabel(r"$\Delta t$ [s]")
    ax.set_ylabel(r"max $|\Delta \mu / \mu_0|$")
    ax.set_title("磁镜中磁矩误差随步长的收敛")
    fig.savefig("figures/mu_mirror.png", dpi=150)
    plt.close(fig)


def plot_convergence():
    field = MirrorField(B0=1.0, bm=2.0, L=1.0)
    e = lambda x, y, z: np.zeros(3)
    v = np.array([1e5, 0.0, 5e4])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([0.01, 0.0, 0.0]), v)
    # Use more timesteps and a finer reference so the slope is not dominated
    # by reference error or phase error.
    dt_list = [2e-10, 4e-10, 8e-10, 1.6e-9, 3.2e-9]
    result = convergence_test(field, e, p, dt_ref=2.5e-11, n_steps_ref=32000, dt_list=dt_list)

    fig, ax = plt.subplots()
    ax.loglog(result["dt_list"], result["errors"], "o-")
    ax.set_xlabel(r"$\Delta t$ [s]")
    ax.set_ylabel("位置误差")
    ax.set_title("Boris 算法收敛性")
    slope = np.polyfit(np.log(result["dt_list"]), np.log(result["errors"]), 1)[0]
    ax.text(0.1, 0.9, f"斜率 = {slope:.2f}", transform=ax.transAxes)
    fig.savefig("figures/convergence.png", dpi=150)
    plt.close(fig)


def plot_banana_vs_gc():
    field = TokamakField(B0=5.0, R0=10.0, a=1.0, q0=1.0, q1=2.0)
    # 选取被磁镜效应捕获的粒子：垂直速度大、平行速度小
    v = np.array([2e5, 0.0, 5e3])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.2, 0.0, 0.0]), v)

    # 导心近似用于展示完整香蕉轨道；Boris 全轨道因 μ 漂移不适合直接跑完整 bounce。
    result_gc = guiding_center_integrate(field, p, dt=5e-8, n_steps=200000)

    R_g = np.sqrt(result_gc["trajectory"][:, 0] ** 2 + result_gc["trajectory"][:, 1] ** 2)
    Z_g = result_gc["trajectory"][:, 2]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(R_g - 10.0, Z_g, lw=0.6)
    ax.set_xlabel("R - R0")
    ax.set_ylabel("Z")
    ax.set_title("香蕉轨道（导心近似）")
    fig.savefig("figures/banana_vs_gc.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ensure_figures_dir()
    plot_energy_conservation_tokamak()
    plot_mu_conservation_mirror()
    plot_convergence()
    plot_banana_vs_gc()
    print("验证图已保存到 figures/ 目录")
