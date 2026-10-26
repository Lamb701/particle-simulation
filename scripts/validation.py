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
    e = lambda x, y, z: np.zeros(3)
    v = np.array([1e5, 0.0, 5e4])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.2, 0.0, 0.0]), v)
    result = boris_integrate(field, e, p, dt=1e-10, n_steps=5000)
    vel = result["velocity"]
    times = result["times"]
    energies = np.array([kinetic_energy(vv, PROTON_MASS) for vv in vel])

    fig, ax = plt.subplots()
    ax.plot(times, (energies - energies[0]) / energies[0])
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\Delta E / E_0$")
    ax.set_title("托卡马克中的能量守恒")
    fig.savefig("figures/energy_tokamak.png", dpi=150)
    plt.close(fig)


def plot_mu_conservation_mirror():
    field = MirrorField(B0=1.0, bm=4.0, L=1.0)
    e = lambda x, y, z: np.zeros(3)
    v = np.array([1e4, 0.0, 5e4])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([0.01, 0.0, 0.0]), v)
    result = boris_integrate(field, e, p, dt=1e-9, n_steps=5000)
    traj = result["trajectory"]
    vel = result["velocity"]
    times = result["times"]
    mus = np.array([magnetic_moment(vv, field, *xx) for vv, xx in zip(vel, traj)])

    fig, ax = plt.subplots()
    ax.plot(times, (mus - mus[0]) / mus[0])
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\Delta \mu / \mu_0$")
    ax.set_title("磁镜中的磁矩守恒")
    fig.savefig("figures/mu_mirror.png", dpi=150)
    plt.close(fig)


def plot_convergence():
    field = MirrorField(B0=1.0, bm=2.0, L=1.0)
    e = lambda x, y, z: np.zeros(3)
    v = np.array([1e5, 0.0, 5e4])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([0.01, 0.0, 0.0]), v)
    dt_list = [2e-10, 4e-10, 8e-10]
    result = convergence_test(field, e, p, dt_ref=1e-10, n_steps_ref=8000, dt_list=dt_list)

    fig, ax = plt.subplots()
    ax.loglog(result["dt_list"], result["errors"], "o-")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("位置误差")
    ax.set_title("Boris 算法收敛性")
    slope = np.polyfit(np.log(result["dt_list"]), np.log(result["errors"]), 1)[0]
    ax.text(0.1, 0.9, f"斜率 = {slope:.2f}", transform=ax.transAxes)
    fig.savefig("figures/convergence.png", dpi=150)
    plt.close(fig)


def plot_banana_vs_gc():
    field = TokamakField(B0=5.0, R0=10.0, a=1.0, q0=1.0, q1=2.0)
    e = lambda x, y, z: np.zeros(3)
    # 选取被磁镜效应捕获的粒子：垂直速度大、平行速度小
    v = np.array([2e5, 0.0, 5e3])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.2, 0.0, 0.0]), v)

    dt = 1e-9
    n_steps_boris = 450000
    result_boris = boris_integrate(field, e, p, dt=dt, n_steps=n_steps_boris)

    B = field(*p.position)
    B_mag = np.linalg.norm(B)
    v_para = np.dot(v, B) / B_mag
    v_perp_sq = float(v @ v) - v_para ** 2
    mu = 0.5 * v_perp_sq / B_mag

    # 导心推进可用较大步长
    result_gc = guiding_center_integrate(field, p, dt=2e-9, n_steps=n_steps_boris // 2, mu=mu)

    R0 = 10.0
    R_b = np.sqrt(result_boris["trajectory"][:, 0] ** 2 + result_boris["trajectory"][:, 1] ** 2)
    Z_b = result_boris["trajectory"][:, 2]
    r_b = np.sqrt((R_b - R0) ** 2 + Z_b ** 2)
    theta_b = np.arctan2(Z_b, R_b - R0)

    R_g = np.sqrt(result_gc["trajectory"][:, 0] ** 2 + result_gc["trajectory"][:, 1] ** 2)
    Z_g = result_gc["trajectory"][:, 2]
    r_g = np.sqrt((R_g - R0) ** 2 + Z_g ** 2)
    theta_g = np.arctan2(Z_g, R_g - R0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # R-Z 投影
    axes[0].plot(R_b[::20], Z_b[::20], label="Boris 全轨道", lw=0.2, alpha=0.7)
    axes[0].plot(R_g[::10], Z_g[::10], label="导心近似", lw=0.8)
    axes[0].set_xlabel("R")
    axes[0].set_ylabel("Z")
    axes[0].set_title("R-Z 投影")
    axes[0].legend()
    axes[0].set_aspect("equal")

    # 极向截面：香蕉轨道的标准展示方式（用导心近似，避免拉莫尔回旋噪声）
    theta_g_unwrap = np.unwrap(theta_g[::10])
    axes[1].plot(theta_g_unwrap, r_g[::10], label="导心近似", lw=0.8)
    axes[1].set_xlabel(r"$\theta$")
    axes[1].set_ylabel("r")
    axes[1].set_title("极向截面 (r, θ)")
    axes[1].legend()

    fig.suptitle("香蕉轨道：Boris 全轨道与导心近似")
    fig.tight_layout()
    fig.savefig("figures/banana_vs_gc.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ensure_figures_dir()
    plot_energy_conservation_tokamak()
    plot_mu_conservation_mirror()
    plot_convergence()
    plot_banana_vs_gc()
    print("验证图已保存到 figures/ 目录")
