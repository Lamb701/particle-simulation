import os
import sys
from pathlib import Path

# Allow running this script directly from the repo root: python scripts/validation.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

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
    ax.set_title("Energy conservation in tokamak")
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
    ax.set_title("Magnetic moment conservation in mirror")
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
    ax.set_ylabel("Position error")
    ax.set_title("Boris convergence")
    slope = np.polyfit(np.log(result["dt_list"]), np.log(result["errors"]), 1)[0]
    ax.text(0.1, 0.9, f"slope = {slope:.2f}", transform=ax.transAxes)
    fig.savefig("figures/convergence.png", dpi=150)
    plt.close(fig)


def plot_banana_vs_gc():
    field = TokamakField(B0=5.0, R0=10.0, a=1.0, q0=1.0, q1=2.0)
    e = lambda x, y, z: np.zeros(3)
    v = np.array([5e4, 0.0, 8e4])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.2, 0.0, 0.0]), v)

    result_boris = boris_integrate(field, e, p, dt=1e-10, n_steps=8000)
    B = field(*p.position)
    B_mag = np.linalg.norm(B)
    v_para = np.dot(v, B) / B_mag
    v_perp_sq = float(v @ v) - v_para ** 2
    mu = 0.5 * v_perp_sq / B_mag
    result_gc = guiding_center_integrate(field, p, dt=1e-10, n_steps=8000, mu=mu)

    fig, ax = plt.subplots()
    R_b = np.sqrt(result_boris["trajectory"][:, 0] ** 2 + result_boris["trajectory"][:, 1] ** 2)
    R_g = np.sqrt(result_gc["trajectory"][:, 0] ** 2 + result_gc["trajectory"][:, 1] ** 2)
    ax.plot(R_b, result_boris["trajectory"][:, 2], label="Boris full orbit", lw=0.8)
    ax.plot(R_g, result_gc["trajectory"][:, 2], label="Guiding center", lw=0.8)
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title("Banana orbit: Boris vs guiding-center")
    ax.legend()
    fig.savefig("figures/banana_vs_gc.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ensure_figures_dir()
    plot_energy_conservation_tokamak()
    plot_mu_conservation_mirror()
    plot_convergence()
    plot_banana_vs_gc()
    print("Validation figures saved to figures/")
