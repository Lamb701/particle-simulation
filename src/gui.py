import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 设置中文字体，按常见系统字体依次尝试
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src.particles import Particle
from src.fields import TokamakField
from src.integrators import boris_integrate
from src.diagnostics import kinetic_energy, magnetic_moment

PROTON_MASS = 1.6726219e-27
PROTON_CHARGE = 1.6021766e-19


def run_gui():
    fig = plt.figure(figsize=(14, 8))
    fig.subplots_adjust(left=0.25)

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_rz = fig.add_subplot(2, 2, 2)
    ax_energy = fig.add_subplot(2, 2, 3)
    ax_mu = fig.add_subplot(2, 2, 4)

    # 默认托卡马克参数
    B0, R0, a, q0, q1 = 5.0, 10.0, 1.0, 1.0, 2.0
    field = TokamakField(B0, R0, a, q0, q1)

    v = np.array([1e5, 0.0, 5e4])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.2, 0.0, 0.0]), v)
    dt = 1e-10
    n_steps = 2000

    result = boris_integrate(field, lambda x, y, z: np.zeros(3), p, dt, n_steps)
    traj = result["trajectory"]
    vel = result["velocity"]
    times = result["times"]

    energies = np.array([kinetic_energy(vv, PROTON_MASS) for vv in vel])
    mus = np.array([magnetic_moment(vv, field, *xx) for vv, xx in zip(vel, traj)])

    line3d, = ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.8)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    line_rz, = ax_rz.plot(np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2), traj[:, 2], lw=0.8)
    ax_rz.set_xlabel("R")
    ax_rz.set_ylabel("Z")

    line_e, = ax_energy.plot(times, (energies - energies[0]) / energies[0])
    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel("$\\Delta E / E_0$")

    line_mu, = ax_mu.plot(times, (mus - mus[0]) / mus[0])
    ax_mu.set_xlabel("t")
    ax_mu.set_ylabel("$\\Delta \\mu / \\mu_0$")

    axcolor = "lightgoldenrodyellow"
    ax_vpara = plt.axes([0.03, 0.75, 0.18, 0.03], facecolor=axcolor)
    ax_vperp = plt.axes([0.03, 0.70, 0.18, 0.03], facecolor=axcolor)
    ax_R = plt.axes([0.03, 0.65, 0.18, 0.03], facecolor=axcolor)
    ax_Z = plt.axes([0.03, 0.60, 0.18, 0.03], facecolor=axcolor)
    ax_steps = plt.axes([0.03, 0.55, 0.18, 0.03], facecolor=axcolor)

    s_vpara = Slider(ax_vpara, "平行速度", 1e4, 1e6, valinit=5e4)
    s_vperp = Slider(ax_vperp, "垂直速度", 1e4, 1e6, valinit=1e5)
    s_R = Slider(ax_R, "大半径 R", 9.5, 11.0, valinit=10.2)
    s_Z = Slider(ax_Z, "高度 Z", -0.5, 0.5, valinit=0.0)
    s_steps = Slider(ax_steps, "步数", 500, 10000, valinit=2000, valstep=100)

    def update(val):
        v_para = s_vpara.val
        v_perp = s_vperp.val
        R0_init = s_R.val
        Z_init = s_Z.val
        n = int(s_steps.val)

        B_local = np.array([0.0, 0.0, field(R0_init, 0.0, Z_init)[2]])
        B_mag = np.linalg.norm(B_local)
        b = B_local / B_mag
        perp_dir = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(perp_dir, b)) > 0.9:
            perp_dir = np.array([0.0, 1.0, 0.0])
        perp_dir = perp_dir - np.dot(perp_dir, b) * b
        perp_dir = perp_dir / np.linalg.norm(perp_dir)

        v_new = v_para * b + v_perp * perp_dir
        p_new = Particle(PROTON_CHARGE, PROTON_MASS, np.array([R0_init, 0.0, Z_init]), v_new)
        res = boris_integrate(field, lambda x, y, z: np.zeros(3), p_new, dt, n)
        t = res["trajectory"]
        vels = res["velocity"]

        line3d.set_data_3d(t[:, 0], t[:, 1], t[:, 2])
        line_rz.set_data(np.sqrt(t[:, 0] ** 2 + t[:, 1] ** 2), t[:, 2])

        es = np.array([kinetic_energy(vv, PROTON_MASS) for vv in vels])
        ms = np.array([magnetic_moment(vv, field, *xx) for vv, xx in zip(vels, t)])
        line_e.set_data(res["times"], (es - es[0]) / es[0])
        line_mu.set_data(res["times"], (ms - ms[0]) / ms[0])

        ax_energy.relim()
        ax_energy.autoscale_view()
        ax_mu.relim()
        ax_mu.autoscale_view()
        fig.canvas.draw_idle()

    s_vpara.on_changed(update)
    s_vperp.on_changed(update)
    s_R.on_changed(update)
    s_Z.on_changed(update)
    s_steps.on_changed(update)

    reset_ax = plt.axes([0.08, 0.45, 0.08, 0.04])
    reset_btn = Button(reset_ax, "重置", color=axcolor, hovercolor="0.975")

    def reset(event):
        s_vpara.reset()
        s_vperp.reset()
        s_R.reset()
        s_Z.reset()
        s_steps.reset()

    reset_btn.on_clicked(reset)

    plt.show()


if __name__ == "__main__":
    run_gui()
