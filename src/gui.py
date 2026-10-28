import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 设置中文字体，按常见系统字体依次尝试
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src.particles import Particle
from src.fields import TokamakField
from src.integrators import boris_integrate, guiding_center_integrate
from src.diagnostics import kinetic_energy, magnetic_moment

PROTON_MASS = 1.6726219e-27
PROTON_CHARGE = 1.6021766e-19


def initial_velocity_from_pitch(field, position, v_para, v_perp):
    """Construct velocity from parallel/perpendicular components at a given position."""
    B_local = field(*position)
    B_mag = np.linalg.norm(B_local)
    b = B_local / B_mag
    perp_dir = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(perp_dir, b)) > 0.9:
        perp_dir = np.array([0.0, 1.0, 0.0])
    perp_dir = perp_dir - np.dot(perp_dir, b) * b
    perp_dir = perp_dir / np.linalg.norm(perp_dir)
    return v_para * b + v_perp * perp_dir


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

    # 默认使用导心近似展示完整香蕉轨道；Boris 全轨道长时间会出现 μ 漂移。
    v_para_init, v_perp_init = 5e3, 2e5
    v_init = initial_velocity_from_pitch(field, (10.2, 0.0, 0.0), v_para_init, v_perp_init)
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.2, 0.0, 0.0]), v_init)
    dt = 1e-7
    n_steps = 100000

    result_gc = guiding_center_integrate(field, p, dt=dt, n_steps=n_steps)
    traj = result_gc["trajectory"]
    times = result_gc["times"]
    vparas = result_gc["v_parallel"]
    mu = result_gc["mu"]

    # GC diagnostics: energy from v_parallel and mu; mu is conserved by construction.
    Bmag = np.array([np.linalg.norm(field(*xx)) for xx in traj])
    energies = 0.5 * PROTON_MASS * (vparas ** 2 + 2 * mu * Bmag)
    mus_gc = np.full_like(times, mu)

    # Also run Boris for comparison: its mu drifts over long times.
    result_boris = boris_integrate(field, lambda x, y, z: np.zeros(3), p, dt=dt, n_steps=n_steps)
    mus_boris = np.array([
        magnetic_moment(vv, field, *xx)
        for vv, xx in zip(result_boris["velocity"], result_boris["trajectory"])
    ])

    line3d, = ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.8)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    line_rz, = ax_rz.plot(np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2) - R0, traj[:, 2], lw=0.8)
    ax_rz.set_xlabel("R - R0")
    ax_rz.set_ylabel("Z")

    line_e, = ax_energy.plot(times, (energies - energies[0]) / energies[0])
    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel("$\\Delta E / E_0$")

    line_mu_boris, = ax_mu.plot(times, (mus_boris - mus_boris[0]) / mus_boris[0], label="Boris")
    line_mu_gc, = ax_mu.plot(times, (mus_gc - mus_gc[0]) / mus_gc[0], label="导心近似")
    ax_mu.set_xlabel("t")
    ax_mu.set_ylabel("$\\Delta \\mu / \\mu_0$")
    ax_mu.legend()

    axcolor = "lightgoldenrodyellow"
    ax_vpara = plt.axes([0.03, 0.75, 0.18, 0.03], facecolor=axcolor)
    ax_vperp = plt.axes([0.03, 0.70, 0.18, 0.03], facecolor=axcolor)
    ax_R = plt.axes([0.03, 0.65, 0.18, 0.03], facecolor=axcolor)
    ax_Z = plt.axes([0.03, 0.60, 0.18, 0.03], facecolor=axcolor)
    ax_steps = plt.axes([0.03, 0.55, 0.18, 0.03], facecolor=axcolor)

    s_vpara = Slider(ax_vpara, "v_par [m/s]", 1e3, 1e6, valinit=5e3)
    s_vperp = Slider(ax_vperp, "v_perp [m/s]", 1e4, 1e6, valinit=2e5)
    # 限制滑块范围，保证 r = sqrt((R-R0)^2 + Z^2) <= a
    s_R = Slider(ax_R, "R [m]", 9.2, 10.8, valinit=10.2)
    s_Z = Slider(ax_Z, "Z [m]", -0.5, 0.5, valinit=0.0)
    s_steps = Slider(ax_steps, "步数", 10000, 200000, valinit=100000, valstep=10000)

    def update(val):
        v_para = s_vpara.val
        v_perp = s_vperp.val
        R0_init = s_R.val
        Z_init = s_Z.val
        n = int(s_steps.val)

        # 用完整磁场向量确定平行/垂直方向，并用 GC 展示香蕉轨道
        v_new = initial_velocity_from_pitch(field, (R0_init, 0.0, Z_init), v_para, v_perp)
        p_new = Particle(PROTON_CHARGE, PROTON_MASS, np.array([R0_init, 0.0, Z_init]), v_new)
        res_gc = guiding_center_integrate(field, p_new, dt=dt, n_steps=n)
        t = res_gc["trajectory"]
        vparas_new = res_gc["v_parallel"]
        mu_new = res_gc["mu"]

        line3d.set_data_3d(t[:, 0], t[:, 1], t[:, 2])
        line_rz.set_data(np.sqrt(t[:, 0] ** 2 + t[:, 1] ** 2) - R0, t[:, 2])

        Bmag_new = np.array([np.linalg.norm(field(*xx)) for xx in t])
        es = 0.5 * PROTON_MASS * (vparas_new ** 2 + 2 * mu_new * Bmag_new)
        ms_gc = np.full_like(res_gc["times"], mu_new)

        # 同时运行 Boris，展示其 μ 漂移
        res_boris = boris_integrate(field, lambda x, y, z: np.zeros(3), p_new, dt=dt, n_steps=n)
        ms_boris = np.array([
            magnetic_moment(vv, field, *xx)
            for vv, xx in zip(res_boris["velocity"], res_boris["trajectory"])
        ])

        line_e.set_data(res_gc["times"], (es - es[0]) / es[0])
        line_mu_boris.set_data(res_boris["times"], (ms_boris - ms_boris[0]) / ms_boris[0])
        line_mu_gc.set_data(res_gc["times"], (ms_gc - ms_gc[0]) / ms_gc[0])

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
