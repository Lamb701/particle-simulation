import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}


def md(cell_id, text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": text.splitlines(True)
    })


def code(cell_id, text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True)
    })


md("title", """# 托卡马克中带电粒子轨道特征汇报

本汇报围绕带电粒子在托卡马克磁场中的运动展开，包含四个部分：
1. 带电粒子运动方程及其无量纲化；
2. 托卡马克磁力线与极向面投影；
3. 改变粒子速度的 pitch angle，观察 passing 与 trapped 轨道的转变；
4. 改变粒子能量，观察轨道与磁力线的偏移。

最后对轨道特征的物理成因进行解释。""")

md("model", """## 1. 模型：带电粒子运动方程无量纲化

带电粒子在电磁场中的运动由洛伦兹力决定：
$$m\\frac{d\\mathbf{v}}{dt} = q(\\mathbf{E} + \\mathbf{v} \\times \\mathbf{B}), \\quad \\frac{d\\mathbf{x}}{dt} = \\mathbf{v}$$

为突出物理参数的作用，对方程进行无量纲化。取：
- 长度：$\\mathbf{x} = R_0 \\tilde{\\mathbf{x}}$
- 时间：$t = \\Omega_c^{-1} \\tilde{t}$，其中 $\\Omega_c = qB_0/m$ 为回旋频率
- 速度：$\\mathbf{v} = v_0 \\tilde{\\mathbf{v}}$，$v_0$ 为特征速度
- 磁场：$\\mathbf{B} = B_0 \\tilde{\\mathbf{B}}$，$\\mathbf{E} = E_0 \\tilde{\\mathbf{E}}$

代入得：
$$\\frac{d\\tilde{\\mathbf{v}}}{d\\tilde{t}} = \\frac{E_0}{v_0 B_0} \\tilde{\\mathbf{E}} + \\tilde{\\mathbf{v}} \\times \\tilde{\\mathbf{B}}$$

若 $E_0 = 0$，则简化为：
$$\\frac{d\\tilde{\\mathbf{v}}}{d\\tilde{t}} = \\tilde{\\mathbf{v}} \\times \\tilde{\\mathbf{B}}$$

关键无量纲参数为：
$$\\rho_* = \\frac{v_0}{\\Omega_c R_0} = \\frac{\\rho_L}{R_0}$$

即拉莫尔半径与装置大半径之比。$\\rho_* \\ll 1$ 时导心近似成立，轨道紧贴磁力线；$\\rho_*$ 增大时，轨道与磁力线的偏移显著。""")

code("setup", """import os
import sys
sys.path.insert(0, os.path.abspath('..'))

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.particles import Particle
from src.fields import TokamakField
from src.fieldlines import integrate_fieldline, flux_surface_points
from src.integrators import guiding_center_integrate

PROTON_MASS = 1.6726219e-27
PROTON_CHARGE = 1.6021766e-19

field = TokamakField(B0=5.0, R0=10.0, a=1.0, q0=1.0, q1=2.0)
R0 = 10.0
""")

md("fieldlines", """## 2. 托卡马克磁力线与极向面投影

托卡马克磁力线是绕环向螺旋前进的曲线，由环向场和极向场共同决定。磁力线在三维空间呈螺旋状，在极向面（R-Z 平面）上的投影为闭合曲线。""")

code("plot_fieldlines", """# 积分一条磁力线
x0 = np.array([10.2, 0.0, 0.0])
ds = 1e-4
n_steps = 20000
traj = integrate_fieldline(field, x0, ds, n_steps)

# 画三维磁力线与极向面投影
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('托卡马克磁力线（三维螺旋）')

ax2 = fig.add_subplot(1, 2, 2)
R = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
ax2.plot(R, traj[:, 2], lw=0.8)
# 叠加磁面
for Rs, Zs, r in flux_surface_points(R0, 1.0, n_r=5):
    ax2.plot(Rs, Zs, 'k--', lw=0.5, alpha=0.5)
ax2.set_xlabel('R')
ax2.set_ylabel('Z')
ax2.set_title('磁力线的极向面投影（虚线为磁面）')
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()""")

md("pitch", """## 3. 改变 pitch angle：passing 与 trapped 轨道

固定粒子总速度，改变平行速度占比 $v_\\parallel/v$（即改变 pitch angle），可以观察到粒子从通行（passing）到捕获（trapped）的转变。

- **passing**：$v_\\parallel/v$ 较大，粒子沿磁力线持续前进，形成通行轨道。
- **trapped**：$v_\\parallel/v$ 较小，粒子被磁镜效应捕获，形成香蕉轨道。""")

code("pitch_scan", """# 固定总速度，改变 v_parallel/v
v_total = 1.2e5
pitch_ratios = [0.9, 0.5, 0.1, 0.02]
labels = ['passing (0.9)', '过渡 (0.5)', 'trapped (0.1)', '深 trapped (0.02)']

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, ratio, label in zip(axes, pitch_ratios, labels):
    v_para = ratio * v_total
    v_perp = np.sqrt(v_total**2 - v_para**2)
    # 在初始位置，B 主要沿 y 方向，因此取 v = [v_perp, v_para, 0]
    v = np.array([v_perp, v_para, 0.0])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.4, 0.0, 0.0]), v)
    result = guiding_center_integrate(field, p, dt=1e-7, n_steps=150000)
    R = np.sqrt(result['trajectory'][:, 0]**2 + result['trajectory'][:, 1]**2)
    Z = result['trajectory'][:, 2]
    ax.plot(R - R0, Z, lw=0.6)
    ax.set_xlabel('R - R0')
    ax.set_ylabel('Z')
    ax.set_title(label)
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()""")

md("energy", """## 4. 改变粒子能量：轨道与磁力线的偏移

固定 pitch angle，改变粒子能量。能量越高，拉莫尔半径越大，导心轨道与磁力线的偏移越明显。

这里用同一条磁力线作为参考，比较不同能量下导心轨道与磁力线的偏离。""")

code("energy_scan", """# 固定 pitch angle = 0.1，改变能量
ratio = 0.1
energies = [2e4, 5e4, 1e5, 2e5]
labels = ['2×10⁴ m/s', '5×10⁴ m/s', '1×10⁵ m/s', '2×10⁵ m/s']

# 先画一条参考磁力线
x0 = np.array([10.4, 0.0, 0.0])
fieldline = integrate_fieldline(field, x0, 1e-4, 20000)
R_fl = np.sqrt(fieldline[:, 0]**2 + fieldline[:, 1]**2)
Z_fl = fieldline[:, 2]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, v_total, label in zip(axes, energies, labels):
    v_para = ratio * v_total
    v_perp = np.sqrt(v_total**2 - v_para**2)
    v = np.array([v_perp, v_para, 0.0])
    p = Particle(PROTON_CHARGE, PROTON_MASS, np.array([10.4, 0.0, 0.0]), v)
    result = guiding_center_integrate(field, p, dt=1e-7, n_steps=150000)
    R = np.sqrt(result['trajectory'][:, 0]**2 + result['trajectory'][:, 1]**2)
    Z = result['trajectory'][:, 2]
    ax.plot(R_fl - R0, Z_fl, 'k--', lw=0.8, label='磁力线')
    ax.plot(R - R0, Z, lw=0.6, label='导心轨道')
    ax.set_xlabel('R - R0')
    ax.set_ylabel('Z')
    ax.set_title(label)
    ax.legend()
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()""")

md("physics", """## 5. 轨道特征的物理解释

### 5.1 为什么会有 passing 与 trapped 之分？

托卡马克中磁场强度随位置变化：环内侧磁场强，环外侧磁场弱。当粒子沿磁力线向强场区运动时，磁矩守恒使垂直速度增大、平行速度减小。若初始平行速度太小，粒子会在某处平行速度降为零并被反射，形成 trapped 轨道；若平行速度足够大，粒子可以越过强场区继续沿磁力线运动，形成 passing 轨道。

判据：
$$\\frac{v_\\parallel^2}{v^2} < \\frac{B_{min}}{B_{max}} \\quad \\Rightarrow \\quad \\text{trapped}$$

### 5.2 为什么香蕉轨道是香蕉形？

Trapped 粒子在磁镜点之间来回弹跳，同时受到 $\\nabla B$ 漂移和曲率漂移的影响。这些漂移使导心在径向发生偏移，且偏移方向随粒子在环内侧/外侧的位置变化。在 R-Z 平面上，粒子轨迹形成一个闭合的弯曲曲线，形状像香蕉，因此称为香蕉轨道。

### 5.3 为什么能量高会偏离磁力线？

拉莫尔半径 $\\rho_L = m v_\\perp / (qB)$ 随能量增加而增大。导心近似要求 $\\rho_L$ 远小于磁场变化尺度（如 $R_0$ 或 $a$）。当能量较高时，$\\rho_L$ 与装置尺度可比，导心轨道与磁力线的偏移显著，漂移效应（$\\nabla B$、曲率）也更加突出。

### 5.4 导心近似与全粒子轨道的关系

导心近似忽略了粒子的快速回旋，只保留缓慢漂移。当 $\\rho_* = \\rho_L/R_0 \\ll 1$ 时，导心近似与全粒子轨道高度一致；当 $\\rho_*$ 较大时，导心近似误差增大，全粒子轨道与磁力线的偏移更加明显。""")

md("summary", """## 6. 总结

本汇报完成了以下工作：
1. 推导了带电粒子运动方程的无量纲形式，明确了关键参数 $\\rho_*$ 的物理意义。
2. 画出了托卡马克磁力线及其极向面投影，并叠加了磁面。
3. 通过改变 pitch angle，展示了 passing 到 trapped 轨道的转变。
4. 通过改变粒子能量，展示了轨道与磁力线偏移的增大趋势。
5. 从磁矩守恒、磁镜反射、漂移运动等角度解释了轨道特征的成因。

这些结果为后续研究真实平衡位形、电场效应和碰撞输运打下了基础。""")

with open('notebooks/complete_report.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print('created notebooks/complete_report.ipynb')
