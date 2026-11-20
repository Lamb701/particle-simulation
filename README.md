# 托卡马克磁场中带电粒子轨道的数值研究

本项目基于单粒子轨道模型，数值研究带电粒子在托卡马克及磁镜磁场中的运动特性，包括通行（passing）与捕获（trapped）轨道的转变、香蕉轨道的形成及其宽度随能量的变化。数值方法采用 Boris 推进器（全粒子轨道）与导心近似（四阶 Runge-Kutta，漂移尺度轨道），二者在短时标下相互校验。

适用范围：单粒子、静磁场条件；磁场为圆截面大环径比解析模型，未求解 Grad-Shafranov 平衡，不含自洽电磁场与碰撞效应。本工作属轨道物理与数值方法的教学性研究。

## 代码结构

- `src/fields.py`：托卡马克与磁镜解析磁场。
- `src/integrators.py`：Boris 推进器与导心推进器。
- `src/diagnostics.py`：能量、磁矩、passing/trapped 判据与收敛测试。
- `src/fieldlines.py`：磁力线积分、磁面与绕组角（用于测定安全因子 q）。
- `src/gui.py`：基于 matplotlib 的交互界面，可调节参数实时观察轨道。
- `scripts/validation.py`：数值验证图（能量守恒、磁矩收敛、二阶收敛）。
- `scripts/report_figures.py`：复现报告全部图件。
- `notebooks/complete_report.ipynb` / `complete_report.html`：完整报告，含推导与图件；HTML 可离线阅读。

## 运行方式

```bash
pip install -r requirements.txt
pytest                        # 单元测试
python -m src.gui             # 交互界面
python scripts/validation.py  # 数值验证图
python scripts/report_figures.py  # 报告图件
```

报告可直接打开 `notebooks/complete_report.html`，或以 Jupyter 打开 `complete_report.ipynb`。

## 主要结论

- 捕获分界 $\xi_c=\sqrt{1-B_{min}/B_{max}}$，本算例约 0.277，与数值扫描的跳变位置一致。
- 磁力线 $\theta$–$\phi$ 绕组斜率等于 $1/q$，拟合值与所设安全因子剖面吻合，构成对磁场模型的独立校验。
- 香蕉宽度随能量近似线性增长，量级符合 $q\rho_L/\sqrt{\varepsilon}$。
- Boris 保持能量守恒但长时标磁矩漂移，导心近似以 $\mu$ 为守恒量；故长轨道用导心、短轨道用 Boris，二者短时一致。

## 后续工作

以 EFIT 平衡替换解析磁场、引入径向电场与 $E\times B$ 漂移、加入碰撞算子以过渡到输运问题。相关接口已预留，尚未实现。
