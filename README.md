# 托卡马克粒子轨道小项目

算单个带电粒子在托卡马克 / 磁镜磁场里的轨道。起因是这学期上等离子体模拟和等离子体导论，想把课上的 passing / trapped、香蕉轨道真正算出来看看，就把之前写的 Boris 推进器接着做成了这个小项目。

是单粒子、静磁场的教学演示，不是完整等离子体模拟（没有自洽场、碰撞）。磁场用解析模型，不是 Grad-Shafranov 平衡解。

## 里面有什么

- `src/fields.py`：托卡马克和磁镜的解析磁场。
- `src/integrators.py`：Boris 推进器 + 导心（RK4）推进器。
- `src/diagnostics.py`：能量、磁矩、passing/trapped 判据、收敛测试。
- `src/fieldlines.py`：磁力线积分、磁面、绕组角（用来测 q）。
- `src/gui.py`：matplotlib 滑块界面，调参数实时看轨道。
- `scripts/validation.py`：数值验证图（能量 / 磁矩 / 收敛）。
- `scripts/report_figures.py`：一键复现汇报用的全部图。
- `notebooks/complete_report.ipynb` / `.html`：完整汇报，含推导和图。html 不用跑代码就能看。

## 怎么跑

```bash
pip install -r requirements.txt
pytest                      # 单元测试
python -m src.gui           # 交互界面
python scripts/validation.py
python scripts/report_figures.py
```

汇报直接打开 `notebooks/complete_report.html`，或 `jupyter notebook notebooks/complete_report.ipynb`。

## 几个我自己觉得有用的结论

- 捕获分界 $\xi_c=\sqrt{1-B_{min}/B_{max}}$，本算例约 0.277，数值扫描的跳变和它对上了。
- 磁力线 $\theta$-$\phi$ 绕组斜率 = 1/q，拟合出的 q 和我设的剖面吻合，等于查了一遍磁场模型。
- 香蕉宽度随能量线性增长，量级和 $q\rho_L/\sqrt{\varepsilon}$ 一致。
- Boris 保能量但长轨道磁矩会漂，所以长轨道用导心、短轨道用 Boris；两者短时一致。

## 还想做

换真实 EFIT 平衡、加径向电场 / E×B、加碰撞往输运走。接口留好了，还没做。
