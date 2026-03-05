我需要基于PhiFlow和JAX实现一个可微分的超流氦二流体模型PTV数据反演模型，包括下述互相独立的任务：

1.经典流体瑞利流方法可行性验证（01_rbc_validation.py）：
利用二维Rayleigh-Bénard方腔对流模型，首先模拟离散粒子随流体运动一段时间后的粒子位置和流场分布（求解boussinesq近似NS方程+粒子advect方程）作为参考，再对这些参考数据进行反问题反演（改变流场的初始分布直到粒子位置损失函数最小），获得重建后的流场分布，最后将流场分布和参考分布进行对比。

2.超流体热逆流方法可行性验证（02_cf_sim_recon.py）：
利用二维超流氦热逆流模型，首先模拟粒子随常流体运动一段时间的粒子位置和流场分布（包括常流、超流分布，需求解二流体方程+粒子advect方程）作为参考，再对常流体流场进行反演，对比反演流场和参考流场

3.超流体圆柱绕流方法可行性验证（03_cyl_sim_recon.py）：
使用的方法同任务2，但几何区域发生改变（二维管道变为二维管道圆柱绕流）

4.超流体热逆流真实实验数据反演（04_cf_exp_recon.py）：
利用二维超流氦热逆流模型，直接读取真实粒子位置的实验数据作为参考，再对常流体流场进行反演，观察重建后的常流场和超流场

5.超流体圆柱绕流真实实验数据反演（05_cyl_exp_recon.py）：
使用的方法同任务4，但几何区域发生改变（二维管道变为二维管道圆柱绕流）

6.超流体热逆流复杂粒子动力学反演（06_cf_multiclass.py）：
利用二维超流氦热逆流模型，直接读取经过分类的不同真实粒子位置的实验数据作为参考，分别利用针对不同类粒子的动力学方程，对常流体、超流场分别进行反演，观察重建后的常流场和超流场

我现在已经针对任务1、任务3、任务4写好了完整的可运行的python文件（见D1.py、D3.py、D4.py），但都是非常多行的屎山代码，我需要将这些代码进行开源且易读的封装，具体的代码结构树已在末行附上。帮我完全按照D1.py、D3.py、D4.py的程序逻辑（不能丢失任何功能）以及PhiFlow语法，进行整理编写。另外由于任务2和任务3、任务4和任务5差别均只在几何区域发生改变，因此还需要帮我对任务2、任务5的代码仿照任务3、任务4进行生成。任务6的代码只生成必要的基础段落，具体功能暂不实现。



SFHelium-Recon/                  # 项目根目录
├── README.md                    # 项目说明（论文引用、安装方法）
├── requirements.txt             # 依赖库 (phiflow, jax, matplotlib等)
├── setup.py                     # 安装脚本 (方便 import src)
├── data/                        # 数据存放 (不上传到git，通过.gitignore忽略)
│   ├── simulation/              # 模拟生成的 GT 数据
│   └── experiment/              # 真实实验的粒子轨迹数据
│
├── src/                         # 【核心库】所有通用的、经过验证的函数放这里
│   └── sf_recon/                # 包名
│       ├── __init__.py
│       ├── physics/             # 物理模型层
│       │   ├── __init__.py
│       │   ├── helium.py        # SFHelium_step, PropSolver (超流氦)
│       │   ├── normal.py        # boussinesq_step (RBC, 普通流体)
│       │   └── boundaries.py    # 各种边界条件, sponge layer, cooling zone
│       │
│       ├── solvers/             # 求解器层
│       │   ├── __init__.py
│       │   ├── projection.py    # joint_pressure_projection (包括各种修正版)
│       │   └── poisson.py       # 基础的 laplace solver
│       │
│       ├── inversion/           # 反演核心层 (JAX相关)
│       │   ├── __init__.py
│       │   ├── loss.py          # loss_function, regularization terms
│       │   ├── differentiable.py# JAX scan loops, physical_step_logic
│       │   └── optimizer.py     # 优化器配置 (L-BFGS-B 等)
│       │
│       └── utils/               # 工具层
│           ├── __init__.py
│           ├── io.py            # 数据加载 (Load markers, Save vtk/npz)
│           ├── particles.py     # Advection, PointCloud 处理
│           └── viz.py           # 画图代码 (流线图、动图生成)
│
├── configs/                     # 【配置层】将参数抽离
│   ├── rbc_config.yaml          # RBC 的 Nx, Ny, Ra 等参数
│   ├── cf_sim_config.yaml       # 热逆流模拟参数
│   └── cylinder_config.yaml     # 圆柱绕流参数
│
├── tasks/                 # 【实验脚本层】针对 6 个任务的运行脚本
│   ├── 01_rbc_validation.py     # 任务1：RBC 验证
│   ├── 02_cf_sim_recon.py       # 任务2：热逆流模拟反演
│   ├── 03_cyl_sim_recon.py      # 任务3：圆柱模拟反演
│   ├── 04_cf_exp_recon.py       # 任务4：热逆流真实实验反演
│   ├── 05_cyl_exp_recon.py      # 任务5：圆柱真实实验反演
│   └── 06_cf_multiclass.py      # 任务6：多分类/双盲反演
│
└── notebooks/                   # 【调试层】
    ├── debug_physics.ipynb      # 专门调试 SFHelium_step 稳定性
    └── visualize_results.ipynb  # 加载 tasks 跑出的结果画论文图

