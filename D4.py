# %pip install --quiet phiflow
# %pip install --quiet pyvista
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# # ==========================================
# # 0. CPU 并行设置 (必须在 import jax/phiflow 之前运行)
# # ==========================================
# # 启用 JAX 的 64 位浮点支持，提高物理模拟精度
# os.environ["JAX_ENABLE_X64"] = "True"
# # 强制使用 8 个线程进行计算
# os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"
# # 针对底层数学库
# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["MKL_NUM_THREADS"] = "8"
# # 强制 JAX 使用 CPU 平台
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

# ==========================================
# 0. GPU 并行设置 (必须在 import jax/phiflow 之前运行)
# ==========================================
# 启用 JAX 的 64 位浮点支持，提高物理模拟精度
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORM_NAME"] = "gpu"

# ==========================================
# 1. 导入库与环境验证
# ==========================================
from phi.jax.flow import *
import jax
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# print(f"JAX Backend: {jax.devices()[0].platform}")
# print(f"Device Info: {jax.devices()[0].device_kind}")
# print("CPU Parallelism Configured: 8 Threads")
# 设置64位精度
math.set_global_precision(64)

# --- 验证是否成功挂载到 GPU ---
try:
    # 获取当前可用设备列表
    devices = jax.devices()
    print(f"\n======== JAX Backend Status ========")
    print(f"Platform: {jax.default_backend()}")
    print(f"Device Count: {jax.device_count()}")
    print(f"Devices: {devices}")

    # 再次确认是否为 GPU
    if any(d.platform == 'gpu' for d in devices):
        print("✅ SUCCESS: Running on CUDA/GPU.")
    else:
        # 如果 JAX_PLATFORM_NAME="gpu" 生效，通常不会运行到这里，直接会在 import 时报错
        print("❌ WARNING: Not using GPU. Check CUDA/cuDNN installation.")
    print("====================================\n")
except RuntimeError as e:
    print(f"\n❌ ERROR: Failed to initialize GPU backend: {e}")
    print("Please make sure 'jax[cuda]' is installed correctly matching your CUDA version.")
    exit(1) # 强制退出，避免用 CPU 跑死

# ==========================================
# 2. 物理参数与求解器配置
# ==========================================
# --- 物理系数定义 ---
VISCOSITY_N_COEFFS = [7.42116853e-04, -2.72872406e-03, 4.18086013e-03, -3.40692439e-03,
               1.55634875e-03, -3.77813201e-04, 3.80801796e-05]

RHO_N_COEFFS = [1.80183130e-01, -2.48558984e+01, 4.01282654e+02, -2.48742413e+03,
                7.85114896e+03, -1.42666651e+04, 1.57974461e+04, -1.08279105e+04,
                4.48686634e+03, -1.02983154e+03, 1.00498992e+02]

RHO_S_COEFFS = [144.94795, 25.18243857, -406.9535624, 2509.83455, -7886.879518,
                14286.34985, -15784.94029, 10803.13498, -4471.875738, 1025.612443, -100.0317144]

ENTROPY_COEFFS = [6.57311375e+06, -4.38113590e+07, 1.30656739e+08, -2.29500549e+08,
                  2.62852401e+08, -2.05055435e+08, 1.10322083e+08, -4.04131685e+07,
                  9.64576203e+06, -1.35448196e+06, 8.49760567e+04]

C1_COEFFS = [-37875.4603432003, 161027.467671278, -292620.794399981, 294653.759992904,
             -177568.113431148, 64045.2095387393, -12802.0813284325, 1094.15041473176]

C2_COEFFS = [2421.03719768316, -7059.46514061684, 8208.38710509775, -4758.60939944746,
             1376.23800786335, -158.752493718344]

C3_COEFFS = [-4.18764389500096, 8.88295312215358, -5.42845535938227, 1.49351622326385]

B = 1.008

KAPPA = 9.97e-8

# 压力求解器配置
# 使用 CG (共轭梯度法) 求解线性方程组 (Poisson方程)
PRESSURE_SOLVER = Solve(
    'CG',
    abs_tol=1e-10,
    rel_tol=0,
    max_iterations=2000,
    suppress=[phi.math.NotConverged] # 忽略偶尔的未完全收敛警告，避免中断训练
)

# ==========================================
# 3. 定义物理模型 (JIT 编译)
# ==========================================

# 定义热源与边界条件值
# 空间配置
Lx, Ly = 0.016, 0.320
Wx, Wy = 0.016, 0.008
Nx, Ny = 16, 320
WNx, WNy = 16, 8
DOMAIN = dict(x=Nx, y=Ny, bounds=Box(x=Lx, y=Ly))
WINDOMAIN = dict(x=WNx, y=WNy, bounds=Box(x=(0, Wx), y=(Ly/2-Wy/2, Ly/2+Wy/2)))
MARKERS = 800
# 时间配置
DT = 2e-3             # 时间步长
STEPS = 640            # 总时间步数
PRE_STEPS = 0        # 预先计算步数以达到稳定状态
FREQ =  120

HEAT_SOURCE_INTENSITY = 1.85
FTP = 1450/((145.4070**2) * (1559**4) * (2.17**3)) * ((HEAT_SOURCE_INTENSITY/2.17)**5.7 * (1-(HEAT_SOURCE_INTENSITY/2.17)**5.7))**(-3.0)
# FTP = A_lambda/(rho^2 s_lambda^4 T_lambda^3) [t^5.7(1-t^5.7)]^(-3)



COLD_SOURCE_INTENSITY = 1.85
PRESSURE_0 = 1948
HEAT_FLUX = 2410.0
Vn_IN = HEAT_FLUX/(145.4070 * 627.0 * HEAT_SOURCE_INTENSITY)
Vs_IN = -(52.86*Vn_IN)/92.54

# 边界条件对象
Vn_BC = {'x': 0, 'y-': vec(x=0, y=Vn_IN), 'y+': ZERO_GRADIENT}
Vs_BC = {'x': extrapolation.PERIODIC, 'y-': vec(x=0, y=Vs_IN), 'y+': ZERO_GRADIENT}
J_BC = {'x': 0, 'y-': vec(x=0, y=0), 'y+': ZERO_GRADIENT}
t_BC_THERMAL = {'x': ZERO_GRADIENT, 'y-': ZERO_GRADIENT, 'y+': ZERO_GRADIENT}
p_BC = {'x': ZERO_GRADIENT, 'y-': ZERO_GRADIENT, 'y+': PRESSURE_0}



from phiml.math import copy_with


@jit_compile
def PropSolver(t):
    """
    根据温度 t 计算超流体氦的各种热物性参数。
    t: CenteredGrid
    返回: 各物理系数的标量或 CenteredGrid
    """
    t = field.maximum(t, 1.7)
    t = field.minimum(t, 2.17)
    # 辅助函数：多项式求和
    def poly_eval(coeffs, t_field):
        res = 0.0
        for power, coeff in enumerate(coeffs):
            res += coeff * (t_field ** power)
        return res

    def poly_deriv(coeffs, t_field):
        res = 0.0
        for power, coeff in enumerate(coeffs):
            if power > 0:
                res += power * coeff * (t_field ** (power-1))
        return res

    VISCOSITY_N = poly_eval(VISCOSITY_N_COEFFS, t)

    RHO_N = poly_eval(RHO_N_COEFFS, t)
    RHO_S = poly_eval(RHO_S_COEFFS, t)
    RHO_N = field.maximum(RHO_N, 0.0)
    RHO_S = field.maximum(RHO_S, 0.0)
    RHO = RHO_N + RHO_S
    RHO = field.maximum(RHO, 100.0)

    dRHO_NdT = poly_deriv(RHO_N_COEFFS, t)
    dRHO_SdT = poly_deriv(RHO_S_COEFFS, t)
    dRHOdT = dRHO_NdT + dRHO_SdT

    ENTROPY = poly_eval(ENTROPY_COEFFS, t)
    dSdT = poly_deriv(ENTROPY_COEFFS, t)

    C1 = poly_eval(C1_COEFFS, t)
    alpha_v = C1/1e6 * RHO_S * ENTROPY * t

    C2 = poly_eval(C2_COEFFS, t)
    beta_v = C2/1e8

    C3 = poly_eval(C3_COEFFS, t)
    gamma_v = math.exp(math.log(10) * C3) * (100**4.5)

    return VISCOSITY_N, RHO_N, RHO_S, RHO, dRHOdT, ENTROPY, dSdT, alpha_v, beta_v, gamma_v

@jit_compile
def SFHelium_step(vn, vs, p, t, L, dt):

    # A. 更新物性参数
    VISCOSITY_N, RHO_N, RHO_S, RHO, dRHOdT, ENTROPY, dSdT, alpha_v, beta_v, gamma_v = PropSolver(t)

    y_coords = t.points.vector['y']
    sponge_top = Ly-Ly/Ny
    sponge_bottom = Ly/Ny
    steepness = 20000.0
    # cooling_zone = field.cast(y_coords > 0.95 * Ly, t.values.dtype)
    cooling_zone = math.sigmoid((y_coords - sponge_top) * steepness)
    cooling_rate = 50.0
    heating_zone = math.sigmoid((sponge_bottom - y_coords) * steepness)



    # t_low, t_up = field.shift(t, offsets=(-1,1), dims='y')
    # t = t_low[0] + heating_zone * FTP * HEAT_FLUX**(3.4) * Ly/Ny
    t = t - cooling_zone * cooling_rate * (t - COLD_SOURCE_INTENSITY) * dt

    # --- 1. 对流 (Advection) ---
    # 使用 ? 格式以获得二阶精度，减小数值扩散。
    # 正常流体携带自身动量，超流体携带自身动量，温度随正常流体输运。
    # vn = advect.semi_lagrangian(vn, vn, dt, integrator=advect.rk4)
    # vs = advect.semi_lagrangian(vs, vs, dt, integrator=advect.rk4)
    # t  = advect.mac_cormack(t, vn, dt)


    vn = advect.semi_lagrangian(vn, vn, dt)
    vs = advect.semi_lagrangian(vs, vs, dt)
    t = advect.semi_lagrangian(t, vn, dt)


    # --- 2. 扩散 (Diffusion) ---
    # 仅正常流体具有黏性
    laplacian_vn = field.laplace(vn)
    vn = vn + (VISCOSITY_N * dt * laplacian_vn.at(t)).at(vn)

    # --- 3. 施加源项 (化学势与相互摩擦) ---

    # 为了避免形状不匹配 (Incompatible shapes)，我们在计算相互作用项时
    # 统一将所有量采样到目标网格上 (Centered 或 Staggered)。

    # 3.1 准备基础变量
    # 相对速度 (Staggered -> Centered -> Staggered 确保对齐)
    vn_at_vs = vn.at(t).at(vs)
    vs_at_vn = vs.at(t).at(vn)

    vns = vn - vs_at_vn # 在 vn 网格上的相对速度
    vsn = vs - vn_at_vs # 在 vs 网格上的相对速度

    # 动能项 (在 Centered 网格计算，方便求梯度)
    # k = 0.5 * (vn - vs)^2
    vns_centered = vns.at(t)
    kinetic_energy = 0.5 * field.vec_squared(vns_centered)

    vs_mag = field.vec_length(vs.at(t))
    # 避免除以零
    vs_mag = field.maximum(vs_mag, 1e-10)

    # 3.2 化学势梯度 (Grad Chemical Potential)
    # 计算梯度时指定 extrapolation，确保边界处梯度正确
    grad_k_n = field.spatial_gradient(kinetic_energy, Vn_BC).at(vn)
    grad_t_n = field.spatial_gradient(t, Vn_BC).at(vn)

    grad_k_s = field.spatial_gradient(kinetic_energy, Vs_BC).at(vs)
    grad_t_s = field.spatial_gradient(t, Vs_BC).at(vs)

    # 计算化学势力的系数 (采样到对应的速度网格)
    # term_n = - (rho_s/rho_n * S) * grad T - (rho_s/rho) * grad K
    coeff_S_n = (RHO_S/RHO_N * ENTROPY).at(vn)
    coeff_K_n = (RHO_S/RHO).at(vn)
    CHEM_POTENTIAL_n = - coeff_S_n * grad_t_n - coeff_K_n * grad_k_n

    # term_s = - S * grad T - (rho_n/rho) * grad K
    coeff_S_s = ENTROPY.at(vs)
    coeff_K_s = (RHO_N/RHO).at(vs)
    CHEM_POTENTIAL_s = - coeff_S_s * grad_t_s - coeff_K_s * grad_k_s

    # 更新中间速度
    vn = vn + CHEM_POTENTIAL_n * dt
    vs = vs - CHEM_POTENTIAL_s * dt

    # 3.3 量子涡线密度 (Vinen Equation)
    # 这里涉及非线性源项，使用半隐式或显式推进
    # L 是 CenteredGrid

    # 涡线的平流速度 vL
    # vL = vL_term1 + vL_term2
    sqrt_2k = math.sqrt(2*kinetic_energy)
    term1 = -2.4 * KAPPA * ((L.at(t))**0.5)
    term2 = - (RHO_N/RHO) * sqrt_2k.at(t)
    vL_pre = (term1 + term2) / vs_mag.at(t) # 投影到 vs 方向
    vL_vec = vL_pre.at(vs) * vs # 这是一个近似，确保它是矢量场

    # 涡线平流
    L = advect.mac_cormack(L, vL_vec, dt)

    # 涡线生成与衰减 (Source/Sink)
    # 为了数值稳定性，将 L 限制在非负
    L = field.maximum(L, 0.0)
    L_prod   = (alpha_v.at(L) * sqrt_2k.at(L)) * (L**1.5)
    L_decay  = beta_v.at(L) * (L**2)
    L_remain = gamma_v.at(L) * ((sqrt_2k.at(L))**2.5) # 注意原公式幂次

    L_new = L + (L_prod - L_decay + L_remain) * dt
    L = field.maximum(L_new, 0.0)

    # 3.4 相互摩擦力 (Mutual Friction)
    # Fns ~ B * ... * vns
    # L = (alpha_v/beta_v)**2 * (2*kinetic_energy.at(t))
    B_coeff = B / 3.0 * KAPPA * L.at(t)

    Fns_coeff = (B_coeff * (RHO_S/RHO))
    Fsn_coeff = (B_coeff * (RHO_N/RHO))

    Fns = Fns_coeff.at(vn) * vns
    Fsn = Fsn_coeff.at(vs) * vsn

    # vn = vn - Fns * dt
    # vs = vs - Fsn * dt


    vn_safe = math.safe_div((vn + Fns_coeff.at(vn) * dt * vs_at_vn).values, 1 + (Fns_coeff.at(vn) * dt).values)
    vs_safe = math.safe_div((vs + Fsn_coeff.at(vs) * dt * vn_at_vs).values, 1 + (Fsn_coeff.at(vs) * dt).values)

    vn = vn.with_values(vn_safe)
    vs = vs.with_values(vs_safe)

    # Update
    vs_at_vn = vs.at(t).at(vn)
    vns = vn - vs_at_vn
    vns_centered = vns.at(t)
    kinetic_energy = 0.5 * field.vec_squared(vns_centered)
    L = (alpha_v/beta_v)**2 * (2*kinetic_energy.at(t))
    B_coeff = B / 3.0 * KAPPA * L.at(t)
    Fns_coeff = (B_coeff * (RHO_S/RHO))


    # 3.5 能量方程 (温度更新)
    # 耗散项加热
    # Dissipation = Fns * vns ~ coeff * vns^2
    dissipation_energy = (Fns_coeff) * field.vec_squared(vns.at(t))

    # 热容系数
    heat_capacity = t * (ENTROPY * dRHOdT + RHO * dSdT)
    # 避免除以零
    heat_capacity = field.maximum(heat_capacity, 1e-5)

    t_dissipa = dissipation_energy.at(t) / heat_capacity.at(t)
    t = t + t_dissipa * dt

    t = t - cooling_zone * cooling_rate * (t - COLD_SOURCE_INTENSITY) * dt

    # 强制边界条件
    vn = StaggeredGrid(vn, Vn_BC, **DOMAIN)
    vs = StaggeredGrid(vs, Vs_BC, **DOMAIN)
    t = CenteredGrid(t, t_BC_THERMAL, **DOMAIN)
    L = CenteredGrid(L, ZERO_GRADIENT, **DOMAIN)
    p = CenteredGrid(p, ZERO_GRADIENT, **DOMAIN)

    # --- 4. 压力投影 (Pressure Projection) ---

    _, p_n = fluid.make_incompressible(vn, (), PRESSURE_SOLVER)
    _, p_s = fluid.make_incompressible(vs, (), PRESSURE_SOLVER)

    grad_pn = field.spatial_gradient(p_n, vn.extrapolation, at=vn.sampled_at, scheme='green-gauss')
    grad_ps = field.spatial_gradient(p_s, vs.extrapolation, at=vs.sampled_at, scheme='green-gauss')

    vn = (vn - grad_pn).with_extrapolation(vn.extrapolation)
    vs = (vs - grad_ps).with_extrapolation(vs.extrapolation)

    p = RHO.at(p) * (p_n.at(p) + p_s.at(p)) * 0.5

    return vn, vs, p, t, L

import pandas as pd
import pyvista as pv
import numpy as np
from phi.flow import *

import pandas as pd
import numpy as np
from phi.flow import *

@jit_compile
def load_particle_trajectories(file_path, num_particles, max_steps, dt, domain_bounds):
    """
    加载数据并生成用于处理【迟到粒子】的重置掩码。
    (修正版：修复了 Mask 维度不匹配的问题)
    """
    print(f"Loading dense data with Late Entry support from {file_path}...")
    df = pd.read_excel(file_path)
    df = df[df['category'] == 'g2']

    # 物理坐标变换
    offset_y = Ly / 2 - Wy / 2

    df['time_phys'] = df['time'] * (1/FREQ)
    df['pos_x_phys'] = df['pos_x'] * 1.265e-5
    df['pos_y_phys'] = df['pos_y'] * 1.265e-5 + offset_y

    # 生成所有模拟时间点
    all_sim_times = np.linspace(0, max_steps * dt, max_steps + 1)

    # 筛选粒子
    unique_ids = df['trajectory_num'].unique()
    selected_ids = unique_ids[:num_particles]

    tensor_list = []
    loss_mask_list = []
    reset_mask_list = []

    for pid in selected_ids:
        p_data = df[df['trajectory_num'] == pid].sort_values('time_phys')

        t_start = p_data['time_phys'].min()
        t_end = p_data['time_phys'].max()

        # 1. 插值生成全时段轨迹 (GT)
        x_interp = np.interp(all_sim_times, p_data['time_phys'], p_data['pos_x_phys'])
        y_interp = np.interp(all_sim_times, p_data['time_phys'], p_data['pos_y_phys'])
        traj = np.stack([x_interp, y_interp], axis=-1)

        # 2. 生成 Loss Mask (有效性掩码)
        is_active = (all_sim_times >= t_start - 1e-6) & (all_sim_times <= t_end + 1e-6)
        l_mask = is_active.astype(np.float64)

        # 3. 生成 Reset Mask (注入掩码)
        r_mask = np.zeros_like(l_mask)
        if t_start > dt:
            start_idx = np.argmin(np.abs(all_sim_times - t_start))
            if start_idx < max_steps:
                r_mask[start_idx] = 1.0

        tensor_list.append(traj)
        loss_mask_list.append(l_mask)
        reset_mask_list.append(r_mask)

    # --- 关键修改开始 ---

    # 1. 轨迹数据 (GT): 包含 vector 维度 (x, y)
    # Shape: (N, T, 2) -> (T, N, 2)
    np_traj = np.stack(tensor_list, axis=0).transpose(1, 0, 2)
    gt_tensor = math.tensor(np_traj, batch('time') & instance('markers') & channel(vector='x,y'))

    # 2. 掩码数据 (Mask): 不包含 vector 维度 (标量)
    # Shape: (N, T) -> (T, N)
    np_l_mask = np.stack(loss_mask_list, axis=0).transpose(1, 0)
    np_r_mask = np.stack(reset_mask_list, axis=0).transpose(1, 0)

    # 定义 Tensor 时去掉 channel(...)
    loss_mask = math.tensor(np_l_mask, batch('time') & instance('markers'))
    reset_mask = math.tensor(np_r_mask, batch('time') & instance('markers'))

    # --- 关键修改结束 ---

    initial_pos = gt_tensor.time[0]

    print(f"Processed {len(selected_ids)} particles.")
    print(f"GT Shape: {gt_tensor.shape}")     # (time, markers, vector=x,y)
    print(f"Mask Shape: {loss_mask.shape}")   # (time, markers)

    return gt_tensor, loss_mask, reset_mask, initial_pos




# ==========================================
# 初始化真实初始速度场
# ==========================================

# "./1.85K/241mWcm2_6.55s.vtk"

# print("\nGenerating Ground Truth Data...")

# 使用 make_incompressible 确保初始场物理自洽
v0_gt0 = StaggeredGrid(0, Vn_BC, **DOMAIN)
# v0_gt0, _ = fluid.make_incompressible(v0_gt0, [], PRESSURE_SOLVER)

vs0_gt0 = StaggeredGrid(0, Vs_BC, **DOMAIN)
# vs0_gt0, _ = fluid.make_incompressible(vs0_gt0, [], PRESSURE_SOLVER)

# 初始化真实初始温度场
t0_base = CenteredGrid(HEAT_SOURCE_INTENSITY, t_BC_THERMAL, **DOMAIN)
t0_gt0 = t0_base

L0_gt0 = CenteredGrid(0, ZERO_GRADIENT, **DOMAIN)
p0_gt0 = CenteredGrid(3130, p_BC, **DOMAIN) # 初始压力基准

# # 初始化示踪粒子
# # 使用 sample_uniform 在 bounds 内随机撒点
# markers0 = WINDOMAIN['bounds'].sample_uniform(instance(markers=MARKERS))

# # 预热 (Pre-computation)
# print(f"Pre-stepping for {PRE_STEPS} steps...")
# current_v = v0_gt0
# current_vs = vs0_gt0
# current_p = p0_gt0
# current_t = t0_gt0
# current_L = L0_gt0

# ==========================================
# 核心函数：轴对称旋转重采样 (带平移修正版)
# ==========================================
@jit_compile
def resample_axisymmetric_rotated_ij(vtk_mesh, domain_res, domain_bounds, translation=(0.0, 0.0)):
    """
    执行空间重采样，包含旋转、轴对称镜像以及原点平移。

    Args:
        translation (tuple): (shift_x, shift_y).
                             在查询 OpenFOAM 前对 PhiFlow 坐标进行的平移量。
                             例如：若 PhiFlow x在[0, 2], 对称轴在1, 则需 shift_x = -1.0
    """

    # 1. 解析目标分辨率 (Nx, Ny)
    if hasattr(domain_res, 'sizes'):
        if 'x' in domain_res and 'y' in domain_res:
            nx = domain_res.get_size('x')
            ny = domain_res.get_size('y')
        else:
            sizes = domain_res.sizes
            nx, ny = sizes[0], sizes[1]
    else:
        nx, ny = int(domain_res[0]), int(domain_res[1])

    print(f"  [Debug] Target Grid Resolution: X={nx}, Y={ny}")

    # 2. 生成目标网格 (PhiFlow) 的物理坐标点阵
    min_vec = domain_bounds.lower
    max_vec = domain_bounds.upper

    dx = (float(max_vec.vector[0]) - float(min_vec.vector[0])) / nx
    dy = (float(max_vec.vector[1]) - float(min_vec.vector[1])) / ny

    target_x_1d = np.linspace(float(min_vec.vector[0]) + dx/2, float(max_vec.vector[0]) - dx/2, int(nx))
    target_y_1d = np.linspace(float(min_vec.vector[1]) + dy/2, float(max_vec.vector[1]) - dy/2, int(ny))

    # 生成原始坐标网格 (Matrix Indexing)
    xv, yv = np.meshgrid(target_x_1d, target_y_1d, indexing='ij')

    # ---------------------------------------------------------
    # 关键步骤 0: 坐标平移 (修正原点不对齐)
    # ---------------------------------------------------------
    shift_x, shift_y = translation
    print(f"  [Debug] Applying translation: X += {shift_x}, Y += {shift_y}")

    # 修正后的查询坐标
    xv_query = xv + shift_x
    yv_query = yv + shift_y

    # ---------------------------------------------------------
    # 关键步骤 1: 坐标映射 (PhiFlow -> OpenFOAM)
    # ---------------------------------------------------------
    # OpenFOAM X (长轴) <--- PhiFlow Y_query (Height)
    src_x = yv_query.flatten()

    # OpenFOAM Y (径向) <--- abs(PhiFlow X_query) (Width)
    # 只有经过平移修正后，abs() 才能正确实现关于对称轴的镜像
    src_y = np.abs(xv_query.flatten())

    # OpenFOAM Z
    src_z = np.zeros_like(src_x)

    # 构建查询点云
    probe_points = np.column_stack((src_x, src_y, src_z))
    probe_cloud = pv.PolyData(probe_points)

    # 3. 执行采样
    print(f"  [Debug] Probing {len(probe_points)} points from VTK source...")
    sampled_data = probe_cloud.sample(vtk_mesh, tolerance=1e-5)

    # 4. 记录符号矩阵 (基于平移修正后的坐标)
    # 用于修正矢量方向：修正后 x<0 的区域，径向速度反向
    sign_x_flat = np.sign(xv_query.flatten())
    sign_x_flat[sign_x_flat == 0] = 1

    return sampled_data, (nx, ny), sign_x_flat

# ==========================================
# 辅助函数：加载并对齐 (带参数接口)
# ==========================================
@jit_compile
def load_and_align_fields(vtk_path, domain_res, domain_bounds, boundary_conditions, translation=(-Lx/2, 0.0)):

    print(f"Reading VTK: {vtk_path}")
    try:
        source_mesh = pv.read(vtk_path)
    except Exception as e:
        raise IOError(f"无法读取 VTK 文件: {e}")

    # 调用带平移的采样函数
    sampled_mesh, (nx, ny), sign_flat = resample_axisymmetric_rotated_ij(
        source_mesh, domain_res, domain_bounds, translation
    )

    def get_tensor_data(name, is_vector=False):
        if name not in sampled_mesh.point_data:
            print(f"Warning: Field '{name}' missing, filling zeros.")
            shape = (nx, ny, 2) if is_vector else (nx, ny)
            return np.zeros(shape)

        raw_data = sampled_mesh.point_data[name]

        if is_vector:
            src_ux = raw_data[:, 0]
            src_uy = raw_data[:, 1]

            # 矢量修正
            target_vx_flat = src_uy * sign_flat
            target_vy_flat = src_ux

            vx_grid = target_vx_flat.reshape(nx, ny)
            vy_grid = target_vy_flat.reshape(nx, ny)
            return np.stack([vx_grid, vy_grid], axis=-1)
        else:
            return raw_data.reshape(nx, ny)

    # --- 提取数据 ---
    t_np = get_tensor_data('T', is_vector=False)
    l_np = get_tensor_data('L', is_vector=False)
    p_np = get_tensor_data('p', is_vector=False)
    un_np = get_tensor_data('Un', is_vector=True)
    us_np = get_tensor_data('Us', is_vector=True)

    # --- 构建 PhiFlow Grids ---
    t_grid = CenteredGrid(tensor(t_np, spatial('x,y')), boundary_conditions['t'], bounds=domain_bounds)
    l_grid = CenteredGrid(tensor(l_np, spatial('x,y')), boundary_conditions['l'], bounds=domain_bounds)
    p_grid = CenteredGrid(tensor(p_np, spatial('x,y')), boundary_conditions['p'], bounds=domain_bounds)

    # 矢量场转换
    un_grid_centered = CenteredGrid(tensor(un_np, spatial('x,y'), channel(vector='x,y')), boundary_conditions['v'], bounds=domain_bounds)
    target_v_template = StaggeredGrid(0, boundary_conditions['v'], bounds=domain_bounds, resolution=domain_res)
    v_staggered = un_grid_centered.at(target_v_template)

    us_grid_centered = CenteredGrid(tensor(us_np, spatial('x,y'), channel(vector='x,y')), boundary_conditions['vs'], bounds=domain_bounds)
    target_vs_template = StaggeredGrid(0, boundary_conditions['vs'], bounds=domain_bounds, resolution=domain_res)
    vs_staggered = us_grid_centered.at(target_vs_template)

    return v_staggered, vs_staggered, t_grid, l_grid, p_grid

# ==========================================
# 主调用逻辑
# ==========================================

# 确保你的 DOMAIN 定义包含了 bounds
# 例如: WINDOMAIN = dict(resolution=[32, 64], bounds=Box(x=100, y=200))

bcs = {
    'v': Vn_BC,
    'vs': Vs_BC,
    't': t_BC_THERMAL,
    'l': ZERO_GRADIENT,
    'p': p_BC
}

v0_gt0, vs0_gt0, t0_gt0, L0_gt0, p0_gt0 = load_and_align_fields(
    "./1.85K/241mWcm2_6.55s.vtk",
    v0_gt0.resolution, # 你的目标分辨率，例如 [32, 32]
    v0_gt0.bounds,     # 你的物理边界，例如 Box(x=1.0, y=1.0)
    bcs
)
print("Alignment and Initialization successful.")

# # ==========================================
# # 4. 生成 Ground Truth (真实观测数据)
# # ==========================================

# 物理自洽处理 (可选，视 OpenFOAM 数据质量而定)
# 如果 OpenFOAM 数据已经是无散的，这一步可以跳过；否则建议执行以消除转换误差
# v0_gt0, _ = fluid.make_incompressible(v0_gt0, [], PRESSURE_SOLVER)

# 初始化示踪粒子 (保持不变)
markers0 = WINDOMAIN['bounds'].sample_uniform(instance(markers=MARKERS))

# 预热 (Pre-computation)
# 注意：如果你导入的已经是稳定场，PRE_STEPS 可以设为 0
print(f"Pre-stepping for {PRE_STEPS} steps...")
current_v = v0_gt0
current_vs = vs0_gt0
current_p = p0_gt0
current_t = t0_gt0
current_L = L0_gt0


for _ in range(PRE_STEPS):
    current_v, current_vs, current_p, current_t, current_L = SFHelium_step(
        current_v, current_vs, current_p, current_t, current_L, dt=DT
    )

# 保存预热后的状态作为 GT 的起点
v0_gt, vs0_gt, p0_gt, t0_gt, L0_gt = current_v, current_vs, current_p, current_t, current_L
initial_markers = markers0 # 假设粒子在预热阶段不移动，或者这里重置粒子位置



# 假设 Excel 文件路径
EXCEL_PATH = "./1.85K/10.10classified_k4_1.85 K_241mWcm2_120fps_Trajectory_v3.xlsx"  # <--- 请修改这里

# 调用加载函数
# 注意：这里你需要传递正确的 num_particles, max_steps 和 dt
# max_steps 对应你的 STEPS
gt_marker_trajectories_stack, loss_mask_dense, reset_mask_dense,  initial_markers_tensor = load_particle_trajectories(
    EXCEL_PATH,
    num_particles=MARKERS,
    max_steps=STEPS,
    dt=DT,
    domain_bounds=WINDOMAIN['bounds']
)

# 此时 initial_markers_tensor 是一个纯坐标 Tensor
# 我们需要将其转换为 PointCloud 对象以便在 simulation 中使用 (advect.points 需要 PointCloud)
# 但在这里，GT 已经有了，不需要再跑一遍 advect 来生成 GT。
# 只需要在 loss function 里用 initial_markers 来初始化模拟即可。

# 将 Tensor 包装为 PointCloud 对象 (用于 Loss 函数中的初始状态)
initial_markers = PointCloud(
    elements=geom.Point(initial_markers_tensor)
)

print("Ground Truth Data Loaded Successfully.")


# 生成 GT 轨迹
print(f"Simulating Ground Truth for {STEPS} steps...")
# gt_marker_trajectories = [initial_markers]
current_markers = initial_markers

for time_step in range(STEPS):
    current_v, current_vs, current_p, current_t, current_L = SFHelium_step(
        current_v, current_vs, current_p, current_t, current_L, dt=DT
    )
    # 粒子平流：使用 RK4 积分器提高精度
    # current_markers = advect.points(current_markers, current_v, dt=DT, integrator=advect.rk4)
    # gt_marker_trajectories.append(current_markers)

# 转换为张量 (Time, Markers, Vector)
# gt_marker_trajectories_stack = math.stack(gt_marker_trajectories, batch('time'))
print(f"Ground Truth Generated. Shape: {gt_marker_trajectories_stack.shape}")


# ==========================================
# *. 测试流场Vn和Vs
# ==========================================
print("\nRe-simulating to get full flow field evolution...")

# 确保起点是纯粹的 Field，而不是优化器的变量状态
# v_current = v0_reconstructed
v_current = v0_gt
vs_current = vs0_gt
p_current = p0_gt
t_current = t0_gt
L_current = L0_gt

reconstructed_velocities = [v_current]
reconstructed_velocities_s = [vs_current]
reconstructed_p = [p_current]
reconstructed_temperatures = [t_current]
reconstructed_Line = [L_current]
# 注意：这里我们不需要存储 marker，marker 在可视化阶段单独算
# 这样可以解耦“流场计算”和“粒子平流”，避免 Tracer 混入列表

for time_step in range(STEPS):
    v_current, vs_current, p_current, t_current, L_current = SFHelium_step(v_current, vs_current, p_current, t_current, L_current, dt=DT)
    reconstructed_velocities.append(v_current)
    reconstructed_velocities_s.append(vs_current)
    reconstructed_p.append(p_current)
    reconstructed_temperatures.append(t_current)
    reconstructed_Line.append(L_current)
# 堆叠结果 (Time, Y, X, Vector)
v_stack = math.stack(reconstructed_velocities, batch('time'))
vs_stack = math.stack(reconstructed_velocities_s, batch('time'))
# p_stack = math.stack(reconstructed_p, batch('time'))
t_stack = math.stack(reconstructed_temperatures, batch('time'))
L_stack = math.stack(reconstructed_Line, batch('time'))


# ==========================================
# 8. 可视化：双流体流线对比 (Vn vs Vs)
# ==========================================
print("Generating dual-fluid visualization data...")

# --- A. 数据准备 ---

# 1. 处理 Vn (正常流体)
if 'seed' in v_stack.shape:
    v_display = v_stack[{'seed': 0}]
else:
    v_display = v_stack

v_centered = v_display.at_centers()
un_data = v_centered.vector['x'].values.numpy('time, y, x')
vn_data = v_centered.vector['y'].values.numpy('time, y, x')
vn_mag_data = np.sqrt(un_data**2 + vn_data**2)

# 2. 处理 Vs (超流体)
if 'seed' in vs_stack.shape:
    vs_display = vs_stack[{'seed': 0}]
else:
    vs_display = vs_stack

vs_centered = vs_display.at_centers()
us_data = vs_centered.vector['x'].values.numpy('time, y, x')
vs_y_data = vs_centered.vector['y'].values.numpy('time, y, x') # 重命名避免混淆
vs_mag_data = np.sqrt(us_data**2 + vs_y_data**2)

# 3. 计算全局物理网格和绘图范围
x_phys = np.linspace(0, Lx, Nx)
y_phys = np.linspace(0, Ly, Ny)
X_grid, Y_grid = np.meshgrid(x_phys, y_phys)

# 为两个子图分别设置颜色映射范围
Vn_MIN, Vn_MAX = 0, 1*np.max(vn_mag_data)
Vs_MIN, Vs_MAX = 0, 1*np.max(vs_mag_data)

# --- B. 动画画布设置 ---

# --- B. 动画画布设置 ---
import matplotlib as mpl                                                                                              
from matplotlib.colors import ListedColormap

bmap=mpl.cm.twilight_shifted #获取色条
rmap=mpl.cm.twilight
bluecolors=bmap(np.linspace(0,1,256)) #分片操作                                  
redcolors=rmap(np.linspace(0,1,256)) #分片操作                                  
cblue=ListedColormap(bluecolors[:100]) #切片取舍
cred=ListedColormap(redcolors[128:228]) #切片取舍

# 创建一个包含 1 行 2 列子图的画布
# figsize 控制整体图片大小，确保两个长条形流道放得下且比例合适
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# 统一设置子图标题和坐标轴
for ax, title in zip([ax1, ax2], ["Vn (Normal Fluid)", "Vs (Superfluid)"]):
    ax.set_title(title)
    ax.set_xlim(0, Wx)
    ax.set_ylim(Ly/2-Wy/2, Ly/2+Wy/2)
    # ax.set_ylim(0, Ly)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    # 强制设置物理比例，确保流道不变形
    ax.set_aspect('equal', adjustable='box')

# 设置颜色条 (Colorbar)
# 左图 (Vn)
norm_vn = mcolors.Normalize(vmin=Vn_MIN, vmax=Vn_MAX)
sm_vn = cm.ScalarMappable(norm=norm_vn, cmap=cred)
cbar1 = fig.colorbar(sm_vn, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Magnitude (m/s)')

# 右图 (Vs)
norm_vs = mcolors.Normalize(vmin=Vs_MIN, vmax=Vs_MAX)
sm_vs = cm.ScalarMappable(norm=norm_vs, cmap=cblue)
cbar2 = fig.colorbar(sm_vs, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Magnitude (m/s)')

# --- C. 动画更新函数 ---

SKIP_STEPS = 100  # 每隔 10 步绘制一帧

def update(frame):
    # 清除上一帧内容，但保留坐标轴设置
    ax1.clear()
    ax2.clear()

    # 重新设置标题、坐标轴和比例 (clear() 会清除这些)
    for ax, title in zip([ax1, ax2], [f"Vn - T={frame*DT:.5f}s", f"Vs - T={frame*DT:.5f}s"]):
        ax.set_title(title)
        ax.set_xlim(0, Wx)
        ax.set_ylim(Ly/2-Wy/2, Ly/2+Wy/2)
        # ax.set_ylim(0, Ly)
        ax.set_xlabel("X (m)")
        # 只有左图显示 Y 轴标签，避免重叠
        if ax == ax1:
            ax.set_ylabel("Y (m)")
        # 强制物理比例
        ax.set_aspect('equal', adjustable='box')

    # --- 绘制子图 1: Vn ---
    # 1. 速度大小云图 (Contourf)
    # 使用 vmin/vmax*0.8 来增强对比度，视数据情况调整
    ax1.contourf(
        X_grid, Y_grid,
        vn_mag_data[frame],
        levels=50, # 减少 levels 提高渲染速度
        cmap=cred,
        vmin=Vn_MIN,
        vmax=Vn_MAX * 0.8 if Vn_MAX > 0 else 1.0 # 防止最大值为0时报错
    )
    # 2. 流线图 (Streamplot)
    ax1.streamplot(
        X_grid, Y_grid,
        un_data[frame], vn_data[frame],
        color='black', 
        linewidth=1.2, arrowsize=1.2, density=0.8
    )

    # --- 绘制子图 2: Vs ---
    # 1. 速度大小云图
    ax2.contourf(
        X_grid, Y_grid,
        vs_mag_data[frame],
        levels=50,
        cmap=cblue,
        vmin=Vs_MIN,
        vmax=Vs_MAX * 0.8 if Vs_MAX > 0 else 1.0
    )
    # 2. 流线图
    ax2.streamplot(
        X_grid, Y_grid,
        us_data[frame], vs_y_data[frame],
        color='black',
        linewidth=1.2, arrowsize=1.2, density=0.8
    )

    # 返回需要更新的 Artists 对象 (blit=True 时需要，这里 blit=False 可省略或返回空)
    return ax1, ax2

# --- D. 生成并显示动画 ---
print("Rendering dual-fluid animation...")
# 使用 tight_layout 自动调整子图间距，避免标题和坐标轴重叠
plt.tight_layout()

# 创建动画对象
# frames=len(un_data) 确保遍历所有时间步
# interval=200 设置帧间隔为 200ms (即 5 fps)
ani = animation.FuncAnimation(fig, update, frames=range(0, len(un_data), 16), interval=200, blit=False)

# 关闭静态图像显示
plt.close()

# 保存为 MP4 文件 (可选)
# ani.save("dual_fluid_streamline.mp4", fps=5, dpi=150, writer='ffmpeg')

# 在 Notebook 中显示 JavaScript 动画控件
HTML(ani.to_jshtml())

# ==========================================
# 9. 可视化：温度场演化动画 (修复与优化版)
# ==========================================
print("Rendering Temperature Field Animation...")

# A. 准备数据
# ------------------------------------------------
# 选取第一个 seed

if 'seed' in t_stack.shape:
    t_display = t_stack[{'seed': 0}]
else:
    t_display = t_stack

# 提取 Numpy 数据 (Time, Y, X)
t_data_np = t_display.values.numpy('time, y, x')

# 计算全局最大最小值，固定 Colorbar 防止闪烁
vmin_t, vmax_t = np.min(t_data_np), np.max(t_data_np)

# B. 创建动画
# ------------------------------------------------
fig_t, ax_t = plt.subplots(figsize=(5, 6))

# 初始图像
im_t = ax_t.imshow(
    t_data_np[0],
    origin='lower',
    extent=[0, Lx, 0, Ly],
    cmap='inferno',
    vmin=vmin_t, vmax=vmax_t
)

cbar_t = fig_t.colorbar(im_t, ax=ax_t)
cbar_t.set_label('Temperature')

title_t = ax_t.set_title(f"Temperature Field - T=0.0s")
ax_t.set_xlabel("X (m)")
ax_t.set_ylabel("Y (m)")

def update_temp(frame):
    im_t.set_data(t_data_np[frame])
    title_t.set_text(f"Temperature Field - T={frame*DT:.2f}s")
    return im_t, title_t

ani_t = animation.FuncAnimation(fig_t, update_temp, frames=range(0, len(t_data_np), SKIP_STEPS), interval=200, blit=False)
plt.close()

# 保存为 MP4 文件 (可选)
# ani_t.save("temperature_field_animation.mp4", fps=10, dpi=150)

display(HTML(ani_t.to_jshtml()))

import jax
import jax.numpy as jnp

# ==========================================
# 优化方案：JAX Scan + Checkpoint (防泄漏修正版)
# ==========================================
@jit_compile
def physical_step_logic(carry_state_native, time_input_native):
    """
    JAX Scan 循环体
    关键修正：在循环内重新实例化 Grid，避免污染全局 Geometry 缓存
    """
    # 1. 拆包 State
    (v_x_nat, v_y_nat), (vs_x_nat, vs_y_nat), p_nat, t_nat, L_nat, coords_nat = carry_state_native

    # 2. 拆包 Input
    reset_mask_nat, gt_pos_nat = time_input_native

    # 3. 重组 PhiFlow 对象 (使用构造函数而非 with_values)
    # 获取全局常量 (这些是 int/float/tuple，不会导致泄漏)
    bounds = DOMAIN['bounds']
    ext_v = Vn_BC
    ext_vs = Vs_BC # 假设你有定义 Vs 的边界条件变量，如果没有请用 v0_gt.extrapolation
    ext_p = p0_gt.extrapolation
    ext_t = t0_gt.extrapolation
    ext_L = L0_gt.extrapolation

    # --- 重组 V (Staggered) ---
    # 显式构造，确保 geometry 是新的实例
    v_x_tensor = math.tensor(v_x_nat, spatial('x,y')) # Staggered X component shape
    v_y_tensor = math.tensor(v_y_nat, spatial('x,y')) # Staggered Y component shape
    v_values = math.stack([v_x_tensor, v_y_tensor], dual(vector='x,y'))

    v = StaggeredGrid(
        values=v_values,
        extrapolation=ext_v,
        bounds=bounds,
        resolution=spatial(x=Nx, y=Ny)
    )

    # --- 重组 Vs (Staggered) ---
    vs_x_tensor = math.tensor(vs_x_nat, spatial('x,y'))
    vs_y_tensor = math.tensor(vs_y_nat, spatial('x,y'))
    vs_values = math.stack([vs_x_tensor, vs_y_tensor], dual(vector='x,y'))

    vs = StaggeredGrid(
        values=vs_values,
        extrapolation=ext_vs,
        bounds=bounds,
        resolution=spatial(x=Nx, y=Ny)
    )

    # --- 重组 Scalar Fields (Centered) ---
    # 假设标量场分辨率为 (Nx, Ny)
    grid_shape = spatial(x=Nx, y=Ny)

    p = CenteredGrid(
        values=math.tensor(p_nat, grid_shape),
        extrapolation=ext_p,
        bounds=bounds,
        resolution=grid_shape
    )

    t = CenteredGrid(
        values=math.tensor(t_nat, grid_shape),
        extrapolation=ext_t,
        bounds=bounds,
        resolution=grid_shape
    )

    L = CenteredGrid(
        values=math.tensor(L_nat, grid_shape),
        extrapolation=ext_L,
        bounds=bounds,
        resolution=grid_shape
    )

    # --- 重组粒子与输入 ---
    coords = math.tensor(coords_nat, instance('markers') & channel(vector='x,y'))
    reset_mask_t = math.tensor(reset_mask_nat, instance('markers'))
    gt_pos_t = math.tensor(gt_pos_nat, instance('markers') & channel(vector='x,y'))

    # 4. 执行物理计算
    v_next, vs_next, p_next, t_next, L_next = SFHelium_step(
        v, vs, p, t, L, dt=DT
    )

    # 粒子平流
    markers_obj = PointCloud(geom.Point(coords))
    markers_obj = advect.points(markers_obj, v_next, dt=DT, integrator=advect.rk4)
    advected_coords = markers_obj.geometry.center

    # 迟到粒子重置
    next_coords = math.where(reset_mask_t > 0.5, gt_pos_t, advected_coords)

    # 5. 拆包返回值
    new_carry_native = (
        (v_next.vector['x'].values.native(['x', 'y']), v_next.vector['y'].values.native(['x', 'y'])),
        (vs_next.vector['x'].values.native(['x', 'y']), vs_next.vector['y'].values.native(['x', 'y'])),
        p_next.values.native(['x', 'y']),
        t_next.values.native(['x', 'y']),
        L_next.values.native(['x', 'y']),
        next_coords.native(['markers', 'vector'])
    )

    output_native = next_coords.native(['markers', 'vector'])

    return new_carry_native, output_native

# 重新应用 checkpoint
step_fn_checkpointed = jax.checkpoint(physical_step_logic)


# 3. 重写 Loss Function
# ------------------------------------------------
@jit_compile
def loss_function(v_guess_centered):
    # --- A. 准备初始状态 ---
    v_sim = v_guess_centered.at(v0_gt)

    # 拆解初始状态为 Native Arrays
    # v_sim 和 vs0_gt 是交错网格，拆分 x/y
    v_init_tuple = (
        v_sim.vector['x'].values.native(['x', 'y']),
        v_sim.vector['y'].values.native(['x', 'y'])
    )
    vs_init_tuple = (
        vs0_gt.vector['x'].values.native(['x', 'y']),
        vs0_gt.vector['y'].values.native(['x', 'y'])
    )

    state_init_native = (
        v_init_tuple,
        vs_init_tuple,
        p0_gt.values.native(['x', 'y']),
        t0_gt.values.native(['x', 'y']),
        L0_gt.values.native(['x', 'y']),
        initial_markers.geometry.center.native(['markers', 'vector'])
    )

    # --- B. 准备 Scan 输入 ---
    inputs_mask_tensor = reset_mask_dense.time[1:]
    inputs_gt_tensor = gt_marker_trajectories_stack.time[1:]

    inputs_mask_native = inputs_mask_tensor.native(['time', 'markers'])
    inputs_gt_native = inputs_gt_tensor.native(['time', 'markers', 'vector'])

    scan_inputs = (inputs_mask_native, inputs_gt_native)

    # --- C. 执行 JAX Scan ---
    final_state_native, trajectory_stack_native = jax.lax.scan(
        step_fn_checkpointed,
        state_init_native,
        scan_inputs
    )

    # --- D. 计算 Loss ---
    gt_target_native = gt_marker_trajectories_stack.time[1:].native(['time', 'markers', 'vector'])
    loss_mask_native = loss_mask_dense.time[1:].native(['time', 'markers'])

    loss_mask_native = jnp.expand_dims(loss_mask_native, axis=-1)

    diff = trajectory_stack_native - gt_target_native
    weighted_sq_diff = (diff ** 2) * loss_mask_native

    mse_loss = jnp.sum(weighted_sq_diff)
    
    # =========================================================
    # --- E. 正则化项 (Regularization Terms) ---
    # =========================================================

    # 1. 平滑过渡项（Smooth）
    # 分别提取 x 和 y 方向的速度分量 (得到两个标量场)
    u_component = v_guess_centered.vector['x']
    v_component = v_guess_centered.vector['y']
    
    # 分别计算梯度
    # grad_u 包含 (du/dx, du/dy)
    # grad_v 包含 (dv/dx, dv/dy)
    
    factor = 5e4
    grad_u = field.spatial_gradient(u_component)
    grad_v = field.spatial_gradient(v_component)
    
    # 计算所有导数的 L2 Loss 之和
    smoothness_loss = field.l2_loss(grad_u) * factor + field.l2_loss(grad_v)

    # 2. 能量惩罚项 (Energy Penalty)
    # ---------------------------------------------------------
    # 限制 Vn 动能，防止数值爆炸
    vn_vals = v_guess_centered.values.native(['x', 'y', 'vector'])
    energy_loss = 0.5 * jnp.sum(vn_vals ** 2)

    # =========================================================
    # --- F. 总损失与权重 ---
    # =========================================================
    
    alpha = 5e-6     # 平滑度权重
    beta  = 1e-4   # 能量权重

    total_loss = mse_loss + alpha * smoothness_loss + beta * energy_loss

    return total_loss

# 4. 初始化修正 (防止优化变量带有 Batch 维度)
# ------------------------------------------------
print("\nInitializing Optimization Guess...")
# 确保 init_values 只有空间维度，没有 batch/time
# init_values = math.zeros(spatial(x=Nx, y=Ny) & channel(vector='x,y'))
v0_gt_centered = v0_gt.at_centers()
init_values = v0_gt_centered.values

v_guess_proxy = CenteredGrid(
    values=init_values,
    extrapolation=Vn_BC,
    bounds=DOMAIN['bounds']
)



# ==========================================
# 6. 执行优化
# ==========================================
print("\nStarting Optimization (L-BFGS-B)...")
t_start = time.time()

# 优化设置
# max_iterations: 控制优化步数
# x0: 初始猜测
# method: 'L-BFGS-B' 适用于这种高维非线性优化
optimizer = Solve(
    'L-BFGS-B',
    x0=v_guess_proxy,
    max_iterations=100,
    suppress=[phi.math.Diverged, phi.math.NotConverged]
)

# 记录求解过程
with math.SolveTape(record_trajectories=False) as solves:
    result_centered = minimize(loss_function, optimizer)

t_end = time.time()
print(f"Optimization Finished in {t_end - t_start:.2f} s")

# 提取结果
v0_reconstructed = result_centered.at(v0_gt)
final_loss = loss_function(result_centered)

print(f"=== Optimization Result ===")
print(f"Final Loss: {final_loss:.6e}")
print(f"Iterations: {len(solves)}")

# 简单验证：比较重建场与 GT 场的差异
diff_field = v0_reconstructed - v0_gt
diff_mag = field.l2_loss(diff_field)
print(f"Field Reconstruction Error (L2): {diff_mag:.6e}")


# ==========================================
# 7. 重建流场演化与粒子轨迹 (Re-simulation)
# ==========================================
print("\nRe-simulating to get full flow field evolution...")

# 1. 确保起点是纯粹的 Field
# ------------------------------------------------
v_current = v0_reconstructed
vs_current = vs0_gt
p_current = p0_gt
t_current = t0_gt
L_current = L0_gt

# 2. 准备粒子初始状态
# ------------------------------------------------
# 使用加载数据时返回的初始位置
# 注意：即使对于迟到粒子，这里也有初始值 (虽然在 mask 为 0 时无效)
current_markers = initial_markers.geometry.center

# 3. 存储容器
# ------------------------------------------------
# 物理场存储
reconstructed_velocities = [v_current]
reconstructed_velocities_s = [vs_current]
reconstructed_temperatures = [t_current]

# 粒子轨迹存储 (纯数值 numpy array)
# 先存第 0 帧
marker_positions = [current_markers.numpy('markers, vector')]

# 4. 执行循环 (物理演化 + 粒子平流 + 强制重置)
# ------------------------------------------------
# reset_mask_dense: (Time, Markers) - 注意去掉 vector 维度后的形状
# 我们需要按时间步索引它
print("Calculating marker trajectories with reset injection...")

for i in range(1, STEPS + 1):
    # A. 物理场更新
    v_current, vs_current, p_current, t_current, L_current = SFHelium_step(
        v_current, vs_current, p_current, t_current, L_current, dt=DT
    )

    # B. 粒子平流 (Advection)
    # 包装成 PointCloud 进行计算
    markers_obj = PointCloud(geom.Point(current_markers))
    markers_obj = advect.points(markers_obj, v_current, dt=DT, integrator=advect.rk4)
    advected_coords = markers_obj.geometry.center

    # C. 粒子重置 (Reset Injection) - 关键！
    # 获取当前步的 Reset Mask 和 GT 位置
    # reset_mask_dense 形状: (Time=STEPS+1, Markers)
    # gt_tensor_dense 形状: (Time=STEPS+1, Markers, Vector=2)
    mask_at_t = reset_mask_dense.time[i] # Shape: (Markers)
    gt_at_t = gt_marker_trajectories_stack.time[i]    # Shape: (Markers, Vector)

    # 扩展 mask 维度以匹配 vector (如果 mask 是标量)
    # PhiFlow 的 math.where 支持自动广播，但为了稳妥起见，确保维度一致
    # 如果 mask_at_t 是 (N)，gt 是 (N, 2)，自动广播通常没问题

    # 执行重置：如果 mask > 0.5，强制移动到 GT 位置
    current_markers = math.where(mask_at_t > 0.5, gt_at_t, advected_coords)

    # D. 存储数据
    reconstructed_velocities.append(v_current)
    reconstructed_velocities_s.append(vs_current)
    reconstructed_temperatures.append(t_current)

    # 转为 Numpy 存储，节省显存
    marker_positions.append(current_markers.numpy('markers, vector'))

# 5. 堆叠物理场 (Time, Y, X, Vector)
# ------------------------------------------------
v_stack = math.stack(reconstructed_velocities, batch('time'))
vs_stack = math.stack(reconstructed_velocities_s, batch('time'))
# t_stack = math.stack(reconstructed_temperatures, batch('time'))

# 6. 转换粒子轨迹为 Numpy Array
# Shape: (Time, Markers, 2)
marker_positions_np = np.array(marker_positions)

# 7. 获取可见性掩码 (Loss Mask) 用于可视化
# 转换为 numpy，形状 (Time, Markers)
# loss_mask_dense 是 Tensor，形状 (Time, Markers)
visibility_mask_np = loss_mask_dense.numpy('time, markers')

print(f"Re-simulation complete. Trajectory shape: {marker_positions_np.shape}")


# ==========================================
# 可视化：reconstruct 双流体流线对比 (带掩码)
# ==========================================
print("Generating dual-fluid visualization data...")

# --- A. 数据准备 (同前) ---

# 1. 处理 Vn
if 'seed' in v_stack.shape:
    v_recon_display = v_stack[{'seed': 0}]
else:
    v_recon_display = v_stack

v_centered = v_recon_display.at_centers()
un_data = v_centered.vector['x'].values.numpy('time, y, x')
vn_data = v_centered.vector['y'].values.numpy('time, y, x')
vn_mag_data = np.sqrt(un_data**2 + vn_data**2)

# 2. 处理 Vs
if 'seed' in vs_stack.shape:
    vs_recon_display = vs_stack[{'seed': 0}]
else:
    vs_recon_display = vs_stack

vs_centered = vs_recon_display.at_centers()
us_data = vs_centered.vector['x'].values.numpy('time, y, x')
vs_y_data = vs_centered.vector['y'].values.numpy('time, y, x')
vs_mag_data = np.sqrt(us_data**2 + vs_y_data**2)

# 3. 计算网格
x_phys = np.linspace(0, Lx, Nx)
y_phys = np.linspace(0, Ly, Ny)
X_grid, Y_grid = np.meshgrid(x_phys, y_phys)

import matplotlib as mpl                                                                                              
from matplotlib.colors import ListedColormap

# 范围设置
Vn_MIN, Vn_MAX = 0, 1*np.max(vn_mag_data)
Vs_MIN, Vs_MAX = 0, 1*np.max(vs_mag_data)

bmap=mpl.cm.twilight_shifted #获取色条
rmap=mpl.cm.twilight
bluecolors=bmap(np.linspace(0,1,256)) #分片操作                                  
redcolors=rmap(np.linspace(0,1,256)) #分片操作                                  
cblue=ListedColormap(bluecolors[:100]) #切片取舍
cred=ListedColormap(redcolors[128:228]) #切片取舍

# --- B. 动画画布设置 ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for ax, title in zip([ax1, ax2], ["Vn (Normal Fluid)", "Vs (Superfluid)"]):
    ax.set_title(title)
    ax.set_xlim(0, Wx)
    ax.set_ylim(Ly/2-Wy/2, Ly/2+Wy/2)
    # ax.set_ylim(0, Ly)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal', adjustable='box')

# Colorbars
norm_vn = mcolors.Normalize(vmin=Vn_MIN, vmax=Vn_MAX)
cbar1 = fig.colorbar(cm.ScalarMappable(norm=norm_vn, cmap=cred), ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Magnitude (m/s)')

norm_vs = mcolors.Normalize(vmin=Vs_MIN, vmax=Vs_MAX)
cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm_vs, cmap=cblue), ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Magnitude (m/s)')

# --- C. 动画更新函数 (带掩码筛选) ---

def update(frame):
    ax1.clear()
    ax2.clear()

    # 1. 基础设置
    for ax, title in zip([ax1, ax2], [f"Vn - T={frame*DT:.4f}s", f"Vs - T={frame*DT:.4f}s"]):
        ax.set_title(title)
        ax.set_xlim(0, Wx)
        ax.set_ylim(Ly/2-Wy/2, Ly/2+Wy/2)
        # ax.set_ylim(0, Ly)
        ax.set_xlabel("X (m)")
        if ax == ax1: ax.set_ylabel("Y (m)")
        ax.set_aspect('equal', adjustable='box')

    # 2. 筛选当前帧可见的粒子
    # ------------------------------------------------
    # 获取当前帧所有粒子的坐标 (N, 2)
    current_coords = marker_positions_np[frame]
    # 获取当前帧所有粒子的掩码 (N)
    current_mask = visibility_mask_np[frame]

    # 布尔索引筛选: 只保留 mask > 0.5 的粒子
    visible_indices = current_mask > 0.5

    visible_px = current_coords[visible_indices, 0]
    visible_py = current_coords[visible_indices, 1]

    # 打印调试信息 (可选，前几帧)
    # if frame < STEPS:
    #     print(f"Frame {frame}: Showing {len(visible_px)}/{len(current_coords)} particles")

    # --- 绘制子图 1: Vn ---
    ax1.contourf(
        X_grid, Y_grid, vn_mag_data[frame], levels=50, cmap=cred,
        vmin=Vn_MIN, vmax=Vn_MAX * 0.8 if Vn_MAX > 0 else 1.0
    )
    ax1.streamplot(
        X_grid, Y_grid, un_data[frame], vn_data[frame],
        color='black', linewidth=1.2, arrowsize=1.2, density=0.8
    )
    # 仅绘制可见粒子
    if len(visible_px) > 0:
        ax1.scatter(visible_px, visible_py, color='lightblue', s=80, edgecolors='white', linewidths=2, zorder=10, label='Markers')

    # --- 绘制子图 2: Vs ---
    ax2.contourf(
        X_grid, Y_grid, vs_mag_data[frame], levels=50, cmap=cblue,
        vmin=Vs_MIN, vmax=Vs_MAX * 0.8 if Vs_MAX > 0 else 1.0
    )
    ax2.streamplot(
        X_grid, Y_grid, us_data[frame], vs_y_data[frame],
        color='black', linewidth=1.2, arrowsize=1.2, density=0.8
    )
    if len(visible_px) > 0:
        ax2.scatter(visible_px, visible_py, color='lightblue', s=80, edgecolors='white', linewidths=2, zorder=10)

    return ax1, ax2

# --- D. 生成动画 ---
print("Rendering masked animation...")
plt.tight_layout()
ani = animation.FuncAnimation(fig, update, frames=range(0, len(un_data), 16), interval=200, blit=False)
plt.close()

# 保存为gif文件
ani.save("Reconstructed_Field2.gif", fps=5, dpi=150)

# HTML(ani.to_jshtml())

# ==========================================
# 7. 保存优化结果
# ==========================================
import pickle
import numpy as np
import os

SAVE_DIR = "./optimization_results"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\nSaving results to {SAVE_DIR}...")

# ---------------------------------------------------------
# 方法 A: 保存为 NumPy .npz 格式 (推荐用于数据分析)
# ---------------------------------------------------------
# v0_reconstructed 是 StaggeredGrid (交错网格)
# 交错网格的 X 和 Y 分量形状不同，因此我们需要分开提取

# 1. 提取 Staggered 原始数据 (用于精确重启模拟)
# 注意：v0_reconstructed.vector['x'] 是 x 面上的标量场
vx_raw = v0_reconstructed.vector['x'].values.numpy('y,x')
vy_raw = v0_reconstructed.vector['y'].values.numpy('y,x')

# 2. 提取 Centered 插值数据 (用于可视化或通用处理)
# 将交错网格插值到中心点
v_centered = v0_reconstructed.at_centers()
vx_centered = v_centered.vector['x'].values.numpy('y,x')
vy_centered = v_centered.vector['y'].values.numpy('y,x')
v_mag_centered = np.sqrt(vx_centered**2 + vy_centered**2)

# 3. 保存
np.savez(
    os.path.join(SAVE_DIR, "v0_reconstructed_data.npz"),
    # 原始交错数据
    vx_staggered=vx_raw,
    vy_staggered=vy_raw,
    # 中心化数据
    vx_centered=vx_centered,
    vy_centered=vy_centered,
    v_magnitude=v_mag_centered,
    # 辅助信息
    loss=final_loss,
    dt=DT,
    extent=[0, Lx, 0, Ly] # 物理范围
)
print(f"  -> Data saved as .npz (NumPy arrays)")


# ---------------------------------------------------------
# 方法 B: 保存为 Pickle 对象 (推荐用于 PhiFlow 继续加载)
# ---------------------------------------------------------
# 这会保存整个 StaggeredGrid 对象，包含 extrapolation(边界条件) 和 bounds
with open(os.path.join(SAVE_DIR, "v0_reconstructed_obj.pkl"), "wb") as f:
    pickle.dump(v0_reconstructed, f)

print(f"  -> Object saved as .pkl (PhiFlow Grid object)")


# ==========================================
# 附：如何加载这些数据的示例代码
# ==========================================
"""
# 加载示例 1: NumPy
data = np.load("./optimization_results/v0_reconstructed_data.npz")
vx = data['vx_centered']
print("Loaded velocity shape:", vx.shape)

# 加载示例 2: PhiFlow 对象
with open("./optimization_results/v0_reconstructed_obj.pkl", "rb") as f:
    v_loaded = pickle.load(f)

# 验证加载是否成功
from phi.flow import *
# 此时 v_loaded 可以直接放入 SFHelium_step 中使用
print("Loaded Grid type:", type(v_loaded))
"""

# ==========================================
# 10. 可视化：速度大小对比 (高画质版)
# ==========================================
print("\nGenerating Velocity Magnitude Comparison Animation (High Quality)...")

# 1. 准备数据
mag_recon_np = field.vec_length(v_recon_display).values.numpy('time, y, x')
mag_gt_np = field.vec_length(v_display).values.numpy('time, y, x')
diff_mag_np = mag_recon_np - mag_gt_np

# 2. 统一量程
max_val1_mag = 0.5 * np.max(np.abs(mag_gt_np)) + 1e-8
max_err_mag = np.max(np.abs(diff_mag_np)) + 1e-8
full_extent = [0, Lx, 0, Ly]

# 定义视图区域
y_min_view = Ly/2 - 2*Wy
y_max_view = Ly/2 + 2*Wy

# --- 关键修改 1: 提高 DPI ---
fig_comp, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True, dpi=200)
ax1, ax2, ax3 = axes

def add_colorbar(im, ax, label):
    cbar = fig_comp.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    return cbar

# --- 关键修改 2: 使用 'spline36' 插值消除马赛克 ---
# interpolation 可选值:
# 'none' (默认, 马赛克), 'bilinear' (线性模糊), 'spline36' (高阶平滑, 推荐), 'gaussian' (高斯模糊)

# 1. Ground Truth
im1 = ax1.imshow(mag_gt_np[0], origin='lower', cmap='PuOr',
                 vmin=-max_val1_mag, vmax=max_val1_mag,
                 extent=full_extent,
                 interpolation='spline36') # <--- 这里消除了马赛克
ax1.set_title("Ground Truth Vn", fontsize=12)
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_ylim(y_min_view, y_max_view)
ax1.set_aspect('equal') # 或者是 'auto' 来填满画布
add_colorbar(im1, ax1, "Vn (m/s)")

# 2. Reconstructed
im2 = ax2.imshow(mag_recon_np[0], origin='lower', cmap='PuOr',
                 vmin=-max_val1_mag, vmax=max_val1_mag,
                 extent=full_extent,
                 interpolation='spline36') # <--- 这里
ax2.set_title("Reconstructed Vn")
ax2.set_xlabel("X (m)")
ax2.set_ylim(y_min_view, y_max_view)
ax2.set_aspect('equal')
add_colorbar(im2, ax2, "Vn (m/s)")

# 3. Difference
im3 = ax3.imshow(diff_mag_np[0], origin='lower', cmap='PuOr',
                 vmin=-max_val1_mag, vmax=max_val1_mag,
                 extent=full_extent,
                 interpolation='spline36') # <--- 这里
ax3.set_title(f"Difference")
ax3.set_xlabel("X (m)")
ax3.set_ylim(y_min_view, y_max_view)
ax3.set_aspect('equal')
add_colorbar(im3, ax3, "Diff (m/s)")

def update_comp(frame):
    im1.set_data(mag_gt_np[frame])
    im2.set_data(mag_recon_np[frame])
    im3.set_data(diff_mag_np[frame])
    fig_comp.suptitle(f"Vn Comparison - T={frame*DT:.4f}s", fontsize=16)
    return im1, im2, im3

# --- 关键修改 3: 保存时提高比特率 ---
ani_comp = animation.FuncAnimation(fig_comp, update_comp, frames=range(0, len(mag_gt_np), 4), interval=200, blit=False)
plt.close()

# 导出gif
ani_comp.save("vn_comparison_hd.gif", fps=10, dpi=200)

print("Displaying animation...")
# display(HTML(ani_comp.to_jshtml()))