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
# 2. 物理参数与求解器配置
# ==========================================
# 空间配置

Lx, Ly = 0.020, 0.050
Nx = 64
Ny = 64
BOUNDARY = Box(x=Lx, y=Ly) 
DOMAIN = dict(x=Nx, y=Ny, bounds=BOUNDARY)
SPHERE = Sphere(x=Lx/2, y=Ly/2, radius=0.003)
OBSTACLE = Obstacle(SPHERE)
MARKERS = 128

# 时间与物理参数（Water at 25℃）
DT = 1e-7            # 时间步长
STEPS = 50          # 总时间步数
PRE_STEPS = 0     # 预先计算步数以达到稳定状态

HEAT_SOURCE_INTENSITY = 1.94
COLD_SOURCE_INTENSITY = 1.94
PRESSURE_0 = 1948
HEAT_FLUX = 6000

rho_total = 145.5244
rho_n = 68.22
rho_s = 77.31
entropy = 813.4

FTP = 1450/((rho_total**2) * (1559**4) * (2.17**3)) * ((HEAT_SOURCE_INTENSITY/2.17)**5.7 * (1-(HEAT_SOURCE_INTENSITY/2.17)**5.7))**(-3.0)
# FTP = A_lambda/(rho^2 s_lambda^4 T_lambda^3) [t^5.7(1-t^5.7)]^(-3)

Vn_IN = HEAT_FLUX/(rho_total * entropy * HEAT_SOURCE_INTENSITY)
Vs_IN = -(rho_n*Vn_IN)/rho_s

# 边界条件对象
Vn_BC = {'x': 0, 'y-': vec(x=0, y=Vn_IN), 'y+': ZERO_GRADIENT}
Vs_BC = {'x': extrapolation.PERIODIC, 'y-': vec(x=0, y=Vs_IN), 'y+': ZERO_GRADIENT}
J_BC = {'x': 0, 'y-': vec(x=0, y=0), 'y+': ZERO_GRADIENT}
t_BC_THERMAL = {'x': ZERO_GRADIENT, 'y-': ZERO_GRADIENT, 'y+': ZERO_GRADIENT}
p_BC = {'x': ZERO_GRADIENT, 'y-': ZERO_GRADIENT, 'y+': PRESSURE_0}

# ==========================================
# 3. 定义物理模型 (JIT 编译)
# ==========================================

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
    L = advect.semi_lagrangian(L, vL_vec, dt/2)

    # 涡线生成与衰减 (Source/Sink)
    # 为了数值稳定性，将 L 限制在非负
    L = field.maximum(L, 0.0)
    L_prod   = (alpha_v.at(L) * sqrt_2k.at(L)) * (L**1.5)
    L_decay  = beta_v.at(L) * (L**2)
    L_remain = gamma_v.at(L) * ((sqrt_2k.at(L))**2.5) # 注意原公式幂次

    L_new = L + (L_prod - L_decay + L_remain) * dt/2
    L = field.maximum(L_new, 0.0)

    # 3.4 相互摩擦力 (Mutual Friction)
    # Fns ~ B * ... * vns
    # L = (alpha_v/beta_v)**2 * (2*kinetic_energy.at(t))
    B_coeff = B / 3.0 * KAPPA * L.at(t)

    Fns_coeff = (B_coeff * (RHO_S/RHO))
    Fsn_coeff = (B_coeff * (RHO_N/RHO))

    Fns = Fns_coeff.at(vn) * vns
    Fsn = Fsn_coeff.at(vs) * vsn

    vn_safe = math.safe_div((vn + Fns_coeff.at(vn) * dt * vs_at_vn).values, 1 + (Fns_coeff.at(vn) * dt).values)
    vs_safe = math.safe_div((vs + Fsn_coeff.at(vs) * dt * vn_at_vs).values, 1 + (Fsn_coeff.at(vs) * dt).values)

    vn = vn.with_values(vn_safe)
    vs = vs.with_values(vs_safe)
    
    obstacle_mask_vn = resample(~(OBSTACLE.geometry), vn)
    vn = field.safe_mul(obstacle_mask_vn, vn)
    obstacle_mask_vs = resample(~(OBSTACLE.geometry), vs)
    vs = field.safe_mul(obstacle_mask_vs, vs)
    obstacle_mask_t = resample(~(OBSTACLE.geometry), t)
    t = field.safe_mul(obstacle_mask_t, t)
    obstacle_mask_L = resample(~(OBSTACLE.geometry), L)
    L = field.safe_mul(obstacle_mask_L, L)
    
    # 强制边界条件
    vn = StaggeredGrid(vn, Vn_BC, **DOMAIN)
    vs = StaggeredGrid(vs, Vs_BC, **DOMAIN)
    t = CenteredGrid(t, t_BC_THERMAL, **DOMAIN)
    L = CenteredGrid(L, ZERO_GRADIENT, **DOMAIN)
    p = CenteredGrid(p, ZERO_GRADIENT, **DOMAIN)

    # Update
    vs_at_vn = vs.at(t).at(vn)
    vns = vn - vs_at_vn
    vns_centered = vns.at(t)
    kinetic_energy = 0.5 * field.vec_squared(vns_centered)
    sqrt_2k = math.sqrt(2*kinetic_energy)
    term1 = -2.4 * KAPPA * ((L.at(t))**0.5)
    term2 = - (RHO_N/RHO) * sqrt_2k.at(t)
    vL_pre = (term1 + term2) / vs_mag.at(t) # 投影到 vs 方向
    vL_vec = vL_pre.at(vs) * vs # 这是一个近似，确保它是矢量场

    # 涡线平流
    L = advect.semi_lagrangian(L, vL_vec, dt/2)

    # 涡线生成与衰减 (Source/Sink)
    # 为了数值稳定性，将 L 限制在非负
    L = field.maximum(L, 0.0)
    L_prod   = (alpha_v.at(L) * sqrt_2k.at(L)) * (L**1.5)
    L_decay  = beta_v.at(L) * (L**2)
    L_remain = gamma_v.at(L) * ((sqrt_2k.at(L))**2.5) # 注意原公式幂次

    L_new = L + (L_prod - L_decay + L_remain) * dt/2
    L = field.maximum(L_new, 0.0)
    
    # L = (alpha_v/beta_v)**2 * (2*kinetic_energy.at(t))
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

    
    # --- 4. 障碍物边界条件 ---
    
    obstacle_mask_vn = resample(~(OBSTACLE.geometry), vn)
    vn = field.safe_mul(obstacle_mask_vn, vn)
    obstacle_mask_vs = resample(~(OBSTACLE.geometry), vs)
    vs = field.safe_mul(obstacle_mask_vs, vs)
    obstacle_mask_t = resample(~(OBSTACLE.geometry), t)
    t = field.safe_mul(obstacle_mask_t, t)
    obstacle_mask_L = resample(~(OBSTACLE.geometry), L)
    L = field.safe_mul(obstacle_mask_L, L)
    
    # 强制边界条件
    vn = StaggeredGrid(vn, Vn_BC, **DOMAIN)
    vs = StaggeredGrid(vs, Vs_BC, **DOMAIN)
    t = CenteredGrid(t, t_BC_THERMAL, **DOMAIN)
    L = CenteredGrid(L, ZERO_GRADIENT, **DOMAIN)
    p = CenteredGrid(p, ZERO_GRADIENT, **DOMAIN)
    
    # --- 5. 压力投影 (Pressure Projection) ---

    vn, p_n = fluid.make_incompressible(vn, OBSTACLE, PRESSURE_SOLVER)
    vs, p_s = fluid.make_incompressible(vs, OBSTACLE, PRESSURE_SOLVER)

    p = RHO.at(p) * (p_n.at(p) + p_s.at(p)) * 0.5

    return vn, vs, p, t, L

@jit_compile
def constrain_markers_push(markers, sphere_obstacle):
    """
    专用于圆柱/球体障碍物。将内部粒子推到表面。
    """
    center = sphere_obstacle.geometry.center
    radius = sphere_obstacle.geometry.radius
    
    # 计算向量：粒子 -> 圆心
    diff = markers - center
    # 计算距离
    dist = math.vec_length(diff)
    
    # 计算修正后的位置：方向不变，长度强制设为半径 + 一个微小量
    # direction = diff / dist
    # new_pos = center + direction * (radius + epsilon)
    epsilon = 1e-4
    correction = center + (diff / (dist + 1e-6)) * (radius + epsilon)
    
    # 只有当 dist < radius 时才应用修正
    is_inside = dist < radius
    fixed_markers = math.where(is_inside, correction, markers)
    
    return fixed_markers

# ==========================================
# 4. 生成 Ground Truth (真实观测数据)
# ==========================================
print("\nGenerating Ground Truth Data...")

# 初始化真实初始速度场
# 定义随机扰动函数
# v0_gt0 = StaggeredGrid(Noise(scale=0.001), **DOMAIN)

# 直接初始化为零场以简化
v0_gt0 = StaggeredGrid(0, Vn_BC, **DOMAIN)
obstacle_mask_vn0 = resample(~(OBSTACLE.geometry), v0_gt0)
v0_gt0 = field.safe_mul(obstacle_mask_vn0, v0_gt0)
v0_gt0, _ = fluid.make_incompressible(v0_gt0, OBSTACLE, PRESSURE_SOLVER)

vs0_gt0 = StaggeredGrid(0, Vs_BC, **DOMAIN)
obstacle_mask_vs0 = resample(~(OBSTACLE.geometry), vs0_gt0)
vs0_gt0 = field.safe_mul(obstacle_mask_vs0, vs0_gt0)
vs0_gt0, _ = fluid.make_incompressible(vs0_gt0, OBSTACLE, PRESSURE_SOLVER)


t0_gt0 = CenteredGrid(HEAT_SOURCE_INTENSITY, t_BC_THERMAL, **DOMAIN)
obstacle_mask_t0 = resample(~(OBSTACLE.geometry), t0_gt0)
t0_gt0 = field.safe_mul(obstacle_mask_t0, t0_gt0)

L0_gt0 = CenteredGrid(0, ZERO_GRADIENT, **DOMAIN)
obstacle_mask_L0 = resample(~(OBSTACLE.geometry), L0_gt0)
L0_gt0 = field.safe_mul(obstacle_mask_L0, L0_gt0)

p0_gt0 = CenteredGrid(3130, p_BC, **DOMAIN) # 初始压力基准
obstacle_mask_p0 = resample(~(OBSTACLE.geometry), p0_gt0)
p0_gt0 = field.safe_mul(obstacle_mask_p0, p0_gt0)


# 初始化示踪粒子
markers0 = DOMAIN['bounds'].sample_uniform(instance(markers=MARKERS))
markers0 = constrain_markers_push(markers0, OBSTACLE)


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

    
# 生成真实轨迹
print(f"Simulating Ground Truth for {STEPS} steps...")
gt_marker_trajectories = [initial_markers]
current_markers = initial_markers


for time_step in range(STEPS):
    current_v, current_vs, current_p, current_t, current_L = SFHelium_step(
        current_v, current_vs, current_p, current_t, current_L, dt=DT
    )
    current_markers = advect.points(current_markers, current_v, dt=DT, integrator=advect.rk4)
    current_markers = constrain_markers_push(current_markers, OBSTACLE)
    gt_marker_trajectories.append(current_markers)


# 转换为张量 (Time, Markers, Vector)
gt_marker_trajectories_stack = math.stack(gt_marker_trajectories, batch('time'))
print(f"Ground Truth Generated. Frames: {gt_marker_trajectories_stack.time.size}")


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

obstacle_mask_v_current = resample(~(OBSTACLE.geometry), v_current)
v_current = field.safe_mul(obstacle_mask_v_current, v_current)

obstacle_mask_vs_current = resample(~(OBSTACLE.geometry), vs_current)
vs_current = field.safe_mul(obstacle_mask_vs_current, vs_current)

obstacle_mask_p_current = resample(~(OBSTACLE.geometry), p_current)
p_current = field.safe_mul(obstacle_mask_p_current, p_current)

obstacle_mask_t_current = resample(~(OBSTACLE.geometry), t_current)
t_current = field.safe_mul(obstacle_mask_t_current, t_current)

obstacle_mask_L_current = resample(~(OBSTACLE.geometry), L_current)
L_current = field.safe_mul(obstacle_mask_L_current, L_current)

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
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
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
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
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
ani = animation.FuncAnimation(fig, update, frames=range(0, len(un_data), 2), interval=200, blit=False)

# 关闭静态图像显示
plt.close()

# 保存为 MP4 文件 (可选)
# ani.save("dual_fluid_streamline.mp4", fps=5, dpi=150, writer='ffmpeg')

# 在 Notebook 中显示 JavaScript 动画控件
HTML(ani.to_jshtml())

# ==========================================
# 5. 定义 Loss 函数与初始化猜测
# ==========================================
@jit_compile
def loss_function(v_guess_centered):
    """
    输入: CenteredGrid (为了优化器稳定性)
    输出: Scalar Loss (轨迹误差)
    """
    # 变量转换: Centered -> Staggered
    v = v_guess_centered.at(v0_gt)

    markers = initial_markers
    trajectory = [markers]
    

    # 前向模拟
    for _ in range(STEPS):
        v = boussinesq_step(v, dt=DT)
        markers = advect.points(markers, v, dt=DT, integrator=advect.rk4)
        markers = constrain_markers_push(markers, OBSTACLE)
        trajectory.append(markers)

    simulated_trajectories = math.stack(trajectory, batch('time'))

    # 计算 Loss (显式对所有维度求和，确保返回标量)
    diff = simulated_trajectories - gt_marker_trajectories_stack
    loss = math.sum(diff ** 2, dim=diff.shape)
    return loss

# 初始化猜测值 (使用全0场，但保持与 GT 相同的 Batch 维度)
batch_shape = v0_gt.shape.batch
print(f"Initializing guess with batch dims: {batch_shape}")

init_shape = batch_shape & spatial(x=Nx, y=Ny) & channel(vector='x,y')
init_values = math.zeros(init_shape)

v_guess_proxy = CenteredGrid(
    values=init_values,
    extrapolation=V_BC_PIPE,
    bounds=Box(x=Lx, y=Ly)
)
obstacle_mask_guess = resample(~(OBSTACLE.geometry), v_guess_proxy)
v_guess_proxy = field.safe_mul(obstacle_mask_guess, v_guess_proxy)



# ==========================================
# 6. 执行优化
# ==========================================
print("\nStarting Optimization...")
t_start = time.time()

# 使用 L-BFGS-B 优化器
# suppress=[phi.math.Diverged] 用于忽略优化器因精度原因提前停止的报错
with math.SolveTape(record_trajectories=False) as solves:
    result_centered = minimize(
        loss_function,
        Solve('L-BFGS-B', x0=v_guess_proxy, max_iterations=100, suppress=[phi.math.Diverged, phi.math.NotConverged])
    )

total_time = time.time() - t_start
v0_reconstructed = result_centered.at(v0_gt)
final_loss = loss_function(result_centered)

print(f"=== Optimization Complete ===")
print(f"Total Time: {total_time:.2f} s")
print(f"Final Loss: {final_loss:.4e}")


import jax
import jax.numpy as jnp
import time
from phi.jax.flow import *

# ==========================================
# 0. 假设已经定义好的全局变量 (从您的上下文中继承)
# ==========================================
# Nx, Ny, Lx, Ly, DT, STEPS
# Vn_BC, Vs_BC, p_BC, t_BC, L_BC (边界条件)
# DOMAIN (包含 bounds)
# SFHelium_step (物理步进函数)
# v0_gt, vs0_gt, p0_gt, t0_gt, L0_gt (初始场 GT，用于参考或辅助)
# initial_markers (初始粒子位置, 形状为 (markers, vector))
# gt_marker_trajectories_stack (GT 轨迹堆叠, 形状为 (time, markers, vector))

# ==========================================
# 1. 核心 JAX Scan 循环体 (Simulated Data 版)
# ==========================================
# 与真实数据版相比，这里简化了 inputs (不需要 reset mask 和 gt pos)
# 且 markers 直接作为 carry state 的一部分进行传递
@jit_compile
def physical_step_logic_sim(carry_state_native, _):
    """
    JAX Scan 循环体 - 模拟数据适配版
    _ : 占位符，因为不需要每一步的时间相关输入
    """
    # 1. 拆包 State
    # 注意：模拟数据通常不需要中途重置粒子，所以 markers 直接放在 carry 里更新
    (v_x_nat, v_y_nat), (vs_x_nat, vs_y_nat), p_nat, t_nat, L_nat, markers_nat = carry_state_native

    # 2. 获取全局常量 (避免泄漏)
    bounds = DOMAIN['bounds']
    
    # 3. 重组 PhiFlow 对象 (显式构造)
    grid_res = spatial(x=Nx, y=Ny)
    
    # --- 重组 Vn (Staggered) ---
    v_x_tensor = math.tensor(v_x_nat, spatial('x,y'))
    v_y_tensor = math.tensor(v_y_nat, spatial('x,y'))
    v_values = math.stack([v_x_tensor, v_y_tensor], dual(vector='x,y'))
    
    vn = StaggeredGrid(
        values=v_values,
        extrapolation=Vn_BC,
        bounds=bounds,
        resolution=grid_res
    )

    # --- 重组 Vs (Staggered) ---
    vs_x_tensor = math.tensor(vs_x_nat, spatial('x,y'))
    vs_y_tensor = math.tensor(vs_y_nat, spatial('x,y'))
    vs_values = math.stack([vs_x_tensor, vs_y_tensor], dual(vector='x,y'))
    
    vs = StaggeredGrid(
        values=vs_values,
        extrapolation=Vs_BC,
        bounds=bounds,
        resolution=grid_res
    )

    # --- 重组 Scalar Fields (Centered) ---
    p = CenteredGrid(values=math.tensor(p_nat, grid_res), extrapolation=p0_gt.extrapolation, bounds=bounds, resolution=grid_res)
    t = CenteredGrid(values=math.tensor(t_nat, grid_res), extrapolation=t0_gt.extrapolation, bounds=bounds, resolution=grid_res)
    L = CenteredGrid(values=math.tensor(L_nat, grid_res), extrapolation=L0_gt.extrapolation, bounds=bounds, resolution=grid_res)
    
    # --- 重组 Markers ---
    # 模拟数据中 markers 通常只是坐标张量
    # 注意维度：markers_nat 是 native array，转回 tensor
    current_markers = math.tensor(markers_nat, instance('markers') & channel(vector='x,y'))

    # 4. 执行物理计算 (SFHelium Step)
    vn_next, vs_next, p_next, t_next, L_next = SFHelium_step(
        vn, vs, p, t, L, dt=DT
    )

    # 5. 粒子平流 (Advection)
    # 直接对坐标场进行平流，不需要 PointCloud 包装，因为没有重置逻辑
    # advect.advect 适用于 Field，advect.points 适用于几何
    # 在模拟数据中，如果 initial_markers 是 Tensor，这里最方便的是将其视为 PointCloud 的几何中心进行平流
    # 或者使用 advect.mac_cormack / semi_lagrangian 如果是场
    
    # 这里保持与您原代码一致，使用 advect.points
    # 为了兼容性，临时包装成 Geometry，操作完取回 center
    points_obj = geom.Point(current_markers)
    advected_points = advect.points(points_obj, vn_next, dt=DT, integrator=advect.rk4)
    next_markers = advected_points.center # Tensor
    next_markers = constrain_markers_push(next_markers, OBSTACLE)

    # 6. 拆包返回值 (Native)
    new_carry_native = (
        (vn_next.vector['x'].values.native(['x', 'y']), vn_next.vector['y'].values.native(['x', 'y'])),
        (vs_next.vector['x'].values.native(['x', 'y']), vs_next.vector['y'].values.native(['x', 'y'])),
        p_next.values.native(['x', 'y']),
        t_next.values.native(['x', 'y']),
        L_next.values.native(['x', 'y']),
        next_markers.native(['markers', 'vector'])
    )

    # scan 的输出：只需要当前步的粒子位置
    output_native = next_markers.native(['markers', 'vector'])

    return new_carry_native, output_native

# Checkpoint 优化
step_fn_checkpointed = jax.checkpoint(physical_step_logic_sim)


# ==========================================
# 2. Loss 函数 (集成正则化)
# ==========================================
@jit_compile
def loss_function(v_guess_centered):
    """
    输入: CenteredGrid (Vn 的初始猜测)
    假设: 其他场 (Vs, p, t, L) 使用 GT 初始值
    """
    
    # --- A. 准备初始状态 (Init State) ---
    # 将 CenteredGrid 猜测值插值回 StaggeredGrid
    vn_sim = v_guess_centered.at(v0_gt)

    # 拆解初始状态为 Native Arrays (Scan 需要 tuple of arrays)
    v_init_tuple = (
        vn_sim.vector['x'].values.native(['x', 'y']),
        vn_sim.vector['y'].values.native(['x', 'y'])
    )
    # 假设 Vs, P, T, L 的初始值已知且固定 (使用 GT)
    vs_init_tuple = (
        vs0_gt.vector['x'].values.native(['x', 'y']),
        vs0_gt.vector['y'].values.native(['x', 'y'])
    )

    # 初始 State 打包
    # 注意：模拟数据中，initial_markers 直接作为 State 的一部分传入
    state_init_native = (
        v_init_tuple,
        vs_init_tuple,
        p0_gt.values.native(['x', 'y']),
        t0_gt.values.native(['x', 'y']),
        L0_gt.values.native(['x', 'y']),
        initial_markers.native(['markers', 'vector']) # 初始粒子位置
    )

    # --- B. 准备 Scan 输入 (Dummy Inputs) ---
    # 模拟数据通常不需要随时间变化的外部输入 (如 mask)，所以传一个长度为 STEPS 的空数组占位
    # jnp.arange(STEPS) 仅用于驱动循环次数
    scan_inputs = jnp.arange(STEPS)

    # --- C. 执行 JAX Scan ---
    final_state_native, trajectory_stack_native = jax.lax.scan(
        step_fn_checkpointed,
        state_init_native,
        scan_inputs
    )

    # --- D. 计算 MSE Loss ---
    # trajectory_stack_native 形状: (time, markers, vector)
    # gt_marker_trajectories_stack 包含 t=0，切片去掉 t=0 (如果 scan 输出不含初始帧)
    # 通常 scan 输出的是 t=1 到 t=STEPS 的结果
    
    # 假设 GT stack 包含了 t=0...T。需要对齐。
    # 这里假设 scan 输出了 STEPS 步，对应 GT 的 [1:]
    gt_target_native = gt_marker_trajectories_stack.time[1:].native(['time', 'markers', 'vector'])
    
    diff = trajectory_stack_native - gt_target_native
    mse_loss = jnp.sum(diff ** 2)

    # --- E. 正则化项 (Regularization) ---
    
    # 1. 平滑度正则化 (Smoothness)
    # 对 v_guess_centered (输入变量) 计算梯度
    u_comp = v_guess_centered.vector['x']
    v_comp = v_guess_centered.vector['y']
    
    grad_u = field.spatial_gradient(u_comp)
    grad_v = field.spatial_gradient(v_comp)
    
    # 因子需要根据模拟数据的量级调整
    alpha_smooth = 1e-4 
    smoothness_loss = field.l2_loss(grad_u) + field.l2_loss(grad_v)

    # 2. 能量正则化 (Energy Penalty)
    # 防止初始速度过大
    vn_vals_native = v_guess_centered.values.native(['x', 'y', 'vector'])
    beta_energy = 1e-5
    energy_loss = 0.5 * jnp.sum(vn_vals_native ** 2)

    # --- F. 总损失 ---
    total_loss = mse_loss + alpha_smooth * smoothness_loss + beta_energy * energy_loss
    
    return total_loss


# ==========================================
# 3. 初始化优化变量
# ==========================================
print("\nInitializing Optimization Guess...")

# 模拟实验可以直接用全 0 初始化，或者用 GT 的平滑版本
# 这里保持您原代码的逻辑：创建一个与 GT 形状一致的 CenteredGrid
# 注意：v0_gt 可能有 batch 维度，如果有，需要去掉或保持一致

# 假设 v0_gt 是 (x, y, vector)，没有 batch
# 如果 v0_gt 是 Staggered，我们先转为 Centered 作为优化基底
v0_gt_centered = v0_gt.at_centers()

# 构造一个全 0 的猜测 (或者加上微小噪音)
init_values = math.zeros_like(v0_gt_centered.values)

v_guess_proxy = CenteredGrid(
    values=init_values,
    extrapolation=Vn_BC,
    bounds=DOMAIN['bounds']
)

# 如果有障碍物，应用 Mask (可选)
obstacle_mask_guess = resample(~(OBSTACLE.geometry), v_guess_proxy)
v_guess_proxy = field.safe_mul(obstacle_mask_guess, v_guess_proxy)


# ==========================================
# 4. 执行优化 (L-BFGS-B)
# ==========================================
print("\nStarting Optimization (L-BFGS-B)...")
t_start = time.time()

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

# ==========================================
# 5. 结果处理
# ==========================================
# 将优化结果转回 StaggeredGrid 用于显示或保存
v0_reconstructed = result_centered.at(v0_gt)
final_loss = loss_function(result_centered)

print(f"=== Optimization Result ===")
print(f"Final Loss: {final_loss:.6e}")
print(f"Iterations: {len(solves)}")

# 简单验证：比较重建场与 GT 场的差异
diff_field = v0_reconstructed - v0_gt
diff_mag = field.l2_loss(diff_field)
print(f"Field Reconstruction Error (L2): {diff_mag:.6e}")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from IPython.display import HTML

# ==========================================
# 7. 后处理：重建全时序流场 (Re-simulation)
# ==========================================
print("\nRe-simulating to get full flow field evolution...")

# 1. 初始化物理场
# ------------------------------------------------
# Vn 使用优化后的结果，其他场使用 GT 初始值
v_current = v0_reconstructed
vs_current = vs0_gt
p_current = p0_gt
t_current = t0_gt
L_current = L0_gt

obstacle_mask_v_current = resample(~(OBSTACLE.geometry), v_current)
v_current = field.safe_mul(obstacle_mask_v_current, v_current)

obstacle_mask_vs_current = resample(~(OBSTACLE.geometry), vs_current)
vs_current = field.safe_mul(obstacle_mask_vs_current, vs_current)

obstacle_mask_p_current = resample(~(OBSTACLE.geometry), p_current)
p_current = field.safe_mul(obstacle_mask_p_current, p_current)

obstacle_mask_t_current = resample(~(OBSTACLE.geometry), t_current)
t_current = field.safe_mul(obstacle_mask_t_current, t_current)

obstacle_mask_L_current = resample(~(OBSTACLE.geometry), L_current)
L_current = field.safe_mul(obstacle_mask_L_current, L_current)

# 2. 存储容器
# ------------------------------------------------
reconstructed_velocities_n = [v_current]
reconstructed_velocities_s = [vs_current]
# 如果内存允许，也可以存储温度场
# reconstructed_temperatures = [t_current]

# 3. 执行物理演化循环 (仅计算流场)
# ------------------------------------------------
print(f"Simulating {STEPS} steps of fluid dynamics...")

for t in range(STEPS):
    # 执行一步 SFHelium 物理模拟
    v_current, vs_current, p_current, t_current, L_current = SFHelium_step(
        v_current, vs_current, p_current, t_current, L_current, dt=DT
    )
    
    # 存储结果
    reconstructed_velocities_n.append(v_current)
    reconstructed_velocities_s.append(vs_current)

# 4. 堆叠流场结果 (Time, Y, X, Vector)
# ------------------------------------------------
v_stack = math.stack(reconstructed_velocities_n, batch('time'))
vs_stack = math.stack(reconstructed_velocities_s, batch('time'))
print(f"Flow field stack shape: {v_stack.shape}")


# ==========================================
# 8. 后处理：计算粒子轨迹 (Particle Advection)
# ==========================================
print("Calculating marker trajectories for visualization...")

# 1. 准备初始粒子
# ------------------------------------------------
# 直接使用 Tensor，不需要 Geometry/PointCloud 包装
current_markers = initial_markers

# 处理可能的 Batch 维度
if 'seed' in current_markers.shape:
    current_markers = current_markers[{'seed': 0}]
else:
    current_markers = current_markers

# 2. 遍历流场进行平流
# ------------------------------------------------
marker_positions = [current_markers.numpy('markers, vector')]

# 遍历每一帧 (除了最后一帧)
# 注意：粒子通常跟随正常流体 Vn 运动
for i in range(len(reconstructed_velocities_n) - 1):
    # 取出当前帧 Vn
    v_field = reconstructed_velocities_n[i]
    if 'seed' in v_field.shape:
        v_field = v_field[{'seed': 0}]
        
    # 执行平流 (advect.points 直接支持坐标 Tensor)
    current_markers = advect.points(current_markers, v_field, dt=DT, integrator=advect.rk4)
    
    # 如果有障碍物，应用约束 (可选)
    current_markers = constrain_markers_push(current_markers, OBSTACLE)
    
    # 存入列表
    marker_positions.append(current_markers.numpy('markers, vector'))

# 转为 Numpy 数组: (Time, Markers, 2)
marker_positions_np = np.array(marker_positions)
print(f"Particle trajectory shape: {marker_positions_np.shape}")


# ==========================================
# 9. 可视化：双流体流线 + 粒子动画
# ==========================================
print("Generating Dual-Fluid Visualization...")

# --- A. 数据提取与网格准备 ---
# 1. 提取 Vn 数据
if 'seed' in v_stack.shape: v_recon_display = v_stack[{'seed': 0}]
else: v_recon_display = v_stack

v_centered = v_recon_display.at_centers()
un_data = v_centered.vector['x'].values.numpy('time, y, x')
vn_data = v_centered.vector['y'].values.numpy('time, y, x')
vn_mag_data = np.sqrt(un_data**2 + vn_data**2)

# 2. 提取 Vs 数据
if 'seed' in vs_stack.shape: vs_recon_display = vs_stack[{'seed': 0}]
else: vs_recon_display = vs_stack

vs_centered = vs_recon_display.at_centers()
us_data = vs_centered.vector['x'].values.numpy('time, y, x')
vs_y_data = vs_centered.vector['y'].values.numpy('time, y, x')
vs_mag_data = np.sqrt(us_data**2 + vs_y_data**2)

# 3. 物理网格
x_phys = np.linspace(0, Lx, Nx)
y_phys = np.linspace(0, Ly, Ny)
X_grid, Y_grid = np.meshgrid(x_phys, y_phys)

# --- B. 配色方案 ---
import matplotlib as mpl
from matplotlib.colors import ListedColormap

Vn_MIN, Vn_MAX = 0, np.max(vn_mag_data)
Vs_MIN, Vs_MAX = 0, np.max(vs_mag_data)

# 自定义色条 (红暖色 vs 蓝冷色)
bmap = mpl.cm.twilight_shifted
rmap = mpl.cm.twilight
bluecolors = bmap(np.linspace(0, 1, 256))
redcolors = rmap(np.linspace(0, 1, 256))
cblue = ListedColormap(bluecolors[:100])      # Vs 使用冷色调
cred = ListedColormap(redcolors[128:228])     # Vn 使用暖色调

# --- C. 动画初始化 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 设置 Colorbar
norm_vn = mcolors.Normalize(vmin=Vn_MIN, vmax=Vn_MAX)
cbar1 = fig.colorbar(cm.ScalarMappable(norm=norm_vn, cmap=cred), ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('|Vn| (m/s)')

norm_vs = mcolors.Normalize(vmin=Vs_MIN, vmax=Vs_MAX)
cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm_vs, cmap=cblue), ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('|Vs| (m/s)')

# --- D. 动画更新函数 ---
def update(frame):
    ax1.clear()
    ax2.clear()
    
    # 获取当前帧粒子坐标
    px = marker_positions_np[frame, :, 0]
    py = marker_positions_np[frame, :, 1]

    # --- 左图：Vn (Normal Fluid) + Markers ---
    ax1.set_title(f"Vn (Normal) - T={frame*DT:.3f}s")
    # 背景云图
    ax1.contourf(
        X_grid, Y_grid, vn_mag_data[frame], levels=50, cmap=cred,
        vmin=Vn_MIN, vmax=Vn_MAX
    )
    # 流线
    ax1.streamplot(
        X_grid, Y_grid, un_data[frame], vn_data[frame],
        color='black', linewidth=0.8, arrowsize=0.8, density=1.0
    )
    # 粒子 (通常粒子跟随 Vn，所以在 Vn 图中画实心)
    ax1.scatter(px, py, color='cyan', s=30, edgecolors='white', linewidths=0.5, zorder=10, label='Markers')
    
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(0, Ly)
    ax1.set_aspect('equal')

    # --- 右图：Vs (Superfluid) + Markers (对比用) ---
    ax2.set_title(f"Vs (Superfluid) - T={frame*DT:.3f}s")
    ax2.contourf(
        X_grid, Y_grid, vs_mag_data[frame], levels=50, cmap=cblue,
        vmin=Vs_MIN, vmax=Vs_MAX
    )
    ax2.streamplot(
        X_grid, Y_grid, us_data[frame], vs_y_data[frame],
        color='white', linewidth=0.8, arrowsize=0.8, density=1.0
    )
    # 在 Vs 图中也画粒子，方便观察粒子是否与超流体解耦
    ax2.scatter(px, py, color='cyan', s=30, edgecolors='white', alpha=0.6, linewidths=0.5, zorder=10)

    ax2.set_xlim(0, Lx)
    ax2.set_ylim(0, Ly)
    ax2.set_aspect('equal')

    return ax1, ax2

# --- E. 渲染与保存 ---
print("Rendering animation...")
plt.tight_layout()
# interval=200ms -> 5fps
ani = animation.FuncAnimation(fig, update, frames=len(un_data), interval=200, blit=False)
plt.close()

# 保存
# ani.save("Dual_Fluid_Simulation.gif", fps=10, dpi=120)


HTML(ani.to_jshtml())

