import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# ==========================================
# 0. CPU 并行设置 (必须在 import jax/phiflow 之前运行)
# ==========================================
# 启用 JAX 的 64 位浮点支持，提高物理模拟精度
os.environ["JAX_ENABLE_X64"] = "True"
# 强制使用 8 个线程进行计算
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"
# 针对底层数学库
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
# 强制 JAX 使用 CPU 平台
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# ==========================================
# 1. 导入库与环境验证
# ==========================================
from phi.jax.flow import *
import jax
import matplotlib.colors as mcolors
import matplotlib.cm as cm

print(f"JAX Backend: {jax.devices()[0].platform}")
print(f"Device Info: {jax.devices()[0].device_kind}")
print("CPU Parallelism Configured: 8 Threads")
# 设置64位精度
math.set_global_precision(64)

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
Lx, Ly = 0.0004, 0.0016
Nx, Ny = 20, 80
DOMAIN = dict(x=Nx, y=Ny, bounds=Box(x=Lx, y=Ly))
MARKERS = 128
# 时间配置
DT = 1e-6             # 时间步长
STEPS = 5            # 总时间步数
PRE_STEPS = 200        # 预先计算步数以达到稳定状态

HEAT_SOURCE_INTENSITY = 2.0
FTP = 1450/((145.6217**2) * (1559**4) * (2.17**3)) * ((HEAT_SOURCE_INTENSITY/2.17)**5.7 * (1-(HEAT_SOURCE_INTENSITY/2.17)**5.7))**(-3.0)
# FTP = A_lambda/(rho^2 s_lambda^4 T_lambda^3) [t^5.7(1-t^5.7)]^(-3)



COLD_SOURCE_INTENSITY = 2.0
PRESSURE_0 = 3130
HEAT_FLUX = 2170.0
Vn_IN = HEAT_FLUX/(145.6217 * 962.1 * HEAT_SOURCE_INTENSITY)
Vs_IN = -(80.55*Vn_IN)/65.07

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
    
    

    t_low, t_up = field.shift(t, offsets=(-1,1), dims='y')
    t = t_low[0] + heating_zone * FTP * HEAT_FLUX**(3.4) * Ly/Ny
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

# ==========================================
# 4. 生成 Ground Truth (真实观测数据)
# ==========================================
print("\nGenerating Ground Truth Data...")

# 初始化真实初始速度场
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

# 初始化示踪粒子
# 使用 sample_uniform 在 bounds 内随机撒点
markers0 = DOMAIN['bounds'].sample_uniform(instance(markers=MARKERS))

# 预热 (Pre-computation)
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

# 生成 GT 轨迹
print(f"Simulating Ground Truth for {STEPS} steps...")
gt_marker_trajectories = [initial_markers]
current_markers = initial_markers

for time_step in range(STEPS):
    current_v, current_vs, current_p, current_t, current_L = SFHelium_step(
        current_v, current_vs, current_p, current_t, current_L, dt=DT
    )
    # 粒子平流：使用 RK4 积分器提高精度
    current_markers = advect.points(current_markers, current_v, dt=DT, integrator=advect.rk4)
    gt_marker_trajectories.append(current_markers)

# 转换为张量 (Time, Markers, Vector)
gt_marker_trajectories_stack = math.stack(gt_marker_trajectories, batch('time'))
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
p_stack = math.stack(reconstructed_p, batch('time'))
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
sm_vn = cm.ScalarMappable(norm=norm_vn, cmap='viridis')
cbar1 = fig.colorbar(sm_vn, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Magnitude (m/s)')

# 右图 (Vs)
norm_vs = mcolors.Normalize(vmin=Vs_MIN, vmax=Vs_MAX)
sm_vs = cm.ScalarMappable(norm=norm_vs, cmap='viridis')
cbar2 = fig.colorbar(sm_vs, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Magnitude (m/s)')

# --- C. 动画更新函数 ---

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
        cmap='viridis',
        vmin=Vn_MIN,
        vmax=Vn_MAX * 0.8 if Vn_MAX > 0 else 1.0 # 防止最大值为0时报错
    )
    # 2. 流线图 (Streamplot)
    ax1.streamplot(
        X_grid, Y_grid,
        un_data[frame], vn_data[frame],
        color='white', # 在深色云图上用白色流线更清晰
        linewidth=1.0, arrowsize=0.8, density=1.2
    )

    # --- 绘制子图 2: Vs ---
    # 1. 速度大小云图
    ax2.contourf(
        X_grid, Y_grid,
        vs_mag_data[frame],
        levels=50,
        cmap='viridis',
        vmin=Vs_MIN,
        vmax=Vs_MAX * 0.8 if Vs_MAX > 0 else 1.0
    )
    # 2. 流线图
    ax2.streamplot(
        X_grid, Y_grid,
        us_data[frame], vs_y_data[frame],
        color='white',
        linewidth=1.0, arrowsize=0.8, density=1.2
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
ani = animation.FuncAnimation(fig, update, frames=len(un_data), interval=200, blit=False)

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

ani_t = animation.FuncAnimation(fig_t, update_temp, frames=len(t_data_np), interval=200, blit=False)
plt.close()

# 保存为 MP4 文件 (可选)
# ani_t.save("temperature_field_animation.mp4", fps=10, dpi=150)

display(HTML(ani_t.to_jshtml()))

# ==========================================
# 5. 定义 Loss 函数与初始化猜测
# ==========================================

# 优化目标：找到 v_guess_centered，使得从它开始模拟得到的粒子轨迹与 GT 轨迹一致
# 注意：我们这里只优化 vn 的初始场，假设 vs, t, L 已知 (或从预热后的状态开始)

@jit_compile
def loss_function(v_guess_centered):
    """
    计算模拟轨迹与真实轨迹的均方误差 (MSE)。
    输入: CenteredGrid (优化器友好的形式)
    输出: Scalar Loss
    """
    # 1. 变量转换: Centered -> Staggered (恢复物理场结构)
    # 使用 .at() 将猜测的中心网格值投影到 Staggered 网格上，并应用边界条件
    v_sim = v_guess_centered.at(v0_gt)

    # 其他物理量保持 GT 的初始状态 (假设已知)
    t_sim = t0_gt
    vs_sim = vs0_gt
    p_sim = p0_gt
    L_sim = L0_gt

    markers_sim = initial_markers
    trajectory_list = [markers_sim]

    # 2. 可微模拟循环
    for _ in range(STEPS):
        v_sim, vs_sim, p_sim, t_sim, L_sim = SFHelium_step(
            v_sim, vs_sim, p_sim, t_sim, L_sim, dt=DT
        )
        markers_sim = advect.points(markers_sim, v_sim, dt=DT, integrator=advect.rk4)
        trajectory_list.append(markers_sim)

    # 3. 堆叠轨迹
    simulated_trajectories_stack = math.stack(trajectory_list, batch('time'))

    # 4. 计算 Loss
    # MSE = sum((x_sim - x_gt)^2)
    # 对 time, markers, vector 维度求和
    diff = simulated_trajectories_stack - gt_marker_trajectories_stack
    loss = math.sum(diff ** 2, dim=diff.shape)

    return loss

# 初始化猜测值
# 为了避免优化器在全零梯度处卡住，可以添加微小的随机噪声，这里为了复现性使用全零
print("\nInitializing Optimization Guess...")
init_values = math.zeros(v0_gt.shape.batch & spatial(x=Nx, y=Ny) & channel(vector='x,y'))

# 使用 CenteredGrid 作为优化变量，因为它比 StaggeredGrid 更容易处理形状
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
# *. 测试流场Vn和Vs
# ==========================================
print("\nRe-simulating to get full flow field evolution...")

# 确保起点是纯粹的 Field，而不是优化器的变量状态
v_current = v0_reconstructed
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
p_stack = math.stack(reconstructed_p, batch('time'))
t_stack = math.stack(reconstructed_temperatures, batch('time'))
L_stack = math.stack(reconstructed_Line, batch('time'))


# ==========================================
# 可视化：reconstruct 双流体流线对比 (Vn vs Vs)
# ==========================================
print("Generating dual-fluid visualization data...")

# --- A. 数据准备 ---

# 1. 处理 Vn (正常流体)
if 'seed' in v_stack.shape:
    v_recon_display = v_stack[{'seed': 0}]
else:
    v_recon_display = v_stack

v_centered = v_recon_display.at_centers()
un_data = v_centered.vector['x'].values.numpy('time, y, x')
vn_data = v_centered.vector['y'].values.numpy('time, y, x')
vn_mag_data = np.sqrt(un_data**2 + vn_data**2)

# 2. 处理 Vs (超流体)
if 'seed' in vs_stack.shape:
    vs_recon_display = vs_stack[{'seed': 0}]
else:
    vs_recon_display = vs_stack

vs_centered = vs_recon_display.at_centers()
us_data = vs_centered.vector['x'].values.numpy('time, y, x')
vs_y_data = vs_centered.vector['y'].values.numpy('time, y, x') # 重命名避免混淆
vs_mag_data = np.sqrt(us_data**2 + vs_y_data**2)


# B. 单独计算粒子轨迹 (纯数值计算，避免 JAX Tracer 错误)
# ------------------------------------------------
print("Calculating marker trajectories for visualization...")
current_markers = initial_markers
# 从 v_stack 中提取每一帧的速度场进行平流
# 这样避免了在 python 循环中反复调用 JIT 函数，速度更快且稳定
marker_positions = [current_markers.numpy('markers, vector')]

# 遍历每一帧流场 (除了最后一帧，因为它没有下一刻了)
for i in range(len(reconstructed_velocities) - 1):
    # 取出当前帧速度场
    v_field = reconstructed_velocities[i]
    if 'seed' in v_field.shape:
        v_field = v_field[{'seed': 0}]

    # 平流粒子
    current_markers = advect.points(current_markers, v_field, dt=DT, integrator=advect.rk4)
    marker_positions.append(current_markers.numpy('markers, vector'))

marker_positions_np = np.array(marker_positions)




# 3. 计算全局物理网格和绘图范围
x_phys = np.linspace(0, Lx, Nx)
y_phys = np.linspace(0, Ly, Ny)
X_grid, Y_grid = np.meshgrid(x_phys, y_phys)

# 为两个子图分别设置颜色映射范围
Vn_MIN, Vn_MAX = 0, 1*np.max(vn_mag_data)
Vs_MIN, Vs_MAX = 0, 1*np.max(vs_mag_data)

# --- B. 动画画布设置 ---

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
sm_vn = cm.ScalarMappable(norm=norm_vn, cmap='viridis')
cbar1 = fig.colorbar(sm_vn, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Magnitude (m/s)')

# 右图 (Vs)
norm_vs = mcolors.Normalize(vmin=Vs_MIN, vmax=Vs_MAX)
sm_vs = cm.ScalarMappable(norm=norm_vs, cmap='viridis')
cbar2 = fig.colorbar(sm_vs, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Magnitude (m/s)')

# --- C. 动画更新函数 ---

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
        cmap='viridis',
        vmin=Vn_MIN,
        vmax=Vn_MAX * 0.8 if Vn_MAX > 0 else 1.0 # 防止最大值为0时报错
    )
    # 2. 流线图 (Streamplot)
    ax1.streamplot(
        X_grid, Y_grid,
        un_data[frame], vn_data[frame],
        color='white', # 在深色云图上用白色流线更清晰
        linewidth=1.0, arrowsize=0.8, density=1.2
    )
    # 3. 绘制粒子
    px = marker_positions_np[frame, :, 0]
    py = marker_positions_np[frame, :, 1]
    ax1.scatter(px, py, color='orange', s=40, edgecolors='white', zorder=10)

    # --- 绘制子图 2: Vs ---
    # 1. 速度大小云图
    ax2.contourf(
        X_grid, Y_grid,
        vs_mag_data[frame],
        levels=50,
        cmap='viridis',
        vmin=Vs_MIN,
        vmax=Vs_MAX * 0.8 if Vs_MAX > 0 else 1.0
    )
    # 2. 流线图
    ax2.streamplot(
        X_grid, Y_grid,
        us_data[frame], vs_y_data[frame],
        color='white',
        linewidth=1.0, arrowsize=0.8, density=1.2
    )
    # 3. 绘制粒子
    px = marker_positions_np[frame, :, 0]
    py = marker_positions_np[frame, :, 1]
    ax2.scatter(px, py, color='orange', s=40, edgecolors='white', zorder=10)

    # 返回需要更新的 Artists 对象 (blit=True 时需要，这里 blit=False 可省略或返回空)
    return ax1, ax2

# --- D. 生成并显示动画 ---
print("Rendering dual-fluid animation...")
# 使用 tight_layout 自动调整子图间距，避免标题和坐标轴重叠
plt.tight_layout()

# 创建动画对象
# frames=len(un_data) 确保遍历所有时间步
# interval=200 设置帧间隔为 200ms (即 5 fps)
ani = animation.FuncAnimation(fig, update, frames=len(un_data), interval=200, blit=False)

# 关闭静态图像显示
plt.close()

# 保存为 MP4 文件 (可选)
# ani.save("dual_fluid_streamline.mp4", fps=5, dpi=150, writer='ffmpeg')

# 在 Notebook 中显示 JavaScript 动画控件
HTML(ani.to_jshtml())

# ==========================================
# 10. 可视化：速度大小对比 (Reconstructed vs Ground Truth)
# ==========================================
print("\nGenerating Velocity Magnititude Comparison Animation...")
# 3. 计算 Mag 并转为 Numpy
# 2D 速度场的 Mag 是标量场
mag_recon_np = field.vec_length(v_recon_display).values.numpy('time, y, x')
mag_gt_np = field.vec_length(v_display).values.numpy('time, y, x')

# 4. 计算误差场
diff_mag_np = mag_recon_np - mag_gt_np

# --- C. 设置绘图参数 ---
# 统一量程：使用 GT 的最大绝对值，保证对比公平
max_val1_mag = np.max(np.abs(mag_gt_np)) + 1e-8
max_val2_mag = np.max(np.abs(mag_recon_np)) + 1e-8
# 误差量程：通常误差较小，单独计算最大值以便观察细节
max_err_mag = np.max(np.abs(diff_mag_np)) + 1e-8

def add_colorbar(im, ax, label):
    cbar = fig_comp.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    return cbar


# --- D. 生成三联画动画 ---
fig_comp, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
ax1, ax2, ax3 = axes

# 1. Ground Truth
im1 = ax1.imshow(mag_gt_np[0], origin='lower', cmap='PuOr', vmin=-max_val1_mag, vmax=max_val1_mag, extent=[0,5,0,5])
ax1.set_title("Ground Truth Vn", fontsize=12, pad=10)
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
add_colorbar(im1, ax1, "Vn (m/s)")


# 2. Reconstructed
im2 = ax2.imshow(mag_recon_np[0], origin='lower', cmap='PuOr', vmin=-max_val1_mag, vmax=max_val1_mag, extent=[0,5,0,5])
ax2.set_title("Reconstructed Vn")
ax2.set_xlabel("X (m)")
add_colorbar(im2, ax2, "Vn (m/s)")

# 3. Difference
im3 = ax3.imshow(diff_mag_np[0], origin='lower', cmap='PuOr', vmin=-max_val1_mag, vmax=max_val1_mag, extent=[0,5,0,5])
ax3.set_title(f"Difference")
ax3.set_xlabel("X (m)")
add_colorbar(im3, ax3, "Vn (m/s)")

def update_comp(frame):
    # 更新数据
    im1.set_data(mag_gt_np[frame])
    im2.set_data(mag_recon_np[frame])
    im3.set_data(diff_mag_np[frame])

    # 更新标题
    fig_comp.suptitle(f"Vn Comparison - T={frame*DT:.2f}s", fontsize=16)
    return im1, im2, im3

ani_comp = animation.FuncAnimation(fig_comp, update_comp, frames=len(mag_gt_np), interval=200, blit=False)
plt.close()

# 保存为 MP4 文件 (可选)
# ani_comp.save("vn_comparison.mp4", fps=10, dpi=150)

display(HTML(ani_comp.to_jshtml()))