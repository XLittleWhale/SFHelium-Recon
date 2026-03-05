# ==========================================
# 2. 物理参数与求解器配置
# ==========================================
# 空间配置
Lx, Ly = 0.1, 0.1
Nx, Ny = 64, 64
DOMAIN = dict(x=Nx, y=Ny, bounds=Box(x=Lx, y=Ly))
MARKERS = 128

# 时间与物理参数（Water at 25℃）
DT = 0.1            # 时间步长
STEPS = 10          # 总时间步数
PRE_STEPS = 100     # 预先计算步数以达到稳定状态
VISCOSITY = 8.6e-7     # 运动黏度
THERMAL_DIFFUSIVITY = 1.5e-7   # 热扩散系数 (新增)
BUOYANCY_FACTOR = 2.7e-3
BUOYANCY_COEFFS = [2.5e-3, 9e-5, -1e-5]       # 浮力系数 (对应 alpha * g)

# 压力求解器配置 (限制迭代次数以加速 JIT 编译与运行)
# suppress 用于忽略 "未完全收敛" 的警告，这对反向传播通常是可以接受的
PRESSURE_SOLVER = Solve('CG', 1e-10, x0=None, max_iterations=20, suppress=[phi.math.NotConverged])

# ==========================================
# 3. 定义物理模型 (JIT 编译)
# ==========================================

# 定义热源
HEAT_SOURCE_INTENSITY = 10.0
COLD_SOURCE_INTENSITY = -10.0

V_BC = {'x': 0, 'y': 0}
t_BC_THERMAL = {'x': ZERO_GRADIENT, 'y-': HEAT_SOURCE_INTENSITY, 'y+': COLD_SOURCE_INTENSITY}

@jit_compile
def boussinesq_step(v, t, dt):
    """
    Boussinesq 近似求解器：
    v: 速度场 (StaggeredGrid)
    t: 温度场 (CenteredGrid, 作为密度标量)
    """
    # 1. 对流 (Coupled Advection)
    # 速度场平流自身，同时也平流温度场
    v = advect.semi_lagrangian(v, v, dt)
    t = advect.semi_lagrangian(t, v, dt)

    # 2. 扩散 (Diffusion)
    # 分别对 速度(黏性) 和 温度(热传导) 进行显式扩散
    v = diffuse.explicit(v, VISCOSITY, dt)
    t = diffuse.explicit(t, THERMAL_DIFFUSIVITY, dt)

    # 3. 添加浮力 (Buoyancy Force)
    # Boussinesq 近似: 密度变化由温度决定，产生沿重力方向(这里设为Y轴向上)的浮力
    # Force = factor * temperature * (0, 1)
    # 注意: t 是 CenteredGrid, v 是 StaggeredGrid, .at(v) 会自动进行插值

    t_at_v = t.at(v)
    scalar_buoyancy = 0
    for power, coeff in enumerate(BUOYANCY_COEFFS):
        scalar_buoyancy += coeff * (t_at_v ** power)
    buoyancy_force = t_at_v * scalar_buoyancy * (0, 1)

    v = v + buoyancy_force * dt

    # 4. 压力投影 (不可压缩约束)
    v, _ = fluid.make_incompressible(v, [], PRESSURE_SOLVER)

    return v, t

# ==========================================
# 4. 生成 Ground Truth (真实观测数据)
# ==========================================
print("\nGenerating Ground Truth Data...")

# 初始化真实初始速度场
# 定义随机扰动函数
# v0_gt0 = StaggeredGrid(Noise(scale=0.001), **DOMAIN)

# 直接初始化为零场以简化
v0_gt0 = StaggeredGrid(0, V_BC, **DOMAIN)
v0_gt0, _ = fluid.make_incompressible(v0_gt0, [], PRESSURE_SOLVER)

# 初始化真实初始温度场
# 定义线性温度梯度函数
def linear_temp_gradient(x):
    return (COLD_SOURCE_INTENSITY - HEAT_SOURCE_INTENSITY) * (x.vector['y'] / Ly) + HEAT_SOURCE_INTENSITY

t0_base = CenteredGrid(linear_temp_gradient, t_BC_THERMAL, **DOMAIN)
t0_noise = CenteredGrid(Noise(scale=0.1), **DOMAIN)
t0_gt0 = t0_base + t0_noise

# 直接初始化为零场以简化
# t0_gt0 = CenteredGrid(0, t_BC_THERMAL, **DOMAIN)

# 初始化示踪粒子
markers0 = DOMAIN['bounds'].sample_uniform(instance(markers=MARKERS))

# 先预先算几步
for _ in range(PRE_STEPS):
    v0_gt0, t0_gt0 = boussinesq_step(v0_gt0, t0_gt0, dt=DT)
    markers0 = advect.points(markers0, v0_gt0, dt=DT, integrator=advect.rk4)


v0_gt, t0_gt = v0_gt0, t0_gt0
initial_markers = markers0

# 生成真实轨迹
gt_marker_trajectories = [initial_markers]
current_v = v0_gt
current_t = t0_gt
current_markers = initial_markers

for _ in range(STEPS):
    current_v, current_t = boussinesq_step(current_v, current_t, dt=DT)
    current_markers = advect.points(current_markers, current_v, dt=DT, integrator=advect.rk4)
    gt_marker_trajectories.append(current_markers)

# 转换为张量 (Time, Markers, Vector)
gt_marker_trajectories_stack = math.stack(gt_marker_trajectories, batch('time'))
print(f"Ground Truth Generated. Frames: {gt_marker_trajectories_stack.time.size}")

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
    t = t0_gt

    markers = initial_markers
    trajectory = [markers]

    # 前向模拟
    for _ in range(STEPS):
        v, t = boussinesq_step(v, t, dt=DT)
        markers = advect.points(markers, v, dt=DT, integrator=advect.rk4)
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
    extrapolation=V_BC,
    bounds=Box(x=Lx, y=Ly)
)

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

# ==========================================
# 7. 后处理：重建全时序流场
# ==========================================
print("\nRe-simulating to get full flow field evolution...")

# 确保起点是纯粹的 Field，而不是优化器的变量状态
v_current = v0_reconstructed
t_current = t0_gt

reconstructed_velocities = [v_current]
reconstructed_temperatures = [t_current]
# 注意：这里我们不需要存储 marker，marker 在可视化阶段单独算
# 这样可以解耦“流场计算”和“粒子平流”，避免 Tracer 混入列表

for t in range(STEPS):
    v_current, t_current = boussinesq_step(v_current, t_current, dt=DT)
    reconstructed_velocities.append(v_current)
    reconstructed_temperatures.append(t_current)

# 堆叠结果 (Time, Y, X, Vector)
v_stack = math.stack(reconstructed_velocities, batch('time'))
t_stack = math.stack(reconstructed_temperatures, batch('time'))

# ==========================================
# 8. 可视化：流线 + 粒子动画
# ==========================================
print("Generating visualization data...")

# A. 提取流场数据
# ------------------------------------------------
if 'seed' in v_stack.shape:
    v_display = v_stack[{'seed': 0}]
else:
    v_display = v_stack

# 插值到中心并提取分量
v_centered = v_display.at_centers()
u_data = v_centered.vector['x'].values.numpy('time, y, x')
v_data = v_centered.vector['y'].values.numpy('time, y, x')

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

# C. 生成动画
# ------------------------------------------------
print("Rendering Streamline + Marker Animation...")

x_phys = np.linspace(0, Lx, Nx)
y_phys = np.linspace(0, Ly, Ny)
X_grid, Y_grid = np.meshgrid(x_phys, y_phys)

fig, ax = plt.subplots(figsize=(5, 5))

def update(frame):
    ax.clear()

    # 1. 绘制流线
    ax.streamplot(
        X_grid, Y_grid,
        u_data[frame], v_data[frame],
        color='cornflowerblue', linewidth=0.8, arrowsize=1.0, density=1.5
    )

    # 2. 绘制粒子
    px = marker_positions_np[frame, :, 0]
    py = marker_positions_np[frame, :, 1]
    ax.scatter(px, py, color='orange', s=40, edgecolors='white', zorder=10)

    # 3. 设置
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_title(f"Reconstructed Flow - T={frame*DT:.2f}s")
    return ax,

ani = animation.FuncAnimation(fig, update, frames=len(u_data), interval=300, blit=False)
plt.close()

# 保存为 MP4 文件 (可选)
# ani.save("streamline_marker_animation.mp4", fps=10, dpi=150)

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
fig_t, ax_t = plt.subplots(figsize=(7, 6))

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
# 10. 可视化：涡量对比 (Reconstructed vs Ground Truth)
# ==========================================
print("\nGenerating Vorticity Comparison Animation...")

# --- A. 重新生成 Ground Truth 的全时序流场 ---
# 之前我们只存了粒子，现在需要存每一帧的速度场 v 用来算涡量
print("Re-simulating Ground Truth history for comparison...")

# 确保使用初始的 GT 状态
v_gt_curr = v0_gt
t_gt_curr = t0_gt
gt_velocity_list = [v_gt_curr]

# 重新跑一遍物理循环 (仅前向推理，不涉及梯度，速度很快)
for _ in range(STEPS):
    v_gt_curr, t_gt_curr = boussinesq_step(v_gt_curr, t_gt_curr, dt=DT)
    gt_velocity_list.append(v_gt_curr)

# 堆叠为 FieldStack
v_gt_stack = math.stack(gt_velocity_list, batch('time'))

# --- B. 计算涡量 (Curl) ---
print("Calculating Vorticity...")

# 1. 获取 Reconstructed 数据 (来自 Section 7 的 v_stack)
if 'seed' in v_stack.shape:
    v_recon_display = v_stack[{'seed': 0}]
else:
    v_recon_display = v_stack

# 2. 获取 Ground Truth 数据
if 'seed' in v_gt_stack.shape:
    v_gt_display = v_gt_stack[{'seed': 0}]
else:
    v_gt_display = v_gt_stack

# 3. 计算 Curl 并转为 Numpy
# 2D 速度场的 Curl 是标量场
curl_recon_np = v_recon_display.curl().values.numpy('time, y, x')
curl_gt_np = v_gt_display.curl().values.numpy('time, y, x')

# 4. 计算误差场
diff_curl_np = curl_recon_np - curl_gt_np

# --- C. 设置绘图参数 ---
# 统一量程：使用 GT 的最大绝对值，保证对比公平
max_val1 = np.max(np.abs(curl_gt_np)) + 1e-5
max_val2 = np.max(np.abs(curl_recon_np)) + 1e-5
# 误差量程：通常误差较小，单独计算最大值以便观察细节
max_err = np.max(np.abs(diff_curl_np)) + 1e-5

def add_colorbar(im, ax, label):
    cbar = fig_comp.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    return cbar


# --- D. 生成三联画动画 ---
fig_comp, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
ax1, ax2, ax3 = axes

# 1. Ground Truth
im1 = ax1.imshow(curl_gt_np[0], origin='lower', cmap='PuOr', vmin=-max_val1, vmax=max_val1, extent=[0,Lx,0,Ly])
ax1.set_title("Ground Truth Vorticity", fontsize=12, pad=10)
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
add_colorbar(im1, ax1, "Vorticity (1/s)")


# 2. Reconstructed
im2 = ax2.imshow(curl_recon_np[0], origin='lower', cmap='PuOr', vmin=-max_val1, vmax=max_val1, extent=[0,Lx,0,Ly])
ax2.set_title("Reconstructed Vorticity")
ax2.set_xlabel("X (m)")
add_colorbar(im2, ax2, "Vorticity (1/s)")

# 3. Difference
im3 = ax3.imshow(diff_curl_np[0], origin='lower', cmap='PuOr', vmin=-max_val1, vmax=max_val1, extent=[0,Lx,0,Ly])
ax3.set_title(f"Difference (Max Err: {max_err:.2e})")
ax3.set_xlabel("X (m)")
add_colorbar(im3, ax3, "Vorticity (1/s)")

def update_comp(frame):
    # 更新数据
    im1.set_data(curl_gt_np[frame])
    im2.set_data(curl_recon_np[frame])
    im3.set_data(diff_curl_np[frame])

    # 更新标题
    fig_comp.suptitle(f"Vorticity Comparison - T={frame*DT:.2f}s", fontsize=16)
    return im1, im2, im3

ani_comp = animation.FuncAnimation(fig_comp, update_comp, frames=len(curl_gt_np), interval=200, blit=False)
plt.close()

# 保存为 MP4 文件 (可选)
# ani_comp.save("vorticity_comparison.mp4", fps=10, dpi=150)

display(HTML(ani_comp.to_jshtml()))