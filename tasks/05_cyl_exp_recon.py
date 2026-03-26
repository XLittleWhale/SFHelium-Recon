"""
Task 05: Experimental cylinder-flow reconstruction.

This script follows the Task-04 workflow for real particle trajectories
but uses cylinder geometry (with obstacle) as in Task 03:
- initialise all physics fields from scratch (no VTK reference data),
- load experimental PTV particle trajectories from CSV,
- reconstruct the initial normal-fluid field with L-BFGS-B,
- save particle trajectories and field time series for analysis.
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR  = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from phi.jax.flow import *
from phi.jax.flow import (
    Box, StaggeredGrid, CenteredGrid, ZERO_GRADIENT,
    advect, math, jit_compile, spatial, channel, dual, geom,
    PointCloud, minimize, instance, Sphere, Obstacle, resample,
)

from sf_recon.physics import helium, boundaries
from sf_recon.utils import io, particles
from sf_recon.utils.continuous_field import ContinuousVelocityFieldFitter
from sf_recon.utils.guess import (
    build_obstacle_aware_inflow_prior,
    native_to_centered_grid,
)
from sf_recon.utils.saving import (
    simple_to_numpy  as _simple_to_numpy,
    extract_time_series_for_vn as _extract_time_series_for_vn,
    ensure_HW as _ensure_HW,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_centered_velocity(velocity_grid, coords_nat):
    """Sample a centered velocity field at particle coordinates."""
    coords = math.tensor(coords_nat, instance('markers') & channel(vector='x,y'))
    sampled = velocity_grid.at(PointCloud(geom.Point(coords))).values
    return sampled.native(['markers', 'vector'])

def _compute_material_acceleration(prev_grid, next_grid, dt):
    """Approximate material acceleration using a first-order time difference."""
    return (next_grid - prev_grid) * (1.0 / max(float(dt), 1e-12))

def _sanitize_native(x, clip=1e6):
    """Bound invalid values to keep JAX scans numerically stable."""
    return jnp.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)

def _advance_particle_state(coords_nat, vel_nat, vn_now, vs_now, vn_prev, vs_prev, dt, tau, rho_n, rho_s, rho_total):
    """Advance particle position and velocity using the relaxation model."""
    vn_now_center = vn_now.at_centers()
    vs_now_center = vs_now.at_centers()
    dvn_dt = _compute_material_acceleration(vn_prev.at_centers(), vn_now_center, dt)
    dvs_dt = _compute_material_acceleration(vs_prev.at_centers(), vs_now_center, dt)
    vn_sample = _sample_centered_velocity(vn_now_center, coords_nat)
    dvn_sample = _sample_centered_velocity(dvn_dt, coords_nat)
    dvs_sample = _sample_centered_velocity(dvs_dt, coords_nat)
    acc = (
        (vn_sample - vel_nat) / float(tau)
        + (float(rho_n) / float(rho_total)) * dvn_sample
        + (float(rho_s) / float(rho_total)) * dvs_sample
    )
    next_vel = vel_nat + float(dt) * acc
    next_coords = coords_nat + float(dt) * next_vel
    return next_coords, next_vel

def _center_guess_to_bc_staggered(v_guess_centered, v_template, vn_bc):
    """Convert a centered guess into a staggered field that respects `Vn_BC`."""
    centered_clean = CenteredGrid(
        values=v_guess_centered.values,
        extrapolation=ZERO_GRADIENT,
        bounds=v_guess_centered.bounds,
        resolution=v_guess_centered.resolution,
    )
    staggered_template = StaggeredGrid(
        0,
        vn_bc,
        bounds=v_template.bounds,
        resolution=v_template.resolution,
    )
    return centered_clean.at(staggered_template)

def _configure_geometry(args):
    """Build cylinder domain and obstacle configuration."""
    Lx, Ly = 0.023, 0.2
    RAD    = 0.001
    Wy     = 0.0227                       # observation-window height
    Nx, Ny = 30, 200                     # grid resolution (≈ Lx/Ly aspect)

    SPHERE   = Sphere(x=Lx / 2, y=Ly/2 - Wy/2 +  Wy * 0.56, radius=RAD)
    OBSTACLE = Obstacle(SPHERE)
    DOMAIN   = dict(x=Nx, y=Ny, bounds=Box(x=Lx, y=Ly))

    # Observation window (full width, ±Wy/2 around cylinder centre)
    cyl_y  = Ly/2 - Wy/2 + Wy * 0.56
    WINDOMAIN = dict(
        x=Nx, y=Ny,
        bounds=Box(x=(0, Lx), y=(Ly/2 - Wy / 2, Ly/2 + Wy / 2)),
    )

    return {
        'Lx': Lx, 'Ly': Ly, 'Nx': Nx, 'Ny': Ny,
        'RAD': RAD, 'Wy': Wy,
        'SPHERE': SPHERE, 'OBSTACLE': OBSTACLE,
        'MARKERS': args.num_particles,
        'DT': args.dt,
        'STEPS': args.steps,
        'PRE_STEPS': 0,
        'DOMAIN':     DOMAIN,
        'WINDOMAIN':  WINDOMAIN,
    }

def _prepare_save_array(x, nx, ny):
    """Convert a PhiFlow object to a NumPy array with consistent (H, W) shape."""
    try:
        arr = _simple_to_numpy(x)
    except Exception:
        arr = None
    return _ensure_HW(arr, nx, ny)

def _build_save_dict(config, gt_np, recon_traj, loss_mask_np, reset_mask_np, field_series):
    """Assemble the ``.npz`` payload saved by Task 05."""
    marker_bounds = particles.marker_window_bounds(gt_np, loss_mask_np)
    Lx, Ly = config['Lx'], config['Ly']
    Wy     = config['Wy']
    cyl_y  = Ly/2 - Wy/2 + Wy * 0.56
    if marker_bounds is None:
        marker_x_min, marker_x_max = 0.0, Lx
        marker_y_min, marker_y_max = Ly/2 - Wy / 2, Ly/2 + Wy / 2
    else:
        marker_x_min, marker_x_max, marker_y_min, marker_y_max = marker_bounds

    v_gt_u, v_gt_v, v_gt_speed, v_gt_vec, \
        vs_gt_u, vs_gt_v, vs_gt_speed, vs_gt_vec, t_gt = field_series['gt']
    v_recon_u, v_recon_v, v_recon_speed, v_recon_vec, \
        vs_recon_u, vs_recon_v, vs_recon_speed, vs_recon_vec, t_recon = \
        field_series['recon']
    obs_vel = config.get('obs_vel_np')
    recon_vel = config.get('recon_vel_np')
    vel_mask = config.get('vel_mask_np')
    supervision_mode = config.get('supervision_mode', 'sparse-observation')
    supervision_positions = config.get('supervision_positions_np')
    supervision_velocities = config.get('supervision_velocities_np')
    supervision_loss_mask = config.get('supervision_loss_mask_np')
    supervision_vel_mask = config.get('supervision_vel_mask_np')

    Nx, Ny = config['Nx'], config['Ny']
    return dict(
        success=1,
        Lx=Lx, Ly=Ly, Wx=Lx, Wy=Wy, RAD=config['RAD'],
        cyl_center_x=Lx / 2, cyl_center_y=cyl_y,
        window_x_min=0.0, window_x_max=Lx, window_y_min=Ly/2 - Wy / 2, window_y_max=Ly/2 + Wy / 2,
        marker_x_min=marker_x_min, marker_x_max=marker_x_max,
        marker_y_min=marker_y_min, marker_y_max=marker_y_max,
        pre_steps=config['PRE_STEPS'],
        gt=gt_np, recon=recon_traj, gt_vel=obs_vel, recon_vel=recon_vel,
        loss_mask=loss_mask_np, reset_mask=reset_mask_np, vel_mask=vel_mask,
        supervision_mode=supervision_mode, supervision_positions=supervision_positions,
        supervision_velocities=supervision_velocities, supervision_loss_mask=supervision_loss_mask,
        supervision_vel_mask=supervision_vel_mask,
        v_gt_u=_prepare_save_array(v_gt_u, Nx, Ny), v_gt_v=_prepare_save_array(v_gt_v, Nx, Ny),
        v_gt_speed=_prepare_save_array(v_gt_speed, Nx, Ny), v_gt_vec=_prepare_save_array(v_gt_vec, Nx, Ny),
        v_recon_u=_prepare_save_array(v_recon_u, Nx, Ny), v_recon_v=_prepare_save_array(v_recon_v, Nx, Ny),
        v_recon_speed=_prepare_save_array(v_recon_speed, Nx, Ny), v_recon_vec=_prepare_save_array(v_recon_vec, Nx, Ny),
        vs_gt_u=_prepare_save_array(vs_gt_u, Nx, Ny), vs_gt_v=_prepare_save_array(vs_gt_v, Nx, Ny),
        vs_gt_speed=_prepare_save_array(vs_gt_speed, Nx, Ny), vs_gt_vec=_prepare_save_array(vs_gt_vec, Nx, Ny),
        vs_recon_u=_prepare_save_array(vs_recon_u, Nx, Ny), vs_recon_v=_prepare_save_array(vs_recon_v, Nx, Ny),
        vs_recon_speed=_prepare_save_array(vs_recon_speed, Nx, Ny), vs_recon_vec=_prepare_save_array(vs_recon_vec, Nx, Ny),
        t_gt=_prepare_save_array(t_gt, Nx, Ny), t_recon=_prepare_save_array(t_recon, Nx, Ny),
    )


# ========================================================================
# Main
# ========================================================================

def main():
    """Run the experimental cylinder-flow reconstruction workflow."""
    
    # ------------------------------------------------------------------
    # IMPORTANT: RE-BALANCED WEIGHTS TO ENSURE VORTICITY CAPTURE
    # ------------------------------------------------------------------
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 自适应权重初始化：设定各部分损失项的初始目标贡献量 (Target Importances)
    # 取代了原本写死的 1e8, 1e0, 1e2, 1e0
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    TARGET_MSE_LOSS = 10.0      # Expect position MSE to contribute ~10.0 at iter 0
    TARGET_VEL_LOSS = 1.0       # Expect velocity loss to contribute ~1.0 at iter 0
    TARGET_VORT_LOSS = 10.0      # Expect vorticity loss to contribute ~5.0 at iter 0
    TARGET_DIV_LOSS = 1.0       # Expect divergence loss to contribute ~1.0 at iter 0
    
    DEFAULT_POS_LOSS_SCALE = 1e-5
    DEFAULT_VEL_LOSS_SCALE = 1
    DEFAULT_VORT_LOSS_SCALE = 1
    DEFAULT_PRIOR_REG_WEIGHT = 1e-4
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 对 y 方向速度为负时的惩罚放大系数
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    DEFAULT_NEG_Y_PENALTY = 1
    
    DEFAULT_FALLBACK_GD_ITERS = 25
    DEFAULT_FALLBACK_GD_LR = 3e-3
    DEFAULT_MIN_LOSS_IMPROVE_ABS = 1e-2
    DEFAULT_MIN_LOSS_IMPROVE_REL = 1e-7
    DEFAULT_RANDOM_SEARCH_TRIALS = 16
    DEFAULT_RANDOM_SEARCH_SCALE = 1e-3
    DEFAULT_MIN_OBS = 2
    DEFAULT_SMOOTH_PARTICLES = False
    DEFAULT_SMOOTH_RADIUS = 2
    DEFAULT_SMOOTH_SIGMA = 1.0
    
    RUN_INVERSION = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str,
                        default='data/experiment/PTV_CylinderA6_Results.csv',
                        help='PTV particle tracking CSV file')
    parser.add_argument('--lbfgs-iters', type=int, default=2000,
                        help='Max L-BFGS-B iterations')
    parser.add_argument('--num-particles', type=int, default=5174,
                        help='Number of particle trajectories to load')
    parser.add_argument('--steps', type=int, default=1,
                        help='Number of simulation steps used for inversion')
    parser.add_argument('--dt', type=float, default=1e-6,
                        help='Time step size')
    parser.add_argument('--freq', type=float, default=1000.0,
                        help='PTV acquisition frequency (fps)')
    parser.add_argument('--particle-resample-freq', type=float, default=1000.0,
                        help='Target particle preprocessing frequency (Hz)')
    parser.add_argument('--particle-tau', type=float, default=1e-3,
                        help='Particle relaxation time')
    parser.add_argument('--fit-continuous-field', action='store_true',
                        help='Fit a continuous-time velocity field from sparse observations')
    parser.add_argument('--fit-num-particles', type=int, default=5174,
                        help='Trajectories used for continuous-field fitting')
    parser.add_argument('--fit-duration', type=float, default=0.002,
                        help='Physical duration for raw observation import')
    args = parser.parse_args()

    math.use('jax')
    math.set_global_precision(64)

    print('Task 05: starting')

    # ------------------------------------------------------------------
    # 1. Geometry and Parameters Configuration
    # ------------------------------------------------------------------
    config   = _configure_geometry(args)
    Lx, Ly   = config['Lx'], config['Ly']
    Nx, Ny   = config['Nx'], config['Ny']
    RAD      = config['RAD']
    SPHERE   = config['SPHERE']
    OBSTACLE = config['OBSTACLE']
    MARKERS  = config['MARKERS']
    DT       = config['DT']
    STEPS    = config['STEPS']
    PRE_STEPS = config['PRE_STEPS']
    DOMAIN    = config['DOMAIN']
    WINDOMAIN = config['WINDOMAIN']
    
    POS_LOSS_SCALE = max(float(DEFAULT_POS_LOSS_SCALE), 1e-12)
    VEL_LOSS_SCALE = max(float(DEFAULT_VEL_LOSS_SCALE), 1e-12)
    VORT_LOSS_SCALE = max(float(DEFAULT_VORT_LOSS_SCALE), 1e-12)
    PRIOR_REG_WEIGHT = max(float(DEFAULT_PRIOR_REG_WEIGHT), 0.0)
    MIN_LOSS_IMPROVE_ABS = float(DEFAULT_MIN_LOSS_IMPROVE_ABS)
    MIN_LOSS_IMPROVE_REL = float(DEFAULT_MIN_LOSS_IMPROVE_REL)

    # ------------------------------------------------------------------
    # 2. Time grids and Experimental Data Loading
    # (We load data FIRST so we can deduce the true physical boundaries!)
    # ------------------------------------------------------------------
    inv_steps = max(int(STEPS), 1)
    inv_dt = float(DT)
    inv_total_time = inv_steps * inv_dt

    fit_dt = 1.0 / max(float(args.particle_resample_freq), 1e-12)
    fit_total_time = (float(args.fit_duration) if args.fit_duration is not None 
                      else max(inv_total_time, inv_steps * (1.0 / max(float(args.freq), 1e-12))))
    fit_steps = max(int(round(fit_total_time / fit_dt)), 1)
    
    fit_markers = max(int(args.fit_num_particles if args.fit_num_particles is not None else MARKERS), 1)
    obs_steps, obs_dt = fit_steps, fit_dt
    obs_markers = fit_markers if args.fit_continuous_field else int(MARKERS)

    gt_tensor, loss_mask, reset_mask, initial_pos, gt_vel_tensor, vel_mask = \
        particles.load_experimental_particle_data(
            args.csv_path, num_particles=obs_markers, max_steps=obs_steps, dt=obs_dt,
            domain_bounds=WINDOMAIN['bounds'], freq=args.freq, category=None,
            shift_to_zero=True, smooth=DEFAULT_SMOOTH_PARTICLES, smooth_radius=DEFAULT_SMOOTH_RADIUS,
            smooth_sigma=DEFAULT_SMOOTH_SIGMA, min_obs=DEFAULT_MIN_OBS, position_scale=-3.2125e-5,
            offset_x=0.0222, offset_y=0.0227+0.1-0.0227/2,
        )

    gt_all_native     = gt_tensor.native(['time', 'markers', 'vector'])
    gt_vel_native     = gt_vel_tensor.native(['time', 'markers', 'vector'])
    loss_mask_native  = loss_mask.native(['time', 'markers'])
    reset_mask_native = reset_mask.native(['time', 'markers'])
    vel_mask_native   = vel_mask.native(['time', 'markers'])

    # =====================================================================
    # COMPUTE TRUE BACKGROUND VELOCITY GLOBALLY (Mean detrending)
    # =====================================================================
    valid_global_idx = (vel_mask_native > 0.5) & np.isfinite(gt_vel_native[..., 0]) & np.isfinite(gt_vel_native[..., 1])
    valid_global_vels = gt_vel_native[valid_global_idx]
    if valid_global_vels.shape[0] > 0:
        bg_u = float(np.mean(valid_global_vels[:, 0]))
        bg_v = float(np.mean(valid_global_vels[:, 1]))
    else:
        bg_u, bg_v = 0.0, 0.026 # Fallback roughly to heat flux scale
    bg_vel_np = np.array([bg_u, bg_v], dtype=np.float64)
    print(f'[INFO] Calculated Experimental Background Velocity: u={bg_u:.6f}, v={bg_v:.6f}')
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 生成并保存粒子速度的分布图 (直方图)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if valid_global_vels.shape[0] > 0:
        import matplotlib.pyplot as plt
        import os
        
        plt.figure(figsize=(12, 5))
        
        # 1. 绘制 u 分量分布
        plt.subplot(1, 2, 1)
        plt.hist(valid_global_vels[:, 0], bins=100, color='royalblue', alpha=0.7)
        plt.axvline(bg_u, color='red', linestyle='dashed', linewidth=2, label=f'Median: {bg_u:.4f}')
        plt.axvline(np.mean(valid_global_vels[:, 0]), color='orange', linestyle='dotted', linewidth=2, label=f'Mean: {np.mean(valid_global_vels[:, 0]):.4f}')
        plt.title('Distribution of u (Horizontal Velocity)')
        plt.xlabel('u [m/s]')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 2. 绘制 v 分量分布
        plt.subplot(1, 2, 2)
        plt.hist(valid_global_vels[:, 1], bins=100, color='seagreen', alpha=0.7)
        plt.axvline(bg_v, color='red', linestyle='dashed', linewidth=2, label=f'Median: {bg_v:.4f}')
        plt.axvline(np.mean(valid_global_vels[:, 1]), color='orange', linestyle='dotted', linewidth=2, label=f'Mean: {np.mean(valid_global_vels[:, 1]):.4f}')
        plt.title('Distribution of v (Vertical Velocity)')
        plt.xlabel('v [m/s]')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        dist_path = 'data/experiment/velocity_distribution.png'
        os.makedirs(os.path.dirname(dist_path), exist_ok=True)
        plt.savefig(dist_path, dpi=150)
        plt.close()
        print(f'[INFO] Saved velocity distribution plot to: {dist_path}')
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # =====================================================================

    # ------------------------------------------------------------------
    # 3. Initialise physics fields with MATCHED boundaries
    # This prevents the massive t=0 shock between experiment and solver.
    # ------------------------------------------------------------------
    HEAT_SOURCE_INTENSITY = 2.00
    HEAT_FLUX   = 9100
    DENSITY     = 145.6217
    ENTROPY     = 962.1
    DENSITY_N   = 80.55
    DENSITY_S   = 65.07
    RHO_TOTAL   = DENSITY
    PRESSURE_0  = 3130
    Vn_IN = HEAT_FLUX / (DENSITY * ENTROPY * HEAT_SOURCE_INTENSITY)
    Vs_IN = -(DENSITY_N * Vn_IN) / DENSITY_S
    print(f'[INFO] Initializing Physical Bounds with Vn_IN = {Vn_IN:.6f} m/s')
    
    Vn_BC, Vs_BC, J_BC, t_BC_THERMAL, p_BC = boundaries.get_cylinder_bcs(
        Vn_IN=Vn_IN, Vs_IN=Vs_IN, PRESSURE_0=PRESSURE_0,
    )
    
    v0_gt0  = StaggeredGrid(lambda x,y: vec(x=0, y=(6 * Vn_IN / Lx**2) * x * (Lx - x)), Vn_BC, **DOMAIN)
    vs0_gt0 = StaggeredGrid((0, Vs_IN), Vs_BC, **DOMAIN)
    p0_gt0  = CenteredGrid(PRESSURE_0, p_BC, **DOMAIN)
    t0_gt0  = CenteredGrid(HEAT_SOURCE_INTENSITY, t_BC_THERMAL, **DOMAIN)
    L0_gt0  = CenteredGrid(0, ZERO_GRADIENT, **DOMAIN)

    obstacle_mask_vn = resample(~(OBSTACLE.geometry), v0_gt0)
    v0_gt0  = field.safe_mul(obstacle_mask_vn, v0_gt0)
    obstacle_mask_vs = resample(~(OBSTACLE.geometry), vs0_gt0)
    vs0_gt0 = field.safe_mul(obstacle_mask_vs, vs0_gt0)
    obstacle_mask_t  = resample(~(OBSTACLE.geometry), t0_gt0)
    t0_gt0  = field.safe_mul(obstacle_mask_t, t0_gt0)
    obstacle_mask_L  = resample(~(OBSTACLE.geometry), L0_gt0)
    L0_gt0  = field.safe_mul(obstacle_mask_L, L0_gt0)

    if PRE_STEPS > 0:
        curr_v, curr_vs, curr_p, curr_t, curr_L = (v0_gt0, vs0_gt0, p0_gt0, t0_gt0, L0_gt0)
        for _ in range(PRE_STEPS):
            curr_v, curr_vs, curr_p, curr_t, curr_L = helium.SFHelium_step(
                curr_v, curr_vs, curr_p, curr_t, curr_L, dt=DT, DOMAIN=DOMAIN, 
                OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL,
            )
        v0_gt0, vs0_gt0, p0_gt0, t0_gt0, L0_gt0 = (curr_v, curr_vs, curr_p, curr_t, curr_L)

    supervision_pos_native = gt_all_native
    supervision_vel_native = gt_vel_native
    supervision_loss_mask_native = loss_mask_native
    supervision_vel_mask_native = vel_mask_native
    supervision_mode = 'sparse-observation'
    supervision_init_pos_native = gt_all_native[0]
    supervision_init_vel_native = gt_vel_native[0]
    
    continuous_recon_traj, continuous_recon_vel = None, None
    continuous_vn_u_series, continuous_vn_v_series, continuous_vn_speed_series = None, None, None
    continuous_recon_loss_mask, continuous_recon_reset_mask = None, None
    
    fit_times = None
    observation_times = np.linspace(0.0, fit_total_time, obs_steps + 1, dtype=np.float64)
    supervision_times = np.linspace(0.0, inv_total_time, inv_steps + 1, dtype=np.float64)

    if not args.fit_continuous_field:
        right_idx = np.searchsorted(observation_times, supervision_times, side='left')
        right_idx = np.clip(right_idx, 0, max(len(observation_times) - 1, 0))
        left_idx = np.clip(right_idx - 1, 0, max(len(observation_times) - 1, 0))
        right_dt = np.abs(observation_times[right_idx] - supervision_times)
        left_dt = np.abs(observation_times[left_idx] - supervision_times)
        obs_indices = np.where(right_dt <= left_dt, right_idx, left_idx)
        
        supervision_pos_native = gt_all_native[obs_indices]
        supervision_vel_native = gt_vel_native[obs_indices]
        supervision_loss_mask_native = loss_mask_native[obs_indices]
        supervision_vel_mask_native = vel_mask_native[obs_indices]
        supervision_init_pos_native = supervision_pos_native[0]
        supervision_init_vel_native = supervision_vel_native[0]

    if args.fit_continuous_field:
        fit_times = np.linspace(0.0, fit_total_time, fit_steps + 1, dtype=np.float64)
        
        spatial_range = Lx
        time_range = max(fit_total_time, 1e-6)
        rbf_time_scale = spatial_range / time_range
        scaled_fit_times = fit_times * rbf_time_scale
        
        gt_np = particles.tensor_time_markers_to_numpy(gt_tensor)
        gt_vel_np = particles.tensor_time_marker_velocity_to_numpy(gt_vel_tensor)
        loss_mask_np = particles.tensor_time_marker_mask_to_numpy(loss_mask)
        reset_mask_np = particles.tensor_time_marker_mask_to_numpy(reset_mask)
        vel_mask_np = particles.tensor_time_marker_mask_to_numpy(vel_mask)

        use_fit_markers = min(int(fit_markers), int(gt_np.shape[1]))
        gt_np = gt_np[:, :use_fit_markers, :]
        gt_vel_np = gt_vel_np[:, :use_fit_markers, :]
        vel_mask_np = vel_mask_np[:, :use_fit_markers]

        valid_idx = (vel_mask_np > 0.5) & np.isfinite(gt_vel_np[..., 0]) & np.isfinite(gt_vel_np[..., 1])
        gt_vel_fluct_np = gt_vel_np.copy()
        gt_vel_fluct_np[valid_idx] -= bg_vel_np

        fitter = ContinuousVelocityFieldFitter(
            smoothing=1e-16,
            neighbors=5,
            kernel = 'gaussian',
            epsilon = 250
            # kernel='thin_plate_spline'
        )
        
        fitter.fit(
            positions=gt_np,
            velocities=gt_vel_fluct_np, 
            mask=vel_mask_np,
            times=scaled_fit_times,
        )

        def custom_rollout_with_bg(rbf_fitter, initial_positions, query_times_raw, bg_vel):
            initial_positions = np.asarray(initial_positions, dtype=np.float64)
            query_times_raw = np.asarray(query_times_raw, dtype=np.float64)
            traj = np.zeros((query_times_raw.size, initial_positions.shape[0], 2), dtype=np.float64)
            vel = np.zeros_like(traj)
            
            traj[0] = initial_positions
            t0_scaled = np.full((initial_positions.shape[0],), query_times_raw[0] * rbf_time_scale, dtype=np.float64)
            vel[0] = rbf_fitter.sample_velocity(initial_positions, t0_scaled) + bg_vel
            
            for i in range(1, query_times_raw.size):
                dt = float(query_times_raw[i] - query_times_raw[i - 1])
                t_prev_scaled = np.full((initial_positions.shape[0],), query_times_raw[i - 1] * rbf_time_scale, dtype=np.float64)
                t_curr_scaled = np.full((initial_positions.shape[0],), query_times_raw[i] * rbf_time_scale, dtype=np.float64)
                
                vel_prev = rbf_fitter.sample_velocity(traj[i - 1], t_prev_scaled) + bg_vel
                traj[i] = traj[i - 1] + dt * vel_prev
                vel[i] = rbf_fitter.sample_velocity(traj[i], t_curr_scaled) + bg_vel
            return traj, vel

        fit_recon_traj_all, fit_recon_vel_all = custom_rollout_with_bg(fitter, gt_np[0], fit_times, bg_vel_np)

        selected_markers = min(int(MARKERS), int(use_fit_markers))
        selected_init_pos = np.asarray(fit_recon_traj_all[0, :selected_markers], dtype=np.float64)
        selected_init_vel = np.asarray(fit_recon_vel_all[0, :selected_markers], dtype=np.float64)
        
        recon_traj, recon_vel = custom_rollout_with_bg(fitter, selected_init_pos, supervision_times, bg_vel_np)
        continuous_recon_traj = recon_traj
        continuous_recon_vel = recon_vel

        supervision_pos_native = np.asarray(recon_traj, dtype=np.float64)
        supervision_vel_native = np.asarray(recon_vel, dtype=np.float64)
        supervision_mode = 'continuous-field'
        supervision_init_pos_native = supervision_pos_native[0]
        supervision_init_vel_native = np.asarray(selected_init_vel, dtype=np.float64)

        x_centers = np.linspace(Lx / (2 * Nx), Lx - Lx / (2 * Nx), Nx, dtype=np.float64)
        y_centers = np.linspace(Ly / (2 * Ny), Ly - Ly / (2 * Ny), Ny, dtype=np.float64)
        XX, YY = np.meshgrid(x_centers, y_centers, indexing='ij')
        grid_points = np.stack([XX, YY], axis=-1).reshape(-1, 2)

        vn_u_series, vn_v_series, vn_speed_series = [], [], []
        obstacle_mask_grid = ((grid_points[:, 0] - float(SPHERE.center.vector[0])) ** 2 + (grid_points[:, 1] - float(SPHERE.center.vector[1])) ** 2) >= float(SPHERE.radius) ** 2
        obstacle_mask_grid = obstacle_mask_grid.reshape(Nx, Ny)

        for t_value in supervision_times:
            t_scaled = float(t_value * rbf_time_scale)
            grid_vel_fluct = fitter.sample_velocity(
                grid_points, np.full((grid_points.shape[0],), t_scaled, dtype=np.float64),
            ).reshape(Nx, Ny, 2)
            
            grid_vel = grid_vel_fluct + bg_vel_np
            grid_vel[..., 0] = np.where(obstacle_mask_grid, grid_vel[..., 0], 0.0)
            grid_vel[..., 1] = np.where(obstacle_mask_grid, grid_vel[..., 1], 0.0)
            vn_u_series.append(grid_vel[..., 0])
            vn_v_series.append(grid_vel[..., 1])
            vn_speed_series.append(np.sqrt(grid_vel[..., 0] ** 2 + grid_vel[..., 1] ** 2))
            
        vn_u_series = np.stack(vn_u_series, axis=0)
        vn_v_series = np.stack(vn_v_series, axis=0)
        vn_speed_series = np.stack(vn_speed_series, axis=0)
        continuous_vn_u_series = vn_u_series
        continuous_vn_v_series = vn_v_series
        continuous_vn_speed_series = vn_speed_series

        finite_mask = np.isfinite(recon_traj[..., 0]) & np.isfinite(recon_traj[..., 1])
        recon_loss_mask = finite_mask.astype(np.float64)
        recon_reset_mask = np.zeros_like(recon_loss_mask)
        recon_reset_mask[0, :] = 1.0
        continuous_recon_loss_mask = recon_loss_mask
        continuous_recon_reset_mask = recon_reset_mask
        supervision_loss_mask_native = recon_loss_mask
        supervision_vel_mask_native = recon_loss_mask

    # Obstacle geometry constants
    obstacle_center_np = np.array([float(SPHERE.center.vector[0]), float(SPHERE.center.vector[1])])
    obstacle_radius = float(SPHERE.radius)

    obs_dx = gt_all_native[..., 0] - obstacle_center_np[0]
    obs_dy = gt_all_native[..., 1] - obstacle_center_np[1]
    observed_inside_obstacle = (obs_dx ** 2 + obs_dy ** 2) < (obstacle_radius ** 2)
    valid_observation_mask = 1.0 - observed_inside_obstacle.astype(np.float64)
    loss_mask_native = loss_mask_native * valid_observation_mask
    vel_mask_native = vel_mask_native * valid_observation_mask

    sup_dx = supervision_pos_native[..., 0] - obstacle_center_np[0]
    sup_dy = supervision_pos_native[..., 1] - obstacle_center_np[1]
    supervision_inside_obstacle = (sup_dx ** 2 + sup_dy ** 2) < (obstacle_radius ** 2)
    valid_supervision_mask = 1.0 - supervision_inside_obstacle.astype(np.float64)
    supervision_loss_mask_native = supervision_loss_mask_native * valid_supervision_mask
    supervision_vel_mask_native = supervision_vel_mask_native * valid_supervision_mask

    # ------------------------------------------------------------------
    # 4. Inversion: L-BFGS-B direct field optimisation
    # ------------------------------------------------------------------
    v0_reconstructed = None

    if RUN_INVERSION:
        prior_native = build_obstacle_aware_inflow_prior(
            Nx, Ny, Lx, Ly, Vn_IN,
            center_xy=(float(SPHERE.center.vector[0]), float(SPHERE.center.vector[1])),
            radius=float(SPHERE.radius),
        )
        prior_grid_centered = native_to_centered_grid(
            prior_native, Nx, Ny, DOMAIN['bounds'], Vn_BC,
        )
        prior_values_native = jnp.asarray(
            prior_grid_centered.values.native(prior_grid_centered.values.shape)
        )

        def physical_step_logic(carry_state_native, scan_input):
            (v_x_nat, v_y_nat), (vs_x_nat, vs_y_nat), \
                p_nat, t_nat, L_nat, coords_nat, vel_nat = carry_state_native

            bounds     = DOMAIN['bounds']
            grid_shape = spatial(x=Nx, y=Ny)

            v_values   = math.stack([math.tensor(v_x_nat, spatial('x,y')), math.tensor(v_y_nat, spatial('x,y'))], dual(vector='x,y'))
            vn = StaggeredGrid(values=v_values, extrapolation=Vn_BC, bounds=bounds, resolution=grid_shape)

            vs_values  = math.stack([math.tensor(vs_x_nat, spatial('x,y')), math.tensor(vs_y_nat, spatial('x,y'))], dual(vector='x,y'))
            vs = StaggeredGrid(values=vs_values, extrapolation=Vs_BC, bounds=bounds, resolution=grid_shape)

            p = CenteredGrid(values=math.tensor(p_nat, grid_shape), extrapolation=p_BC, bounds=bounds, resolution=grid_shape)
            t = CenteredGrid(values=math.tensor(t_nat, grid_shape), extrapolation=t_BC_THERMAL, bounds=bounds, resolution=grid_shape)
            L = CenteredGrid(values=math.tensor(L_nat, grid_shape), extrapolation=ZERO_GRADIENT, bounds=bounds, resolution=grid_shape)

            coords = math.tensor(coords_nat, instance('markers') & channel(vector='x,y'))

            vn_prev, vs_prev = vn, vs
            vn_next, vs_next, p_next, t_next, L_next = helium.SFHelium_step(
                vn, vs, p, t, L,
                dt=inv_dt, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE,
                Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL,
            )

            next_coords_free, next_vel_free = _advance_particle_state(
                coords_nat, vel_nat, vn_next, vs_next, vn_prev, vs_prev,
                inv_dt, args.particle_tau, DENSITY_N, DENSITY_S, RHO_TOTAL,
            )
            next_coords_free = _sanitize_native(next_coords_free)
            next_vel_free = _sanitize_native(next_vel_free)

            obs_c, obs_r = jnp.array(obstacle_center_np), obstacle_radius
            diff = next_coords_free - obs_c
            dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1, keepdims=True) + 1e-12)
            correction = obs_c + (diff / (dist + 1e-6)) * (obs_r + 1e-4)
            next_coords_free = jnp.where(dist < obs_r, correction, next_coords_free)

            new_carry = (
                (_sanitize_native(vn_next.vector['x'].values.native(['x', 'y'])),
                 _sanitize_native(vn_next.vector['y'].values.native(['x', 'y']))),
                (_sanitize_native(vs_next.vector['x'].values.native(['x', 'y'])),
                 _sanitize_native(vs_next.vector['y'].values.native(['x', 'y']))),
                _sanitize_native(p_next.values.native(['x', 'y'])),
                _sanitize_native(t_next.values.native(['x', 'y'])),
                _sanitize_native(L_next.values.native(['x', 'y'])),
                next_coords_free, next_vel_free,
            )
            return new_carry, (next_coords_free, next_vel_free)

        step_fn_checkpointed = jax.checkpoint(physical_step_logic)

        @jax.jit
        def compute_particle_vorticity(positions, velocities, search_radius=0.005):
            dx = positions[:, 0:1] - positions[None, :, 0] 
            dy = positions[:, 1:2] - positions[None, :, 1] 
            dist_sq = dx**2 + dy**2
            
            mask = (dist_sq > 1e-8) & (dist_sq < search_radius**2)
            weights = jnp.where(mask, 1.0 / (dist_sq + 1e-8), 0.0)
            
            du = velocities[:, 0:1] - velocities[None, :, 0]
            dv = velocities[:, 1:2] - velocities[None, :, 1]
            
            du_dy = jnp.sum(weights * du * dy, axis=1, keepdims=True) / jnp.sum(weights * dy**2 + 1e-8, axis=1, keepdims=True)
            dv_dx = jnp.sum(weights * dv * dx, axis=1, keepdims=True) / jnp.sum(weights * dx**2 + 1e-8, axis=1, keepdims=True)
            return jnp.squeeze(dv_dx - du_dy, axis=-1)

        @jit_compile
        def loss_terms(v_guess_centered):
            """Returns the raw unweighted loss components"""
            v_sim    = _center_guess_to_bc_staggered(v_guess_centered, v0_gt0, Vn_BC)
            
            # --- DIVERGENCE PENALTY ---
            div_grid = field.divergence(v_sim)
            div_native = _sanitize_native(div_grid.values.native(['x', 'y']))
            div_loss = jnp.mean(div_native ** 2)

            obs_mask = resample(~(OBSTACLE.geometry), v_sim)
            v_sim    = field.safe_mul(obs_mask, v_sim)

            state_init_native = (
                (v_sim.vector['x'].values.native(['x', 'y']), v_sim.vector['y'].values.native(['x', 'y'])),
                (vs0_gt0.vector['x'].values.native(['x', 'y']), vs0_gt0.vector['y'].values.native(['x', 'y'])),
                p0_gt0.values.native(['x', 'y']),
                t0_gt0.values.native(['x', 'y']),
                L0_gt0.values.native(['x', 'y']),
                supervision_init_pos_native, supervision_init_vel_native,
            )

            scan_inputs = (supervision_pos_native[1:], supervision_vel_native[1:])
            _, outputs_native = jax.lax.scan(step_fn_checkpointed, state_init_native, scan_inputs)
            trajectory_stack_native, simulated_vel_native = outputs_native

            nonfinite_count = jnp.sum(~jnp.isfinite(trajectory_stack_native)) + jnp.sum(~jnp.isfinite(simulated_vel_native))
            nonfinite_penalty = 1e2 * nonfinite_count

            gt_target_native   = _sanitize_native(supervision_pos_native[1:])
            trajectory_stack_native = _sanitize_native(trajectory_stack_native)
            simulated_vel_native = _sanitize_native(simulated_vel_native)
            gt_vel_target_native = _sanitize_native(supervision_vel_native[1:])
            loss_mask_expanded = supervision_loss_mask_native[1:, :, None]

            # MSE
            diff = jnp.clip((trajectory_stack_native - gt_target_native) * loss_mask_expanded, -1e3, 1e3) / POS_LOSS_SCALE
            mse_loss = jnp.sum(diff ** 2) / jnp.maximum(jnp.sum(loss_mask_expanded), 1.0)

            # --- VELOCITY LOSS ---
            vel_mask_expanded = supervision_vel_mask_native[1:, :, None]
            bg_vel_jnp = jnp.array([bg_u, bg_v], dtype=gt_vel_target_native.dtype)
            
            gt_vel_fluct = gt_vel_target_native - bg_vel_jnp
            gt_speed_fluct = jnp.sqrt(gt_vel_fluct[..., 0]**2 + gt_vel_fluct[..., 1]**2)
            
            epsilon_vel = 1e-8 
            weight = 1.0 / (gt_speed_fluct + epsilon_vel)
            
            raw_vel_diff = simulated_vel_native - gt_vel_target_native
            
            x_multiplier = jnp.ones_like(gt_vel_target_native[..., 0])
            y_multiplier = jnp.where(
                gt_vel_target_native[..., 1] < 0.0, 
                float(DEFAULT_NEG_Y_PENALTY), 
                1.0
            )
            directional_multiplier = jnp.stack([x_multiplier, y_multiplier], axis=-1)

            vel_diff = raw_vel_diff * vel_mask_expanded * directional_multiplier #* weight[..., None] 
            vel_diff = jnp.clip(vel_diff, -1e3, 1e3) / VEL_LOSS_SCALE
            vel_loss = jnp.sum(vel_diff ** 2) / jnp.maximum(jnp.sum(vel_mask_expanded), 1.0)

            # VORTICITY
            vmap_vorticity = jax.vmap(compute_particle_vorticity, in_axes=(0, 0))
            sim_vorticity = vmap_vorticity(trajectory_stack_native, simulated_vel_native)
            gt_vorticity = vmap_vorticity(gt_target_native, gt_vel_target_native)
            vorticity_mask = supervision_vel_mask_native[1:, :] 
            
            vort_diff = jnp.clip((sim_vorticity - gt_vorticity) * vorticity_mask, -1e3, 1e3) / VORT_LOSS_SCALE
            vort_loss = jnp.sum(vort_diff ** 2) / jnp.maximum(jnp.sum(vorticity_mask), 1.0)

            return mse_loss, vel_loss, vort_loss, div_loss, nonfinite_penalty

        print('\nInitializing optimization guess...')
        guess_shape = v0_gt0.at_centers().values.shape
        noise_scale = 5e-4
        noise = math.random_uniform(guess_shape, low=-noise_scale, high=noise_scale)
        init_values_dense = native_to_centered_grid(prior_native, Nx, Ny, DOMAIN['bounds'], Vn_BC).values + noise
        init_values = jnp.asarray(init_values_dense.native(init_values_dense.shape))

        init_values_tensor = math.tensor(init_values, spatial('x,y'), channel(vector='x,y'))
        v_guess_proxy = CenteredGrid(values=init_values_tensor, extrapolation=Vn_BC, bounds=DOMAIN['bounds'], resolution=spatial(x=Nx, y=Ny))

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 计算自适应权重 (Adaptive Weights Computation)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        print('\nEvaluating initial loss terms for adaptive weighting...')
        try:
            t0 = loss_terms(v_guess_proxy)
            init_mse = float(t0[0])
            init_vel = float(t0[1])
            init_vort = float(t0[2])
            init_div = float(t0[3])
            init_nonfinite = float(t0[4])
            
            print(f'Initial unweighted terms: mse={init_mse:.6e}, vel={init_vel:.6e}, vort={init_vort:.6e}, div={init_div:.6e}, nonfinite={init_nonfinite:.6e}')

            # 防止分母为0
            eps = 1e-12
            MSE_WEIGHT = TARGET_MSE_LOSS / (init_mse + eps)
            VEL_WEIGHT = TARGET_VEL_LOSS / (init_vel + eps)
            VORTICITY_WEIGHT = TARGET_VORT_LOSS / (init_vort + eps)
            DIV_WEIGHT = TARGET_DIV_LOSS / (init_div + eps)

            print(f'[INFO] Computed Adaptive Weights: MSE={MSE_WEIGHT:.2e}, VEL={VEL_WEIGHT:.2e}, VORT={VORTICITY_WEIGHT:.2e}, DIV={DIV_WEIGHT:.2e}')
        except Exception as exc:
            print(f'Initial loss evaluation failed: {exc}. Falling back to default static weights.')
            MSE_WEIGHT = 1e8
            VEL_WEIGHT = 1e0
            VORTICITY_WEIGHT = 1e2
            DIV_WEIGHT = 1e0

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 在计算出自适应权重后，组装并 JIT 编译最终的 loss_function
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        @jit_compile
        def loss_function(v_guess_centered):
            mse, vel, vort, div, nonfinite = loss_terms(v_guess_centered)
            v_native = jnp.asarray(v_guess_centered.values.native(v_guess_centered.values.shape))
            prior_reg = jnp.mean((v_native - prior_values_native) ** 2)
            
            return (
                MSE_WEIGHT * mse
                + VEL_WEIGHT * vel
                + VORTICITY_WEIGHT * vort  
                + DIV_WEIGHT * div         
                + nonfinite
                + PRIOR_REG_WEIGHT * prior_reg
            )

        @jit_compile
        def loss_from_native(v_native):
            v_tensor = math.tensor(v_native, spatial('x,y'), channel(vector='x,y'))
            v_grid = CenteredGrid(values=v_tensor, extrapolation=Vn_BC, bounds=DOMAIN['bounds'], resolution=spatial(x=Nx, y=Ny))
            return loss_function(v_grid)

        try:
            init_loss = loss_function(v_guess_proxy)
            print(f'Initial Weighted loss: {float(init_loss):.6e}')
        except Exception as exc:
            print(f'Initial weighted loss evaluation failed: {exc}')

        grad_diag_is_finite = False
        try:
            warm_native_diag = jnp.asarray(v_guess_proxy.values.native(v_guess_proxy.values.shape))
            diag_loss, diag_grad = jax.value_and_grad(loss_from_native)(warm_native_diag)
            diag_grad_clean = jnp.nan_to_num(diag_grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_diag_is_finite = np.isfinite(float(jnp.linalg.norm(diag_grad))) and np.isfinite(float(jnp.max(jnp.abs(diag_grad))))
            print(f'Warm-start gradient diagnostics: loss={float(diag_loss):.6e}, grad_l2={float(jnp.linalg.norm(diag_grad_clean)):.6e}')
        except Exception as exc:
            print(f'Warm-start gradient diagnostics failed: {exc}')

        # ---- D. L-BFGS-B optimisation ------------------------------------
        result_centered = v_guess_proxy
        initial_loss_value = None
        if args.lbfgs_iters > 0:
            print(f'\nStarting Optimization (L-BFGS-B, max_iter={args.lbfgs_iters})...')
            t_start = time.time()
            optimizer = Solve('L-BFGS-B', x0=v_guess_proxy, max_iterations=args.lbfgs_iters, suppress=[phi.math.Diverged, phi.math.NotConverged])
            result_centered = minimize(loss_function, optimizer)
            print(f'Optimization finished in {time.time() - t_start:.2f} s')

            try:
                initial_loss_value = float(loss_function(v_guess_proxy))
                warm_native = jnp.asarray(v_guess_proxy.values.native(v_guess_proxy.values.shape))
                cand_native = jnp.asarray(result_centered.values.native(result_centered.values.shape))
                cand_delta_l2 = float(jnp.linalg.norm(cand_native - warm_native))
                min_improve = max(MIN_LOSS_IMPROVE_ABS, abs(initial_loss_value) * MIN_LOSS_IMPROVE_REL)
                
                best_loss, best_alpha, best_grid = initial_loss_value, 0.0, v_guess_proxy

                for alpha in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
                    blend_native = warm_native + alpha * (cand_native - warm_native)
                    blend_tensor = math.tensor(blend_native, spatial('x,y'), channel(vector='x,y'))
                    blend_grid = CenteredGrid(values=blend_tensor, extrapolation=Vn_BC, bounds=DOMAIN['bounds'], resolution=spatial(x=Nx, y=Ny))
                    blend_loss = float(loss_function(blend_grid))
                    if np.isfinite(blend_loss) and blend_loss < (best_loss - min_improve):
                        best_loss, best_alpha, best_grid = blend_loss, alpha, blend_grid

                if best_alpha <= 0.0:
                    print('L-BFGS-B did not improve objective. Using warm-start.')
                else:
                    print(f'Accepted L-BFGS-B step: alpha={best_alpha:.2f}, loss={best_loss:.6e}')
                    result_centered = best_grid
            except Exception as exc:
                print(f'Could not validate optimization result ({exc}); using warm-start.')
                result_centered = v_guess_proxy
        else:
            print('\nSkipping L-BFGS-B because --lbfgs-iters=0; using warmup result directly.')

        v0_reconstructed = _center_guess_to_bc_staggered(result_centered, v0_gt0, Vn_BC)
        obs_mask = resample(~(OBSTACLE.geometry), v0_reconstructed)
        v0_reconstructed = field.safe_mul(obs_mask, v0_reconstructed)

        final_loss  = loss_function(result_centered)
        final_terms = loss_terms(result_centered)
        print('=== Optimization Result ===')
        print(f'Final Loss: {float(final_loss):.6e}')
        print(f'Final unweighted terms: mse={float(final_terms[0]):.6e}, vel={float(final_terms[1]):.6e}, vor={float(final_terms[2]):.6e}, div={float(final_terms[3]):.6e}')

    # ------------------------------------------------------------------
    # 5. Reconstruct marker trajectories from the optimised field
    # ------------------------------------------------------------------
    recon_traj, recon_vel = None, None
    if args.fit_continuous_field and continuous_recon_traj is not None:
        recon_traj = np.asarray(continuous_recon_traj, dtype=float)
        recon_vel = np.asarray(continuous_recon_vel, dtype=float) if continuous_recon_vel is not None else None
        
    if v0_reconstructed is not None:
        try:
            initial_supervision_markers = PointCloud(geom.Point(math.tensor(supervision_init_pos_native, instance('markers') & channel(vector='x,y'))))
            recon_traj_list = [initial_supervision_markers]
            recon_vel_list = [np.asarray(supervision_init_vel_native, dtype=float)]
            v_curr, vs_curr, p_curr, t_curr, L_curr = v0_reconstructed, vs0_gt0, p0_gt0, t0_gt0, L0_gt0
            coords_curr = np.asarray(supervision_init_pos_native, dtype=float)
            vel_curr = np.asarray(supervision_init_vel_native, dtype=float)
            
            for step in range(inv_steps):
                v_prev, vs_prev = v_curr, vs_curr
                v_curr, vs_curr, p_curr, t_curr, L_curr = helium.SFHelium_step(v_curr, vs_curr, p_curr, t_curr, L_curr, dt=inv_dt, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)
                coords_np, vel_np = _advance_particle_state(coords_curr, vel_curr, v_curr, vs_curr, v_prev, vs_prev, inv_dt, args.particle_tau, DENSITY_N, DENSITY_S, RHO_TOTAL)
                coords_tensor = math.tensor(coords_np, instance('markers') & channel(vector='x,y'))
                constrained = helium.constrain_markers_push(coords_tensor, OBSTACLE)
                coords_np = constrained.native(['markers', 'vector']) if hasattr(constrained, 'native') else np.asarray(constrained)
                coords_curr, vel_curr = coords_np, vel_np
                recon_traj_list.append(PointCloud(geom.Point(math.tensor(coords_np, instance('markers') & channel(vector='x,y')))))
                recon_vel_list.append(np.asarray(vel_np, dtype=float))
                
            recon_traj = particles.pointcloud_list_to_numpy(recon_traj_list)
            recon_vel = np.stack(recon_vel_list, axis=0)
        except Exception as exc:
            print(f'Reconstruction rollout failed: {exc}')

    if recon_traj is None and supervision_pos_native is not None:
        recon_traj = np.asarray(supervision_pos_native, dtype=float)
    if recon_vel is None and supervision_vel_native is not None:
        recon_vel = np.asarray(supervision_vel_native, dtype=float)

    # ------------------------------------------------------------------
    # 6. Extract field time series and save
    # ------------------------------------------------------------------
    v_gt_u, v_gt_v, v_gt_speed, v_gt_vec, vs_gt_u, vs_gt_v, vs_gt_speed, vs_gt_vec, t_gt = \
        _extract_time_series_for_vn(v0_gt0, vs0_gt0, p0_gt0, t0_gt0, L0_gt0, inv_steps, inv_dt, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)

    v_recon_u = v_recon_v = v_recon_speed = v_recon_vec = None
    vs_recon_u = vs_recon_v = vs_recon_speed = vs_recon_vec = None
    t_recon = None
    if args.fit_continuous_field and continuous_vn_u_series is not None:
        v_recon_u, v_recon_v, v_recon_speed = continuous_vn_u_series, continuous_vn_v_series, continuous_vn_speed_series
    if v0_reconstructed is not None:
        v_recon_u, v_recon_v, v_recon_speed, v_recon_vec, vs_recon_u, vs_recon_v, vs_recon_speed, vs_recon_vec, t_recon = \
            _extract_time_series_for_vn(v0_reconstructed, vs0_gt0, p0_gt0, t0_gt0, L0_gt0, inv_steps, inv_dt, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)

    # Stable NumPy outputs
    gt_np         = particles.tensor_time_markers_to_numpy(gt_tensor)
    loss_mask_to_save = particles.tensor_time_marker_mask_to_numpy(loss_mask)
    reset_mask_to_save = particles.tensor_time_marker_mask_to_numpy(reset_mask)
    if args.fit_continuous_field and continuous_recon_loss_mask is not None:
        loss_mask_to_save, reset_mask_to_save = continuous_recon_loss_mask, continuous_recon_reset_mask

    config.update({
        'obs_vel_np': gt_vel_native, 'recon_vel_np': recon_vel, 'vel_mask_np': vel_mask_native,
        'supervision_mode': supervision_mode,
        'supervision_positions_np': np.asarray(supervision_pos_native, dtype=float),
        'supervision_velocities_np': np.asarray(supervision_vel_native, dtype=float),
        'supervision_loss_mask_np': np.asarray(supervision_loss_mask_native, dtype=float),
        'supervision_vel_mask_np': np.asarray(supervision_vel_mask_native, dtype=float),
        'supervision_pts': np.asarray(supervision_pos_native, dtype=float),
        'supervision_pts_vel': np.asarray(supervision_vel_native, dtype=float),
        'supervision_pts_mask': np.asarray(supervision_loss_mask_native, dtype=float),
        'sim_pts': np.asarray(recon_traj, dtype=float) if recon_traj is not None else None,
        'sim_pts_vel': np.asarray(recon_vel, dtype=float) if recon_vel is not None else None,
    })

    field_series = {
        'gt':    (v_gt_u, v_gt_v, v_gt_speed, v_gt_vec, vs_gt_u, vs_gt_v, vs_gt_speed, vs_gt_vec, t_gt),
        'recon': (v_recon_u, v_recon_v, v_recon_speed, v_recon_vec, vs_recon_u, vs_recon_v, vs_recon_speed, vs_recon_vec, t_recon),
    }
    save_dict = _build_save_dict(config, gt_np, recon_traj, loss_mask_to_save, reset_mask_to_save, field_series)
    
    steps_len = int(inv_steps) + 1
    save_dict.update({
        'supervision_pts': np.asarray(supervision_pos_native[:steps_len], dtype=float) if supervision_pos_native is not None else None,
        'supervision_pts_vel': np.asarray(supervision_vel_native[:steps_len], dtype=float) if supervision_vel_native is not None else None,
        'supervision_pts_mask': np.asarray(supervision_loss_mask_native[:steps_len], dtype=float) if supervision_loss_mask_native is not None else None,
        'sim_pts': np.asarray(recon_traj[:steps_len], dtype=float) if recon_traj is not None else None,
        'sim_pts_vel': np.asarray(recon_vel[:steps_len], dtype=float) if recon_vel is not None else None,
    })
    if fit_times is not None:
        save_dict.update({'fit_continuous_field': 1, 'fit_times': fit_times, 'supervision_times': supervision_times})
    if continuous_recon_loss_mask is not None:
        save_dict.update({'recon_loss_mask': continuous_recon_loss_mask, 'recon_reset_mask': continuous_recon_reset_mask})
        
    io.save_npz('data/experiment/cyl_exp_recon.npz', **save_dict)
    print('Task 05 complete. Data saved to data/experiment/cyl_exp_recon.npz')

if __name__ == '__main__':
    main()