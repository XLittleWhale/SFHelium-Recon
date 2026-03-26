"""
Task 03: Cylinder flow simulation/inversion (uses SFHelium model)

Optimization follows the proven D4.py approach:
    - Direct CenteredGrid field as optimization variable
    - L-BFGS-B via phiflow minimize()
    - jax.lax.scan + jax.checkpoint for memory-efficient forward simulation
    - Marker position/velocity loss
    - Obstacle (Sphere) constraint applied after each advection step
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

from phi.jax.flow import *
from phi.jax.flow import Box, StaggeredGrid, CenteredGrid, ZERO_GRADIENT, instance, advect, math, batch, jit_compile, spatial, channel, dual, geom, PointCloud, minimize, Sphere, Obstacle, resample
from sf_recon.physics import helium, boundaries
from sf_recon.utils import viz, io
from sf_recon.utils.guess import (
    build_obstacle_aware_inflow_prior,
    native_to_centered_grid,
)
from sf_recon.utils.load import load_csv_to_grids_cyl
from sf_recon.utils.saving import (
    simple_to_numpy as _simple_to_numpy,
    stack_if_possible as _stack_if_possible,
    extract_time_series_for_vn as _extract_time_series_for_vn,
    ensure_HW as _ensure_HW,
)
import argparse
import numpy as np
import time
import jax
import jax.numpy as jnp


def _pointcloud_list_to_numpy(pc_list):
    """Convert a list of PointCloud objects to a (T, N, 2) numpy array."""
    coords = []
    for pc in pc_list:
        if hasattr(pc, 'geometry'):
            c = pc.geometry.center.numpy(['markers', 'vector'])
        elif hasattr(pc, 'center'):
            c = pc.center.numpy(['markers', 'vector'])
        else:
            c = np.asarray(pc)
        coords.append(np.asarray(c))
    return np.stack(coords, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-csv', type=str, default=None, help='Optional OpenFOAM-export CSV to initialize fields')
    parser.add_argument('--skip-inversion', action='store_true', help='Skip the inversion step')
    parser.add_argument('--lbfgs-iters', type=int, default=100, help='Max L-BFGS-B iterations')
    parser.add_argument('--outer-iters', type=int, default=3, help='Outer neural optimization iterations')
    parser.add_argument('--outer-lr', type=float, default=1e-3, help='Learning rate for neural initialization optimization')
    parser.add_argument('--pretrain-lbfgs-iters', type=int, default=10, help='Inner L-BFGS-B steps used during outer-loop neural optimization')
    parser.add_argument('--nn-width', type=int, default=64, help='Hidden width of the neural initialization MLP')
    parser.add_argument('--nn-depth', type=int, default=3, help='Number of hidden layers of the neural initialization MLP')
    parser.add_argument('--warmstart-candidates', type=int, default=4, help='Number of neural warm-start candidates to evaluate before final L-BFGS-B')
    parser.add_argument('--warmstart-noise', type=float, default=0.05, help='Noise scale for sampling neural warm-start candidates')
    args = parser.parse_args()

    math.use('jax')
    math.set_global_precision(64)

    print('Task 03: starting')
    # ==========================================
    # Domain and simulation parameters
    # ==========================================
    Lx, Ly = 0.02, 0.2
    Nx, Ny = 40, 200
    RAD = 0.004  #0.003175  #0.004
    MARKERS = 4096
    DT = 1e-6
    STEPS = 1
    MSE_WEIGHT = 1e11
    VEL_WEIGHT = 0
    DOMAIN = dict(x=Nx, y=Ny, bounds=Box(x=Lx, y=Ly))
    SPHERE = Sphere(x=Lx/2, y=Ly/2, radius=RAD)
    OBSTACLE = Obstacle(SPHERE)

    # Physical parameters
    HEAT_SOURCE_INTENSITY = 1.94
    HEAT_FLUX = 6000  #10000  #3000  #6000
    DENSITY = 145.5244
    ENTROPY = 813.4
    DENSITY_N = 68.22
    DENSITY_S = 77.31
    Vn_IN = HEAT_FLUX / (DENSITY * ENTROPY * HEAT_SOURCE_INTENSITY)
    Vs_IN = -(DENSITY_N * Vn_IN) / DENSITY_S
    Vn_BC, Vs_BC, J_BC, t_BC_THERMAL, p_BC = boundaries.get_cylinder_bcs(Vn_IN=Vn_IN, Vs_IN=Vs_IN, PRESSURE_0=3130)

    # initialize GT fields
    def _init_fields_from_csv_or_default(init_csv=None):
        if init_csv is None:
            v0 = StaggeredGrid(0, Vn_BC, **DOMAIN)
            vs0 = StaggeredGrid(0, Vs_BC, **DOMAIN)
            p0 = CenteredGrid(3130, p_BC, **DOMAIN)
            t0 = CenteredGrid(HEAT_SOURCE_INTENSITY, t_BC_THERMAL, **DOMAIN)
            L0 = CenteredGrid(0, ZERO_GRADIENT, **DOMAIN)
            return v0, vs0, p0, t0, L0
        try:
            un0_np, un1_np, us0_np, us1_np, p_np, t_np, L_np, count_total = load_csv_to_grids_cyl(
                init_csv, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny
            )
            print(f'Loaded CSV init from {init_csv} with {count_total} samples')
            t_grid = CenteredGrid(tensor(t_np.T, spatial('x,y')), t_BC_THERMAL, bounds=Box(x=Lx, y=Ly))
            p_grid = CenteredGrid(tensor(p_np.T, spatial('x,y')), p_BC, bounds=Box(x=Lx, y=Ly))
            L_grid = CenteredGrid(tensor(L_np.T, spatial('x,y')), ZERO_GRADIENT, bounds=Box(x=Lx, y=Ly))
            un_center = CenteredGrid(tensor(np.stack([un0_np.T, un1_np.T], axis=-1), spatial('x,y'), channel(vector='x,y')), Vn_BC, bounds=Box(x=Lx, y=Ly))
            vs_center = CenteredGrid(tensor(np.stack([us0_np.T, us1_np.T], axis=-1), spatial('x,y'), channel(vector='x,y')), Vs_BC, bounds=Box(x=Lx, y=Ly))
            target_v_template = StaggeredGrid(0, Vn_BC, **DOMAIN)
            v_stag = un_center.at(target_v_template)
            target_vs_template = StaggeredGrid(0, Vs_BC, **DOMAIN)
            vs_stag = vs_center.at(target_vs_template)
            return v_stag, vs_stag, p_grid, t_grid, L_grid
        except Exception as exc:
            print(f'Failed to load CSV for initialization; using defaults. Reason: {exc}')
            return StaggeredGrid(0, Vn_BC, **DOMAIN), StaggeredGrid(0, Vs_BC, **DOMAIN), CenteredGrid(3130, p_BC, **DOMAIN), CenteredGrid(HEAT_SOURCE_INTENSITY, t_BC_THERMAL, **DOMAIN), CenteredGrid(0, ZERO_GRADIENT, **DOMAIN)

    
    v0_gt0, vs0_gt0, p0_gt0, t0_gt0, L0_gt0 = _init_fields_from_csv_or_default(getattr(args, 'init_csv', None))

    if OBSTACLE is not None:
        obstacle_mask_vn = resample(~(OBSTACLE.geometry), v0_gt0)
        v0_gt0 = field.safe_mul(obstacle_mask_vn, v0_gt0)
        obstacle_mask_vs = resample(~(OBSTACLE.geometry), vs0_gt0)
        vs0_gt0 = field.safe_mul(obstacle_mask_vs, vs0_gt0)
        obstacle_mask_t = resample(~(OBSTACLE.geometry), t0_gt0)
        t0_gt0 = field.safe_mul(obstacle_mask_t, t0_gt0)
        obstacle_mask_L = resample(~(OBSTACLE.geometry), L0_gt0)
        L0_gt0 = field.safe_mul(obstacle_mask_L, L0_gt0)

    # Markers Initialization
    markers0 = DOMAIN['bounds'].sample_uniform(instance(markers=MARKERS))
    if OBSTACLE is not None:
        markers0 = helium.constrain_markers_push(markers0, OBSTACLE)
    # Wrap as PointCloud for advect.points compatibility
    initial_markers = PointCloud(geom.Point(markers0))

    # ==========================================
    # Simulate GT marker trajectories
    # ==========================================
    curr_vn = v0_gt0
    curr_vs = vs0_gt0
    curr_p = p0_gt0
    curr_t = t0_gt0
    curr_L = L0_gt0
    gt_traj = [initial_markers]
    current_markers = initial_markers
    for _ in range(STEPS):
        curr_vn, curr_vs, curr_p, curr_t, curr_L = helium.SFHelium_step(
            curr_vn, curr_vs, curr_p, curr_t, curr_L,
            dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE,
            Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL
        )
        current_markers = advect.points(current_markers, curr_vn, dt=DT, integrator=advect.rk4)
        if OBSTACLE is not None:
            constrained_coords = helium.constrain_markers_push(current_markers.geometry.center, OBSTACLE)
            current_markers = PointCloud(geom.Point(constrained_coords))
        gt_traj.append(current_markers)

    gt_stack = math.stack(gt_traj, batch('time'))
    print('GT Generation Done.')

    # Build GT trajectory as native array for scan-based loss
    # Shape: (STEPS+1, MARKERS, 2)
    gt_native_list = []
    for step_pc in gt_traj:
        if hasattr(step_pc, 'geometry'):
            gt_native_list.append(step_pc.geometry.center.native(['markers', 'vector']))
        else:
            gt_native_list.append(step_pc.native(['markers', 'vector']))
    gt_all_native = jnp.stack(gt_native_list, axis=0)  # (STEPS+1, MARKERS, 2)

    # Obstacle geometry constants for use inside scan loop
    obstacle_center_np = np.array([float(SPHERE.center.vector[0]), float(SPHERE.center.vector[1])])
    obstacle_radius = float(SPHERE.radius)

    v0_reconstructed = None

    # ==========================================
    # Inversion: L-BFGS-B direct field optimization (D4.py approach)
    # ==========================================
    if not getattr(args, 'skip_inversion', False):

        prior_native = build_obstacle_aware_inflow_prior(
            Nx, Ny, Lx, Ly, Vn_IN,
            center_xy=(float(SPHERE.center.vector[0]), float(SPHERE.center.vector[1])),
            radius=float(SPHERE.radius)
        )

        # ------------------------------------------------------------------
        # A. Define physical_step_logic for jax.lax.scan + jax.checkpoint
        # ------------------------------------------------------------------
        def physical_step_logic(carry_state_native, time_input_native):
            """Single time-step for jax.lax.scan (native JAX arrays)."""
            # 1. Unpack carry state
            (v_x_nat, v_y_nat), (vs_x_nat, vs_y_nat), p_nat, t_nat, L_nat, coords_nat = carry_state_native

            # 2. Reconstruct PhiFlow Grid objects
            bounds = DOMAIN['bounds']
            grid_shape = spatial(x=Nx, y=Ny)

            v_x_tensor = math.tensor(v_x_nat, spatial('x,y'))
            v_y_tensor = math.tensor(v_y_nat, spatial('x,y'))
            v_values = math.stack([v_x_tensor, v_y_tensor], dual(vector='x,y'))
            v = StaggeredGrid(values=v_values, extrapolation=Vn_BC, bounds=bounds, resolution=grid_shape)

            vs_x_tensor = math.tensor(vs_x_nat, spatial('x,y'))
            vs_y_tensor = math.tensor(vs_y_nat, spatial('x,y'))
            vs_values = math.stack([vs_x_tensor, vs_y_tensor], dual(vector='x,y'))
            vs = StaggeredGrid(values=vs_values, extrapolation=Vs_BC, bounds=bounds, resolution=grid_shape)

            p = CenteredGrid(values=math.tensor(p_nat, grid_shape), extrapolation=p_BC, bounds=bounds, resolution=grid_shape)
            t = CenteredGrid(values=math.tensor(t_nat, grid_shape), extrapolation=t_BC_THERMAL, bounds=bounds, resolution=grid_shape)
            L = CenteredGrid(values=math.tensor(L_nat, grid_shape), extrapolation=ZERO_GRADIENT, bounds=bounds, resolution=grid_shape)

            coords = math.tensor(coords_nat, instance('markers') & channel(vector='x,y'))

            # 3. Physics step (with obstacle)
            v_next, vs_next, p_next, t_next, L_next = helium.SFHelium_step(
                v, vs, p, t, L,
                dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE,
                Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL
            )

            # 4. Advect markers
            markers_obj = PointCloud(geom.Point(coords))
            markers_obj = advect.points(markers_obj, v_next, dt=DT, integrator=advect.rk4)
            advected_coords = markers_obj.geometry.center

            # 5. Obstacle constraint on markers (push outside sphere)
            # Pure JAX arithmetic for scan compatibility
            obs_c = jnp.array(obstacle_center_np)
            obs_r = obstacle_radius
            coords_jax = advected_coords.native(['markers', 'vector'])
            diff_obs = coords_jax - obs_c
            dist_obs = jnp.sqrt(jnp.sum(diff_obs ** 2, axis=-1, keepdims=True))
            epsilon = 1e-4
            correction = obs_c + (diff_obs / (dist_obs + 1e-6)) * (obs_r + epsilon)
            is_inside = dist_obs < obs_r
            coords_jax = jnp.where(is_inside, correction, coords_jax)

            # 6. Pack back to native arrays
            new_carry_native = (
                (v_next.vector['x'].values.native(['x', 'y']),
                 v_next.vector['y'].values.native(['x', 'y'])),
                (vs_next.vector['x'].values.native(['x', 'y']),
                 vs_next.vector['y'].values.native(['x', 'y'])),
                p_next.values.native(['x', 'y']),
                t_next.values.native(['x', 'y']),
                L_next.values.native(['x', 'y']),
                coords_jax
            )
            output_native = coords_jax
            return new_carry_native, output_native

        # Apply gradient checkpointing
        step_fn_checkpointed = jax.checkpoint(physical_step_logic)

        # ------------------------------------------------------------------
        # B. Loss function with L-BFGS-B compatible signature
        # ------------------------------------------------------------------
        @jit_compile
        def loss_terms(v_guess_centered):
            """
            Loss = marker position + marker velocity.
            v_guess_centered: CenteredGrid (Nx, Ny, vector=2)
            """
            # A. Convert guess to StaggeredGrid & apply obstacle mask
            v_sim = v_guess_centered.at(v0_gt0)
            obstacle_mask_vn = resample(~(OBSTACLE.geometry), v_sim)
            v_sim = field.safe_mul(obstacle_mask_vn, v_sim)

            # B. Pack initial state as native arrays
            v_init_tuple = (
                v_sim.vector['x'].values.native(['x', 'y']),
                v_sim.vector['y'].values.native(['x', 'y'])
            )
            vs_init_tuple = (
                vs0_gt0.vector['x'].values.native(['x', 'y']),
                vs0_gt0.vector['y'].values.native(['x', 'y'])
            )

            init_coords = initial_markers.geometry.center
            state_init_native = (
                v_init_tuple,
                vs_init_tuple,
                p0_gt0.values.native(['x', 'y']),
                t0_gt0.values.native(['x', 'y']),
                L0_gt0.values.native(['x', 'y']),
                init_coords.native(['markers', 'vector'])
            )

            # C. Dummy scan inputs
            scan_inputs = jnp.arange(STEPS)

            # D. Run forward simulation via jax.lax.scan
            final_state_native, trajectory_stack_native = jax.lax.scan(
                step_fn_checkpointed, state_init_native, scan_inputs
            )
            # trajectory_stack_native: (STEPS, MARKERS, 2)

            # E. Marker MSE loss
            gt_target_native = gt_all_native[1:]  # (STEPS, MARKERS, 2)
            diff = trajectory_stack_native - gt_target_native
            mse_loss = jnp.sum(diff ** 2)

            prev_pred_native = jnp.concatenate([
                init_coords.native(['markers', 'vector'])[None, ...],
                trajectory_stack_native[:-1]
            ], axis=0)
            prev_gt_native = gt_all_native[:-1]
            pred_vel_native = trajectory_stack_native - prev_pred_native
            gt_vel_native = gt_target_native - prev_gt_native
            vel_loss = jnp.sum((pred_vel_native - gt_vel_native) ** 2)

            return mse_loss, vel_loss

        @jit_compile
        def loss_function(v_guess_centered):
            mse_loss, vel_loss = loss_terms(v_guess_centered)
            total_loss = (
                MSE_WEIGHT * mse_loss
                + VEL_WEIGHT * vel_loss
            )

            return total_loss

        guess_shape = v0_gt0.at_centers().values.shape
        noise_scale = 0.005
        noise = math.random_uniform(guess_shape, low=-noise_scale, high=noise_scale)
        init_values = native_to_centered_grid(prior_native, Nx, Ny, DOMAIN['bounds'], Vn_BC).values + noise
        candidate_grid = CenteredGrid(
            values=init_values,
            extrapolation=Vn_BC,
            bounds=DOMAIN['bounds']
        )
        print(f'Warm-start prior [cylinder wake]: loss={float(loss_function(candidate_grid)):.6e}')
        v_guess_proxy = CenteredGrid(
            values=init_values,
            extrapolation=Vn_BC,
            bounds=DOMAIN['bounds']
        )

        # Evaluate initial loss
        try:
            init_loss = loss_function(v_guess_proxy)
            print(f'Initial loss: {float(init_loss):.6e}')
            init_terms = loss_terms(v_guess_proxy)
            print(
                'Initial terms: '
                f'mse={float(init_terms[0]):.6e}, '
                f'vel={float(init_terms[1]):.6e}'
            )
        except Exception as e:
            print(f'Initial loss evaluation failed: {e}')

        print(f'\nStarting Optimization (L-BFGS-B, max_iter={args.lbfgs_iters})...')
        t_start = time.time()

        optimizer = Solve(
            'L-BFGS-B',
            x0=v_guess_proxy,
            max_iterations=args.lbfgs_iters,
            # abs_tol = 1e-16,
            suppress=[phi.math.Diverged, phi.math.NotConverged]
        )

        with math.SolveTape(record_trajectories=False) as solves:
            result_centered = minimize(loss_function, optimizer)

        t_end = time.time()
        print(f'Optimization finished in {t_end - t_start:.2f} s')

        # Extract result
        v0_reconstructed = result_centered.at(v0_gt0)
        # Apply obstacle mask
        obstacle_mask_vn = resample(~(OBSTACLE.geometry), v0_reconstructed)
        v0_reconstructed = field.safe_mul(obstacle_mask_vn, v0_reconstructed)

        final_loss = loss_function(result_centered)
        final_terms = loss_terms(result_centered)

        print(f'=== Optimization Result ===')
        print(f'Final Loss: {float(final_loss):.6e}')
        print(
            'Final terms: '
            f'mse={float(final_terms[0]):.6e}, '
            f'vel={float(final_terms[1]):.6e}'
        )

        # Compare with GT
        diff_field = v0_reconstructed - v0_gt0
        diff_mag = field.l2_loss(diff_field)
        print(f'Field Reconstruction Error (L2): {float(diff_mag):.6e}')

    else:
        v0_reconstructed = None

    # ==============================================================================
    # Prepare and save GT and reconstruction for visualization
    # ==============================================================================

    # Extract per-frame GT fields for inspection
    v_gt_u, v_gt_v, v_gt_speed, v_gt_vec, vs_gt_u, vs_gt_v, vs_gt_speed, vs_gt_vec, t_gt = _extract_time_series_for_vn(
        v0_gt0, vs0_gt0, p0_gt0, t0_gt0, L0_gt0, STEPS, DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)

    # Reconstruction fields (if inversion ran)
    recon_traj = None
    v_recon_u = v_recon_v = v_recon_speed = v_recon_vec = None
    vs_recon_u = vs_recon_v = vs_recon_speed = vs_recon_vec = None
    t_recon = None
    if v0_reconstructed is not None:
        v_recon_u, v_recon_v, v_recon_speed, v_recon_vec, vs_recon_u, vs_recon_v, vs_recon_speed, vs_recon_vec, t_recon = _extract_time_series_for_vn(
            v0_reconstructed, vs0_gt0, p0_gt0, t0_gt0, L0_gt0, STEPS, DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)

        # Reconstruct particle trajectories advected by reconstructed normal fluid
        try:
            markers_rec = initial_markers
            recon_traj_list = [markers_rec]
            v_curr = v0_reconstructed
            vs_curr = vs0_gt0
            p_curr = p0_gt0
            t_curr = t0_gt0
            L_curr = L0_gt0

            if OBSTACLE is not None:
                obstacle_mask_vn = resample(~(OBSTACLE.geometry), v_curr)
                v_curr = field.safe_mul(obstacle_mask_vn, v_curr)
                obstacle_mask_vs = resample(~(OBSTACLE.geometry), vs_curr)
                vs_curr = field.safe_mul(obstacle_mask_vs, vs_curr)
                obstacle_mask_t = resample(~(OBSTACLE.geometry), t_curr)
                t_curr = field.safe_mul(obstacle_mask_t, t_curr)
                obstacle_mask_L = resample(~(OBSTACLE.geometry), L_curr)
                L_curr = field.safe_mul(obstacle_mask_L, L_curr)
            
            for _ in range(STEPS):
                v_curr, vs_curr, p_curr, t_curr, L_curr = helium.SFHelium_step(
                    v_curr, vs_curr, p_curr, t_curr, L_curr,
                    dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE,
                    Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL
                )
                markers_rec = advect.points(markers_rec, v_curr, dt=DT, integrator=advect.rk4)
                if OBSTACLE is not None:
                    constrained_coords = helium.constrain_markers_push(markers_rec.geometry.center, OBSTACLE)
                    markers_rec = PointCloud(geom.Point(constrained_coords))
                recon_traj_list.append(markers_rec)
            recon_stack = math.stack(recon_traj_list, batch('time'))
            recon_traj = _pointcloud_list_to_numpy(recon_traj_list)
        except Exception:
            recon_traj = None

    # Normalize and prepare temperature arrays
    t_gt = _simple_to_numpy(t_gt) if 't_gt' in locals() else None
    t_recon = _simple_to_numpy(t_recon) if 't_recon' in locals() else None
    if t_gt is None and v_gt_speed is not None:
        try:
            t_gt = np.full(np.asarray(v_gt_speed).shape, np.nan, dtype=float)
        except Exception:
            t_gt = None
    if t_recon is None and v_recon_speed is not None:
        try:
            t_recon = np.full(np.asarray(v_recon_speed).shape, np.nan, dtype=float)
        except Exception:
            t_recon = None

    # Ensure arrays are numpy and have shape (T, H, W) or (H, W) before saving
    def _prepare_save(x):
        try:
            a = _simple_to_numpy(x)
        except Exception:
            a = None
        return _ensure_HW(a, Nx, Ny)

    v_gt_u = _prepare_save(v_gt_u)
    v_gt_v = _prepare_save(v_gt_v)
    v_gt_speed = _prepare_save(v_gt_speed)
    v_gt_vec = _prepare_save(v_gt_vec)
    vs_gt_u = _prepare_save(vs_gt_u)
    vs_gt_v = _prepare_save(vs_gt_v)
    vs_gt_speed = _prepare_save(vs_gt_speed)
    vs_gt_vec = _prepare_save(vs_gt_vec)

    v_recon_u = _prepare_save(v_recon_u)
    v_recon_v = _prepare_save(v_recon_v)
    v_recon_speed = _prepare_save(v_recon_speed)
    v_recon_vec = _prepare_save(v_recon_vec)
    vs_recon_u = _prepare_save(vs_recon_u)
    vs_recon_v = _prepare_save(vs_recon_v)
    vs_recon_speed = _prepare_save(vs_recon_speed)
    vs_recon_vec = _prepare_save(vs_recon_vec)

    t_gt = _prepare_save(t_gt)
    t_recon = _prepare_save(t_recon)

    gt_np = _pointcloud_list_to_numpy(gt_traj)
    save_dict = dict(
        success=1, gt=gt_np, recon=recon_traj,
        v_gt_speed=v_gt_speed, v_recon_speed=v_recon_speed, v_gt_vec=v_gt_vec, v_recon_vec=v_recon_vec,
        v_gt_u=v_gt_u, v_gt_v=v_gt_v, v_recon_u=v_recon_u, v_recon_v=v_recon_v,
        vs_gt_speed=vs_gt_speed, vs_recon_speed=vs_recon_speed, vs_gt_vec=vs_gt_vec, vs_recon_vec=vs_recon_vec,
        vs_gt_u=vs_gt_u, vs_gt_v=vs_gt_v, vs_recon_u=vs_recon_u, vs_recon_v=vs_recon_v,
        t_gt=t_gt, t_recon=t_recon
    )
    io.save_npz('data/simulation/cyl_v0_recon.npz', **save_dict)

    print('Task 03 complete. Data saved to data/simulation/cyl_v0_recon.npz')

if __name__ == '__main__':
    main()
