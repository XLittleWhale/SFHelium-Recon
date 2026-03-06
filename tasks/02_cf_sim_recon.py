"""
Task 02: Superfluid helium counterflow simulation + inversion

This script generates a Ground Truth (GT) using the SFHelium model and
then performs inversion for the normal-fluid initial field `vn`.

Optimization follows the proven D4.py approach:
  - Direct CenteredGrid field as optimization variable
  - L-BFGS-B via phiflow minimize()
  - jax.lax.scan + jax.checkpoint for memory-efficient forward simulation
  - Marker MSE loss + smoothness + energy regularization
"""
from phi.jax.flow import *
from phi.jax.flow import Box, StaggeredGrid, CenteredGrid, ZERO_GRADIENT, instance, advect, math, batch, jit_compile, spatial, channel, dual, geom, PointCloud, minimize
from sf_recon.physics import helium, boundaries
from sf_recon.utils import io
from sf_recon.utils.load import load_csv_to_grids_cf
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
    args = parser.parse_args()

    math.use('jax')
    math.set_global_precision(64)

    print('Task 02: starting')
    # ==========================================
    # Domain & simulation parameters
    # ==========================================
    Lx, Ly = 0.0004, 0.0016
    Nx, Ny = 20, 80
    MARKERS = 512
    DT = 1e-6
    STEPS = 5
    PRE_STEPS = 5
    MSE_WEIGHT = 1e10
    SMOOTH_WEIGHT = 0   # smoothness weight
    ENERGY_WEIGHT  = 0   # energy weight
    ZERO_WEIGHT    = 0  # anti-zero weight (prevent trivial all-zero solution)
    DOMAIN = dict(x=Nx, y=Ny, bounds=Box(x=Lx, y=Ly))
    OBSTACLE = None

    # Physical parameters
    HEAT_SOURCE_INTENSITY = 2.0
    HEAT_FLUX = 2170.0
    DENSITY = 145.6217
    ENTROPY = 962.1
    DENSITY_N = 80.55
    DENSITY_S = 65.07
    Vn_IN = HEAT_FLUX / (DENSITY * ENTROPY * HEAT_SOURCE_INTENSITY)
    Vs_IN = -(DENSITY_N * Vn_IN) / DENSITY_S
    Vn_BC, Vs_BC, J_BC, t_BC_THERMAL, p_BC = boundaries.get_sf_bcs(Vn_IN=Vn_IN, Vs_IN=Vs_IN, PRESSURE_0=3130)

    # ==========================================
    # Initialize GT fields
    # ==========================================
    if args.init_csv is not None:
        try:
            un0_np, un1_np, us0_np, us1_np, p_np, t_np, L_np, count_total = load_csv_to_grids_cf(
                args.init_csv, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny
            )
            print(f'Loaded CSV init from {args.init_csv} with {count_total} samples')
            t0_gt0 = CenteredGrid(tensor(t_np.T, spatial('x,y')), t_BC_THERMAL, bounds=Box(x=Lx, y=Ly))
            p0_gt0 = CenteredGrid(tensor(p_np.T, spatial('x,y')), p_BC, bounds=Box(x=Lx, y=Ly))
            L0_gt0 = CenteredGrid(tensor(L_np.T, spatial('x,y')), ZERO_GRADIENT, bounds=Box(x=Lx, y=Ly))
            un_center = CenteredGrid(tensor(np.stack([un0_np.T, un1_np.T], axis=-1), spatial('x,y'), channel(vector='x,y')), Vn_BC, bounds=Box(x=Lx, y=Ly))
            vs_center = CenteredGrid(tensor(np.stack([us0_np.T, us1_np.T], axis=-1), spatial('x,y'), channel(vector='x,y')), Vs_BC, bounds=Box(x=Lx, y=Ly))
            target_v_template = StaggeredGrid(0, Vn_BC, **DOMAIN)
            v0_gt0 = un_center.at(target_v_template)
            target_vs_template = StaggeredGrid(0, Vs_BC, **DOMAIN)
            vs0_gt0 = vs_center.at(target_vs_template)
        except Exception as exc:
            print(f'Failed to load CSV, falling back to zeros. Reason: {exc}')
            v0_gt0 = StaggeredGrid(0, Vn_BC, **DOMAIN)
            vs0_gt0 = StaggeredGrid(0, Vs_BC, **DOMAIN)
            p0_gt0 = CenteredGrid(3130, p_BC, **DOMAIN)
            t0_gt0 = CenteredGrid(HEAT_SOURCE_INTENSITY, t_BC_THERMAL, **DOMAIN)
            L0_gt0 = CenteredGrid(0, ZERO_GRADIENT, **DOMAIN)
    else:
        v0_gt0 = StaggeredGrid(0, Vn_BC, **DOMAIN)
        vs0_gt0 = StaggeredGrid(0, Vs_BC, **DOMAIN)
        p0_gt0 = CenteredGrid(3130, p_BC, **DOMAIN)
        t0_gt0 = CenteredGrid(HEAT_SOURCE_INTENSITY, t_BC_THERMAL, **DOMAIN)
        L0_gt0 = CenteredGrid(0, ZERO_GRADIENT, **DOMAIN)

    # Markers Initialization — wrap as PointCloud for advect.points compatibility
    markers0_coords = DOMAIN['bounds'].sample_uniform(instance(markers=MARKERS))
    markers0 = PointCloud(geom.Point(markers0_coords))

    # Pre-steps to reach steady state
    curr_vn = v0_gt0
    curr_vs = vs0_gt0
    curr_p = p0_gt0
    curr_t = t0_gt0
    curr_L = L0_gt0
    for _ in range(PRE_STEPS):
        curr_vn, curr_vs, curr_p, curr_t, curr_L = helium.SFHelium_step(
            curr_vn, curr_vs, curr_p, curr_t, curr_L,
            dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE,
            Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL
        )
    v0_gt0, vs0_gt0, p0_gt0, t0_gt0, L0_gt0 = curr_vn, curr_vs, curr_p, curr_t, curr_L
    print('Pre-steps completed')
    initial_markers = markers0

    # ==========================================
    # Simulate GT marker trajectories
    # ==========================================
    gt_traj = [initial_markers]
    current_markers = initial_markers
    cvn, cvs, cp, ct, cL = v0_gt0, vs0_gt0, p0_gt0, t0_gt0, L0_gt0
    for _ in range(STEPS):
        cvn, cvs, cp, ct, cL = helium.SFHelium_step(
            cvn, cvs, cp, ct, cL,
            dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE,
            Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL
        )
        current_markers = advect.points(current_markers, cvn, dt=DT, integrator=advect.rk4)
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

    v0_reconstructed = None

    # ==========================================
    # Inversion: L-BFGS-B direct field optimization (D4.py approach)
    # ==========================================
    if not getattr(args, 'skip_inversion', False):

        # ------------------------------------------------------------------
        # A. Define physical_step_logic for jax.lax.scan + jax.checkpoint
        # ------------------------------------------------------------------
        # This function operates on native JAX arrays to be compatible with
        # jax.lax.scan. It reconstructs PhiFlow grids internally, runs one
        # physics step, advects markers, and returns native arrays.
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

            # 3. Physics step
            v_next, vs_next, p_next, t_next, L_next = helium.SFHelium_step(
                v, vs, p, t, L,
                dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE,
                Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL
            )

            # 4. Advect markers
            markers_obj = PointCloud(geom.Point(coords))
            markers_obj = advect.points(markers_obj, v_next, dt=DT, integrator=advect.rk4)
            advected_coords = markers_obj.geometry.center

            # 5. Pack back to native arrays
            new_carry_native = (
                (v_next.vector['x'].values.native(['x', 'y']),
                 v_next.vector['y'].values.native(['x', 'y'])),
                (vs_next.vector['x'].values.native(['x', 'y']),
                 vs_next.vector['y'].values.native(['x', 'y'])),
                p_next.values.native(['x', 'y']),
                t_next.values.native(['x', 'y']),
                L_next.values.native(['x', 'y']),
                advected_coords.native(['markers', 'vector'])
            )
            output_native = advected_coords.native(['markers', 'vector'])
            return new_carry_native, output_native

        # Apply gradient checkpointing
        step_fn_checkpointed = jax.checkpoint(physical_step_logic)

        # ------------------------------------------------------------------
        # B. Loss function with L-BFGS-B compatible signature
        # ------------------------------------------------------------------
        @jit_compile
        def loss_function(v_guess_centered):
            """
            Loss = marker MSE + smoothness + energy.
            v_guess_centered: CenteredGrid (Nx, Ny, vector=2)
            """
            # A. Convert guess to StaggeredGrid for physics
            v_sim = v_guess_centered.at(v0_gt0)

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

            # C. Dummy scan inputs (no reset mask needed for synthetic GT)
            scan_inputs = jnp.arange(STEPS)

            # D. Run forward simulation via jax.lax.scan
            final_state_native, trajectory_stack_native = jax.lax.scan(
                step_fn_checkpointed, state_init_native, scan_inputs
            )
            # trajectory_stack_native: (STEPS, MARKERS, 2) — predicted positions at t=1..STEPS

            # E. Marker MSE loss
            gt_target_native = gt_all_native[1:]  # (STEPS, MARKERS, 2)
            diff = trajectory_stack_native - gt_target_native
            mse_loss = jnp.sum(diff ** 2)

            # F. Smoothness regularization (gradient penalty on the guess field)
            u_component = v_guess_centered.vector['x']
            v_component = v_guess_centered.vector['y']
            grad_u = field.spatial_gradient(u_component)
            grad_v = field.spatial_gradient(v_component)
            smoothness_loss = field.l2_loss(grad_u) + field.l2_loss(grad_v)

            # G. Energy penalty (limit total kinetic energy)
            vn_vals = v_guess_centered.values.native(['x', 'y', 'vector'])
            energy_loss = 0.5 * jnp.sum(vn_vals ** 2)

            # H. Anti-zero penalty (prevent trivial all-zero solution)
            mean_sq = jnp.mean(vn_vals ** 2)
            anti_zero_loss = 1.0 / (mean_sq + 1e-12)

            # I. Total loss with weights
            total_loss = MSE_WEIGHT * mse_loss + SMOOTH_WEIGHT * smoothness_loss + ENERGY_WEIGHT * energy_loss + ZERO_WEIGHT * anti_zero_loss


            return total_loss

        # ------------------------------------------------------------------
        # C. Initialize guess & run L-BFGS-B optimization
        # ------------------------------------------------------------------
        print('\nInitializing optimization guess...')
        v0_gt0_centered = v0_gt0.at_centers()
        # Start from zero/zero+noise/base/base+noise as initial guess
        zero_values = math.zeros(spatial(x=Nx, y=Ny) & channel(vector='x,y'))
        base_values = v0_gt0.at_centers().values
        
        guess_shape = v0_gt0.at_centers().values.shape
        noise_scale = 0.0005
        noise = math.random_uniform(guess_shape, low=-noise_scale, high=noise_scale)
        
        # init_values = zero_values + noise
        init_values = base_values + noise
        
        v_guess_proxy = CenteredGrid(
            values=init_values,
            extrapolation=Vn_BC,
            bounds=DOMAIN['bounds']
        )

        # Evaluate initial loss
        try:
            init_loss = loss_function(v_guess_proxy)
            print(f'Initial loss: {float(init_loss):.6e}')
        except Exception as e:
            print(f'Initial loss evaluation failed: {e}')

        print(f'\nStarting Optimization (L-BFGS-B, max_iter={args.lbfgs_iters})...')
        t_start = time.time()

        optimizer = Solve(
            'L-BFGS-B',
            x0=v_guess_proxy,
            max_iterations=args.lbfgs_iters,
            suppress=[phi.math.Diverged, phi.math.NotConverged]
        )

        with math.SolveTape(record_trajectories=False) as solves:
            result_centered = minimize(loss_function, optimizer)

        t_end = time.time()
        print(f'Optimization finished in {t_end - t_start:.2f} s')

        # Extract result
        v0_reconstructed = result_centered.at(v0_gt0)
        final_loss = loss_function(result_centered)

        print(f'=== Optimization Result ===')
        print(f'Final Loss: {float(final_loss):.6e}')

        # Compare with GT
        diff_field = v0_reconstructed - v0_gt0
        diff_mag = field.l2_loss(diff_field)
        print(f'Field Reconstruction Error (L2): {float(diff_mag):.6e}')

    else:
        v0_reconstructed = None

    # ==============================================================================
    # Prepare and save GT and reconstruction for visualization
    # ==============================================================================

    # Re-simulate GT and recon to extract per-frame vn/vs components and speeds
    v_gt_u, v_gt_v, v_gt_speed, v_gt_vec, vs_gt_u, vs_gt_v, vs_gt_speed, vs_gt_vec, t_gt = _extract_time_series_for_vn(
        v0_gt0, vs0_gt0, p0_gt0, t0_gt0, L0_gt0, STEPS, DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)

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
            for _ in range(STEPS):
                v_curr, vs_curr, p_curr, t_curr, L_curr = helium.SFHelium_step(
                    v_curr, vs_curr, p_curr, t_curr, L_curr,
                    dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE,
                    Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL
                )
                markers_rec = advect.points(markers_rec, v_curr, dt=DT, integrator=advect.rk4)
                recon_traj_list.append(markers_rec)
            recon_stack = math.stack(recon_traj_list, batch('time'))
            recon_traj = _pointcloud_list_to_numpy(recon_traj_list)
        except Exception:
            recon_traj = None

    # Normalize temperature arrays and fill placeholders if needed
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
    io.save_npz('data/simulation/cf_v0_recon.npz', **save_dict)

    print('Task 02 complete. Data saved to data/simulation/cf_v0_recon.npz')


if __name__ == '__main__':
    main()


