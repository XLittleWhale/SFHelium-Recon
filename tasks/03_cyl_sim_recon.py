"""
Task 03: Cylinder flow simulation/inversion (uses SFHelium model)

"""
from phi.jax.flow import *
from sf_recon.physics import helium, boundaries
from sf_recon.utils import viz, io
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

def main():
    # Choose initialization with external CSV if provided
    # Choose whether to skip inversion
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-csv', type=str, default=None, help='Optional OpenFOAM-export CSV to initialize fields')
    parser.add_argument('--skip-inversion', action='store_true', help='Skip the inversion step')
    parser.add_argument('--nn-iters', type=int, default=500, help='Training iterations for U-Net optimization')
    parser.add_argument('--nn-lr', type=float, default=5e-4, help='Learning rate for U-Net optimization')
    parser.add_argument('--train-steps', type=int, default=None, help='Subsample time steps for faster training')
    parser.add_argument('--train-markers', type=int, default=None, help='Subsample markers for faster training')
    args = parser.parse_args()

    math.use('jax')
    math.set_global_precision(64)

    print('Task 03: starting')
    # Domain and simulation parameters (tunable)
    Lx, Ly = 0.02, 0.2
    Nx, Ny = 40, 200
    RAD = 0.004
    MARKERS = 512
    DT = 1e-5
    STEPS = 10
    LOSS_SCALE = 1e6    # Changed from 1e12 to 1e9 so that initial physics loss is O(1) instead of O(1000)
    VEL_SCALE = 0.2  # Smooth clamp for predicted velocity
    REG_WEIGHT = 1  # Small weight decay to keep gradients active
    SUP_WEIGHT = 1e3 # Small supervised weight for anchoring only 
    PHYS_WEIGHT = 1e8  # Full physics backprop now that gradients are stabilized
    U_REF = 0.02

    # Physical parameters
    HEAT_SOURCE_INTENSITY = 1.94
    HEAT_FLUX = 6000
    DENSITY = 145.5244
    ENTROPY = 813.4
    DENSITY_N = 68.22
    DENSITY_S = 77.31
    Vn_IN = HEAT_FLUX / (DENSITY * ENTROPY * HEAT_SOURCE_INTENSITY)
    Vs_IN = -(DENSITY_N * Vn_IN) / DENSITY_S
    Vn_BC, Vs_BC, J_BC, t_BC_THERMAL, p_BC = boundaries.get_cylinder_bcs(Vn_IN=Vn_IN, Vs_IN=Vs_IN, PRESSURE_0=3130)
    DOMAIN = dict(x=Nx, y=Ny, bounds=Box(x=Lx, y=Ly))
    SPHERE = Sphere(x=Lx/2, y=Ly/2, radius=RAD)
    OBSTACLE = Obstacle(SPHERE)

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

    # Simulate GT
    curr_vn = v0_gt0
    curr_vs = vs0_gt0
    curr_p = p0_gt0
    curr_t = t0_gt0
    curr_L = L0_gt0
    gt_traj = [markers0]
    for _ in range(STEPS):
        curr_vn, curr_vs, curr_p, curr_t, curr_L = helium.SFHelium_step(curr_vn, curr_vs, curr_p, curr_t, curr_L, dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)
        markers0 = advect.points(markers0, curr_vn, dt=DT, integrator=advect.rk4)
        try:
            if OBSTACLE is not None:
                markers0 = helium.constrain_markers_push(markers0, OBSTACLE)
        except Exception:
            pass
        gt_traj.append(markers0)

    gt_stack = math.stack(gt_traj, batch('time'))
    gt_disp = []
    for idx in range(len(gt_traj) - 1):
        delta = gt_traj[idx + 1] - gt_traj[idx]
        delta = math.where(math.is_finite(delta), delta, 0.0)
        gt_disp.append(delta / DT)
    gt_disp_stack = math.stack(gt_disp, batch('time_disp')) if gt_disp else None
    print('GT Generation Done.')

    def _subsample_markers(markers_tensor):
        if args.train_markers is None or args.train_markers >= MARKERS:
            return markers_tensor
        stride = max(1, MARKERS // args.train_markers)
        try:
            return markers_tensor.markers[::stride]
        except Exception:
            return markers_tensor

    train_steps = STEPS if args.train_steps is None else max(1, min(STEPS, args.train_steps))
    train_gt_traj = [_subsample_markers(step) for step in gt_traj[:train_steps + 1]]
    train_gt_stack = math.stack(train_gt_traj, batch('time'))

    # Inversion: run optimization unless user requested skip
    v0_reconstructed = None
    def _sanitize_staggered(v_field):
        try:
            vals = v_field.values
            vals = math.where(math.is_finite(vals), vals, 0.0)
            return StaggeredGrid(values=vals, extrapolation=v_field.extrapolation, bounds=v_field.bounds)
        except Exception:
            return v_field
    def _sanitize_centered(c_field):
        try:
            vals = c_field.values
            vals = math.where(math.is_finite(vals), vals, 0.0)
            return CenteredGrid(values=vals, extrapolation=c_field.extrapolation, bounds=c_field.bounds)
        except Exception:
            return c_field
    if not getattr(args, 'skip_inversion', False):
        def _loss_function_raw(v_guess_centered):
            vn = v_guess_centered.at(v0_gt0)
            markers = train_gt_traj[0]
            traj = [markers]
            v_curr = vn
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
                markers = helium.constrain_markers_push(markers, OBSTACLE)

            for _ in range(train_steps):
                v_curr, vs_curr, p_curr, t_curr, L_curr = helium.SFHelium_step(
                    v_curr, vs_curr, p_curr, t_curr, L_curr,
                    dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE,
                    Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL
                )
                markers = advect.points(markers, v_curr, dt=DT, integrator=advect.rk4)
                if OBSTACLE is not None:
                    markers = helium.constrain_markers_push(markers, OBSTACLE)
                traj.append(markers)

            sim = math.stack(traj, batch('time'))
            diff = sim - train_gt_stack
            diff = math.where(math.is_finite(diff), diff, 0.0)
            # Average over time/markers and rescale to keep loss near 1
            loss = math.mean(diff ** 2, dim=diff.shape) * LOSS_SCALE
            return math.where(math.is_finite(loss), loss, math.tensor(1e6))

        loss_function = jit_compile(_loss_function_raw)

        guess_shape = v0_gt0.at_centers().values.shape
        noise_scale = 0.005
        noise = math.random_uniform(guess_shape, low=-noise_scale, high=noise_scale)
        base_guess = v0_gt0.at_centers().values
        # Initialize from GT + noise so it's easy to tune
        v_guess = CenteredGrid(values=base_guess + noise, extrapolation=Vn_BC, bounds=Box(x=Lx, y=Ly))

        print('Starting U-Net training')

        # U-Net training (2D) following PhiML Networks examples
        from phiml import nn as phiml_nn
        net = phiml_nn.u_net(in_channels=2, out_channels=2, in_spatial=2)

        def _ensure_net_parameters(network):
            if getattr(network, 'parameters', None) is not None:
                return
            init_fn = getattr(network, 'initialize', None)
            if callable(init_fn):
                try:
                    init_fn()
                except Exception:
                    pass
            if getattr(network, 'parameters', None) is None:
                try:
                    from phiml.backend.jax import stax_nets as _stax_nets
                    rnd_key = _stax_nets.JAX.rnd_key
                    _stax_nets.JAX.rnd_key, init_key = _stax_nets.random.split(rnd_key)
                    _, params64 = network._initialize(init_key, input_shape=network._input_shape)
                    network.parameters = params64
                except Exception:
                    pass

        _ensure_net_parameters(net)
        optimizer = phiml_nn.adam(net, learning_rate=args.nn_lr)
        train_x = v_guess.values

        try:
            init_pred = math.native_call(net, train_x)
            init_pred_centered = CenteredGrid(values=init_pred, extrapolation=Vn_BC, bounds=Box(x=Lx, y=Ly))
            init_loss = _loss_function_raw(init_pred_centered)
            print(f'Initial loss (net): {float(init_loss):.10g}')
        except Exception:
            pass

        def unet_loss(x_tensor, dyn_sup_weight):
            # Prediction: velocity field -> supervised match to initial guess
            raw_pred = math.native_call(net, x_tensor)
            raw_pred = math.where(math.is_finite(raw_pred), raw_pred, 0.0)
            # Bound velocities without triggering JAX tracer conversion
            # pred = math.clip(pred, -VEL_SCALE, VEL_SCALE)
            pred = U_REF * raw_pred
            pred_centered = CenteredGrid(values=pred, extrapolation=Vn_BC, bounds=Box(x=Lx, y=Ly))
            physics = _loss_function_raw(pred_centered)
            physics = math.where(math.is_finite(physics), physics, math.tensor(0.0))
            reg = math.mean(raw_pred ** 2, dim=raw_pred.shape)
            sup = math.mean((pred - x_tensor) ** 2, dim=pred.shape)
            
            # Using clean physics loss. Scale is handled globally by LOSS_SCALE now.
            return dyn_sup_weight * sup + REG_WEIGHT * reg + PHYS_WEIGHT * physics

        unet_loss = jit_compile(unet_loss)
        
        current_sup_weight = SUP_WEIGHT
        
        for it in range(args.nn_iters):
            if it > 0 and it % 100 == 0:
                current_sup_weight *= 0.5
                print(f'->Step {it}: Reducing SUP_WEIGHT to {current_sup_weight}')
            
            dyn_sup_weight = math.tensor(current_sup_weight)
            
            loss_value = phiml_nn.update_weights(net, optimizer, unet_loss, train_x, dyn_sup_weight)
            if not math.all(math.is_finite(loss_value)).all:
                print(f'U-Net iter {it + 1}/{args.nn_iters}: loss became non-finite, stopping early.')
                break
            if (it + 1) % 20 == 0:
                try:
                    raw_pred_dbg = math.native_call(net, train_x)
                    # pred_dbg = math.clip(pred_dbg, -VEL_SCALE, VEL_SCALE)
                    pred_dbg = raw_pred_dbg * U_REF
                    physics_dbg = _loss_function_raw(CenteredGrid(values=pred_dbg, extrapolation=Vn_BC, bounds=Box(x=Lx, y=Ly)))
                    sup_dbg = SUP_WEIGHT * math.mean((pred_dbg - train_x) ** 2, dim=pred_dbg.shape)
                    print(
                        f'U-Net iter {it + 1}/{args.nn_iters}: '
                        f'loss={float(loss_value):.10g}, '
                        f'sup={float(sup_dbg):.10g}, '
                        f'physics={float(physics_dbg):.10g}'
                    )
                except Exception:
                    print(f'U-Net iter {it + 1}/{args.nn_iters}: loss={float(loss_value):.10g}')

        pred_centered = _sanitize_centered(CenteredGrid(values=math.native_call(net, train_x) * U_REF, extrapolation=Vn_BC, bounds=Box(x=Lx, y=Ly)))
        v0_reconstructed = _sanitize_staggered(pred_centered.at(v0_gt0))
        try:
            final_loss = _loss_function_raw(pred_centered)
            print(f'Final loss: {float(final_loss):.10g}')
        except Exception:
            pass
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
            markers_rec = gt_traj[0]
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
                markers_rec = helium.constrain_markers_push(markers_rec, OBSTACLE)
            
            for _ in range(STEPS):
                v_curr, vs_curr, p_curr, t_curr, L_curr = helium.SFHelium_step(
                    v_curr, vs_curr, p_curr, t_curr, L_curr,
                    dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE,
                    Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL
                )
                markers_rec = advect.points(markers_rec, v_curr, dt=DT, integrator=advect.rk4)
                if OBSTACLE is not None:
                    markers_rec = helium.constrain_markers_push(markers_rec, OBSTACLE)
                recon_traj_list.append(markers_rec)
            recon_stack = math.stack(recon_traj_list, batch('time'))
            recon_traj = _simple_to_numpy(recon_stack)
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

    gt_np = _simple_to_numpy(gt_stack)
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
