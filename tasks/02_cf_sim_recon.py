"""
Task 02: Superfluid helium counterflow simulation + inversion

This script generates a Ground Truth (GT) using the SFHelium model and
then performs inversion for the normal-fluid initial field `vn`.
"""
from phi.jax.flow import *
from phi.jax.flow import Box, StaggeredGrid, CenteredGrid, ZERO_GRADIENT, instance, advect, math, batch, jit_compile, spatial, channel, minimize, Solve
from sf_recon.physics import helium, boundaries
from sf_recon.utils import io
from sf_recon.utils.load import load_csv_to_grids_cf
from sf_recon.utils.saving import (
    simple_to_numpy as _simple_to_numpy,
    stack_if_possible as _stack_if_possible,
    extract_time_series_for_vn as _extract_time_series_for_vn,
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
    args = parser.parse_args()

    print('Task 02: starting')
    # Domain & simulation parameters (tunable)
    Lx, Ly = 0.0004, 0.0016
    Nx, Ny = 20, 80
    MARKERS = 128
    DT = 1e-6
    STEPS = 5
    PRE_STEPS = 5

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
    DOMAIN = dict(x=Nx, y=Ny, bounds=Box(x=Lx, y=Ly))
    OBSTACLE = None

    # initialize GT fields
    if args.init_csv is not None:
        try:
            un0_np, un1_np, us0_np, us1_np, p_np, t_np, L_np, count_total = load_csv_to_grids_cf(
                args.init_csv, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny
            )
            print(f'Loaded CSV init from {args.init_csv} with {count_total} samples')
            # build grids (note: CenteredGrid expects (Nx,Ny) ordering for tensor)
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

    # Markers Initialization
    markers0 = DOMAIN['bounds'].sample_uniform(instance(markers=MARKERS))

    # Pre-steps to reach steady state
    curr_vn = v0_gt0
    curr_vs = vs0_gt0
    curr_p = p0_gt0
    curr_t = t0_gt0
    curr_L = L0_gt0
    for _ in range(PRE_STEPS):
        curr_vn, curr_vs, curr_p, curr_t, curr_L = helium.SFHelium_step(curr_vn, curr_vs, curr_p, curr_t, curr_L, dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)
    v0_gt, vs0_gt, p0_gt, t0_gt, L0_gt = curr_vn, curr_vs, curr_p, curr_t, curr_L
    print('Pre-steps completed')
    initial_markers = markers0

    # Simulate GT
    gt_traj = [initial_markers]
    current_markers = initial_markers
    cvn, cvs, cp, ct, cL = v0_gt, vs0_gt, p0_gt, t0_gt, L0_gt
    for _ in range(STEPS):
        cvn, cvs, cp, ct, cL = helium.SFHelium_step(cvn, cvs, cp, ct, cL, dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)
        current_markers = advect.points(current_markers, cvn, dt=DT, integrator=advect.rk4)
        gt_traj.append(current_markers)

    gt_stack = math.stack(gt_traj, batch('time'))
    print('GT Generation Done.')

    # Define loss: invert only the normal-fluid initial centered field
    @jit_compile
    def loss_function(v_guess_centered):
        vn = v_guess_centered.at(v0_gt)
        markers = gt_traj[0]
        traj = [markers]
        v_curr = vn
        vs_curr = vs0_gt
        p_curr = p0_gt
        t_curr = t0_gt
        L_curr = L0_gt
        for _ in range(STEPS):
            v_curr, vs_curr, p_curr, t_curr, L_curr = helium.SFHelium_step(v_curr, vs_curr, p_curr, t_curr, L_curr, dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)
            markers = advect.points(markers, v_curr, dt=DT, integrator=advect.rk4)
            traj.append(markers)
        sim = math.stack(traj, batch('time'))
        diff = sim - gt_stack
        return math.sum(diff ** 2, dim=diff.shape)

    # Initialize guess (zero-centered field)
    batch_shape = v0_gt.shape.batch
    init_shape = batch_shape & spatial(x=Nx, y=Ny) & channel(vector='x,y')
    init_values = math.zeros(init_shape)
    v_guess = CenteredGrid(values=init_values, extrapolation=Vn_BC, bounds=Box(x=Lx, y=Ly))

    # Run optimization
    print('Starting optimization')
    with math.SolveTape(record_trajectories=False) as solves:
        result_centered = minimize(loss_function, Solve('L-BFGS-B', x0=v_guess, max_iterations=2000))
    print('Optimization completed')

    v0_reconstructed = result_centered.at(v0_gt)

    # ==============================================================================
    # Prepare and save GT and reconstruction for visualization
    # ==============================================================================

    # Re-simulate GT and recon to extract per-frame vn/vs components and speeds
    v_gt_u, v_gt_v, v_gt_speed, v_gt_vec, vs_gt_u, vs_gt_v, vs_gt_speed, vs_gt_vec, t_gt = _extract_time_series_for_vn(
        v0_gt, vs0_gt, p0_gt, t0_gt, L0_gt, STEPS, DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)
    v_recon_u, v_recon_v, v_recon_speed, v_recon_vec, vs_recon_u, vs_recon_v, vs_recon_speed, vs_recon_vec, t_recon = _extract_time_series_for_vn(
        v0_reconstructed, vs0_gt, p0_gt, t0_gt, L0_gt, STEPS, DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)

    # Reconstruct particle trajectories advected by reconstructed normal fluid
    recon_traj = None
    try:
        markers_rec = initial_markers
        recon_traj_list = [markers_rec]
        v_curr = v0_reconstructed
        vs_curr = vs0_gt
        p_curr = p0_gt
        t_curr = t0_gt
        L_curr = L0_gt
        for _ in range(STEPS):
            v_curr, vs_curr, p_curr, t_curr, L_curr = helium.SFHelium_step(v_curr, vs_curr, p_curr, t_curr, L_curr, dt=DT, DOMAIN=DOMAIN, OBSTACLE=OBSTACLE, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL)
            markers_rec = advect.points(markers_rec, v_curr, dt=DT, integrator=advect.rk4)
            recon_traj_list.append(markers_rec)
        recon_stack = math.stack(recon_traj_list, batch('time'))
        recon_traj = _simple_to_numpy(recon_stack)
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

    gt_np = _simple_to_numpy(gt_stack)
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


