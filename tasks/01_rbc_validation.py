"""
Task 01: Rayleigh-Bénard Convection validation (RBC)

This script uses src.sf_recon.physics.normal to generate GT and run inversion.
"""
from phi.jax.flow import *
from sf_recon.physics import normal
from sf_recon.utils import viz, io
from sf_recon.utils.saving import (
    simple_to_numpy as _simple_to_numpy,
    stack_if_possible as _stack_if_possible,
    extract_time_series_for_rbc as _extract_time_series,
)
import numpy as np
import time

def main():
    # Parameters
    Lx, Ly = 0.1, 0.1
    Nx, Ny = 64, 64
    MARKERS = 128
    DT = 0.1
    STEPS = 10
    PRE_STEPS = 10

    # Generate GT
    print('Task 01: starting')
    v0_gt, t0_gt, initial_markers, gt_stack = normal.generate_rbc_gt(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, MARKERS=MARKERS, DT=DT, STEPS=STEPS, PRE_STEPS=PRE_STEPS)
    print('GT Generation Done.')


    # Loss (JIT)
    @jit_compile
    def loss_function(v_guess_centered):
        v = v_guess_centered.at(v0_gt)
        markers = initial_markers
        traj = [markers]
        for _ in range(STEPS):
            v, _ = normal.boussinesq_step(v, t0_gt, dt=DT)
            markers = advect.points(markers, v, dt=DT, integrator=advect.rk4)
            traj.append(markers)
        sim = math.stack(traj, batch('time'))
        diff = sim - gt_stack
        return math.sum(diff**2, dim=diff.shape)

    # Initialize guess
    batch_shape = v0_gt.shape.batch
    init_shape = batch_shape & spatial(x=Nx, y=Ny) & channel(vector='x,y')
    init_values = math.zeros(init_shape)
    v_guess = CenteredGrid(values=init_values, extrapolation={'x':0,'y':0}, bounds=Box(x=Lx,y=Ly))

    # Run optimization
    print('Starting optimization')
    with math.SolveTape(record_trajectories=False) as solves:
        result = minimize(loss_function, Solve('L-BFGS-B', x0=v_guess, max_iterations=50))
    v_recon = result.at(v0_gt)
    print('Optimization completed.')

    ##########################################################
    # Prepare and save GT and reconstruction for visualization
    ##########################################################

    # Convert particle stacks
    gt_np = _simple_to_numpy(gt_stack)

    # Extract GT time series
    v_gt_u, v_gt_v, v_gt_speed, v_gt_vec = _extract_time_series(v0_gt, t0_gt, STEPS, DT)

    # Extract recon time series by re-simulating from reconstructed initial field
    v_recon_u, v_recon_v, v_recon_speed, v_recon_vec = _extract_time_series(v_recon, t0_gt, STEPS, DT)


    # Simulate particles advected by reconstructed field (best-effort)
    # recon_traj: convert particle stack to numpy
    recon_traj = None
    try:
        markers_rec = initial_markers
        recon_traj_list = [markers_rec]
        for _ in range(STEPS):
            markers_rec = advect.points(markers_rec, v_recon, dt=DT, integrator=advect.rk4)
            recon_traj_list.append(markers_rec)
        recon_stack = math.stack(recon_traj_list, batch('time'))
        recon_traj = _simple_to_numpy(recon_stack)
    except Exception:
        recon_traj = None

    io.save_npz('data/simulation/rbc_v0_recon.npz', success=1, gt=gt_np, recon=recon_traj,
                v_gt_speed=v_gt_speed, v_recon_speed=v_recon_speed, v_gt_vec=v_gt_vec, v_recon_vec=v_recon_vec,
                v_gt_u=v_gt_u, v_gt_v=v_gt_v, v_recon_u=v_recon_u, v_recon_v=v_recon_v)

    print('Task 01 complete. Data Saved to data/simulation/rbc_v0_recon.npz')

if __name__ == '__main__':
    main()
