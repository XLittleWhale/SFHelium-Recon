from phi.jax.flow import *
import jax
import jax.numpy as jnp
from ..physics import helium, normal

@jit_compile
def run_forward_sim_simulated(v_init_centered, vs_init, p_init, t_init, L_init, initial_markers, steps, dt, DOMAIN, SF_step=helium.SFHelium_step):
    """Run forward simulation for simulated-data inversion using JAX scan.

    Returns trajectory tensor (time, markers, vector)
    """
    # Convert Centered guess to staggered aligned with v_init_centered.at(v_init) usage externally
    vn0 = v_init_centered

    # Prepare native state for scan (store native arrays)
    state = (
        (vn0.vector['x'].values.native(['x', 'y']), vn0.vector['y'].values.native(['x', 'y'])),
        (vs_init.vector['x'].values.native(['x', 'y']), vs_init.vector['y'].values.native(['x', 'y'])),
        p_init.values.native(['x', 'y']),
        t_init.values.native(['x', 'y']),
        L_init.values.native(['x', 'y']),
        initial_markers.native(['markers', 'vector'])
    )

    def step_fn(carry, _):
        (v_tuple, vs_tuple, p_nat, t_nat, L_nat, markers_nat) = carry
        # Reconstruct grids
        grid_res = spatial(x=DOMAIN['x'], y=DOMAIN['y'])
        bounds = DOMAIN['bounds']
        vn = StaggeredGrid(values=math.stack([math.tensor(v_tuple[0], spatial('x,y')), math.tensor(v_tuple[1], spatial('x,y'))], dual(vector='x,y')), bounds=bounds)
        vs = StaggeredGrid(values=math.stack([math.tensor(vs_tuple[0], spatial('x,y')), math.tensor(vs_tuple[1], spatial('x,y'))], dual(vector='x,y')), bounds=bounds)
        p = CenteredGrid(values=math.tensor(p_nat, spatial('x,y')), bounds=bounds)
        t = CenteredGrid(values=math.tensor(t_nat, spatial('x,y')), bounds=bounds)
        L = CenteredGrid(values=math.tensor(L_nat, spatial('x,y')), bounds=bounds)
        markers = math.tensor(markers_nat, instance('markers') & channel(vector='x,y'))

        vn_next, vs_next, p_next, t_next, L_next = SF_step(vn, vs, p, t, L, dt, DOMAIN)
        advected = advect.points(geom.Point(markers), vn_next, dt=dt, integrator=advect.rk4).center
        new_carry = (
            (vn_next.vector['x'].values.native(['x', 'y']), vn_next.vector['y'].values.native(['x', 'y'])),
            (vs_next.vector['x'].values.native(['x', 'y']), vs_next.vector['y'].values.native(['x', 'y'])),
            p_next.values.native(['x', 'y']),
            t_next.values.native(['x', 'y']),
            L_next.values.native(['x', 'y']),
            advected.native(['markers', 'vector'])
        )
        return new_carry, advected

    final_state, traj = jax.lax.scan(step_fn, state, jnp.arange(steps))
    return traj


@jit_compile
def run_forward_sim_experiment(v_guess_centered, vs_init, p_init, t_init, L_init, initial_markers_pointcloud, reset_mask, gt_positions, steps, dt, DOMAIN, SF_step=helium.SFHelium_step):
    """Run forward sim for experimental data with resets (late-entry particles).

    Returns simulated trajectory stack (time, markers, vector)
    """
    # Prepare native initial state
    vn0 = v_guess_centered
    state = (
        (vn0.vector['x'].values.native(['x', 'y']), vn0.vector['y'].values.native(['x', 'y'])),
        (vs_init.vector['x'].values.native(['x', 'y']), vs_init.vector['y'].values.native(['x', 'y'])),
        p_init.values.native(['x', 'y']),
        t_init.values.native(['x', 'y']),
        L_init.values.native(['x', 'y']),
        initial_markers_pointcloud.geometry.center.native(['markers', 'vector'])
    )

    # Prepare scan inputs: reset_mask (time, markers), gt_positions (time, markers, vector)
    reset_native = reset_mask.native(['time', 'markers'])
    gt_native = gt_positions.native(['time', 'markers', 'vector'])

    def step_fn(carry, inputs):
        (v_tuple, vs_tuple, p_nat, t_nat, L_nat, markers_nat) = carry
        reset_t, gt_t = inputs
        bounds = DOMAIN['bounds']

        vn = StaggeredGrid(values=math.stack([math.tensor(v_tuple[0], spatial('x,y')), math.tensor(v_tuple[1], spatial('x,y'))], dual(vector='x,y')), bounds=bounds)
        vs = StaggeredGrid(values=math.stack([math.tensor(vs_tuple[0], spatial('x,y')), math.tensor(vs_tuple[1], spatial('x,y'))], dual(vector='x,y')), bounds=bounds)
        p = CenteredGrid(values=math.tensor(p_nat, spatial('x,y')), bounds=bounds)
        t = CenteredGrid(values=math.tensor(t_nat, spatial('x,y')), bounds=bounds)
        L = CenteredGrid(values=math.tensor(L_nat, spatial('x,y')), bounds=bounds)

        markers = math.tensor(markers_nat, instance('markers') & channel(vector='x,y'))

        vn_next, vs_next, p_next, t_next, L_next = SF_step(vn, vs, p, t, L, dt, DOMAIN)

        advected = advect.points(geom.Point(markers), vn_next, dt=dt, integrator=advect.rk4).center

        # Reset late-entry particles: if reset mask > 0.5, replace advected with GT pos
        gt_t_tensor = math.tensor(gt_t, instance('markers') & channel(vector='x,y'))
        reset_t_tensor = math.tensor(reset_t, instance('markers'))
        next_markers = math.where(reset_t_tensor > 0.5, gt_t_tensor, advected)

        new_carry = (
            (vn_next.vector['x'].values.native(['x', 'y']), vn_next.vector['y'].values.native(['x', 'y'])),
            (vs_next.vector['x'].values.native(['x', 'y']), vs_next.vector['y'].values.native(['x', 'y'])),
            p_next.values.native(['x', 'y']),
            t_next.values.native(['x', 'y']),
            L_next.values.native(['x', 'y']),
            next_markers.native(['markers', 'vector'])
        )

        return new_carry, next_markers

    scan_inputs = (reset_native, gt_native)
    final_state, traj = jax.lax.scan(step_fn, state, scan_inputs)
    return traj
