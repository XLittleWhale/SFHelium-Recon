from phi.jax.flow import *
import numpy as np
import matplotlib.pyplot as plt

# Boussinesq solver (Using Water as reference)
@jit_compile
def boussinesq_step(v, t, dt, VISCOSITY=8.6e-7, THERMAL_DIFFUSIVITY=1.5e-7, BUOYANCY_COEFFS=[2.5e-3,9e-5,-1e-5], PRESSURE_SOLVER=None):
    v = advect.semi_lagrangian(v, v, dt)
    t = advect.semi_lagrangian(t, v, dt)
    v = diffuse.explicit(v, VISCOSITY, dt)
    t = diffuse.explicit(t, THERMAL_DIFFUSIVITY, dt)
    t_at_v = t.at(v)
    scalar_buoyancy = 0
    for power, coeff in enumerate(BUOYANCY_COEFFS):
        scalar_buoyancy += coeff * (t_at_v ** power)
    buoyancy_force = t_at_v * scalar_buoyancy * (0, 1)
    v = v + buoyancy_force * dt
    if PRESSURE_SOLVER is None:
        v, _ = fluid.make_incompressible(v, [], Solve('CG', 1e-10, max_iterations=50))
    else:
        v, _ = fluid.make_incompressible(v, [], PRESSURE_SOLVER)
    return v, t

# Generate GT using parameters
def generate_rbc_gt(Lx=0.1, Ly=0.1, Nx=64, Ny=64, MARKERS=128, DT=0.1, STEPS=10, PRE_STEPS=100,
                    HEAT=10.0, COLD=-10.0, V_BC={'x':0,'y':0}):
    DOMAIN = dict(x=Nx, y=Ny, bounds=Box(x=Lx, y=Ly))

    v0 = StaggeredGrid(0, V_BC, **DOMAIN)
    v0, _ = fluid.make_incompressible(v0, [], Solve('CG',1e-10,max_iterations=20))

    def linear_temp_gradient(x):
        return (COLD - HEAT) * (x.vector['y'] / Ly) + HEAT
    
    t0 = CenteredGrid(linear_temp_gradient, {'x': ZERO_GRADIENT, 'y-': HEAT, 'y+': COLD}, **DOMAIN)
    t0 = t0 + CenteredGrid(Noise(scale=0.1), **DOMAIN)

    markers = DOMAIN['bounds'].sample_uniform(instance(markers=MARKERS))

    for _ in range(PRE_STEPS):
        v0, t0 = boussinesq_step(v0, t0, dt=DT)
        markers = advect.points(markers, v0, dt=DT, integrator=advect.rk4)
    gt_traj = [markers]
    curr_v, curr_t, curr_markers = v0, t0, markers

    for _ in range(STEPS):
        curr_v, curr_t = boussinesq_step(curr_v, curr_t, dt=DT)
        curr_markers = advect.points(curr_markers, curr_v, dt=DT, integrator=advect.rk4)
        gt_traj.append(curr_markers)
    gt_stack = math.stack(gt_traj, batch('time'))
    
    return v0, t0, markers, gt_stack
