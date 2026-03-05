from phi.jax.flow import *

def joint_pressure_projection(v, obstacle=None, solver=None):
    """Wrapper around PhiFlow's make_incompressible to centralize options.

    Returns (v_projected, pressure_field)
    """
    if solver is None:
        solver = Solve('CG', abs_tol=1e-10, rel_tol=0, max_iterations=500, suppress=[phi.math.NotConverged])
    v_proj, p = fluid.make_incompressible(v, obstacle or (), solver)
    return v_proj, p
