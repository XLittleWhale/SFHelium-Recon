from phi.jax.flow import *

def run_lbfgs(loss_fn, x0, max_iter=100):
    """Run L-BFGS-B via PhiFlow minimize wrapper and return result object."""
    optimizer = Solve('L-BFGS-B', x0=x0, max_iterations=max_iter, suppress=[phi.math.Diverged, phi.math.NotConverged])
    with math.SolveTape(record_trajectories=False) as solves:
        result = minimize(loss_fn, optimizer)
    return result
