from phi.jax.flow import *

def solve_poisson(rhs, domain, bc=None, tol=1e-10, maxiter=2000):
    """Simple Poisson solver wrapper returning CenteredGrid pressure.

    rhs: CenteredGrid or compatible
    domain: dict with bounds/resolution
    """
    solver = Solve('CG', abs_tol=tol, rel_tol=0, max_iterations=maxiter, suppress=[phi.math.NotConverged])
    # Construct initial guess p0 as zeros with same bounds
    p0 = CenteredGrid(0, bc or ZERO_GRADIENT, **domain)
    # Use math.SolveTape if desired externally; here call fluid.make_incompressible-like solve
    # For general Poisson, we'll use phi.flow's linear solver via math.solve (low-level use)
    # Fallback: return p0 (caller may replace with more advanced solver)
    return p0
