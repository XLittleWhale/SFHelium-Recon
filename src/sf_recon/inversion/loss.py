from phi.jax.flow import *

def mse_loss(simulated_traj, gt_traj, mask=None):
    """Mean squared error between simulated and ground-truth trajectories.

    simulated_traj, gt_traj: Tensors with shape (time, markers, vector)
    mask: optional (time, markers) to weight/ignore missing tracks
    """
    diff = simulated_traj - gt_traj
    if mask is None:
        return math.sum(diff ** 2, dim=diff.shape)
    else:
        m = mask[Ellipsis, None]
        return math.sum((diff ** 2) * m)

def smoothness_loss(field_grid):
    """Spatial gradient L2 loss for a CenteredGrid or Grid component."""
    grads = field.spatial_gradient(field_grid)
    return field.l2_loss(grads)

def energy_loss(v_centered):
    """Simple kinetic-energy penalty for a centered velocity field."""
    vals = v_centered.values
    return 0.5 * math.sum(vals ** 2)
