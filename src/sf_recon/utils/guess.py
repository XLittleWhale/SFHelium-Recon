"""Helpers for building warm-start velocity-field guesses.

These utilities are intentionally lightweight:
- They only use information that is typically available before inversion.
- They return native JAX arrays or PhiFlow-centered values that can be reused
  by multiple tasks.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import stax

from phi.jax.flow import CenteredGrid, tensor, spatial, channel


def build_center_coordinate_features(nx: int, ny: int, lx: float, ly: float) -> np.ndarray:
    """Return normalized center coordinates with shape ``(nx * ny, 2)``."""
    x = np.linspace(0.5 * lx / nx, lx - 0.5 * lx / nx, nx, dtype=np.float64)
    y = np.linspace(0.5 * ly / ny, ly - 0.5 * ly / ny, ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    coords = np.stack([xx / max(lx, 1e-12), yy / max(ly, 1e-12)], axis=-1)
    return coords.reshape(nx * ny, 2)


def native_to_centered_grid(native_values, nx: int, ny: int, bounds, extrapolation) -> CenteredGrid:
    """Convert native ``(nx * ny, 2)`` values to a PhiFlow ``CenteredGrid``."""
    values_reshaped = jnp.asarray(native_values).reshape(nx, ny, 2)
    values_tensor = tensor(values_reshaped, spatial('x,y'), channel(vector='x,y'))
    return CenteredGrid(values=values_tensor, extrapolation=extrapolation, bounds=bounds)


def build_mlp(hidden_layers: list[int]):
    """Build a small tanh MLP that maps coordinates to 2D velocity values."""
    layers = []
    for width in hidden_layers:
        layers.extend([stax.Dense(width), stax.Tanh])
    layers.append(stax.Dense(2))
    return stax.serial(*layers)


def add_param_noise(params, key, scale: float):
    """Perturb network parameters with Gaussian noise."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    noisy_leaves = [p + scale * jax.random.normal(k, p.shape, dtype=p.dtype) for p, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, noisy_leaves)


def build_uniform_inflow_prior(num_points: int, vn_in: float, dtype=jnp.float64):
    """Return a uniform +y inflow prior with shape ``(num_points, 2)``."""
    return jnp.tile(jnp.array([[0.0, vn_in]], dtype=dtype), (num_points, 1))


def build_obstacle_aware_inflow_prior(nx: int, ny: int, lx: float, ly: float, vn_in: float, center_xy, radius: float):
    """Build a weak obstacle-aware inflow prior.

    This is not a precise flow solution. It only encodes two simple assumptions:
    - the dominant flow is along +y,
    - velocity tends to drop near the obstacle.
    """
    coords = build_center_coordinate_features(nx, ny, lx, ly)
    x = coords[:, 0] * lx
    y = coords[:, 1] * ly
    cx, cy = center_xy
    dx = x - cx
    dy = y - cy
    r2 = dx ** 2 + dy ** 2
    sigma2 = max(radius ** 2, 1e-12)
    wake = np.exp(-r2 / (2.0 * sigma2))
    u = np.zeros_like(x)
    v = np.full_like(y, vn_in) * (1.0 - 0.65 * wake)
    return jnp.asarray(np.stack([u, v], axis=-1))


def build_counterflow_inflow_prior(num_points: int, vn_in: float, dtype=jnp.float64):
    """Return a weak counterflow prior aligned with the inlet direction."""
    return build_uniform_inflow_prior(num_points, vn_in, dtype=dtype)
