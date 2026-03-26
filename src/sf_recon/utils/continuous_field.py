from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator
from numpy.linalg import LinAlgError


class ContinuousVelocityFieldFitter:
    """Fit a continuous 2D velocity field from sparse particle trajectories.

    The model fits two scalar RBF interpolants,
    $u = u(x, y, t)$ and $v = v(x, y, t)$,
    over spatio-temporal coordinates ``(x, y, t)``.

    Design goals:
    - work directly with sparse particle positions / velocities,
    - provide dense time queries even when physical data arrive every 0.1s,
    - stay lightweight and repository-friendly.
    """

    def __init__(self, smoothing: float = 1e-8, neighbors: int | None = None, kernel: str = 'thin_plate_spline', epsilon: float | None = None):
        self.smoothing = float(smoothing)
        self.neighbors = neighbors
        self.kernel = kernel
        self.epsilon = epsilon
        self._rbf_u: RBFInterpolator | None = None
        self._rbf_v: RBFInterpolator | None = None
        self._is_fit = False
        self._fit_points: np.ndarray | None = None
        self._fit_u: np.ndarray | None = None
        self._fit_v: np.ndarray | None = None
        self._fallback_mode: str | None = None

    def _build_interpolator(self, fit_points: np.ndarray, values: np.ndarray) -> RBFInterpolator:
        """Build a robust RBF interpolator with graceful fallbacks."""
        n_points = fit_points.shape[0]
        candidate_specs = [
            dict(kernel=self.kernel, degree=None, smoothing=self.smoothing, neighbors=self.neighbors),
            dict(kernel=self.kernel, degree=0, smoothing=max(self.smoothing, 1e-6), neighbors=None),
            dict(kernel='linear', degree=0, smoothing=max(self.smoothing, 1e-6), neighbors=None),
            dict(kernel='multiquadric', degree=0, smoothing=max(self.smoothing, 1e-4), neighbors=None, epsilon=(self.epsilon or 1.0)),
        ]

        last_error = None
        for spec in candidate_specs:
            kwargs = dict(
                y=fit_points,
                d=values,
                smoothing=spec['smoothing'],
                neighbors=(spec['neighbors'] if spec['neighbors'] is None else min(int(spec['neighbors']), n_points)),
                kernel=spec['kernel'],
            )
            if spec.get('degree', None) is not None:
                kwargs['degree'] = spec['degree']
            if spec.get('epsilon', None) is not None:
                kwargs['epsilon'] = spec['epsilon']
            try:
                return RBFInterpolator(**kwargs)
            except (LinAlgError, ValueError) as exc:
                last_error = exc

        raise last_error if last_error is not None else RuntimeError('Failed to build continuous-field interpolator.')

    def _sample_nearest(self, positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Fallback nearest-sample velocity lookup when RBF fitting is ill-conditioned."""
        if self._fit_points is None or self._fit_u is None or self._fit_v is None:
            raise RuntimeError('ContinuousVelocityFieldFitter has no fitted samples for fallback evaluation.')
        query = np.concatenate([positions, times[:, None]], axis=-1)
        deltas = self._fit_points[None, :, :] - query[:, None, :]
        distances = np.sum(deltas ** 2, axis=-1)
        nearest = np.argmin(distances, axis=1)
        return np.stack([self._fit_u[nearest], self._fit_v[nearest]], axis=-1)

    def fit(self, positions: np.ndarray, velocities: np.ndarray, mask: np.ndarray | None = None, times: np.ndarray | None = None) -> 'ContinuousVelocityFieldFitter':
        positions = np.asarray(positions, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        if positions.shape != velocities.shape or positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError('positions and velocities must both have shape (T, N, 2).')

        t_steps, n_markers, _ = positions.shape
        if times is None:
            times = np.arange(t_steps, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if times.ndim != 1 or times.shape[0] != t_steps:
            raise ValueError('times must have shape (T,).')

        if mask is None:
            valid = np.isfinite(positions[..., 0]) & np.isfinite(positions[..., 1]) & np.isfinite(velocities[..., 0]) & np.isfinite(velocities[..., 1])
        else:
            mask = np.asarray(mask, dtype=np.float64)
            if mask.shape != positions.shape[:2]:
                raise ValueError('mask must have shape (T, N).')
            valid = (mask > 0.5)
            valid &= np.isfinite(positions[..., 0]) & np.isfinite(positions[..., 1]) & np.isfinite(velocities[..., 0]) & np.isfinite(velocities[..., 1])

        tt = np.broadcast_to(times[:, None], (t_steps, n_markers))
        points = np.stack([positions[..., 0], positions[..., 1], tt], axis=-1)
        fit_points = points[valid]
        fit_u = velocities[..., 0][valid]
        fit_v = velocities[..., 1][valid]

        if fit_points.shape[0] < 4:
            raise ValueError('Need at least 4 valid spatio-temporal samples to fit a continuous field.')

        self._fit_points = fit_points
        self._fit_u = fit_u
        self._fit_v = fit_v
        self._fallback_mode = None
        try:
            self._rbf_u = self._build_interpolator(fit_points, fit_u)
            self._rbf_v = self._build_interpolator(fit_points, fit_v)
        except (LinAlgError, ValueError):
            self._rbf_u = None
            self._rbf_v = None
            self._fallback_mode = 'nearest'
        self._is_fit = True
        return self

    def sample_velocity(self, positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        if not self._is_fit or self._rbf_u is None or self._rbf_v is None:
            raise RuntimeError('ContinuousVelocityFieldFitter must be fit before sampling.')

        positions = np.asarray(positions, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)

        if positions.ndim == 2 and positions.shape[-1] == 2:
            if times.ndim == 0:
                times = np.full((positions.shape[0],), float(times), dtype=np.float64)
            if self._fallback_mode == 'nearest' or self._rbf_u is None or self._rbf_v is None:
                return self._sample_nearest(positions, times)
            query = np.concatenate([positions, times[:, None]], axis=-1)
            try:
                u = self._rbf_u(query)
                v = self._rbf_v(query)
            except (LinAlgError, ValueError):
                self._fallback_mode = 'nearest'
                return self._sample_nearest(positions, times)
            return np.stack([u, v], axis=-1)

        if positions.ndim == 3 and positions.shape[-1] == 2:
            if times.ndim == 1:
                times = np.broadcast_to(times[:, None], positions.shape[:2])
            flat_positions = positions.reshape(-1, 2)
            flat_times = times.reshape(-1)
            if self._fallback_mode == 'nearest' or self._rbf_u is None or self._rbf_v is None:
                return self._sample_nearest(flat_positions, flat_times).reshape(*positions.shape[:2], 2)
            query = np.concatenate([flat_positions, flat_times[:, None]], axis=-1)
            try:
                u = self._rbf_u(query).reshape(positions.shape[:2])
                v = self._rbf_v(query).reshape(positions.shape[:2])
            except (LinAlgError, ValueError):
                self._fallback_mode = 'nearest'
                return self._sample_nearest(flat_positions, flat_times).reshape(*positions.shape[:2], 2)
            return np.stack([u, v], axis=-1)

        raise ValueError('positions must have shape (N, 2) or (T, N, 2).')

    def rollout_trajectories(self, initial_positions: np.ndarray, query_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fit:
            raise RuntimeError('ContinuousVelocityFieldFitter must be fit before rollout.')
        initial_positions = np.asarray(initial_positions, dtype=np.float64)
        query_times = np.asarray(query_times, dtype=np.float64)
        if initial_positions.ndim != 2 or initial_positions.shape[-1] != 2:
            raise ValueError('initial_positions must have shape (N, 2).')
        if query_times.ndim != 1 or query_times.size < 1:
            raise ValueError('query_times must have shape (T,) with T >= 1.')

        traj = np.zeros((query_times.size, initial_positions.shape[0], 2), dtype=np.float64)
        vel = np.zeros_like(traj)
        traj[0] = initial_positions
        vel[0] = self.sample_velocity(initial_positions, np.full((initial_positions.shape[0],), query_times[0], dtype=np.float64))

        for i in range(1, query_times.size):
            dt = float(query_times[i] - query_times[i - 1])
            vel_prev = self.sample_velocity(traj[i - 1], np.full((initial_positions.shape[0],), query_times[i - 1], dtype=np.float64))
            traj[i] = traj[i - 1] + dt * vel_prev
            vel[i] = self.sample_velocity(traj[i], np.full((initial_positions.shape[0],), query_times[i], dtype=np.float64))
        return traj, vel
