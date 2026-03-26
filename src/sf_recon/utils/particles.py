from phi.jax.flow import *
import numpy as np
import pandas as pd


def _gaussian_kernel1d(radius, sigma):
    """Build a normalized 1D Gaussian kernel."""
    if radius <= 0 or sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel_sum = np.sum(kernel)
    return kernel / kernel_sum if kernel_sum > 0 else np.array([1.0], dtype=np.float64)


def _smooth_series(values, valid_mask, radius=2, sigma=1.0):
    """Smooth a sparse vector-valued time series with mask awareness."""
    values = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=np.float64)
    if values.ndim != 2 or values.shape[-1] != 2 or radius <= 0 or sigma <= 0:
        return values.copy()
    kernel = _gaussian_kernel1d(radius, sigma)
    smoothed = values.copy()
    for dim in range(values.shape[-1]):
        numerator = np.convolve(values[:, dim] * valid_mask, kernel, mode='same')
        denominator = np.convolve(valid_mask, kernel, mode='same')
        smoothed[:, dim] = np.where(denominator > 1e-12, numerator / denominator, values[:, dim])
    return smoothed


def _build_dense_track_from_grouped(
    grouped,
    all_sim_times,
    shift_to_zero=False,
    min_obs=2,
    smooth=False,
    smooth_radius=2,
    smooth_sigma=1.0,
):
    """Convert grouped particle observations to dense sampled position/velocity arrays."""
    grouped = grouped.sort_values('time_phys').copy()
    observed_times = grouped['time_phys'].to_numpy(dtype=np.float64)
    observed_x = grouped['pos_x_phys'].to_numpy(dtype=np.float64)
    observed_y = grouped['pos_y_phys'].to_numpy(dtype=np.float64)

    if observed_times.size < min_obs:
        return None

    if shift_to_zero:
        observed_times = observed_times - observed_times[0]

    valid_window = (observed_times >= 0.0) & (observed_times <= float(all_sim_times[-1]) + 1e-12)
    observed_times = observed_times[valid_window]
    observed_x = observed_x[valid_window]
    observed_y = observed_y[valid_window]
    if observed_times.size < min_obs:
        return None

    observed_indices = np.unique(np.clip(np.searchsorted(all_sim_times, observed_times, side='left'), 0, len(all_sim_times) - 1))
    if observed_indices.size < min_obs:
        return None

    sampled_times = all_sim_times[observed_indices]
    sampled_x = np.interp(sampled_times, observed_times, observed_x)
    sampled_y = np.interp(sampled_times, observed_times, observed_y)

    traj = np.zeros((len(all_sim_times), 2), dtype=np.float64)
    traj[:, 0] = np.interp(all_sim_times, sampled_times, sampled_x)
    traj[:, 1] = np.interp(all_sim_times, sampled_times, sampled_y)

    loss_mask = np.zeros(len(all_sim_times), dtype=np.float64)
    reset_mask = np.zeros(len(all_sim_times), dtype=np.float64)
    loss_mask[observed_indices] = 1.0
    reset_mask[observed_indices[0]] = 1.0
    if observed_indices.size > 1:
        gaps = observed_indices[1:] - observed_indices[:-1]
        reset_mask[observed_indices[1:][gaps > 1]] = 1.0

    traj[observed_indices, 0] = sampled_x
    traj[observed_indices, 1] = sampled_y

    vel = np.zeros_like(traj)
    vel_mask = loss_mask.copy()
    if 'u_phys' in grouped.columns and 'v_phys' in grouped.columns:
        observed_u = grouped['u_phys'].to_numpy(dtype=np.float64)[valid_window]
        observed_v = grouped['v_phys'].to_numpy(dtype=np.float64)[valid_window]
        sampled_u = np.interp(sampled_times, observed_times, observed_u)
        sampled_v = np.interp(sampled_times, observed_times, observed_v)
        vel[:, 0] = np.interp(all_sim_times, sampled_times, sampled_u)
        vel[:, 1] = np.interp(all_sim_times, sampled_times, sampled_v)
        vel[observed_indices, 0] = sampled_u
        vel[observed_indices, 1] = sampled_v
    else:
        native_dt = max(float(all_sim_times[1] - all_sim_times[0]) if len(all_sim_times) > 1 else 0.0, 1e-12)
        vel[1:] = (traj[1:] - traj[:-1]) / native_dt
        vel[0] = vel[1] if len(vel) > 1 else 0.0

    if smooth:
        traj = _smooth_series(traj, loss_mask, radius=smooth_radius, sigma=smooth_sigma)
        vel = _smooth_series(vel, vel_mask, radius=smooth_radius, sigma=smooth_sigma)
        traj[observed_indices, 0] = sampled_x
        traj[observed_indices, 1] = sampled_y

    return traj, loss_mask, reset_mask, vel, vel_mask


def load_experimental_particle_data(
    file_path,
    num_particles,
    max_steps,
    dt,
    domain_bounds,
    freq=120,
    category='g2',
    shift_to_zero=True,
    smooth=False,
    smooth_radius=2,
    smooth_sigma=1.0,
    min_obs=2,
    position_scale=1.265e-5,
    offset_x=0.0,
    offset_y=None,
):
    """Load experimental particle data with dense sampling on the simulation grid.

    Supports both Excel files (Task04-style) and CSV files (Task05-style).

    Time interpretation:
    - if ``freq`` is provided, ``time`` is treated as frame index and converted by ``time / freq``;
    - if ``freq`` is ``None``, ``time`` is treated as already being in physical seconds.
    """
    file_path_lower = str(file_path).lower()
    if file_path_lower.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    rename_map = {}
    if 'traj_id' in df.columns and 'trajectory_num' not in df.columns:
        rename_map['traj_id'] = 'trajectory_num'
    if 'frame' in df.columns and 'time' not in df.columns:
        rename_map['frame'] = 'time'
    if 'x' in df.columns and 'pos_x' not in df.columns:
        rename_map['x'] = 'pos_x'
    if 'y' in df.columns and 'pos_y' not in df.columns:
        rename_map['y'] = 'pos_y'
    if rename_map:
        df = df.rename(columns=rename_map)

    if category is not None and 'category' in df.columns:
        df = df[df['category'] == category]

    has_velocity = 'u(mm/s)' in df.columns and 'v(mm/s)' in df.columns
    upper_y = float(domain_bounds.upper.vector[1])
    lower_y = float(domain_bounds.lower.vector[1])
    offset_y_value = upper_y / 2.0 - (upper_y - lower_y) / 2.0 if offset_y is None else float(offset_y)

    time_values = df['time'].astype(float)
    if freq is None:
        df['time_phys'] = time_values
    else:
        df['time_phys'] = time_values * (1.0 / float(freq))
    df['pos_x_phys'] = df['pos_x'].astype(float) * float(position_scale) + float(offset_x)
    df['pos_y_phys'] = df['pos_y'].astype(float) * float(position_scale) + offset_y_value
    if has_velocity:
        df['u_phys'] = df['u(mm/s)'].astype(float) * 1e-3
        df['v_phys'] = df['v(mm/s)'].astype(float) * 1e-3

    all_sim_times = np.linspace(0.0, max_steps * dt, max_steps + 1, dtype=np.float64)
    unique_ids = df['trajectory_num'].dropna().unique()

    tensor_list = []
    loss_mask_list = []
    reset_mask_list = []
    vel_list = []
    vel_mask_list = []

    for pid in unique_ids:
        if len(tensor_list) >= num_particles:
            break
        p_data = df[df['trajectory_num'] == pid].sort_values('time_phys').copy()
        if p_data.empty:
            continue
        agg_map = {'time_phys': 'mean', 'pos_x_phys': 'mean', 'pos_y_phys': 'mean'}
        if has_velocity:
            agg_map.update({'u_phys': 'mean', 'v_phys': 'mean'})
        grouped = p_data.groupby('time_phys', as_index=False).agg(agg_map).sort_values('time_phys')
        dense_track = _build_dense_track_from_grouped(
            grouped,
            all_sim_times,
            shift_to_zero=shift_to_zero,
            min_obs=min_obs,
            smooth=smooth,
            smooth_radius=smooth_radius,
            smooth_sigma=smooth_sigma,
        )
        if dense_track is None:
            continue
        traj, loss_mask, reset_mask, vel, vel_mask = dense_track
        tensor_list.append(traj)
        loss_mask_list.append(loss_mask)
        reset_mask_list.append(reset_mask)
        vel_list.append(vel)
        vel_mask_list.append(vel_mask)

    if not tensor_list:
        raise ValueError(f'No valid particle trajectories were loaded from {file_path}.')

    np_traj = np.stack(tensor_list, axis=0).transpose(1, 0, 2)
    np_loss_mask = np.stack(loss_mask_list, axis=0).transpose(1, 0)
    np_reset_mask = np.stack(reset_mask_list, axis=0).transpose(1, 0)
    np_vel = np.stack(vel_list, axis=0).transpose(1, 0, 2)
    np_vel_mask = np.stack(vel_mask_list, axis=0).transpose(1, 0)

    gt_tensor = math.tensor(np_traj, batch('time') & instance('markers') & channel(vector='x,y'))
    loss_mask = math.tensor(np_loss_mask, batch('time') & instance('markers'))
    reset_mask = math.tensor(np_reset_mask, batch('time') & instance('markers'))
    vel_tensor = math.tensor(np_vel, batch('time') & instance('markers') & channel(vector='x,y'))
    vel_mask = math.tensor(np_vel_mask, batch('time') & instance('markers'))
    initial_pos = gt_tensor.time[0]
    return gt_tensor, loss_mask, reset_mask, initial_pos, vel_tensor, vel_mask


def pointcloud_list_to_numpy(pointcloud_list):
    """Convert a list of marker clouds to a stable ``(T, N, 2)`` NumPy array."""
    coords = []
    for pointcloud in pointcloud_list:
        if hasattr(pointcloud, 'geometry'):
            center = pointcloud.geometry.center.numpy(['markers', 'vector'])
        elif hasattr(pointcloud, 'center'):
            center = pointcloud.center.numpy(['markers', 'vector'])
        else:
            center = np.asarray(pointcloud)
        coords.append(np.asarray(center, dtype=float))
    return np.stack(coords, axis=0)


def tensor_time_markers_to_numpy(tensor_obj):
    """Convert a time / marker tensor to a stable ``(T, N, 2)`` NumPy array."""
    if tensor_obj is None:
        return None
    try:
        return np.asarray(tensor_obj.native(['time', 'markers', 'vector']), dtype=float)
    except Exception:
        try:
            return np.asarray(tensor_obj.numpy(['time', 'markers', 'vector']), dtype=float)
        except Exception:
            try:
                return np.asarray(tensor_obj, dtype=float)
            except Exception:
                return None


def tensor_time_marker_mask_to_numpy(tensor_obj):
    """Convert a time / marker mask tensor to a stable ``(T, N)`` NumPy array."""
    if tensor_obj is None:
        return None
    try:
        return np.asarray(tensor_obj.native(['time', 'markers']), dtype=float)
    except Exception:
        try:
            return np.asarray(tensor_obj.numpy(['time', 'markers']), dtype=float)
        except Exception:
            try:
                return np.asarray(tensor_obj, dtype=float)
            except Exception:
                return None


def tensor_time_marker_velocity_to_numpy(tensor_obj):
    """Convert a time / marker / vector velocity tensor to ``(T, N, 2)`` NumPy."""
    if tensor_obj is None:
        return None
    try:
        return np.asarray(tensor_obj.native(['time', 'markers', 'vector']), dtype=float)
    except Exception:
        try:
            return np.asarray(tensor_obj.numpy(['time', 'markers', 'vector']), dtype=float)
        except Exception:
            try:
                return np.asarray(tensor_obj, dtype=float)
            except Exception:
                return None


def marker_window_bounds(markers_np, mask_np=None):
    """Return effective marker bounds as ``(x_min, x_max, y_min, y_max)``."""
    if markers_np is None:
        return None
    points = np.asarray(markers_np, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 2:
        return None
    valid = np.isfinite(points[..., 0]) & np.isfinite(points[..., 1])
    if mask_np is not None:
        try:
            mask_arr = np.asarray(mask_np, dtype=float)
            if mask_arr.shape == points.shape[:2]:
                valid &= mask_arr > 0.5
        except Exception:
            pass
    if not np.any(valid):
        return None
    x_vals = points[..., 0][valid]
    y_vals = points[..., 1][valid]
    return float(np.min(x_vals)), float(np.max(x_vals)), float(np.min(y_vals)), float(np.max(y_vals))

def load_particle_trajectories_csv(
    file_path,
    num_particles,
    max_steps,
    dt,
    domain_bounds,
    freq=10,
    scale=2.45e-5,
    offset_x=0.0,
    offset_y=0.0,
    min_obs=2,
):
    """Load experimental particle trajectories from a CSV file.

    Unlike :func:`load_particle_trajectories` (Excel / Excel-specific columns
    with a ``category`` filter), this function works with a generic CSV that
    has columns ``trajectory_num, time, pos_x, pos_y`` and applies no
    category filter.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    num_particles : int
        Maximum number of particle trajectories to keep.
    max_steps : int
        Number of simulation time-steps (trajectory length = max_steps + 1).
    dt : float
        Physical time-step size.
    domain_bounds : Box
        Observation window bounds used only for reference (not for filtering).
    freq : int
        Acquisition frame rate (frames per second).
    scale : float
        Pixel-to-physical conversion factor (metres / pixel).
    offset_x : float
        Constant offset added to the physical x-coordinate.
    offset_y : float
        Constant offset added to the physical y-coordinate.
    min_obs : int
        Minimum number of distinct observed frames to keep a trajectory.

    Returns
    -------
    gt_tensor : phiflow Tensor
        ``(time, markers, vector)``
    loss_mask : phiflow Tensor
        ``(time, markers)`` — 1 at real observation frames, 0 elsewhere.
    reset_mask : phiflow Tensor
        ``(time, markers)`` — 1 at first appearance / re-appearance after gap.
    initial_pos : phiflow Tensor
        Marker positions at ``time[0]``.
    """
    df = pd.read_csv(file_path)

    # Convert pixel coordinates to physical coordinates
    df['time_phys'] = df['time'] * (1.0 / freq)
    df['pos_x_phys'] = df['pos_x'] * scale + offset_x
    df['pos_y_phys'] = df['pos_y'] * scale + offset_y

    all_sim_times = np.linspace(0, max_steps * dt, max_steps + 1)
    unique_ids = df['trajectory_num'].unique()

    tensor_list = []
    loss_mask_list = []
    reset_mask_list = []

    def _time_to_index(time_value):
        return int(np.argmin(np.abs(all_sim_times - time_value)))

    kept_particles = 0
    for pid in unique_ids:
        if kept_particles >= num_particles:
            break
        p_data = df[df['trajectory_num'] == pid].sort_values('time_phys').copy()
        if p_data.empty:
            continue

        # Map each observation to the nearest simulation frame index.
        p_data['sim_idx'] = p_data['time_phys'].map(_time_to_index)
        p_grouped = (
            p_data.groupby('sim_idx', as_index=False)
            .agg({'time_phys': 'mean', 'pos_x_phys': 'mean', 'pos_y_phys': 'mean'})
            .sort_values('sim_idx')
        )

        observed_indices = p_grouped['sim_idx'].to_numpy(dtype=int)
        observed_x = p_grouped['pos_x_phys'].to_numpy(dtype=float)
        observed_y = p_grouped['pos_y_phys'].to_numpy(dtype=float)

        if observed_indices.size < min_obs:
            continue

        traj = np.zeros((max_steps + 1, 2), dtype=np.float64)
        l_mask = np.zeros(max_steps + 1, dtype=np.float64)
        r_mask = np.zeros_like(l_mask)

        x_interp = np.interp(all_sim_times, all_sim_times[observed_indices], observed_x)
        y_interp = np.interp(all_sim_times, all_sim_times[observed_indices], observed_y)
        traj[:, 0] = x_interp
        traj[:, 1] = y_interp
        l_mask[observed_indices] = 1.0

        previous_idx = None
        for obs_idx in observed_indices:
            if obs_idx <= 0:
                previous_idx = obs_idx
                continue
            if previous_idx is None or obs_idx - previous_idx > 1:
                r_mask[obs_idx] = 1.0
            previous_idx = obs_idx

        # Pin exact measured positions at observation frames.
        traj[observed_indices, 0] = observed_x
        traj[observed_indices, 1] = observed_y

        tensor_list.append(traj)
        loss_mask_list.append(l_mask)
        reset_mask_list.append(r_mask)
        kept_particles += 1

    if not tensor_list:
        raise ValueError(f'No valid particle trajectories loaded from {file_path}.')

    np_traj = np.stack(tensor_list, axis=0).transpose(1, 0, 2)
    gt_tensor = math.tensor(np_traj, batch('time') & instance('markers') & channel(vector='x,y'))
    np_l_mask = np.stack(loss_mask_list, axis=0).transpose(1, 0)
    np_r_mask = np.stack(reset_mask_list, axis=0).transpose(1, 0)
    loss_mask = math.tensor(np_l_mask, batch('time') & instance('markers'))
    reset_mask = math.tensor(np_r_mask, batch('time') & instance('markers'))
    initial_pos = gt_tensor.time[0]
    return gt_tensor, loss_mask, reset_mask, initial_pos


def load_particle_trajectories(file_path, num_particles, max_steps, dt, domain_bounds, freq=120):
    """Load experimental particle trajectories from Excel.

    This function intentionally stays outside JIT because it uses pandas and
    NumPy / pandas processing on host arrays.

    Sparse tracks are treated as sparse supervision:
    - ``loss_mask`` is 1 only at frames with a real observation,
    - ``reset_mask`` is 1 when a marker first appears or reappears after a gap,
    - missing spans are *not* interpreted as continuous observed trajectories.
    """
    df = pd.read_excel(file_path)
    df = df[df['category'] == 'g2']
    # df = df[df['category'].isin(['g2', 'g3', 'g4'])]
    upper_y = float(domain_bounds.upper.vector[1])
    lower_y = float(domain_bounds.lower.vector[1])
    offset_y = upper_y / 2.0 - (upper_y - lower_y) / 2.0
    df['time_phys'] = df['time'] * (1/freq)
    df['pos_x_phys'] = df['pos_x'] * 1.265e-5
    df['pos_y_phys'] = df['pos_y'] * 1.265e-5 + offset_y
    all_sim_times = np.linspace(0, max_steps * dt, max_steps + 1)
    unique_ids = df['trajectory_num'].unique()
    tensor_list = []
    loss_mask_list = []
    reset_mask_list = []

    def _time_to_index(time_value):
        return int(np.argmin(np.abs(all_sim_times - time_value)))

    kept_particles = 0
    for pid in unique_ids:
        if kept_particles >= num_particles:
            break
        p_data = df[df['trajectory_num'] == pid].sort_values('time_phys').copy()
        if p_data.empty:
            continue

        # Collapse repeated observations that map to the same simulation frame.
        p_data['sim_idx'] = p_data['time_phys'].map(_time_to_index)
        p_grouped = (
            p_data.groupby('sim_idx', as_index=False)
            .agg({'time_phys': 'mean', 'pos_x_phys': 'mean', 'pos_y_phys': 'mean'})
            .sort_values('sim_idx')
        )

        observed_indices = p_grouped['sim_idx'].to_numpy(dtype=int)
        observed_x = p_grouped['pos_x_phys'].to_numpy(dtype=float)
        observed_y = p_grouped['pos_y_phys'].to_numpy(dtype=float)

        # Keep only markers with at least two real observation frames.
        if observed_indices.size < 2:
            continue

        # Keep a dense trajectory array for downstream tensor shape consistency,
        # but only mark real observation frames as supervised.
        traj = np.zeros((max_steps + 1, 2), dtype=np.float64)
        l_mask = np.zeros(max_steps + 1, dtype=np.float64)
        r_mask = np.zeros_like(l_mask)

        x_interp = np.interp(all_sim_times, all_sim_times[observed_indices], observed_x)
        y_interp = np.interp(all_sim_times, all_sim_times[observed_indices], observed_y)
        traj[:, 0] = x_interp
        traj[:, 1] = y_interp
        l_mask[observed_indices] = 1.0

        previous_idx = None
        for obs_idx in observed_indices:
            if obs_idx <= 0:
                previous_idx = obs_idx
                continue
            if previous_idx is None or obs_idx - previous_idx > 1:
                r_mask[obs_idx] = 1.0
            previous_idx = obs_idx

        # Ensure observation frames use the exact measured positions.
        traj[observed_indices, 0] = observed_x
        traj[observed_indices, 1] = observed_y

        tensor_list.append(traj)
        loss_mask_list.append(l_mask)
        reset_mask_list.append(r_mask)
        kept_particles += 1

    if not tensor_list:
        raise ValueError('No valid particle trajectories were loaded from the Excel file.')

    np_traj = np.stack(tensor_list, axis=0).transpose(1, 0, 2)
    gt_tensor = math.tensor(np_traj, batch('time') & instance('markers') & channel(vector='x,y'))
    np_l_mask = np.stack(loss_mask_list, axis=0).transpose(1, 0)
    np_r_mask = np.stack(reset_mask_list, axis=0).transpose(1, 0)
    loss_mask = math.tensor(np_l_mask, batch('time') & instance('markers'))
    reset_mask = math.tensor(np_r_mask, batch('time') & instance('markers'))
    initial_pos = gt_tensor.time[0]
    return gt_tensor, loss_mask, reset_mask, initial_pos


def load_particle_trajectories_with_velocity(file_path, num_particles, max_steps, dt, domain_bounds, freq=120):
    """Load experimental particle trajectories and optional per-frame velocities from Excel.

    Expected velocity columns are ``u(mm/s)`` and ``v(mm/s)``. When present,
    they are converted to SI units via a factor of ``1e-3`` and aligned to the
    nearest simulation frame, just like the marker positions.

    Returns
    -------
    gt_tensor : phiflow Tensor
        ``(time, markers, vector)`` marker positions.
    loss_mask : phiflow Tensor
        ``(time, markers)`` position observation mask.
    reset_mask : phiflow Tensor
        ``(time, markers)`` re-entry mask.
    initial_pos : phiflow Tensor
        Marker positions at ``time[0]``.
    vel_tensor : phiflow Tensor or None
        ``(time, markers, vector)`` observed particle velocities in m/s.
    vel_mask : phiflow Tensor or None
        ``(time, markers)`` velocity observation mask.
    """
    return load_experimental_particle_data(
        file_path,
        num_particles=num_particles,
        max_steps=max_steps,
        dt=dt,
        domain_bounds=domain_bounds,
        freq=freq,
        category='g2',
        shift_to_zero=True,
        smooth=False,
        min_obs=2,
    )
