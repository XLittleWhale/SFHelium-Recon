from phi.jax.flow import *
import numpy as np
import pandas as pd


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

def load_particle_trajectories(file_path, num_particles, max_steps, dt, domain_bounds, freq=120):
    """Load experimental particle trajectories from Excel.

    This function intentionally stays outside JIT because it uses pandas and
    NumPy interpolation on host arrays.
    """
    df = pd.read_excel(file_path)
    df = df[df['category'] == 'g2']
    upper_y = float(domain_bounds.upper.vector[1])
    lower_y = float(domain_bounds.lower.vector[1])
    offset_y = upper_y / 2.0 - (upper_y - lower_y) / 2.0
    df['time_phys'] = df['time'] * (1/freq)
    df['pos_x_phys'] = df['pos_x'] * 1.265e-5
    df['pos_y_phys'] = df['pos_y'] * 1.265e-5 + offset_y
    all_sim_times = np.linspace(0, max_steps * dt, max_steps + 1)
    unique_ids = df['trajectory_num'].unique()
    selected_ids = unique_ids[:num_particles]
    tensor_list = []
    loss_mask_list = []
    reset_mask_list = []
    for pid in selected_ids:
        p_data = df[df['trajectory_num'] == pid].sort_values('time_phys')
        t_start = p_data['time_phys'].min()
        t_end = p_data['time_phys'].max()
        x_interp = np.interp(all_sim_times, p_data['time_phys'], p_data['pos_x_phys'])
        y_interp = np.interp(all_sim_times, p_data['time_phys'], p_data['pos_y_phys'])
        traj = np.stack([x_interp, y_interp], axis=-1)
        is_active = (all_sim_times >= t_start - 1e-6) & (all_sim_times <= t_end + 1e-6)
        l_mask = is_active.astype(np.float64)
        r_mask = np.zeros_like(l_mask)
        if t_start > dt:
            start_idx = np.argmin(np.abs(all_sim_times - t_start))
            if start_idx < max_steps:
                r_mask[start_idx] = 1.0
        tensor_list.append(traj)
        loss_mask_list.append(l_mask)
        reset_mask_list.append(r_mask)
    np_traj = np.stack(tensor_list, axis=0).transpose(1, 0, 2)
    gt_tensor = math.tensor(np_traj, batch('time') & instance('markers') & channel(vector='x,y'))
    np_l_mask = np.stack(loss_mask_list, axis=0).transpose(1, 0)
    np_r_mask = np.stack(reset_mask_list, axis=0).transpose(1, 0)
    loss_mask = math.tensor(np_l_mask, batch('time') & instance('markers'))
    reset_mask = math.tensor(np_r_mask, batch('time') & instance('markers'))
    initial_pos = gt_tensor.time[0]
    return gt_tensor, loss_mask, reset_mask, initial_pos
