from phi.jax.flow import *
import numpy as np
import pandas as pd

@jit_compile
def load_particle_trajectories(file_path, num_particles, max_steps, dt, domain_bounds, freq=120):
    df = pd.read_excel(file_path)
    df = df[df['category'] == 'g2']
    offset_y = domain_bounds.upper.vector[1]/2 - (domain_bounds.upper.vector[1]-domain_bounds.lower.vector[1])/2
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
