import numpy as np
import pyvista as pv
from phi.jax.flow import *

def resample_axisymmetric_rotated_ij(vtk_mesh, domain_res, domain_bounds, translation=(0.0, 0.0)):
    """Resample an axisymmetric VTK mesh into PhiFlow grid ordering with optional translation.

    Returns (sampled_mesh, (nx, ny), sign_x_flat)
    """
    # domain_res may be spatial(...) or tuple
    if hasattr(domain_res, 'sizes'):
        nx = domain_res.get_size('x')
        ny = domain_res.get_size('y')
    else:
        nx, ny = int(domain_res[0]), int(domain_res[1])

    min_vec = domain_bounds.lower
    max_vec = domain_bounds.upper

    dx = (float(max_vec.vector[0]) - float(min_vec.vector[0])) / nx
    dy = (float(max_vec.vector[1]) - float(min_vec.vector[1])) / ny

    target_x_1d = np.linspace(float(min_vec.vector[0]) + dx/2, float(max_vec.vector[0]) - dx/2, int(nx))
    target_y_1d = np.linspace(float(min_vec.vector[1]) + dy/2, float(max_vec.vector[1]) - dy/2, int(ny))

    xv, yv = np.meshgrid(target_x_1d, target_y_1d, indexing='ij')
    shift_x, shift_y = translation
    xv_query = xv + shift_x
    yv_query = yv + shift_y

    src_x = yv_query.flatten()
    src_y = np.abs(xv_query.flatten())
    src_z = np.zeros_like(src_x)

    probe_points = np.column_stack((src_x, src_y, src_z))
    probe_cloud = pv.PolyData(probe_points)
    sampled_data = probe_cloud.sample(vtk_mesh, tolerance=1e-5)

    sign_x_flat = np.sign(xv_query.flatten())
    sign_x_flat[sign_x_flat == 0] = 1

    return sampled_data, (nx, ny), sign_x_flat


def load_and_align_fields(vtk_path, domain_res, domain_bounds, boundary_conditions, translation=(-0.0, 0.0)):
    """Load VTK and align fields into PhiFlow Grids.

    Returns: v_staggered, vs_staggered, t_grid, L_grid, p_grid
    """
    source_mesh = pv.read(vtk_path)
    sampled_mesh, (nx, ny), sign_flat = resample_axisymmetric_rotated_ij(source_mesh, domain_res, domain_bounds, translation)
    grid_res = spatial(x=nx, y=ny)

    def get_tensor_data(name, is_vector=False):
        if name not in sampled_mesh.point_data:
            if is_vector:
                return np.zeros((nx, ny, 2))
            else:
                return np.zeros((nx, ny))
        raw_data = sampled_mesh.point_data[name]
        if is_vector:
            src_ux = raw_data[:, 0]
            src_uy = raw_data[:, 1]
            target_vx_flat = src_uy * sign_flat
            target_vy_flat = src_ux
            vx_grid = target_vx_flat.reshape(nx, ny)
            vy_grid = target_vy_flat.reshape(nx, ny)
            return np.stack([vx_grid, vy_grid], axis=-1)
        else:
            return raw_data.reshape(nx, ny)

    t_np = get_tensor_data('T', is_vector=False)
    l_np = get_tensor_data('L', is_vector=False)
    p_np = get_tensor_data('p', is_vector=False)
    un_np = get_tensor_data('Un', is_vector=True)
    us_np = get_tensor_data('Us', is_vector=True)

    t_grid = CenteredGrid(tensor(t_np, spatial('x,y')), boundary_conditions['t'], bounds=domain_bounds)
    l_grid = CenteredGrid(tensor(l_np, spatial('x,y')), boundary_conditions['l'], bounds=domain_bounds)
    p_grid = CenteredGrid(tensor(p_np, spatial('x,y')), boundary_conditions['p'], bounds=domain_bounds)

    un_grid_centered = CenteredGrid(tensor(un_np, spatial('x,y'), channel(vector='x,y')), boundary_conditions['v'], bounds=domain_bounds)
    target_v_template = StaggeredGrid(0, boundary_conditions['v'], bounds=domain_bounds, resolution=grid_res)
    v_staggered = un_grid_centered.at(target_v_template)

    us_grid_centered = CenteredGrid(tensor(us_np, spatial('x,y'), channel(vector='x,y')), boundary_conditions['vs'], bounds=domain_bounds)
    target_vs_template = StaggeredGrid(0, boundary_conditions['vs'], bounds=domain_bounds, resolution=grid_res)
    vs_staggered = us_grid_centered.at(target_vs_template)

    return v_staggered, vs_staggered, t_grid, l_grid, p_grid
