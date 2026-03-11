"""
Task 04: Experimental counterflow reconstruction.

This script follows the D4-style workflow for real particle trajectories:
- load an experimental VTK field snapshot as a physics-consistent initial state,
- load experimental particle trajectories from Excel,
- reconstruct the initial normal-fluid field with L-BFGS-B,
- save particle trajectories and field time series for analysis.
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
	sys.path.insert(0, SRC_DIR)

from phi.jax.flow import *
from phi.jax.flow import (
	Box, StaggeredGrid, CenteredGrid, ZERO_GRADIENT,
	advect, math, jit_compile, spatial, channel, dual, geom, PointCloud,
	minimize, instance
)

from sf_recon.physics import helium, boundaries
from sf_recon.utils import io, particles, vtk as vtk_utils
from sf_recon.utils.guess import (
	add_param_noise,
	build_center_coordinate_features,
	build_counterflow_inflow_prior,
	build_mlp,
	build_uniform_inflow_prior,
	native_to_centered_grid,
)
from sf_recon.utils.saving import (
	simple_to_numpy as _simple_to_numpy,
	extract_time_series_for_vn as _extract_time_series_for_vn,
	ensure_HW as _ensure_HW,
)


def _configure_geometry(args):
	"""Build the fixed geometry, time-step settings, and inversion weights."""
	Lx, Ly = 0.016, 0.320
	Wx, Wy = 0.016, 0.008
	Nx, Ny = 16, 320
	WNx, WNy = 16, 8
	return {
		'Lx': Lx,
		'Ly': Ly,
		'Wx': Wx,
		'Wy': Wy,
		'Nx': Nx,
		'Ny': Ny,
		'WNx': WNx,
		'WNy': WNy,
		'MARKERS': args.num_particles,
		'DT': args.dt,
		'STEPS': args.steps,
		'PRE_STEPS': args.pre_steps,
		'MSE_WEIGHT': 1.0,
		'VEL_WEIGHT': 1.0,
		'SMOOTH_WEIGHT': 5e-5,
		'ENERGY_WEIGHT': 1,
		'ZERO_WEIGHT': 0.0,
		'DOMAIN': dict(x=Nx, y=Ny, bounds=Box(x=Lx, y=Ly)),
		'WINDOMAIN': dict(x=WNx, y=WNy, bounds=Box(x=(0, Wx), y=(Ly / 2 - Wy / 2, Ly / 2 + Wy / 2))),
	}


def _prepare_save_array(x, nx, ny):
	"""Convert a PhiFlow object to a NumPy array with consistent grid shape."""
	try:
		arr = _simple_to_numpy(x)
	except Exception:
		arr = None
	return _ensure_HW(arr, nx, ny)


def _build_save_dict(config, gt_np, recon_traj, loss_mask_np, reset_mask_np, field_series):
	"""Assemble the `.npz` payload saved by Task 04."""
	marker_bounds = particles.marker_window_bounds(gt_np, loss_mask_np)
	if marker_bounds is None:
		marker_x_min, marker_x_max = 0.0, config['Wx']
		marker_y_min = config['Ly'] / 2 - config['Wy'] / 2
		marker_y_max = config['Ly'] / 2 + config['Wy'] / 2
	else:
		marker_x_min, marker_x_max, marker_y_min, marker_y_max = marker_bounds

	v_gt_u, v_gt_v, v_gt_speed, v_gt_vec, vs_gt_u, vs_gt_v, vs_gt_speed, vs_gt_vec, t_gt = field_series['gt']
	v_recon_u, v_recon_v, v_recon_speed, v_recon_vec, vs_recon_u, vs_recon_v, vs_recon_speed, vs_recon_vec, t_recon = field_series['recon']

	return dict(
		success=1,
		Lx=config['Lx'],
		Ly=config['Ly'],
		Wx=config['Wx'],
		Wy=config['Wy'],
		window_x_min=0.0,
		window_x_max=config['Wx'],
		window_y_min=config['Ly'] / 2 - config['Wy'] / 2,
		window_y_max=config['Ly'] / 2 + config['Wy'] / 2,
		marker_x_min=marker_x_min,
		marker_x_max=marker_x_max,
		marker_y_min=marker_y_min,
		marker_y_max=marker_y_max,
		pre_steps=config['PRE_STEPS'],
		gt=gt_np,
		recon=recon_traj,
		loss_mask=loss_mask_np,
		reset_mask=reset_mask_np,
		v_gt_u=_prepare_save_array(v_gt_u, config['Nx'], config['Ny']),
		v_gt_v=_prepare_save_array(v_gt_v, config['Nx'], config['Ny']),
		v_gt_speed=_prepare_save_array(v_gt_speed, config['Nx'], config['Ny']),
		v_gt_vec=_prepare_save_array(v_gt_vec, config['Nx'], config['Ny']),
		v_recon_u=_prepare_save_array(v_recon_u, config['Nx'], config['Ny']),
		v_recon_v=_prepare_save_array(v_recon_v, config['Nx'], config['Ny']),
		v_recon_speed=_prepare_save_array(v_recon_speed, config['Nx'], config['Ny']),
		v_recon_vec=_prepare_save_array(v_recon_vec, config['Nx'], config['Ny']),
		vs_gt_u=_prepare_save_array(vs_gt_u, config['Nx'], config['Ny']),
		vs_gt_v=_prepare_save_array(vs_gt_v, config['Nx'], config['Ny']),
		vs_gt_speed=_prepare_save_array(vs_gt_speed, config['Nx'], config['Ny']),
		vs_gt_vec=_prepare_save_array(vs_gt_vec, config['Nx'], config['Ny']),
		vs_recon_u=_prepare_save_array(vs_recon_u, config['Nx'], config['Ny']),
		vs_recon_v=_prepare_save_array(vs_recon_v, config['Nx'], config['Ny']),
		vs_recon_speed=_prepare_save_array(vs_recon_speed, config['Nx'], config['Ny']),
		vs_recon_vec=_prepare_save_array(vs_recon_vec, config['Nx'], config['Ny']),
		t_gt=_prepare_save_array(t_gt, config['Nx'], config['Ny']),
		t_recon=_prepare_save_array(t_recon, config['Nx'], config['Ny']),
	)


def main():
	"""Run the experimental counterflow reconstruction workflow."""
	parser = argparse.ArgumentParser()
	parser.add_argument('--vtk-path', type=str, default='data/openfoam/241mWcm2_6.55s.vtk', help='VTK snapshot used to initialize the physics state')
	parser.add_argument('--excel-path', type=str, default='data/experiment/1.85K/10.10classified_k4_1.85 K_241mWcm2_120fps_Trajectory_v3.xlsx', help='Experimental trajectory Excel file')
	parser.add_argument('--skip-inversion', action='store_true', help='Skip the inversion step')
	parser.add_argument('--lbfgs-iters', type=int, default=100, help='Max L-BFGS-B iterations')
	parser.add_argument('--outer-iters', type=int, default=2, help='Warm-start search rounds')
	parser.add_argument('--nn-width', type=int, default=64, help='Hidden width of the warm-start MLP')
	parser.add_argument('--nn-depth', type=int, default=3, help='Number of hidden layers of the warm-start MLP')
	parser.add_argument('--warmstart-candidates', type=int, default=4, help='Noisy neural warm-start candidates per round')
	parser.add_argument('--warmstart-noise', type=float, default=0.005, help='Noise scale for neural warm-start candidates')
	parser.add_argument('--num-particles', type=int, default=64, help='Number of particle trajectories to load')
	parser.add_argument('--steps', type=int, default=10, help='Number of simulation steps used for inversion')
	parser.add_argument('--dt', type=float, default=1e-2, help='Time step size')
	parser.add_argument('--freq', type=int, default=120, help='Acquisition frequency of the experiment')
	parser.add_argument('--pre-steps', type=int, default=0, help='Number of pre-simulation steps before inversion, matching the D4 workflow')
	args = parser.parse_args()

	math.use('jax')
	math.set_global_precision(64)

	print('Task 04: starting')

	# Geometry and weights intentionally mirror the validated D4 setup.
	config = _configure_geometry(args)
	Lx, Ly = config['Lx'], config['Ly']
	Wx, Wy = config['Wx'], config['Wy']
	Nx, Ny = config['Nx'], config['Ny']
	MARKERS = config['MARKERS']
	DT = config['DT']
	STEPS = config['STEPS']
	PRE_STEPS = config['PRE_STEPS']
	MSE_WEIGHT = config['MSE_WEIGHT']
	VEL_WEIGHT = config['VEL_WEIGHT']
	SMOOTH_WEIGHT = config['SMOOTH_WEIGHT']
	ENERGY_WEIGHT = config['ENERGY_WEIGHT']
	ZERO_WEIGHT = config['ZERO_WEIGHT']
	DOMAIN = config['DOMAIN']
	WINDOMAIN = config['WINDOMAIN']

	HEAT_SOURCE_INTENSITY = 1.85
	PRESSURE_0 = 1948.0
	HEAT_FLUX = 2410.0
	DENSITY = 145.4070
	ENTROPY = 627.0
	DENSITY_N = 52.86
	DENSITY_S = 92.54
	Vn_IN = HEAT_FLUX / (DENSITY * ENTROPY * HEAT_SOURCE_INTENSITY)
	Vs_IN = -(DENSITY_N * Vn_IN) / DENSITY_S
	Vn_BC, Vs_BC, J_BC, t_BC_THERMAL, p_BC = boundaries.get_sf_bcs(Vn_IN=Vn_IN, Vs_IN=Vs_IN, PRESSURE_0=PRESSURE_0)
	boundary_conditions = {'v': Vn_BC, 'vs': Vs_BC, 't': t_BC_THERMAL, 'l': ZERO_GRADIENT, 'p': p_BC}

	# Build the initial physics state from the experimental VTK snapshot.
	v0_gt0, vs0_gt0, t0_gt0, L0_gt0, p0_gt0 = vtk_utils.load_and_align_fields(
		args.vtk_path,
		(Nx, Ny),
		DOMAIN['bounds'],
		boundary_conditions,
		translation=(-Lx/2, 0.0)
	)
	print(f'Loaded VTK fields from {args.vtk_path}')

	# Optionally pre-step the imported state to match the original D4 workflow.
	print(f'Pre-stepping for {PRE_STEPS} steps...')
	current_v = v0_gt0
	current_vs = vs0_gt0
	current_p = p0_gt0
	current_t = t0_gt0
	current_L = L0_gt0
	for _ in range(PRE_STEPS):
		current_v, current_vs, current_p, current_t, current_L = helium.SFHelium_step(
			current_v, current_vs, current_p, current_t, current_L,
			dt=DT, DOMAIN=DOMAIN, OBSTACLE=None,
			Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL,
		)
	v0_gt0, vs0_gt0, p0_gt0, t0_gt0, L0_gt0 = current_v, current_vs, current_p, current_t, current_L

	# Load experimental marker tracks inside the observation window.
	gt_tensor, loss_mask, reset_mask, initial_pos = particles.load_particle_trajectories(
		args.excel_path,
		num_particles=MARKERS,
		max_steps=STEPS,
		dt=DT,
		domain_bounds=WINDOMAIN['bounds'],
		freq=args.freq,
	)
	print(f'Loaded particle trajectories from {args.excel_path}')

	initial_markers = PointCloud(geom.Point(initial_pos))
	gt_all_native = gt_tensor.native(['time', 'markers', 'vector'])
	loss_mask_native = loss_mask.native(['time', 'markers'])
	reset_mask_native = reset_mask.native(['time', 'markers'])

	# Reconstruct the initial normal-fluid field when inversion is enabled.
	v0_reconstructed = None

	if not getattr(args, 'skip_inversion', False):
		coord_features_np = build_center_coordinate_features(Nx, Ny, Lx, Ly)
		coord_features_native = jnp.asarray(coord_features_np)
		hidden_layers = [args.nn_width] * args.nn_depth
		net_init_fun, net_apply_fun = build_mlp(hidden_layers)
		_, net_params = net_init_fun(jax.random.PRNGKey(0), (-1, 2))
		uniform_native = build_uniform_inflow_prior(Nx * Ny, Vn_IN, dtype=jnp.float64)
		prior_native = build_counterflow_inflow_prior(Nx * Ny, Vn_IN, dtype=jnp.float64)

		def physical_step_logic(carry_state_native, scan_input):
			"""Single scan step with reset support for late-entry particles."""
			(v_x_nat, v_y_nat), (vs_x_nat, vs_y_nat), p_nat, t_nat, L_nat, coords_nat = carry_state_native
			reset_t, gt_t = scan_input

			bounds = DOMAIN['bounds']
			grid_shape = spatial(x=Nx, y=Ny)

			v_x_tensor = math.tensor(v_x_nat, spatial('x,y'))
			v_y_tensor = math.tensor(v_y_nat, spatial('x,y'))
			v_values = math.stack([v_x_tensor, v_y_tensor], dual(vector='x,y'))
			vn = StaggeredGrid(values=v_values, extrapolation=Vn_BC, bounds=bounds, resolution=grid_shape)

			vs_x_tensor = math.tensor(vs_x_nat, spatial('x,y'))
			vs_y_tensor = math.tensor(vs_y_nat, spatial('x,y'))
			vs_values = math.stack([vs_x_tensor, vs_y_tensor], dual(vector='x,y'))
			vs = StaggeredGrid(values=vs_values, extrapolation=Vs_BC, bounds=bounds, resolution=grid_shape)

			p = CenteredGrid(values=math.tensor(p_nat, grid_shape), extrapolation=p_BC, bounds=bounds, resolution=grid_shape)
			t = CenteredGrid(values=math.tensor(t_nat, grid_shape), extrapolation=t_BC_THERMAL, bounds=bounds, resolution=grid_shape)
			L = CenteredGrid(values=math.tensor(L_nat, grid_shape), extrapolation=ZERO_GRADIENT, bounds=bounds, resolution=grid_shape)
			coords = math.tensor(coords_nat, instance('markers') & channel(vector='x,y'))

			vn_next, vs_next, p_next, t_next, L_next = helium.SFHelium_step(
				vn, vs, p, t, L,
				dt=DT, DOMAIN=DOMAIN, OBSTACLE=None,
				Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL,
			)

			markers_obj = PointCloud(geom.Point(coords))
			markers_obj = advect.points(markers_obj, vn_next, dt=DT, integrator=advect.rk4)
			advected_coords = markers_obj.geometry.center.native(['markers', 'vector'])
			next_coords = jnp.where(reset_t[..., None] > 0.5, gt_t, advected_coords)

			new_carry_native = (
				(vn_next.vector['x'].values.native(['x', 'y']), vn_next.vector['y'].values.native(['x', 'y'])),
				(vs_next.vector['x'].values.native(['x', 'y']), vs_next.vector['y'].values.native(['x', 'y'])),
				p_next.values.native(['x', 'y']),
				t_next.values.native(['x', 'y']),
				L_next.values.native(['x', 'y']),
				next_coords,
			)
			return new_carry_native, next_coords

		step_fn_checkpointed = jax.checkpoint(physical_step_logic)

		@jit_compile
		def loss_terms(v_guess_centered):
			"""Compute trajectory and regularization losses for experimental data."""
			v_sim = v_guess_centered.at(v0_gt0)
			v_init_tuple = (
				v_sim.vector['x'].values.native(['x', 'y']),
				v_sim.vector['y'].values.native(['x', 'y']),
			)
			vs_init_tuple = (
				vs0_gt0.vector['x'].values.native(['x', 'y']),
				vs0_gt0.vector['y'].values.native(['x', 'y']),
			)
			init_coords = initial_markers.geometry.center.native(['markers', 'vector'])
			state_init_native = (
				v_init_tuple,
				vs_init_tuple,
				p0_gt0.values.native(['x', 'y']),
				t0_gt0.values.native(['x', 'y']),
				L0_gt0.values.native(['x', 'y']),
				init_coords,
			)

			scan_inputs = (reset_mask_native[1:], gt_all_native[1:])
			_, trajectory_stack_native = jax.lax.scan(step_fn_checkpointed, state_init_native, scan_inputs)

			gt_target_native = gt_all_native[1:]
			loss_mask_expanded = loss_mask_native[1:, :, None]
			diff = (trajectory_stack_native - gt_target_native) * loss_mask_expanded
			mse_loss = jnp.sum(diff ** 2)

			prev_pred_native = jnp.concatenate([init_coords[None, ...], trajectory_stack_native[:-1]], axis=0)
			prev_gt_native = gt_all_native[:-1]
			pred_vel_native = (trajectory_stack_native - prev_pred_native) * loss_mask_expanded
			gt_vel_native = (gt_target_native - prev_gt_native) * loss_mask_expanded
			vel_loss = jnp.sum((pred_vel_native - gt_vel_native) ** 2)

			u_component = v_guess_centered.vector['x']
			v_component = v_guess_centered.vector['y']
			grad_u = field.spatial_gradient(u_component)
			grad_v = field.spatial_gradient(v_component)
			smoothness_loss = field.l2_loss(grad_u) + field.l2_loss(grad_v)

			vn_vals = v_guess_centered.values.native(['x', 'y', 'vector'])
			energy_loss = 0.5 * jnp.sum(vn_vals ** 2)
			mean_sq = jnp.mean(vn_vals ** 2)
			anti_zero_loss = 1.0 / (mean_sq + 1e-12)
			return mse_loss, vel_loss, smoothness_loss, energy_loss, anti_zero_loss

		@jit_compile
		def loss_function(v_guess_centered):
			mse_loss, vel_loss, smoothness_loss, energy_loss, anti_zero_loss = loss_terms(v_guess_centered)
			return (
				MSE_WEIGHT * mse_loss
				+ VEL_WEIGHT * vel_loss
				+ SMOOTH_WEIGHT * smoothness_loss
				+ ENERGY_WEIGHT * energy_loss
				+ ZERO_WEIGHT * anti_zero_loss
			)

		def network_to_init_values(net_params_local):
			return jnp.asarray(net_apply_fun(net_params_local, coord_features_native))

		def candidate_native_fields():
			nn_native = network_to_init_values(net_params)
			yield 'nn/base', nn_native
			yield 'uniform/inflow', uniform_native
			yield 'prior/counterflow', prior_native
			yield 'mix/nn-uniform', 0.5 * nn_native + 0.5 * uniform_native
			yield 'mix/nn-prior', 0.5 * nn_native + 0.5 * prior_native

		print('\nInitializing optimization guess...')
		base_values = v0_gt0.at_centers().values
		guess_shape = base_values.shape
		noise_scale = 5e-4
		noise = math.random_uniform(guess_shape, low=-noise_scale, high=noise_scale)

		best_loss = None
		best_init_values = None
		search_key = jax.random.PRNGKey(0)

		for label, candidate_native in candidate_native_fields():
			candidate_values = native_to_centered_grid(candidate_native, Nx, Ny, DOMAIN['bounds'], Vn_BC).values + noise
			candidate_grid = CenteredGrid(values=candidate_values, extrapolation=Vn_BC, bounds=DOMAIN['bounds'])
			candidate_loss = loss_function(candidate_grid)
			candidate_loss_float = float(candidate_loss)
			print(f'Warm-start candidate [{label}]: loss={candidate_loss_float:.6e}')
			if best_loss is None or candidate_loss_float < best_loss:
				best_loss = candidate_loss_float
				best_init_values = candidate_values

		total_random_candidates = max(0, args.outer_iters * args.warmstart_candidates)
		for outer_step in range(total_random_candidates):
			search_key, candidate_key = jax.random.split(search_key)
			candidate_params = add_param_noise(net_params, candidate_key, args.warmstart_noise)
			nn_native = network_to_init_values(candidate_params)
			mix_ratio = 0.35 + 0.3 * ((outer_step % 3) / 2.0)
			candidate_native = mix_ratio * nn_native + (1.0 - mix_ratio) * prior_native
			candidate_values = native_to_centered_grid(candidate_native, Nx, Ny, DOMAIN['bounds'], Vn_BC).values + noise
			candidate_grid = CenteredGrid(values=candidate_values, extrapolation=Vn_BC, bounds=DOMAIN['bounds'])
			candidate_loss = loss_function(candidate_grid)
			candidate_loss_float = float(candidate_loss)
			print(f'Warm-start candidate [nn+prior {outer_step + 1}/{total_random_candidates}]: loss={candidate_loss_float:.6e}')
			if best_loss is None or candidate_loss_float < best_loss:
				best_loss = candidate_loss_float
				best_init_values = candidate_values

		init_values = best_init_values if best_init_values is not None else (base_values + noise)
		v_guess_proxy = CenteredGrid(values=init_values, extrapolation=Vn_BC, bounds=DOMAIN['bounds'])

		try:
			init_loss = loss_function(v_guess_proxy)
			print(f'Initial loss: {float(init_loss):.6e}')
			init_terms = loss_terms(v_guess_proxy)
			print(
				'Initial terms: '
				f'mse={float(init_terms[0]):.6e}, '
				f'vel={float(init_terms[1]):.6e}, '
				f'smooth={float(init_terms[2]):.6e}, '
				f'energy={float(init_terms[3]):.6e}, '
				f'zero={float(init_terms[4]):.6e}'
			)
		except Exception as exc:
			print(f'Initial loss evaluation failed: {exc}')

		print(f'\nStarting Optimization (L-BFGS-B, max_iter={args.lbfgs_iters})...')
		t_start = time.time()
		optimizer = Solve(
			'L-BFGS-B',
			x0=v_guess_proxy,
			max_iterations=args.lbfgs_iters,
			suppress=[phi.math.Diverged, phi.math.NotConverged],
		)
		with math.SolveTape(record_trajectories=False):
			result_centered = minimize(loss_function, optimizer)
		print(f'Optimization finished in {time.time() - t_start:.2f} s')

		v0_reconstructed = result_centered.at(v0_gt0)
		final_loss = loss_function(result_centered)
		final_terms = loss_terms(result_centered)
		print('=== Optimization Result ===')
		print(f'Final Loss: {float(final_loss):.6e}')
		print(
			'Final terms: '
			f'mse={float(final_terms[0]):.6e}, '
			f'vel={float(final_terms[1]):.6e}, '
			f'smooth={float(final_terms[2]):.6e}, '
			f'energy={float(final_terms[3]):.6e}, '
			f'zero={float(final_terms[4]):.6e}'
		)

	recon_traj = None
	if v0_reconstructed is not None:
		try:
			recon_traj_list = [initial_markers]
			v_curr = v0_reconstructed
			vs_curr = vs0_gt0
			p_curr = p0_gt0
			t_curr = t0_gt0
			L_curr = L0_gt0
			markers_curr = initial_markers
			for step in range(STEPS):
				v_curr, vs_curr, p_curr, t_curr, L_curr = helium.SFHelium_step(
					v_curr, vs_curr, p_curr, t_curr, L_curr,
					dt=DT, DOMAIN=DOMAIN, OBSTACLE=None,
					Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL,
				)
				markers_curr = advect.points(markers_curr, v_curr, dt=DT, integrator=advect.rk4)
				coords_np = markers_curr.geometry.center.native(['markers', 'vector'])
				coords_np = jnp.where(reset_mask_native[step + 1][..., None] > 0.5, gt_all_native[step + 1], coords_np)
				markers_curr = PointCloud(geom.Point(math.tensor(coords_np, instance('markers') & channel(vector='x,y'))))
				recon_traj_list.append(markers_curr)
			recon_traj = particles.pointcloud_list_to_numpy(recon_traj_list)
		except Exception:
			recon_traj = None

	v_gt_u, v_gt_v, v_gt_speed, v_gt_vec, vs_gt_u, vs_gt_v, vs_gt_speed, vs_gt_vec, t_gt = _extract_time_series_for_vn(
		v0_gt0, vs0_gt0, p0_gt0, t0_gt0, L0_gt0, STEPS, DT,
		DOMAIN=DOMAIN, OBSTACLE=None, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL,
	)

	v_recon_u = v_recon_v = v_recon_speed = v_recon_vec = None
	vs_recon_u = vs_recon_v = vs_recon_speed = vs_recon_vec = None
	t_recon = None
	if v0_reconstructed is not None:
		v_recon_u, v_recon_v, v_recon_speed, v_recon_vec, vs_recon_u, vs_recon_v, vs_recon_speed, vs_recon_vec, t_recon = _extract_time_series_for_vn(
			v0_reconstructed, vs0_gt0, p0_gt0, t0_gt0, L0_gt0, STEPS, DT,
			DOMAIN=DOMAIN, OBSTACLE=None, Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL,
		)

	# Prepare stable NumPy outputs for post-processing and visualization.
	gt_np = particles.tensor_time_markers_to_numpy(gt_tensor)
	loss_mask_np = particles.tensor_time_marker_mask_to_numpy(loss_mask)
	reset_mask_np = particles.tensor_time_marker_mask_to_numpy(reset_mask)
	field_series = {
		'gt': (v_gt_u, v_gt_v, v_gt_speed, v_gt_vec, vs_gt_u, vs_gt_v, vs_gt_speed, vs_gt_vec, t_gt),
		'recon': (v_recon_u, v_recon_v, v_recon_speed, v_recon_vec, vs_recon_u, vs_recon_v, vs_recon_speed, vs_recon_vec, t_recon),
	}
	save_dict = _build_save_dict(config, gt_np, recon_traj, loss_mask_np, reset_mask_np, field_series)
	io.save_npz('data/experiment/cf_exp_recon.npz', **save_dict)
	print('Task 04 complete. Data saved to data/experiment/cf_exp_recon.npz')


if __name__ == '__main__':
	main()
