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
	build_counterflow_inflow_prior,
	native_to_centered_grid,
)
from sf_recon.utils.saving import (
	simple_to_numpy as _simple_to_numpy,
	extract_time_series_for_vn as _extract_time_series_for_vn,
	ensure_HW as _ensure_HW,
)


def _sample_centered_velocity(velocity_grid, coords_nat):
	"""Sample a centered velocity field at particle coordinates."""
	coords = math.tensor(coords_nat, instance('markers') & channel(vector='x,y'))
	sampled = velocity_grid.at(PointCloud(geom.Point(coords))).values
	return sampled.native(['markers', 'vector'])


def _compute_material_acceleration(prev_grid, next_grid, dt):
	"""Approximate material acceleration using a first-order time difference."""
	return (next_grid - prev_grid) * (1.0 / max(float(dt), 1e-12))


def _advance_particle_state(coords_nat, vel_nat, vn_now, vs_now, vn_prev, vs_prev, dt, tau, rho_n, rho_s, rho_total):
	"""Advance particle position and velocity using the relaxation model."""
	vn_now_center = vn_now.at_centers()
	vs_now_center = vs_now.at_centers()
	dvn_dt = _compute_material_acceleration(vn_prev.at_centers(), vn_now_center, dt)
	dvs_dt = _compute_material_acceleration(vs_prev.at_centers(), vs_now_center, dt)
	vn_sample = _sample_centered_velocity(vn_now_center, coords_nat)
	dvn_sample = _sample_centered_velocity(dvn_dt, coords_nat)
	dvs_sample = _sample_centered_velocity(dvs_dt, coords_nat)
	acc = (
		(vn_sample - vel_nat) / float(tau)
		+ (float(rho_n) / float(rho_total)) * dvn_sample
		+ (float(rho_s) / float(rho_total)) * dvs_sample
	)
	next_vel = vel_nat + float(dt) * acc
	next_coords = coords_nat + float(dt) * next_vel
	return next_coords, next_vel


def _center_guess_to_bc_staggered(v_guess_centered, v_template, vn_bc):
	"""Convert a centered guess into a staggered field that respects `Vn_BC`."""
	centered_clean = CenteredGrid(
		values=v_guess_centered.values,
		extrapolation=ZERO_GRADIENT,
		bounds=v_guess_centered.bounds,
		resolution=v_guess_centered.resolution,
	)
	staggered_template = StaggeredGrid(
		0,
		vn_bc,
		bounds=v_template.bounds,
		resolution=v_template.resolution,
	)
	return centered_clean.at(staggered_template)


def _configure_geometry(args):
	"""Build the fixed geometry and time-step settings."""
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
		'MSE_WEIGHT': 2e6,
		'VEL_WEIGHT': 0,
		'DOMAIN': dict(x=Nx, y=Ny, bounds=Box(x=Lx, y=Ly)),
		'WINDOMAIN': dict(x=WNx, y=WNy, bounds=Box(x=(Lx / 2 - Wx / 2, Lx / 2 + Wx / 2), y=(Ly / 2 - Wy / 2, Ly / 2 + Wy / 2))),
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
			# marker_x_min, marker_x_max = 0.0, config['Wx']
			marker_x_min = config['Lx'] / 2 - config['Wx'] / 2
			marker_x_max = config['Lx'] / 2 + config['Wx'] / 2
			marker_y_min = config['Ly'] / 2 - config['Wy'] / 2
			marker_y_max = config['Ly'] / 2 + config['Wy'] / 2
	else:
		marker_x_min, marker_x_max, marker_y_min, marker_y_max = marker_bounds

	v_gt_u, v_gt_v, v_gt_speed, v_gt_vec, vs_gt_u, vs_gt_v, vs_gt_speed, vs_gt_vec, t_gt = field_series['gt']
	v_recon_u, v_recon_v, v_recon_speed, v_recon_vec, vs_recon_u, vs_recon_v, vs_recon_speed, vs_recon_vec, t_recon = field_series['recon']
	obs_vel = config.get('obs_vel_np')
	recon_vel = config.get('recon_vel_np')
	vel_mask = config.get('vel_mask_np')

	return dict(
		success=1,
		Lx=config['Lx'],
		Ly=config['Ly'],
		Wx=config['Wx'],
		Wy=config['Wy'],
		# window_x_min=0.0,
		# window_x_max=config['Wx'],
        window_x_min=config['Lx'] / 2 - config['Wx'] / 2,
		window_x_max=config['Lx'] / 2 + config['Wx'] / 2,
		window_y_min=config['Ly'] / 2 - config['Wy'] / 2,
		window_y_max=config['Ly'] / 2 + config['Wy'] / 2,
		marker_x_min=marker_x_min,
		marker_x_max=marker_x_max,
		marker_y_min=marker_y_min,
		marker_y_max=marker_y_max,
		pre_steps=config['PRE_STEPS'],
		gt=gt_np,
		recon=recon_traj,
		gt_vel=obs_vel,
		recon_vel=recon_vel,
		loss_mask=loss_mask_np,
		reset_mask=reset_mask_np,
		vel_mask=vel_mask,
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
	parser.add_argument('--num-particles', type=int, default=4096, help='Number of particle trajectories to load')
	parser.add_argument('--steps', type=int, default=10, help='Number of simulation steps used for inversion')
	parser.add_argument('--dt', type=float, default=1e-2, help='Time step size')
	parser.add_argument('--freq', type=int, default=120, help='Acquisition frequency of the experiment')
	parser.add_argument('--pre-steps', type=int, default=0, help='Number of pre-simulation steps before inversion, matching the D4 workflow')
	parser.add_argument('--particle-tau', type=float, default=5e-3, help='Particle relaxation time in the velocity-state model')
	parser.add_argument('--smooth-particles', action='store_true', help='Enable optional Gaussian smoothing for processed particle trajectories and velocities')
	parser.add_argument('--smooth-radius', type=int, default=2, help='Half-width of the Gaussian smoothing kernel in frames')
	parser.add_argument('--smooth-sigma', type=float, default=1.0, help='Standard deviation of the Gaussian smoothing kernel in frames')
	args = parser.parse_args()

	math.use('jax')
	math.set_global_precision(64)

	print('Task 04: starting')

	# Geometry intentionally mirrors the validated D4 setup.
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
	DOMAIN = config['DOMAIN']
	WINDOMAIN = config['WINDOMAIN']

	HEAT_SOURCE_INTENSITY = 1.85
	PRESSURE_0 = 1948.0
	HEAT_FLUX = 2410.0
	DENSITY = 145.4070
	ENTROPY = 627.0
	DENSITY_N = 52.86
	DENSITY_S = 92.54
	RHO_TOTAL = DENSITY
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
	gt_tensor, loss_mask, reset_mask, initial_pos, gt_vel_tensor, vel_mask = particles.load_experimental_particle_data(
		args.excel_path,
		num_particles=MARKERS,
		max_steps=STEPS,
		dt=DT,
		domain_bounds=WINDOMAIN['bounds'],
		freq=args.freq,
		shift_to_zero=True,
		smooth=args.smooth_particles,
		smooth_radius=args.smooth_radius,
		smooth_sigma=args.smooth_sigma,
	)
	print(f'Loaded particle trajectories from {args.excel_path}')

	initial_markers = PointCloud(geom.Point(initial_pos))
	gt_all_native = gt_tensor.native(['time', 'markers', 'vector'])
	gt_vel_native = gt_vel_tensor.native(['time', 'markers', 'vector'])
	loss_mask_native = loss_mask.native(['time', 'markers'])
	reset_mask_native = reset_mask.native(['time', 'markers'])
	vel_mask_native = vel_mask.native(['time', 'markers'])

	# Reconstruct the initial normal-fluid field when inversion is enabled.
	v0_reconstructed = None

	if not getattr(args, 'skip_inversion', False):
		prior_native = build_counterflow_inflow_prior(Nx * Ny, Vn_IN, dtype=jnp.float64)

		def physical_step_logic(carry_state_native, scan_input):
			"""Single scan step with particle position/velocity dynamics."""
			(v_x_nat, v_y_nat), (vs_x_nat, vs_y_nat), p_nat, t_nat, L_nat, coords_nat, vel_nat = carry_state_native
			reset_t, gt_t, gt_vel_t = scan_input

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

			vn_prev = vn
			vs_prev = vs
			vn_next, vs_next, p_next, t_next, L_next = helium.SFHelium_step(
				vn, vs, p, t, L,
				dt=DT, DOMAIN=DOMAIN, OBSTACLE=None,
				Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL,
			)
			next_coords_free, next_vel_free = _advance_particle_state(
				coords_nat,
				vel_nat,
				vn_next,
				vs_next,
				vn_prev,
				vs_prev,
				DT,
				args.particle_tau,
				DENSITY_N,
				DENSITY_S,
				RHO_TOTAL,
			)
			next_coords = jnp.where(reset_t[..., None] > 0.5, gt_t, next_coords_free)
			next_vel = jnp.where(reset_t[..., None] > 0.5, gt_vel_t, next_vel_free)

			new_carry_native = (
				(vn_next.vector['x'].values.native(['x', 'y']), vn_next.vector['y'].values.native(['x', 'y'])),
				(vs_next.vector['x'].values.native(['x', 'y']), vs_next.vector['y'].values.native(['x', 'y'])),
				p_next.values.native(['x', 'y']),
				t_next.values.native(['x', 'y']),
				L_next.values.native(['x', 'y']),
				next_coords,
				next_vel,
			)
			return new_carry_native, (next_coords, next_vel)

		step_fn_checkpointed = jax.checkpoint(physical_step_logic)

		@jit_compile
		def loss_terms(v_guess_centered):
			"""Compute trajectory losses for experimental data."""
			v_sim = _center_guess_to_bc_staggered(v_guess_centered, v0_gt0, Vn_BC)
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
				gt_vel_native[0],
			)

			scan_inputs = (reset_mask_native[1:], gt_all_native[1:], gt_vel_native[1:])
			_, outputs_native = jax.lax.scan(step_fn_checkpointed, state_init_native, scan_inputs)
			trajectory_stack_native, simulated_vel_native = outputs_native

			gt_target_native = gt_all_native[1:]
			loss_mask_expanded = loss_mask_native[1:, :, None]
			diff = (trajectory_stack_native - gt_target_native) * loss_mask_expanded
			mse_loss = jnp.sum(diff ** 2)

			vel_mask_expanded = vel_mask_native[1:, :, None]
			vel_loss = jnp.sum(((simulated_vel_native - gt_vel_native[1:]) * vel_mask_expanded) ** 2)

			return mse_loss, vel_loss

		@jit_compile
		def loss_function(v_guess_centered):
			mse_loss, vel_loss = loss_terms(v_guess_centered)
			return (
				MSE_WEIGHT * mse_loss
				+ VEL_WEIGHT * vel_loss
			)

		print('\nInitializing optimization guess...')
		guess_shape = v0_gt0.at_centers().values.shape
		noise_scale = 5e-4
		noise = math.random_uniform(guess_shape, low=-noise_scale, high=noise_scale)
		init_values = native_to_centered_grid(prior_native, Nx, Ny, DOMAIN['bounds'], Vn_BC).values + noise
		candidate_grid = CenteredGrid(values=init_values, extrapolation=Vn_BC, bounds=DOMAIN['bounds'])
		print(f'Warm-start prior [counterflow]: loss={float(loss_function(candidate_grid)):.6e}')
		v_guess_proxy = CenteredGrid(values=init_values, extrapolation=Vn_BC, bounds=DOMAIN['bounds'])

		try:
			init_loss = loss_function(v_guess_proxy)
			print(f'Initial loss: {float(init_loss):.6e}')
			init_terms = loss_terms(v_guess_proxy)
			print(
				'Initial terms: '
				f'mse={float(init_terms[0]):.6e}, '
				f'vel={float(init_terms[1]):.6e}'
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

		v0_reconstructed = _center_guess_to_bc_staggered(result_centered, v0_gt0, Vn_BC)
		final_loss = loss_function(result_centered)
		final_terms = loss_terms(result_centered)
		print('=== Optimization Result ===')
		print(f'Final Loss: {float(final_loss):.6e}')
		print(
			'Final terms: '
			f'mse={float(final_terms[0]):.6e}, '
			f'vel={float(final_terms[1]):.6e}'
		)

	recon_traj = None
	recon_vel = None
	if v0_reconstructed is not None:
		try:
			recon_traj_list = [initial_markers]
			recon_vel_list = [np.asarray(gt_vel_native[0], dtype=float)]
			v_curr = v0_reconstructed
			vs_curr = vs0_gt0
			p_curr = p0_gt0
			t_curr = t0_gt0
			L_curr = L0_gt0
			markers_curr = initial_markers
			coords_curr = initial_markers.geometry.center.native(['markers', 'vector'])
			vel_curr = gt_vel_native[0]
			for step in range(STEPS):
				v_prev = v_curr
				vs_prev = vs_curr
				v_curr, vs_curr, p_curr, t_curr, L_curr = helium.SFHelium_step(
					v_curr, vs_curr, p_curr, t_curr, L_curr,
					dt=DT, DOMAIN=DOMAIN, OBSTACLE=None,
					Vn_BC=Vn_BC, Vs_BC=Vs_BC, t_BC_THERMAL=t_BC_THERMAL,
				)
				coords_np, vel_np = _advance_particle_state(
					coords_curr,
					vel_curr,
					v_curr,
					vs_curr,
					v_prev,
					vs_prev,
					DT,
					args.particle_tau,
					DENSITY_N,
					DENSITY_S,
					RHO_TOTAL,
				)
				coords_np = jnp.where(reset_mask_native[step + 1][..., None] > 0.5, gt_all_native[step + 1], coords_np)
				vel_np = jnp.where(reset_mask_native[step + 1][..., None] > 0.5, gt_vel_native[step + 1], vel_np)
				coords_curr = coords_np
				vel_curr = vel_np
				markers_curr = PointCloud(geom.Point(math.tensor(coords_np, instance('markers') & channel(vector='x,y'))))
				recon_traj_list.append(markers_curr)
				recon_vel_list.append(np.asarray(vel_np, dtype=float))
			recon_traj = particles.pointcloud_list_to_numpy(recon_traj_list)
			recon_vel = np.stack(recon_vel_list, axis=0)
		except Exception:
			recon_traj = None
			recon_vel = None

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
	gt_vel_np = particles.tensor_time_marker_velocity_to_numpy(gt_vel_tensor)
	loss_mask_np = particles.tensor_time_marker_mask_to_numpy(loss_mask)
	reset_mask_np = particles.tensor_time_marker_mask_to_numpy(reset_mask)
	vel_mask_np = particles.tensor_time_marker_mask_to_numpy(vel_mask)
	config['obs_vel_np'] = gt_vel_np
	config['recon_vel_np'] = recon_vel
	config['vel_mask_np'] = vel_mask_np
	field_series = {
		'gt': (v_gt_u, v_gt_v, v_gt_speed, v_gt_vec, vs_gt_u, vs_gt_v, vs_gt_speed, vs_gt_vec, t_gt),
		'recon': (v_recon_u, v_recon_v, v_recon_speed, v_recon_vec, vs_recon_u, vs_recon_v, vs_recon_speed, vs_recon_vec, t_recon),
	}
	save_dict = _build_save_dict(config, gt_np, recon_traj, loss_mask_np, reset_mask_np, field_series)
	io.save_npz('data/experiment/cf_exp_recon.npz', **save_dict)
	print('Task 04 complete. Data saved to data/experiment/cf_exp_recon.npz')


if __name__ == '__main__':
	main()
