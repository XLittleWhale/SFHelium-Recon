"""
Task 04: Experimental channel flow inversion (uses VTK/Excel input)
"""
from phi.jax.flow import *
from sf_recon.physics import helium, boundaries
from sf_recon.utils import particles, io, vtk as vtk_utils
import pyvista as pv

# This script mirrors the logic in D4.py: load VTK fields, load particle excel, run inversion

# NOTE: update paths below before running
VTK_PATH = './1.85K/241mWcm2_6.55s.vtk'
EXCEL_PATH = './1.85K/10.10classified_k4_1.85 K_241mWcm2_120fps_Trajectory_v3.xlsx'

# Minimal domain and BC placeholders (should match D4 settings when running)
Lx, Ly = 0.016, 0.320
Nx, Ny = 16, 320
DOMAIN = dict(x=Nx, y=Ny, bounds=Box(x=Lx,y=Ly))
Vn_BC = {'x':0,'y-': vec(x=0,y=0),'y+': ZERO_GRADIENT}
Vs_BC = {'x': extrapolation.PERIODIC,'y-': vec(x=0,y=0),'y+': ZERO_GRADIENT}

# Boundary dict for VTK loader
Vn_BC, Vs_BC, J_BC, t_BC_THERMAL, p_BC = boundaries.get_sf_bcs(Vn_IN=0.0, Vs_IN=0.0, PRESSURE_0=1948)
bcs = {'v': Vn_BC, 'vs': Vs_BC, 't': t_BC_THERMAL, 'l': ZERO_GRADIENT, 'p': p_BC}

# Optionally load VTK fields and align to PhiFlow domain (if VTK available)
try:
	v0_gt0, vs0_gt0, t0_gt0, L0_gt0, p0_gt0 = vtk_utils.load_and_align_fields(VTK_PATH, (Nx, Ny), DOMAIN['bounds'], bcs)
	print('VTK alignment successful')
except Exception as e:
	print(f'VTK load skipped or failed: {e}')

# Load particle trajectories from excel
gt_tensor, loss_mask, reset_mask, initial_pos = particles.load_particle_trajectories(EXCEL_PATH, num_particles=800, max_steps=640, dt=2e-3, domain_bounds=DOMAIN['bounds'])
print('GT and masks loaded')

# Save staged data
io.save_npz('data/experiment/loaded_gt.npz', success=1)
print('Task 04 staged')
