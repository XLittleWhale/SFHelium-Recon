from phi.jax.flow import *
from .boundaries import get_sf_bcs
import numpy as np

# Properties of SFHelium
VISCOSITY_N_COEFFS = [7.42116853e-04, -2.72872406e-03, 4.18086013e-03, -3.40692439e-03,
               1.55634875e-03, -3.77813201e-04, 3.80801796e-05]

RHO_N_COEFFS = [1.80183130e-01, -2.48558984e+01, 4.01282654e+02, -2.48742413e+03,
                7.85114896e+03, -1.42666651e+04, 1.57974461e+04, -1.08279105e+04,
                4.48686634e+03, -1.02983154e+03, 1.00498992e+02]

RHO_S_COEFFS = [144.94795, 25.18243857, -406.9535624, 2509.83455, -7886.879518,
                14286.34985, -15784.94029, 10803.13498, -4471.875738, 1025.612443, -100.0317144]

ENTROPY_COEFFS = [6.57311375e+06, -4.38113590e+07, 1.30656739e+08, -2.29500549e+08,
                  2.62852401e+08, -2.05055435e+08, 1.10322083e+08, -4.04131685e+07,
                  9.64576203e+06, -1.35448196e+06, 8.49760567e+04]

C1_COEFFS = [-37875.4603432003, 161027.467671278, -292620.794399981, 294653.759992904,
             -177568.113431148, 64045.2095387393, -12802.0813284325, 1094.15041473176]

C2_COEFFS = [2421.03719768316, -7059.46514061684, 8208.38710509775, -4758.60939944746,
             1376.23800786335, -158.752493718344]

C3_COEFFS = [-4.18764389500096, 8.88295312215358, -5.42845535938227, 1.49351622326385]

B = 1.008
KAPPA = 9.97e-8

COLD_SOURCE_INTENSITY = 1.85  #1.94

# Pressure solver default (forward + gradient solve)
PRESSURE_SOLVER = Solve(
    'CG',
    abs_tol=1e-10,
    rel_tol=0,
    max_iterations=2000,
    suppress=[phi.math.NotConverged, phi.math.Diverged],
    gradient_solve=Solve(
        'CG',
        abs_tol=1e-10,
        rel_tol=0,
        max_iterations=2000,
        suppress=[phi.math.NotConverged, phi.math.Diverged]
    )
)

# PropSolver and SFHelium_step
@jit_compile
def PropSolver(t):
    t = field.maximum(t, 1.7)
    t = field.minimum(t, 2.17)
    def poly_eval(coeffs, t_field):
        res = 0.0
        for power, coeff in enumerate(coeffs):
            res += coeff * (t_field ** power)
        return res
    def poly_deriv(coeffs, t_field):
        res = 0.0
        for power, coeff in enumerate(coeffs):
            if power > 0:
                res += power * coeff * (t_field ** (power-1))
        return res
    VISCOSITY_N = poly_eval(VISCOSITY_N_COEFFS, t)
    RHO_N = poly_eval(RHO_N_COEFFS, t)
    RHO_S = poly_eval(RHO_S_COEFFS, t)
    RHO_N = field.maximum(RHO_N, 0.0)
    RHO_S = field.maximum(RHO_S, 0.0)
    RHO = RHO_N + RHO_S
    RHO = field.maximum(RHO, 100.0)
    dRHO_NdT = poly_deriv(RHO_N_COEFFS, t)
    dRHO_SdT = poly_deriv(RHO_S_COEFFS, t)
    dRHOdT = dRHO_NdT + dRHO_SdT
    ENTROPY = poly_eval(ENTROPY_COEFFS, t)
    dSdT = poly_deriv(ENTROPY_COEFFS, t)
    C1 = poly_eval(C1_COEFFS, t)
    alpha_v = C1/1e6 * RHO_S * ENTROPY * t
    C2 = poly_eval(C2_COEFFS, t)
    beta_v = C2/1e8
    C3 = poly_eval(C3_COEFFS, t)
    gamma_v = math.exp(math.log(10) * C3) * (100**4.5)
    return VISCOSITY_N, RHO_N, RHO_S, RHO, dRHOdT, ENTROPY, dSdT, alpha_v, beta_v, gamma_v

def SFHelium_step(vn, vs, p, t, L, dt, DOMAIN=None, OBSTACLE=None, Vn_BC=None, Vs_BC=None, t_BC_THERMAL=None, PRESSURE_SOLVER=PRESSURE_SOLVER):
    # Prevent numerical issues
    EPSILON = 1e-8
    MAX_VEL_FORCE = 500.0
    MAX_W_MAG_FOR_L = 100.0

    # 1. Get physical properties (Stop gradient on temp to avoid explosive prop derivatives)
    t_sg = math.stop_gradient(t)
    VISCOSITY_N, RHO_N, RHO_S, RHO, dRHOdT, ENTROPY, dSdT, alpha_v, beta_v, gamma_v = PropSolver(t_sg)
    
    RHO   = field.maximum(RHO, EPSILON)
    RHO_N = field.maximum(RHO_N, EPSILON)
    RHO_S = field.maximum(RHO_S, EPSILON)
    
    # 2. BCs adjustment 
    y_coords = t.points.vector['y']
    
    ## for task02
    # Ly = 0.0016
    # Ny = 80
    # SUB_STEPS = 5
    
    ## for task03
    # Ly = 0.5
    # Ny = 200
    # SUB_STEPS = 20
    
    ## for task04
    Ly = 0.320
    Ny = 320
    SUB_STEPS = 20

    sponge_top = Ly - Ly/Ny
    steepness = 20000.0
    cooling_zone = math.sigmoid((y_coords - sponge_top) * steepness)
    cooling_rate = 50.0
    
    t = t - cooling_zone * cooling_rate * (t - COLD_SOURCE_INTENSITY) * dt/2

    # 3. Advection
    vn = advect.semi_lagrangian(vn, vn, dt)
    vs = advect.semi_lagrangian(vs, vs, dt)
    t  = advect.semi_lagrangian(t,  vn, dt)

    # 4. Viscosity
    laplacian_vn = field.laplace(vn)
    vn = vn + (VISCOSITY_N * dt * laplacian_vn.at(t)).at(vn)

    # 5. Thermal-mechanical effects
    grad_k_n = field.spatial_gradient(0.5 * field.vec_squared(vn.at(t)), Vn_BC).at(vn)
    grad_t_n = field.spatial_gradient(t, Vn_BC).at(vn)
    grad_k_s = field.spatial_gradient(0.5 * field.vec_squared(vs.at(t)), Vs_BC).at(vs)
    grad_t_s = field.spatial_gradient(t, Vs_BC).at(vs)

    coeff_S_n = (RHO_S/RHO_N * ENTROPY).at(vn)
    coeff_K_n = (RHO_S/RHO).at(vn)
    CHEM_POTENTIAL_n = - coeff_S_n * grad_t_n - coeff_K_n * grad_k_n

    coeff_S_s = ENTROPY.at(vs)
    coeff_K_s = (RHO_N/RHO).at(vs)
    CHEM_POTENTIAL_s = - coeff_S_s * grad_t_s - coeff_K_s * grad_k_s

    CHEM_POTENTIAL_n = field.minimum(field.maximum(CHEM_POTENTIAL_n, -MAX_VEL_FORCE), MAX_VEL_FORCE)
    CHEM_POTENTIAL_s = field.minimum(field.maximum(CHEM_POTENTIAL_s, -MAX_VEL_FORCE), MAX_VEL_FORCE)

    # Cut off gradients through complex thermodynamics to stabilize macroscopic inversion
    CHEM_POTENTIAL_n = math.stop_gradient(CHEM_POTENTIAL_n)
    CHEM_POTENTIAL_s = math.stop_gradient(CHEM_POTENTIAL_s)

    vn = vn + CHEM_POTENTIAL_n * dt
    vs = vs - CHEM_POTENTIAL_s * dt

    # ==============================================================================
    # 6. Mutual friction and Vinen equation (Sub-cycling)
    # ==============================================================================
    sub_steps = SUB_STEPS
    dt_sub = dt / sub_steps
    
    rho_ratio_s = (RHO_S / RHO)
    rho_ratio_n = (RHO_N / RHO)

    for _ in range(sub_steps):
        # --- A. Initialization ---
        vn_centered = vn.at(t)
        vs_centered = vs.at(t)
        L_centered  = L.at(t)
        
        vns_vec = vn_centered - vs_centered
        w_sq = field.vec_squared(vns_vec)
        w_mag = field.vec_length(vns_vec)
        vs_mag = field.vec_length(vs_centered)
        w_mag_safe = field.maximum(w_mag, EPSILON)
        vs_mag_safe = field.maximum(vs_mag, EPSILON)
        
        # --- B. Vinen Equation ---
        safe_L_sqrt = math.sqrt(field.maximum(L_centered, 0.0))
        
        term1 = -2.4 * KAPPA * safe_L_sqrt
        term2 = -(RHO_N/RHO) * w_mag_safe # 使用 safe mag
        
        vL_scalar = (term1 + term2) / vs_mag_safe
        
        MAX_V_VAL = 100.0
        vL_scalar = field.minimum(field.maximum(vL_scalar, -MAX_V_VAL), MAX_V_VAL)
        
        vL_vec = vL_scalar * vs_centered # assuming vL along vs direction
        
        # Advection of L
        L = advect.semi_lagrangian(L, vL_vec, dt_sub)
        MAX_L_VAL = 1e8
        L = field.minimum(field.maximum(L, 0.0), MAX_L_VAL)
        L_centered = L.at(t)
        
        # Production and decay terms
        w_mag_for_prod = field.minimum(w_mag_safe, MAX_W_MAG_FOR_L)
        
        term_prod = alpha_v * w_mag_for_prod * (L_centered ** 1.5)
        term_nucl = gamma_v * (w_mag_for_prod ** 2.5) 
        
        L_source = term_prod + term_nucl
        denom = 1.0 + dt_sub * beta_v * L_centered
        denom = field.maximum(denom, EPSILON)
        
        L_new_center = (L_centered + dt_sub * L_source) / denom
        L_new_center = field.minimum(field.maximum(L_new_center, 0.0), MAX_L_VAL)
        
        L = L.with_values(L_new_center)
        
        # --- C. Mutual friction (Analytical Decay) ---
        # Using exponential decay method, unconditionally stable
        
        # Calculate the inverse of the decay time constant: 1/tau_decay ~ Friction Frequency
        # F_friction ~ - (B kappa / 3) * L * w
        freq = (B / 3.0 * KAPPA) * L_new_center
        
        # Computing Decay Factor: exp(-freq * dt)
        # If freq * dt is large, decay -> 0 (completely locked)
        # If freq * dt is small, decay -> 1 (no friction)
        change_factor = 1.0 - field.exp(-freq * dt_sub)
        
        # Interpolate to staggered grid, STOP gradient on friction coefficient to prevent Vinen feedback loop
        alpha_at_vn = math.stop_gradient(change_factor.at(vn))
        alpha_at_vs = math.stop_gradient(change_factor.at(vs))
        
        vs_at_vn = vs.at(t).at(vn)
        vn_at_vs = vn.at(t).at(vs)
        
        vn = vn + (rho_ratio_s.at(vn) * alpha_at_vn) * (vs_at_vn - vn)
        vs = vs + (rho_ratio_n.at(vs) * alpha_at_vs) * (vn_at_vs - vs)

    
    # 7. Thermodynamic Dissipation
    vn_centered_final = vn.at(t)
    vs_centered_final = vs.at(t)
    vns_final = vn_centered_final - vs_centered_final
    
    L_final = L.at(t)
    # L_final = (alpha_v/beta_v)**2 * (field.vec_squared(vns_final).at(t))
    B_coeff_final = (B / 3.0 * KAPPA) * L_final
    Fns_coeff_final = B_coeff_final * (RHO_S/RHO) 
    
    dissipation_energy = Fns_coeff_final * field.vec_squared(vns_final)
    
    heat_capacity = t * (ENTROPY * dRHOdT + RHO * dSdT)
    heat_capacity = field.maximum(heat_capacity, 1e-5)
    
    t_dissipa = dissipation_energy / heat_capacity
    t = t + t_dissipa * dt
    t = t - cooling_zone * cooling_rate * (t - COLD_SOURCE_INTENSITY) * dt/2
    
    # update BCs
    vn = StaggeredGrid(vn, Vn_BC, **DOMAIN)
    vs = StaggeredGrid(vs, Vs_BC, **DOMAIN)
    t = CenteredGrid(t, t_BC_THERMAL, **DOMAIN)
    L = CenteredGrid(L, ZERO_GRADIENT, **DOMAIN)
    p = CenteredGrid(p, ZERO_GRADIENT, **DOMAIN)

    # 8. Obstacle Handling
    if OBSTACLE is not None:
        obstacle_mask_vn = resample(~(OBSTACLE.geometry), vn)
        vn = field.safe_mul(obstacle_mask_vn, vn)
        obstacle_mask_vs = resample(~(OBSTACLE.geometry), vs)
        vs = field.safe_mul(obstacle_mask_vs, vs)
        obstacle_mask_t = resample(~(OBSTACLE.geometry), t)
        t = field.safe_mul(obstacle_mask_t, t)
        obstacle_mask_L = resample(~(OBSTACLE.geometry), L)
        L = field.safe_mul(obstacle_mask_L, L)

    # update BCs
    vn = StaggeredGrid(vn, Vn_BC, **DOMAIN)
    vs = StaggeredGrid(vs, Vs_BC, **DOMAIN)
    t = CenteredGrid(t, t_BC_THERMAL, **DOMAIN)
    L = CenteredGrid(L, ZERO_GRADIENT, **DOMAIN)
    p = CenteredGrid(p, ZERO_GRADIENT, **DOMAIN)
    
    # 9. Pressure Projection
    vn, p_n = fluid.make_incompressible(vn, OBSTACLE or (), PRESSURE_SOLVER)
    vs, p_s = fluid.make_incompressible(vs, OBSTACLE or (), PRESSURE_SOLVER)

    p = RHO.at(p) * (p_n.at(p) + p_s.at(p)) * 0.5
    
    return vn, vs, p, t, L

# Constrain Markers
@jit_compile
def constrain_markers_push(markers, sphere_obstacle):
    center = sphere_obstacle.geometry.center
    radius = sphere_obstacle.geometry.radius
    diff = markers - center
    dist = math.vec_length(diff)
    epsilon = 1e-4
    correction = center + (diff / (dist + 1e-6)) * (radius + epsilon)
    is_inside = dist < radius
    fixed_markers = math.where(is_inside, correction, markers)
    return fixed_markers
