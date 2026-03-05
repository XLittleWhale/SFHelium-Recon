from phi.jax.flow import *

def get_rbc_bcs(heat_value=10.0, cold_value=-10.0, v_bc_val=0):
    V_BC = {'x': v_bc_val, 'y': v_bc_val}
    t_BC_THERMAL = {'x': ZERO_GRADIENT, 'y-': heat_value, 'y+': cold_value}
    return V_BC, t_BC_THERMAL

def get_sf_bcs(Vn_IN, Vs_IN, PRESSURE_0=0.0, heat_grad_zero=True):
    # Simple boundary factory: explicit inlet velocities expected as args

    Vn_BC = {'x': 0, 'y-': vec(x=0, y=Vn_IN), 'y+': ZERO_GRADIENT}
    Vs_BC = {'x': extrapolation.PERIODIC, 'y-': vec(x=0, y=Vs_IN), 'y+': ZERO_GRADIENT}
    J_BC = {'x': 0, 'y-': vec(x=0, y=0), 'y+': ZERO_GRADIENT}
    t_BC_THERMAL = {'x': ZERO_GRADIENT, 'y-': ZERO_GRADIENT, 'y+': ZERO_GRADIENT}
    p_BC = {'x': ZERO_GRADIENT, 'y-': ZERO_GRADIENT, 'y+': PRESSURE_0}
    return Vn_BC, Vs_BC, J_BC, t_BC_THERMAL, p_BC

def get_cylinder_bcs(Vn_IN, Vs_IN, PRESSURE_0=0.0):
    # For cylinder we typically reuse SF boundaries but allow differences
    return get_sf_bcs(Vn_IN=Vn_IN, Vs_IN=Vs_IN, PRESSURE_0=PRESSURE_0)
