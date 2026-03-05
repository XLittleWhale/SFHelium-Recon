"""Shared saving utilities for tasks and notebooks."""
from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import numpy as np

from sf_recon.physics import helium, normal


def simple_to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        return np.array(x)
    except Exception:
        try:
            if hasattr(x, 'values'):
                v = x.values
                try:
                    return np.array(v)
                except Exception:
                    try:
                        if hasattr(v, 'numpy'):
                            return np.asarray(v.numpy())
                    except Exception:
                        pass
                    try:
                        seq = [np.asarray(e) for e in v]
                        return np.stack(seq)
                    except Exception:
                        return None
        except Exception:
            return None
    return None


def stack_if_possible(lst: Iterable[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    try:
        if all(x is None for x in lst):
            return None
        ref = None
        for x in lst:
            if x is not None:
                ref = np.asarray(x)
                break
        if ref is None:
            return None
        ref_shape = ref.shape
        stacked = []
        for x in lst:
            if x is None:
                stacked.append(np.full(ref_shape, np.nan, dtype=float))
            else:
                try:
                    arr = np.asarray(x, dtype=float)
                    if arr.shape != ref_shape:
                        stacked.append(np.full(ref_shape, np.nan, dtype=float))
                    else:
                        stacked.append(arr)
                except Exception:
                    stacked.append(np.full(ref_shape, np.nan, dtype=float))
        return np.stack(stacked)
    except Exception:
        return None


def extract_time_series_for_rbc(v_init, t_init, steps, dt):
    """Run forward simulation from (v_init,t_init) for RBC and return stacked u/v/speed/vec arrays."""
    u_list, v_list, speed_list, vec_list = [], [], [], []
    curr_v, curr_t = v_init, t_init
    for i in range(steps + 1):
        u_arr = v_arr = vec_arr = speed_arr = None
        try:
            vc = curr_v.at_centers()
            ux = vc.vector['x']
            uy = vc.vector['y']
            u_arr = simple_to_numpy(ux.values if hasattr(ux, 'values') else ux)
            v_arr = simple_to_numpy(uy.values if hasattr(uy, 'values') else uy)
            if u_arr is not None and v_arr is not None:
                u_arr = np.asarray(u_arr, dtype=float)
                v_arr = np.asarray(v_arr, dtype=float)
                speed_arr = np.sqrt(u_arr ** 2 + v_arr ** 2)
                vec_arr = np.stack([u_arr, v_arr], axis=-1)
        except Exception:
            try:
                cand = simple_to_numpy(curr_v)
                if isinstance(cand, np.ndarray) and cand.ndim == 3 and cand.shape[-1] == 2:
                    vec_arr = np.asarray(cand, dtype=float)
                    u_arr = vec_arr[..., 0]
                    v_arr = vec_arr[..., 1]
                    speed_arr = np.sqrt(u_arr ** 2 + v_arr ** 2)
            except Exception:
                pass
        u_list.append(u_arr)
        v_list.append(v_arr)
        speed_list.append(speed_arr)
        vec_list.append(vec_arr)
        if i < steps:
            curr_v, curr_t = normal.boussinesq_step(curr_v, curr_t, dt=dt)
    return (
        stack_if_possible(u_list),
        stack_if_possible(v_list),
        stack_if_possible(speed_list),
        stack_if_possible(vec_list),
    )


def extract_time_series_for_vn(v_init, vs_init, p_init, t_init, L_init, steps, dt, **step_kwargs):
    """Run forward simulation from (vn,vs,p,t,L) and return stacked arrays for vn/vs and temperature."""
    u_list, v_list, speed_list, vec_list = [], [], [], []
    vs_u_list, vs_v_list, vs_speed_list, vs_vec_list = [], [], [], []
    t_list = []
    curr_vn, curr_vs, curr_p, curr_t, curr_L = v_init, vs_init, p_init, t_init, L_init
    DOMAIN = step_kwargs.get('DOMAIN', None)
    for i in range(steps + 1):
        # vn
        u_arr = v_arr = vec_arr = speed_arr = None
        try:
            vc = curr_vn.at_centers()
            ux = vc.vector['x']
            uy = vc.vector['y']
            u_arr = simple_to_numpy(ux.values if hasattr(ux, 'values') else ux)
            v_arr = simple_to_numpy(uy.values if hasattr(uy, 'values') else uy)
            if u_arr is not None and v_arr is not None:
                u_arr = np.asarray(u_arr, dtype=float)
                v_arr = np.asarray(v_arr, dtype=float)
                speed_arr = np.sqrt(u_arr ** 2 + v_arr ** 2)
                vec_arr = np.stack([u_arr, v_arr], axis=-1)
        except Exception:
            try:
                cand = simple_to_numpy(curr_vn)
                if isinstance(cand, np.ndarray) and cand.ndim == 3 and cand.shape[-1] == 2:
                    vec_arr = np.asarray(cand, dtype=float)
                    u_arr = vec_arr[..., 0]
                    v_arr = vec_arr[..., 1]
                    speed_arr = np.sqrt(u_arr ** 2 + v_arr ** 2)
            except Exception:
                pass
        u_list.append(u_arr)
        v_list.append(v_arr)
        speed_list.append(speed_arr)
        vec_list.append(vec_arr)
        # vs
        vs_u_arr = vs_v_arr = vs_vec_arr = vs_speed_arr = None
        try:
            vc_vs = curr_vs.at_centers()
            ux_vs = vc_vs.vector['x']
            uy_vs = vc_vs.vector['y']
            vs_u_arr = simple_to_numpy(ux_vs.values if hasattr(ux_vs, 'values') else ux_vs)
            vs_v_arr = simple_to_numpy(uy_vs.values if hasattr(uy_vs, 'values') else uy_vs)
            if vs_u_arr is not None and vs_v_arr is not None:
                vs_u_arr = np.asarray(vs_u_arr, dtype=float)
                vs_v_arr = np.asarray(vs_v_arr, dtype=float)
                vs_speed_arr = np.sqrt(vs_u_arr ** 2 + vs_v_arr ** 2)
                vs_vec_arr = np.stack([vs_u_arr, vs_v_arr], axis=-1)
        except Exception:
            try:
                cand_vs = simple_to_numpy(curr_vs)
                if isinstance(cand_vs, np.ndarray) and cand_vs.ndim == 3 and cand_vs.shape[-1] == 2:
                    vs_vec_arr = np.asarray(cand_vs, dtype=float)
                    vs_u_arr = vs_vec_arr[..., 0]
                    vs_v_arr = vs_vec_arr[..., 1]
                    vs_speed_arr = np.sqrt(vs_u_arr ** 2 + vs_v_arr ** 2)
            except Exception:
                pass
        vs_u_list.append(vs_u_arr)
        vs_v_list.append(vs_v_arr)
        vs_speed_list.append(vs_speed_arr)
        vs_vec_list.append(vs_vec_arr)
        # temperature
        t_arr = None
        try:
            tc = curr_t.at_centers() if hasattr(curr_t, 'at_centers') else curr_t
            t_field = tc.values if hasattr(tc, 'values') else tc
            t_arr = simple_to_numpy(t_field)
            if t_arr is not None:
                t_arr = np.asarray(t_arr, dtype=float)
                if t_arr.ndim == 0:
                    if DOMAIN is not None and isinstance(DOMAIN, dict) and 'x' in DOMAIN and 'y' in DOMAIN:
                        Nx = int(DOMAIN.get('x'))
                        Ny = int(DOMAIN.get('y'))
                        t_arr = np.full((Ny, Nx), float(t_arr), dtype=float)
                    else:
                        t_arr = np.asarray(float(t_arr), dtype=float)
        except Exception:
            t_arr = None
        if t_arr is None and t_init is not None:
            try:
                tin_field = t_init.at_centers() if hasattr(t_init, 'at_centers') else t_init
                tin = simple_to_numpy(tin_field.values if hasattr(tin_field, 'values') else tin_field)
                if tin is not None:
                    tin = np.asarray(tin, dtype=float)
                    if tin.ndim == 0:
                        if DOMAIN is not None and isinstance(DOMAIN, dict) and 'x' in DOMAIN and 'y' in DOMAIN:
                            Nx = int(DOMAIN.get('x'))
                            Ny = int(DOMAIN.get('y'))
                            t_arr = np.full((Ny, Nx), float(tin), dtype=float)
                        else:
                            t_arr = np.asarray(float(tin), dtype=float)
                    else:
                        t_arr = tin
            except Exception:
                t_arr = None
        t_list.append(t_arr)
        if i < steps:
            curr_vn, curr_vs, curr_p, curr_t, curr_L = helium.SFHelium_step(
                curr_vn, curr_vs, curr_p, curr_t, curr_L, dt=dt, **step_kwargs
            )
    return (
        stack_if_possible(u_list), stack_if_possible(v_list), stack_if_possible(speed_list), stack_if_possible(vec_list),
        stack_if_possible(vs_u_list), stack_if_possible(vs_v_list), stack_if_possible(vs_speed_list), stack_if_possible(vs_vec_list),
        stack_if_possible(t_list)
    )


def ensure_HW(arr: Any, Nx: int, Ny: int) -> Any:
    """Ensure array has shape (T, H, W) or (H, W) with (H=W?) transpose if needed."""
    if arr is None:
        return None
    try:
        a = np.asarray(arr)
    except Exception:
        return arr
    try:
        if a.ndim == 3:
            _, A, B = a.shape
            if A == Nx and B == Ny:
                return a.transpose(0, 2, 1)
            return a
        if a.ndim == 2:
            A, B = a.shape
            if A == Nx and B == Ny:
                return a.T
            return a
    except Exception:
        return a
    return a
