"""Shared saving utilities for tasks and notebooks."""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np

from sf_recon.physics import helium, normal


def simple_to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    # Prefer explicit extraction paths to avoid implicit Phi-ML -> NumPy
    # conversion warnings about undefined dimension order.
    try:
        if hasattr(x, 'native') and hasattr(x, 'shape'):
            return np.asarray(x.native(x.shape))
    except Exception:
        pass
    try:
        if hasattr(x, 'numpy'):
            return np.asarray(x.numpy())
    except Exception:
        pass
    try:
        return np.array(x)
    except Exception:
        try:
            if hasattr(x, 'values'):
                v = x.values
                try:
                    if hasattr(v, 'native') and hasattr(v, 'shape'):
                        return np.asarray(v.native(v.shape))
                except Exception:
                    pass
                try:
                    if hasattr(v, 'numpy'):
                        return np.asarray(v.numpy())
                except Exception:
                    pass
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


def _normalize_hw(arr: Any, Nx: int, Ny: int) -> Optional[np.ndarray]:
    """Normalize a 2D spatial array to (Ny, Nx)."""
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=float)
    except Exception:
        return None
    if a.ndim != 2:
        return None
    if a.shape == (Ny, Nx):
        return a
    if a.shape == (Nx, Ny):
        return a.T
    if a.shape == (Ny - 2, Nx - 2):
        return a
    if a.shape == (Nx - 2, Ny - 2):
        return a.T
    return None


def _normalize_vec_hw(arr: Any, Nx: int, Ny: int) -> Optional[np.ndarray]:
    """Normalize a 3D vector array to (Ny, Nx, 2)."""
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=float)
    except Exception:
        return None
    if a.ndim != 3:
        return None
    if a.shape == (Ny, Nx, 2):
        return a
    if a.shape == (Nx, Ny, 2):
        return a.transpose(1, 0, 2)
    if a.shape == (Ny - 2, Nx - 2, 2):
        return a
    if a.shape == (Nx - 2, Ny - 2, 2):
        return a.transpose(1, 0, 2)
    if a.shape == (2, Ny, Nx):
        return a.transpose(1, 2, 0)
    if a.shape == (2, Nx, Ny):
        return a.transpose(2, 1, 0)
    if a.shape == (2, Ny - 2, Nx - 2):
        return a.transpose(1, 2, 0)
    if a.shape == (2, Nx - 2, Ny - 2):
        return a.transpose(2, 1, 0)
    return None


def extract_centered_vector_components(field_obj: Any, Nx: int, Ny: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract centered vector field as (u, v, vec) with vec shaped (Ny, Nx, 2)."""
    base = field_obj.at_centers() if hasattr(field_obj, 'at_centers') else field_obj

    try:
        values = base.values if hasattr(base, 'values') else base
        vec = _normalize_vec_hw(simple_to_numpy(values), Nx, Ny)
        if vec is not None:
            return vec[..., 0], vec[..., 1], vec
    except Exception:
        pass

    try:
        ux = base.vector['x']
        uy = base.vector['y']
        u = _normalize_hw(simple_to_numpy(ux.values if hasattr(ux, 'values') else ux), Nx, Ny)
        v = _normalize_hw(simple_to_numpy(uy.values if hasattr(uy, 'values') else uy), Nx, Ny)
        if u is not None and v is not None:
            return u, v, np.stack([u, v], axis=-1)
    except Exception:
        pass

    try:
        raw = simple_to_numpy(base)
        vec = _normalize_vec_hw(raw, Nx, Ny)
        if vec is not None:
            return vec[..., 0], vec[..., 1], vec
    except Exception:
        pass

    return None, None, None


def extract_centered_scalar_hw(field_obj: Any, Nx: int, Ny: int) -> Optional[np.ndarray]:
    """Extract centered scalar field to shape (Ny, Nx), broadcasting scalars when needed."""
    base = field_obj.at_centers() if hasattr(field_obj, 'at_centers') else field_obj
    try:
        values = base.values if hasattr(base, 'values') else base
        arr = simple_to_numpy(values)
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 0:
            return np.full((Ny, Nx), float(arr), dtype=float)
        return _normalize_hw(arr, Nx, Ny)
    except Exception:
        return None


def prepare_save_array(x: Any, Nx: int, Ny: int) -> Any:
    """Normalize arrays before saving while preserving valid interior-grid shapes."""
    arr = simple_to_numpy(x)
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=float)
    except Exception:
        return arr
    valid_hw = {(Ny, Nx), (Ny - 2, Nx - 2)}
    valid_wh = {(Nx, Ny), (Nx - 2, Ny - 2)}
    if a.ndim == 2:
        if a.shape in valid_hw:
            return a
        if a.shape in valid_wh:
            return a.T
        return a
    if a.ndim == 3:
        if a.shape[:2] in valid_hw:
            return a
        if a.shape[:2] in valid_wh:
            return a.transpose(1, 0, 2)
        if a.shape[1:] in valid_hw:
            return a
        if a.shape[1:] in valid_wh:
            return a.transpose(0, 2, 1)
        return a
    if a.ndim == 4:
        if a.shape[1:3] in valid_hw:
            return a
        if a.shape[1:3] in valid_wh:
            return a.transpose(0, 2, 1, 3)
        return a
    return ensure_HW(a, Nx, Ny)


def centered_field_finite_ratio(field_obj: Any, Nx: int, Ny: int) -> float:
    """Return the fraction of finite entries in a centered vector field."""
    u, v, _vec = extract_centered_vector_components(field_obj, Nx, Ny)
    if u is None or v is None:
        return 0.0
    joined = np.concatenate([np.asarray(u, dtype=float).reshape(-1), np.asarray(v, dtype=float).reshape(-1)], axis=0)
    if joined.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(joined)))


def centered_field_is_finite(field_obj: Any, Nx: int, Ny: int) -> bool:
    """Return whether all extracted entries in a centered vector field are finite."""
    u, v, _vec = extract_centered_vector_components(field_obj, Nx, Ny)
    if u is None or v is None:
        return False
    return bool(np.isfinite(np.asarray(u, dtype=float)).all() and np.isfinite(np.asarray(v, dtype=float)).all())


def stack_series_with_common_shape(items: Sequence[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """Stack a sequence by cropping all valid entries to the smallest common spatial shape."""
    arrays = []
    for item in items:
        if item is None:
            arrays.append(None)
            continue
        try:
            arrays.append(np.asarray(item, dtype=float))
        except Exception:
            arrays.append(None)
    valid = [a for a in arrays if a is not None and a.size > 0 and np.isfinite(a).any()]
    if not valid:
        return None
    ndim = valid[0].ndim
    if ndim == 2:
        target_shape = (min(a.shape[0] for a in valid), min(a.shape[1] for a in valid))
    elif ndim == 3:
        target_shape = (min(a.shape[0] for a in valid), min(a.shape[1] for a in valid), valid[0].shape[2])
    else:
        ref_shape = valid[0].shape
        stacked = []
        for arr in arrays:
            if arr is None or arr.shape != ref_shape:
                stacked.append(np.full(ref_shape, np.nan, dtype=float))
            else:
                stacked.append(arr)
        return np.stack(stacked, axis=0)

    def _crop(arr: Optional[np.ndarray]) -> np.ndarray:
        if arr is None:
            return np.full(target_shape, np.nan, dtype=float)
        if ndim == 2:
            return arr[:target_shape[0], :target_shape[1]]
        return arr[:target_shape[0], :target_shape[1], :target_shape[2]]

    return np.stack([_crop(arr) for arr in arrays], axis=0)


def extract_snapshot_series(field_snapshots: Sequence[Tuple[Any, Any, Any, Any, Any]], Nx: int, Ny: int):
    """Convert mean-field snapshots into stacked NumPy series for saving and plotting."""
    def _valid_hw(arr: Optional[np.ndarray]) -> bool:
        if arr is None:
            return False
        a = np.asarray(arr, dtype=float)
        return bool(a.ndim == 2 and a.shape in {(Ny, Nx), (Ny - 2, Nx - 2)} and np.isfinite(a).all())

    def _fallback_hw(arr: Optional[np.ndarray], fallback: Optional[np.ndarray]) -> np.ndarray:
        if _valid_hw(arr):
            return np.asarray(arr, dtype=float)
        if fallback is not None:
            return np.asarray(fallback, dtype=float).copy()
        return np.full((Ny, Nx), np.nan, dtype=float)

    vn_u_list, vn_v_list, vn_speed_list, vn_vec_list = [], [], [], []
    vs_u_list, vs_v_list, vs_speed_list, vs_vec_list = [], [], [], []
    t_list = []
    last_vn_u = last_vn_v = last_vn_speed = last_t = None
    last_vs_u = last_vs_v = last_vs_speed = None

    for vn, vs, _p, t, _L in field_snapshots:
        vn_u_raw, vn_v_raw, vn_vec_raw = extract_centered_vector_components(vn, Nx, Ny)
        vn_u = _fallback_hw(vn_u_raw, last_vn_u)
        vn_v = _fallback_hw(vn_v_raw, last_vn_v)
        vn_speed_raw = np.sqrt(vn_u ** 2 + vn_v ** 2) if (_valid_hw(vn_u) and _valid_hw(vn_v)) else None
        vn_speed = _fallback_hw(vn_speed_raw, last_vn_speed)
        vn_vec = np.stack([vn_u, vn_v], axis=-1)
        if _valid_hw(vn_u):
            last_vn_u = vn_u.copy()
        if _valid_hw(vn_v):
            last_vn_v = vn_v.copy()
        if _valid_hw(vn_speed):
            last_vn_speed = vn_speed.copy()
        vn_u_list.append(vn_u)
        vn_v_list.append(vn_v)
        vn_speed_list.append(vn_speed)
        vn_vec_list.append(vn_vec)

        vs_u_raw, vs_v_raw, vs_vec_raw = extract_centered_vector_components(vs, Nx, Ny)
        vs_u = _fallback_hw(vs_u_raw, last_vs_u)
        vs_v = _fallback_hw(vs_v_raw, last_vs_v)
        vs_speed_raw = np.sqrt(vs_u ** 2 + vs_v ** 2) if (_valid_hw(vs_u) and _valid_hw(vs_v)) else None
        vs_speed = _fallback_hw(vs_speed_raw, last_vs_speed)
        vs_vec = np.stack([vs_u, vs_v], axis=-1)
        if _valid_hw(vs_u):
            last_vs_u = vs_u.copy()
        if _valid_hw(vs_v):
            last_vs_v = vs_v.copy()
        if _valid_hw(vs_speed):
            last_vs_speed = vs_speed.copy()
        vs_u_list.append(vs_u)
        vs_v_list.append(vs_v)
        vs_speed_list.append(vs_speed)
        vs_vec_list.append(vs_vec)

        t_arr = _fallback_hw(extract_centered_scalar_hw(t, Nx, Ny), last_t)
        if _valid_hw(t_arr):
            last_t = t_arr.copy()
        t_list.append(t_arr)

    return (
        stack_series_with_common_shape(vn_u_list),
        stack_series_with_common_shape(vn_v_list),
        stack_series_with_common_shape(vn_speed_list),
        stack_series_with_common_shape(vn_vec_list),
        stack_series_with_common_shape(vs_u_list),
        stack_series_with_common_shape(vs_v_list),
        stack_series_with_common_shape(vs_speed_list),
        stack_series_with_common_shape(vs_vec_list),
        stack_series_with_common_shape(t_list),
    )


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
