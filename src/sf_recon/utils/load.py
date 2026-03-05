"""Shared CSV loading utilities for task initializations."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import csv
import numpy as np


def _find_col(header: Iterable[str], substrs: Iterable[str]):
    for i, h in enumerate(header):
        lh = h.strip().lower()
        for s in substrs:
            if s in lh:
                return i
    return None


def _require_cols(missing_cols: List[str]):
    if missing_cols:
        raise ValueError(f"Missing CSV columns: {', '.join(missing_cols)}")


def load_csv_to_grids_cf(path: str, Lx: float, Ly: float, Nx: int, Ny: int):
    """Load counterflow CSV and map to (Ny, Nx) grids. Applies point:1 shift by half length."""
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # CSV columns are swapped: un:0<->un:1, us:0<->us:1, point:0<->point:1
        i_un0 = _find_col(header, ['un:1'])
        i_un1 = _find_col(header, ['un:0'])
        i_us0 = _find_col(header, ['us:1'])
        i_us1 = _find_col(header, ['us:0'])
        i_p = _find_col(header, ['p', 'pressure'])
        i_T = _find_col(header, ['t', 'temperature'])
        i_L = _find_col(header, ['l'])
        i_x = _find_col(header, ['points:1'])
        i_y = _find_col(header, ['points:0'])
        missing_cols = []
        if i_x is None:
            missing_cols.append('Points:1')
        if i_y is None:
            missing_cols.append('Points:0')
        if i_un0 is None:
            missing_cols.append('Un:1 (mapped to un0)')
        if i_un1 is None:
            missing_cols.append('Un:0 (mapped to un1)')
        if i_us0 is None:
            missing_cols.append('Us:1 (mapped to us0)')
        if i_us1 is None:
            missing_cols.append('Us:0 (mapped to us1)')
        _require_cols(missing_cols)
        sums = {k: np.zeros((Ny, Nx), dtype=float) for k in ('un0','un1','us0','us1','p','T','L')}
        counts = np.zeros((Ny, Nx), dtype=int)
        for row in reader:
            try:
                x_raw = float(row[i_x])
                y = float(row[i_y])
                x = x_raw + 0.5 * Lx
            except Exception:
                continue
            ix = int(round((x / Lx) * (Nx - 1)))
            iy = int(round((y / Ly) * (Ny - 1)))
            ix = max(0, min(Nx - 1, ix))
            iy = max(0, min(Ny - 1, iy))
            counts[iy, ix] += 1
            def safe_add(idx_col, key):
                if idx_col is None:
                    return
                try:
                    val = float(row[idx_col])
                    sums[key][iy, ix] += val
                except Exception:
                    pass
            safe_add(i_un0, 'un0')
            safe_add(i_un1, 'un1')
            safe_add(i_us0, 'us0')
            safe_add(i_us1, 'us1')
            safe_add(i_p, 'p')
            safe_add(i_T, 'T')
            safe_add(i_L, 'L')
    count_total = int(counts.sum())
    if count_total == 0:
        raise ValueError('No CSV points mapped to grid')
    for k in sums:
        mask = counts > 0
        if np.any(mask):
            arr = np.zeros_like(sums[k])
            arr[mask] = sums[k][mask] / counts[mask]
        else:
            arr = np.zeros_like(sums[k])
        sums[k] = arr
    return sums['un0'], sums['un1'], sums['us0'], sums['us1'], sums['p'], sums['T'], sums['L'], count_total


def load_csv_to_grids_cyl(path: str, Lx: float, Ly: float, Nx: int, Ny: int, *, y_center_csv: float = 0.100):
    """Load cylinder CSV and map to (Ny, Nx) grids. Applies point:1 shift by half length and aligns point:0 center."""
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # CSV columns are swapped: un:0<->un:1, us:0<->us:1, point:0<->point:1
        i_un0 = _find_col(header, ['un:1'])
        i_un1 = _find_col(header, ['un:0'])
        i_us0 = _find_col(header, ['us:1'])
        i_us1 = _find_col(header, ['us:0'])
        i_p = _find_col(header, ['p', 'pressure'])
        i_T = _find_col(header, ['t', 'temperature'])
        i_L = _find_col(header, ['l'])
        i_x = _find_col(header, ['points:1'])
        i_y = _find_col(header, ['points:0'])
        missing_cols = []
        if i_x is None:
            missing_cols.append('Points:1')
        if i_y is None:
            missing_cols.append('Points:0')
        if i_un0 is None:
            missing_cols.append('Un:1 (mapped to un0)')
        if i_un1 is None:
            missing_cols.append('Un:0 (mapped to un1)')
        if i_us0 is None:
            missing_cols.append('Us:1 (mapped to us0)')
        if i_us1 is None:
            missing_cols.append('Us:0 (mapped to us1)')
        _require_cols(missing_cols)
        sums = {k: np.zeros((Ny, Nx), dtype=float) for k in ('un0','un1','us0','us1','p','T','L')}
        counts = np.zeros((Ny, Nx), dtype=int)
        for row in reader:
            try:
                x_raw = float(row[i_x])
                y_raw = float(row[i_y])
                x = x_raw + 0.5 * Lx
                y = y_raw + (0.5 * Ly - y_center_csv)
            except Exception:
                continue
            ix = int(round((x / Lx) * (Nx - 1)))
            iy = int(round((y / Ly) * (Ny - 1)))
            ix = max(0, min(Nx - 1, ix))
            iy = max(0, min(Ny - 1, iy))
            counts[iy, ix] += 1
            def safe_add(idx_col, key):
                if idx_col is None:
                    return
                try:
                    val = float(row[idx_col])
                    sums[key][iy, ix] += val
                except Exception:
                    pass
            safe_add(i_un0, 'un0')
            safe_add(i_un1, 'un1')
            safe_add(i_us0, 'us0')
            safe_add(i_us1, 'us1')
            safe_add(i_p, 'p')
            safe_add(i_T, 'T')
            safe_add(i_L, 'L')
    count_total = int(counts.sum())
    if count_total == 0:
        raise ValueError('No CSV points mapped to grid')
    for k in sums:
        mask = counts > 0
        if np.any(mask):
            arr = np.zeros_like(sums[k])
            arr[mask] = sums[k][mask] / counts[mask]
        else:
            arr = np.zeros_like(sums[k])
        sums[k] = arr
    return sums['un0'], sums['un1'], sums['us0'], sums['us1'], sums['p'], sums['T'], sums['L'], count_total
