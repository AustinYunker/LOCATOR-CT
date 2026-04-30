import os

from dataclasses import dataclass
from typing import Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.ndimage import binary_fill_holes, uniform_filter1d
from skimage.morphology import binary_opening, closing, remove_small_objects, disk
from typing import Tuple, Optional
import atexit

from shmarray import ShmArray, shm_create_from_array, shm_create_empty, shm_attach

@dataclass(frozen=True)
class Stage2Config:
    # Morphology ops
    do_open: bool = False          # opening can fragment; default False
    do_close: bool = True          # closing often helps connect regions
    radius: int = 1                # structuring element radius (disk)

    # Optional per-slice hole fill (fills internal voids not connected to border)
    fill_holes_2d: bool = True

    # Optional per-slice small-object removal (2D CC)
    min_area: int = 0              # 0 disables

    # Z smoothing (applied after slice ops)
    z_smooth_win: int = 0          # 0 disables; e.g., 5

    # Connectivity for 2D CC: 1=4-conn, 2=8-conn
    connectivity_2d: int = 2

_WORK = {}

def _worker_init_stage2(in_meta: ShmArray, out_meta: ShmArray, cfg: Stage2Config, roi_meta: Optional[ShmArray]):
    Min, shm_in = shm_attach(in_meta)
    Mout, shm_out = shm_attach(out_meta)

    Roi = shm_roi = None
    if roi_meta is not None:
        Roi, shm_roi = shm_attach(roi_meta)

    # Precompute footprint once per process (cheap win)
    fp = None
    if cfg.radius and cfg.radius > 0 and (cfg.do_open or cfg.do_close):
        fp = disk(int(cfg.radius))

    _WORK.update({
        "Min": Min,
        "Mout": Mout,
        "Roi": Roi,
        "cfg": cfg,
        "fp": fp,
        "_handles": [shm_in, shm_out, shm_roi],
    })

    def _cleanup():
        for h in _WORK.get("_handles", []):
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass

    atexit.register(_cleanup)



def clean_mask_slice_2d(
    m2d: np.ndarray,
    do_open: bool,
    do_close: bool,
    radius: int,
    fill_holes_2d: bool,
    min_area: int,
    connectivity_2d: int,
) -> np.ndarray:
    m = m2d.astype(bool, copy=False)

    if radius and radius > 0 and (do_open or do_close):
        fp = disk(int(radius))
        if do_open:
            m = binary_opening(m, footprint=fp)
        if do_close:
            m = closing(m, footprint=fp)

    if fill_holes_2d and np.any(m):
        m = binary_fill_holes(m)

    if min_area and min_area > 0:
        m = remove_small_objects(m, min_size=int(min_area), connectivity=int(connectivity_2d))

    return m.astype(np.uint8)


def stage2_worker_clean_slice_shm(
    z: int,
    in_meta: ShmArray,     # uint8 mask (0/1)
    out_meta: ShmArray,    # uint8 mask (0/1)
    cfg: Stage2Config,
    roi_meta: Optional[ShmArray] = None,  # optional uint8/bool ROI; enforced after cleanup
) -> int:
    Min, shm_in = shm_attach(in_meta)
    Mout, shm_out = shm_attach(out_meta)

    Roi = None
    shm_roi = None
    if roi_meta is not None:
        Roi, shm_roi = shm_attach(roi_meta)

    try:
        m2d = Min[z].astype(bool, copy=False)

        m2d_c = clean_mask_slice_2d(
            m2d,
            do_open=cfg.do_open,
            do_close=cfg.do_close,
            radius=cfg.radius,
            fill_holes_2d=cfg.fill_holes_2d,
            min_area=cfg.min_area,
            connectivity_2d=cfg.connectivity_2d,
        )

        if Roi is not None:
            rz = Roi[z].astype(bool, copy=False)
            m2d_c = (m2d_c.astype(bool) & rz).astype(np.uint8)

        Mout[z] = m2d_c
        return int(z)
    finally:
        shm_in.close()
        shm_out.close()
        if shm_roi is not None:
            shm_roi.close()


def z_majority_smooth_u8(mask_u8: np.ndarray, win: int) -> np.ndarray:
    """
    Majority smoothing along z. win should be odd-ish (e.g., 5, 7).
    Keeps voxels that are present in >=50% of the window.
    """
    win = int(win)
    if win <= 1:
        return mask_u8
    m = mask_u8.astype(np.float32)
    m_s = uniform_filter1d(m, size=win, axis=0, mode="nearest")
    return (m_s >= 0.5).astype(np.uint8)


@dataclass(frozen=True)
class Stage2Output:
    mask_u8: np.ndarray   # (Zp,Yp,Xp) uint8
    mask: np.ndarray      # bool
    meta: dict
    run_time: float


def stage2_worker_clean_slice_idx(z: int) -> int:
    Min = _WORK["Min"]
    Mout = _WORK["Mout"]
    Roi = _WORK["Roi"]
    cfg: Stage2Config = _WORK["cfg"]
    fp = _WORK["fp"]

    m = Min[z].astype(bool, copy=False)

    # Morphology (reuse fp if available)
    if fp is not None:
        if cfg.do_open:
            m = binary_opening(m, footprint=fp)
        if cfg.do_close:
            m = closing(m, footprint=fp)

    if cfg.fill_holes_2d and np.any(m):
        m = binary_fill_holes(m)

    if cfg.min_area and cfg.min_area > 0:
        m = remove_small_objects(m, min_size=int(cfg.min_area), connectivity=int(cfg.connectivity_2d))

    if Roi is not None:
        m &= Roi[z].astype(bool, copy=False)

    Mout[z] = m.astype(np.uint8, copy=False)
    return int(z)

def stage2_parallel_morphology_shared_memory(
    mask_u8_in: np.ndarray,             # (Zp,Yp,Xp) uint8 (0/1)
    cfg: Stage2Config,
    roi_mask: Optional[np.ndarray] = None,   # optional (Zp,Yp,Xp) bool/uint8
    max_workers: Optional[int] = None,
) -> Stage2Output:
    mask_u8_in = np.asarray(mask_u8_in, dtype=np.uint8)
    Zp = mask_u8_in.shape[0]

    in_meta, shm_in = shm_create_from_array(mask_u8_in)
    out_meta, shm_out = shm_create_empty(mask_u8_in.shape, dtype=np.uint8)

    roi_meta = None
    shm_roi = None
    if roi_mask is not None:
        roi_u8 = roi_mask.astype(np.uint8, copy=False)
        roi_meta, shm_roi = shm_create_from_array(roi_u8)

    max_workers = max_workers or os.cpu_count() or 1

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init_stage2,
            initargs=(in_meta, out_meta, cfg, roi_meta),
        ) as ex:
            futs = [ex.submit(stage2_worker_clean_slice_idx, z) for z in range(Zp)]
            for fut in as_completed(futs):
                fut.result()

        # pull output back
        Msh, shm_out_att = shm_attach(out_meta)
        mask_u8 = np.array(Msh, copy=True)
        shm_out_att.close()

        # optional z smoothing (in parent; cheap)
        if cfg.z_smooth_win and cfg.z_smooth_win > 1:
            mask_u8 = z_majority_smooth_u8(mask_u8, win=cfg.z_smooth_win)

        meta = {
            "do_open": cfg.do_open,
            "do_close": cfg.do_close,
            "radius": cfg.radius,
            "fill_holes_2d": cfg.fill_holes_2d,
            "min_area": cfg.min_area,
            "connectivity_2d": cfg.connectivity_2d,
            "z_smooth_win": cfg.z_smooth_win,
        }

        return Stage2Output(mask_u8=mask_u8, mask=mask_u8.astype(bool), meta=meta, run_time=0.)

    finally:
        shm_in.close(); shm_in.unlink()
        shm_out.close(); shm_out.unlink()
        if shm_roi is not None:
            shm_roi.close(); shm_roi.unlink()
