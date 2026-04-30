import os

import numpy as np

from skimage.morphology import reconstruction, disk
from skimage.transform import downscale_local_mean
from scipy.ndimage import zoom

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import atexit

from shmarray import ShmArray, shm_create_from_array, shm_create_empty, shm_attach

# Global state in each worker process
_WORK = {}

def _worker_init_stage0(
    in_meta, norm_meta, bg_meta, res_meta,
    do_reconstruction,
    recon_footprint_radius,
    return_residual,
):
    Vp, shm_in = shm_attach(in_meta)
    Vn, shm_norm = shm_attach(norm_meta)

    Vbg = shm_bg = None
    if bg_meta is not None:
        Vbg, shm_bg = shm_attach(bg_meta)

    Vres = shm_res = None
    if res_meta is not None:
        Vres, shm_res = shm_attach(res_meta)

    _WORK.update({
        "Vp": Vp,
        "Vn": Vn,
        "Vbg": Vbg,
        "Vres": Vres,
        "do_reconstruction": do_reconstruction,
        "recon_footprint_radius": recon_footprint_radius,
        "return_residual": return_residual,
        "_shm_handles": [shm_in, shm_norm, shm_bg, shm_res],
    })


    def _cleanup():
        for h in _WORK.get("_shm_handles", []):
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass

    atexit.register(_cleanup)


def _downsample_local_mean(vol: np.ndarray, factors: Tuple[int, int, int]) -> np.ndarray:
    """
    Prefer local-mean downsampling (anti-aliasing-like) for robust thresholding.
    Requires skimage. Falls back to scipy zoom if needed.
    """
    fz, fy, fx = factors
    if fz < 1 or fy < 1 or fx < 1:
        raise ValueError(f"Downsample factors must be >=1, got {factors}.")

    if (fz, fy, fx) == (1, 1, 1):
        return vol
    
    if downscale_local_mean is not None:
        # skimage requires shape divisible by factors; crop minimally if needed
        Z, Y, X = vol.shape
        Zc = (Z // fz) * fz
        Yc = (Y // fy) * fy
        Xc = (X // fx) * fx
        vol_c = vol[:Zc, :Yc, :Xc]
        return downscale_local_mean(vol_c, factors)

    # fallback: scipy zoom (not a true mean downsample, but usable)
    if zoom is None:
        raise ImportError(
            "Neither skimage nor scipy is available for downsampling. "
            "Install scikit-image or scipy."
        )
    zoom_factors = (1.0 / fz, 1.0 / fy, 1.0 / fx)
    return zoom(vol, zoom_factors, order=1)  # linear interpolation

def _reconstruct_background_2d(norm2d: np.ndarray, footprint_radius: int = 1) -> np.ndarray:
    marker = norm2d.copy()
    vmin = float(np.min(norm2d))
    index = 10
    marker[index:-index, index:-index] = vmin
    fp = disk(int(footprint_radius))
    bg = reconstruction(marker, norm2d, footprint=fp)
    
    return bg.astype(np.float32, copy=False)

def stage0_worker_shm(
    z: int,
    in_meta: ShmArray,
    norm_meta: ShmArray,
    bg_meta: Optional[ShmArray],
    res_meta: Optional[ShmArray],
    do_reconstruction: bool,
    recon_footprint_radius: int,
    return_residual: bool,
) -> Tuple[int, float, float]:
    """
    Returns (z, lo, hi) for logging; writes outputs into shared arrays.
    """
    Vp, shm_in = shm_attach(in_meta)
    Vn, shm_norm = shm_attach(norm_meta)
    Vbg = None; shm_bg = None
    Vres = None; shm_res = None

    if bg_meta is not None:
        Vbg, shm_bg = shm_attach(bg_meta)
    if res_meta is not None:
        Vres, shm_res = shm_attach(res_meta)

    try:

        norm = Vp[z]
        Vn[z] = norm

        if do_reconstruction:
            bg = _reconstruct_background_2d(norm, footprint_radius=recon_footprint_radius)
            if Vbg is not None:
                Vbg[z] = bg
            if return_residual and Vres is not None:
                res = np.clip(norm - bg, 0.0, None).astype(np.float32, copy=False)
                #res = norm - bg
                Vres[z] = res.astype(np.float32, copy=False)

        return z
    finally:
        # IMPORTANT: close attachments in worker
        shm_in.close()
        shm_norm.close()
        if shm_bg is not None: shm_bg.close()
        if shm_res is not None: shm_res.close()

def stage0_worker_shm_chunk(
    z0: int,
    z1: int,
    in_meta: ShmArray,
    norm_meta: ShmArray,
    bg_meta: Optional[ShmArray],
    res_meta: Optional[ShmArray],
    do_reconstruction: bool,
    recon_footprint_radius: int,
    return_residual: bool,
):
    Vp, shm_in = shm_attach(in_meta)
    Vn, shm_norm = shm_attach(norm_meta)
    Vbg = None; shm_bg = None
    Vres = None; shm_res = None
    if bg_meta is not None:
        Vbg, shm_bg = shm_attach(bg_meta)
    if res_meta is not None:
        Vres, shm_res = shm_attach(res_meta)

    try:
        for z in range(z0, z1):
            norm = Vp[z]

            Vn[z] = norm

            if do_reconstruction:
                bg = _reconstruct_background_2d(norm, footprint_radius=recon_footprint_radius)
                if Vbg is not None:
                    Vbg[z] = bg
                if return_residual and Vres is not None:
                    Vres[z] = np.clip(norm - bg, 0.0, None).astype(np.float32, copy=False)

        return (z0, z1)
    finally:
        shm_in.close(); shm_norm.close()
        if shm_bg is not None: shm_bg.close()
        if shm_res is not None: shm_res.close()

def stage0_worker_idx(z: int):
    Vp   = _WORK["Vp"]
    Vn   = _WORK["Vn"]
    Vbg  = _WORK["Vbg"]
    Vres = _WORK["Vres"]

    do_recon    = _WORK["do_reconstruction"]
    radius      = _WORK["recon_footprint_radius"]
    do_resid    = _WORK["return_residual"]

    norm = Vp[z]   # already float32

    Vn[z] = norm

    if do_recon:
        bg = _reconstruct_background_2d(
            norm,
            footprint_radius=radius
        )

        if Vbg is not None:
            Vbg[z] = bg

        if do_resid and Vres is not None:
            Vres[z] = np.clip(norm - bg, 0.0, None)


## Config and Code to run

@dataclass(frozen=True)
class Stage0Config:
    factors: Tuple[int,int,int] = (2,4,4)
    crop_to_divisible: bool = True

    p_low: float = 1.0
    p_high: float = 99.0

    do_reconstruction: bool = True
    recon_footprint_radius: int = 1
    return_residual: bool = True
    do_circle_crop: bool = True

    max_workers: Optional[int] = None

@dataclass(frozen=True)
class Stage0Output:
    V_proxy: np.ndarray
    V_norm: np.ndarray
    V_bg: Optional[np.ndarray]
    V_residual: Optional[np.ndarray]
    per_slice_stats: Dict[int, Dict[str, float]]
    factors: Tuple[int,int,int]
    original_shape: Tuple[int,int,int]
    run_time: float


def stage0_parallel_preprocess_shared_memory(V: np.ndarray, cfg: Stage0Config) -> Stage0Output:
    
    # 1) Downsample (serial)
    Vp = _downsample_local_mean(V, cfg.factors)

    # 3) clip and rescale volume-wise
    vmin = np.percentile(Vp, cfg.p_low)
    vmax = np.percentile(Vp, cfg.p_high)
    
    Vp = np.clip(Vp, vmin, vmax)
    Vp = ((Vp - vmin) / (vmax - vmin)).astype(np.float32, copy=False)

    if cfg.do_circle_crop:
        # 3) Compute circle crop
        l_x, l_y = Vp.shape[1], Vp.shape[2]
        X, Y = np.ogrid[:l_x, :l_y]
        outer_disk_mask = ((X - l_x / 2) ** 2 + (Y - l_y / 2) ** 2 > (l_x / 2) ** 2).astype(bool)
        Vp[:, outer_disk_mask] = 0

    Zp, Yp, Xp = Vp.shape

    # 2) Create shared buffers
    in_meta, shm_in = shm_create_from_array(Vp)
    norm_meta, shm_norm = shm_create_empty(Vp.shape, dtype=np.float32)

    bg_meta = res_meta = None
    shm_bg = shm_res = None

    if cfg.do_reconstruction:
        bg_meta, shm_bg = shm_create_empty(Vp.shape, dtype=np.float32)
        if cfg.return_residual:
            res_meta, shm_res = shm_create_empty(Vp.shape, dtype=np.float32)

    stats: Dict[int, Dict[str, float]] = {}

    max_workers = cfg.max_workers or os.cpu_count() or 1

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init_stage0,
            initargs=(
                in_meta, norm_meta, bg_meta, res_meta,
                cfg.do_reconstruction,
                cfg.recon_footprint_radius,
                cfg.return_residual,
            ),
        ) as ex:
            

            futs = [ex.submit(stage0_worker_idx, z) for z in range(Zp)]

            for f in as_completed(futs): f.result()

        # 3) Attach in parent to read outputs into normal numpy arrays
        Vn, shm_norm_att = shm_attach(norm_meta)
        Vbg_arr = None
        Vres_arr = None

        Vn_out = np.array(Vn, copy=True)  # copy out so we can free shared memory

        shm_norm_att.close()

        if bg_meta is not None:
            Vbg_sh, shm_bg_att = shm_attach(bg_meta)
            Vbg_arr = np.array(Vbg_sh, copy=True)
            shm_bg_att.close()

        if res_meta is not None:
            Vres_sh, shm_res_att = shm_attach(res_meta)
            Vres_arr = np.array(Vres_sh, copy=True)
            shm_res_att.close()

        return Stage0Output(
            V_proxy=Vp,
            V_norm=Vn_out,
            V_bg=Vbg_arr,
            V_residual=Vres_arr,
            per_slice_stats=stats,
            factors=cfg.factors,
            original_shape=tuple(V.shape),
            run_time=0.0
        )

    finally:
        # 4) Always cleanup shared memory (parent owns unlink)
        shm_in.close(); shm_in.unlink()
        shm_norm.close(); shm_norm.unlink()
        if shm_bg is not None:
            shm_bg.close(); shm_bg.unlink()
        if shm_res is not None:
            shm_res.close(); shm_res.unlink()

