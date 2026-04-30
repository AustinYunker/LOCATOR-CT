import os

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, Any

from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture
from concurrent.futures import ProcessPoolExecutor, as_completed
import atexit

from shmarray import ShmArray, shm_create_from_array, shm_create_empty, shm_attach


_WORK = {}
def _worker_init_stage1(src_meta, out_meta, thr, air_is_low, roi_meta=None):
    V, shm_src = shm_attach(src_meta)
    M, shm_out = shm_attach(out_meta)

    Roi = shm_roi = None
    if roi_meta is not None:
        Roi, shm_roi = shm_attach(roi_meta)

    _WORK.update({
        "V": V,
        "M": M,
        "Roi": Roi,
        "thr": float(thr),
        "air_is_low": bool(air_is_low),
        "_handles": [shm_src, shm_out, shm_roi],
    })

    def _cleanup():
        for h in _WORK.get("_handles", []):
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass

    atexit.register(_cleanup)

@dataclass(frozen=True)
class Stage1Config:
    method: Literal["otsu", "gmm2"] = "otsu"

    # Which proxy signal to threshold:
    # - "norm": normalized proxy intensity
    # - "residual": background-subtracted residual (often better for bright-feature extraction)
    source: Literal["norm", "residual"] = "norm"

    # ROI restriction (recommended for tube scans if you already have a mask)
    # If provided, threshold is computed only over ROI voxels.
    use_roi_for_threshold: bool = True

    # Subsampling for threshold estimation (speeds up and stabilizes for huge volumes)
    max_samples: int = 2_000_000
    seed: int = 0

    # Interpret threshold as separating air/non-air:
    # If air is darker: non_air = I > thr  (most absorption CT)
    # If air is brighter: non_air = I < thr
    air_is_low: bool = True  # typical: air low, material higher

    # Optional clamp: ignore extreme tails when estimating thresholds
    clip_percentiles: Tuple[float, float] = (0.5, 99.5)


def _sample_values_for_threshold(
    V: np.ndarray,
    roi: Optional[np.ndarray],
    max_samples: int,
    seed: int,
    clip_percentiles: Tuple[float, float],
) -> np.ndarray:
    """
    Collect values for threshold estimation with optional ROI restriction and random subsampling.
    """
    if roi is not None:
        vals = V[roi.astype(bool)]
    else:
        vals = V.reshape(-1)

    vals = vals.astype(np.float32, copy=False)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return vals

    lo_p, hi_p = clip_percentiles
    lo = np.percentile(vals, lo_p)
    hi = np.percentile(vals, hi_p)
    vals = vals[(vals >= lo) & (vals <= hi)]

    if vals.size > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(vals.size, size=max_samples, replace=False)
        vals = vals[idx]

    return vals


def compute_global_threshold(
    V: np.ndarray,
    cfg: Stage1Config,
    roi: Optional[np.ndarray] = None,
) -> float:
    """
    Compute a single scalar threshold for the whole proxy volume (or ROI).
    Methods: Otsu, 2-component GMM.
    """

    # threshold using full volume
    if cfg.method == "otsu":
        thr = float(threshold_otsu(V))
        return thr

    # threshold using sample of values
    if cfg.method == "gmm2":

        vals = _sample_values_for_threshold(
            V=V,
            roi=roi if cfg.use_roi_for_threshold else None,
            max_samples=cfg.max_samples,
            seed=cfg.seed,
            clip_percentiles=cfg.clip_percentiles,
        )

        if GaussianMixture is None:
            raise ImportError("scikit-learn is required for GMM-2 thresholding (GaussianMixture).")

        x = vals.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=cfg.seed)
        gmm.fit(x)

        means = gmm.means_.reshape(-1)
        vars_ = gmm.covariances_.reshape(-1)  # since 1D
        weights = gmm.weights_.reshape(-1)

        # Solve for intersection of two 1D Gaussians:
        # w1*N(m1,s1^2) = w2*N(m2,s2^2)
        # -> quadratic in x.
        m1, m2 = means[0], means[1]
        v1, v2 = vars_[0], vars_[1]
        w1, w2 = weights[0], weights[1]

        # Guard against degenerate variances
        v1 = max(v1, 1e-8)
        v2 = max(v2, 1e-8)

        # Coeffs for ax^2 + bx + c = 0
        a = (1.0 / v1) - (1.0 / v2)
        b = (-2.0 * m1 / v1) + (2.0 * m2 / v2)
        c = (m1 * m1 / v1) - (m2 * m2 / v2) + 2.0 * np.log((w2 * np.sqrt(v1)) / (w1 * np.sqrt(v2)))

        if abs(a) < 1e-12:
            # Similar variances: linear solution
            thr = -c / b
        else:
            roots = np.roots([a, b, c])
            roots = np.real(roots[np.isreal(roots)])

            # Choose root between the means if possible
            lo, hi = (min(m1, m2), max(m1, m2))
            between = roots[(roots >= lo) & (roots <= hi)]
            thr = float(between[0]) if between.size > 0 else float(roots[0])

        return float(thr)

    raise ValueError(f"Unknown method: {cfg.method}")


def stage1_worker_apply_threshold_shm(
    z: int,
    src_meta: ShmArray,      # float32 proxy volume (norm or residual)
    out_meta: ShmArray,      # uint8 mask (0/1)
    thr: float,
    air_is_low: bool,
    roi_meta: Optional[ShmArray] = None,  # optional uint8/bool ROI per slice
) -> int:
    """
    Writes mask[z] = 1 for non-air voxels within ROI (if provided), else 0.
    Returns z for bookkeeping.
    """
    V, shm_src = shm_attach(src_meta)
    M, shm_out = shm_attach(out_meta)

    Roi = None
    shm_roi = None
    if roi_meta is not None:
        Roi, shm_roi = shm_attach(roi_meta)

    try:
        s = V[z]  # (Y,X)
        if Roi is not None:
            rz = Roi[z].astype(bool, copy=False)
        else:
            rz = None

        if air_is_low:
            non_air = (s > thr)
        else:
            non_air = (s < thr)

        if rz is not None:
            non_air = non_air & rz

        M[z] = non_air.astype(np.uint8)
        return int(z)
    finally:
        shm_src.close()
        shm_out.close()
        if shm_roi is not None:
            shm_roi.close()

def stage1_worker_apply_threshold_idx(z: int) -> int:
    V   = _WORK["V"]
    M   = _WORK["M"]
    Roi = _WORK["Roi"]
    thr = _WORK["thr"]
    air_is_low = _WORK["air_is_low"]

    s = V[z]  # (Y, X)

    if air_is_low:
        non_air = (s > thr)
    else:
        non_air = (s < thr)

    if Roi is not None:
        non_air &= Roi[z].astype(bool, copy=False)

    M[z] = non_air.astype(np.uint8, copy=False)
    return int(z)



@dataclass(frozen=True)
class Stage1Output:
    threshold: float
    method: str
    source: str
    mask: np.ndarray          # (Zp,Yp,Xp) bool
    mask_u8: np.ndarray       # (Zp,Yp,Xp) uint8
    meta: Dict[str, Any]
    run_time: float


def stage1_parallel_global_threshold_shared_memory(
    V_norm: np.ndarray,
    V_residual: Optional[np.ndarray],
    cfg: Stage1Config,
    roi_mask: Optional[np.ndarray] = None,   # (Zp,Yp,Xp) bool/uint8
    max_workers: Optional[int] = None,
) -> Stage1Output:
    """
    1) Choose source volume (norm or residual)
    2) Compute global threshold (Otsu or GMM-2) over (optional) ROI
    3) Apply threshold slice-by-slice in parallel using shared memory
    """
    if cfg.source == "norm":
        Vsrc = V_norm
    elif cfg.source == "residual":
        if V_residual is None:
            raise ValueError("cfg.source='residual' requested but V_residual is None.")
        Vsrc = V_residual
    else:
        raise ValueError(f"Unknown cfg.source: {cfg.source}")

    Vsrc = np.asarray(Vsrc)
    if Vsrc.dtype != np.float32:
        Vsrc = Vsrc.astype(np.float32, copy=False)
    
    # Compute global threshold on parent (serial)
    thr = compute_global_threshold(Vsrc, cfg, roi=roi_mask)

    # Shared memory: source + output + optional ROI
    src_meta, shm_src = shm_create_from_array(Vsrc)
    out_meta, shm_out = shm_create_empty(Vsrc.shape, dtype=np.uint8)

    roi_meta = None
    shm_roi = None
    if roi_mask is not None:
        roi_u8 = roi_mask.astype(np.uint8, copy=False)
        roi_meta, shm_roi = shm_create_from_array(roi_u8)

    Zp = Vsrc.shape[0]
    max_workers = max_workers or os.cpu_count() or 1

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init_stage1,
            initargs=(src_meta, out_meta, thr, cfg.air_is_low, roi_meta),
        ) as ex:
            futs = [ex.submit(stage1_worker_apply_threshold_idx, z) for z in range(Zp)]
            for fut in as_completed(futs):
                fut.result()

        # Read output mask back into normal numpy
        Msh, shm_out_att = shm_attach(out_meta)
        mask_u8 = np.array(Msh, copy=True)
        shm_out_att.close()

        mask = mask_u8.astype(bool)

        meta = {
            "threshold": float(thr),
            "method": cfg.method,
            "source": cfg.source,
            "air_is_low": bool(cfg.air_is_low),
            "use_roi_for_threshold": bool(cfg.use_roi_for_threshold),
            "clip_percentiles": tuple(cfg.clip_percentiles),
            "max_samples": int(cfg.max_samples),
        }

        return Stage1Output(
            threshold=float(thr),
            method=cfg.method,
            source=cfg.source,
            mask=mask,
            mask_u8=mask_u8,
            meta=meta,
            run_time=0.
        )

    finally:
        # cleanup shm
        shm_src.close(); shm_src.unlink()
        shm_out.close(); shm_out.unlink()
        if shm_roi is not None:
            shm_roi.close(); shm_roi.unlink()

