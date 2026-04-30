from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np

from preprocess import Stage0Config, stage0_parallel_preprocess_shared_memory
from threshold import Stage1Config, stage1_parallel_global_threshold_shared_memory
from morph_cleanup import Stage2Config, stage2_parallel_morphology_shared_memory
from union_bbox import Stage3Config, stage3_parallel_bbox_shared_memory, BBox3D
from refine_bbox import Stage3bConfig, stage3b_refine_bbox_largest_component


def map_bbox_proxy_to_full(
    bbox_p: BBox3D,
    original_shape: Tuple[int,int,int],
    factors: Tuple[int,int,int],
) -> BBox3D:
    fz, fy, fx = factors
    Z, Y, X = original_shape

    b = BBox3D(
        z0=bbox_p.z0 * fz,
        z1=bbox_p.z1 * fz,
        y0=bbox_p.y0 * fy,
        y1=bbox_p.y1 * fy,
        x0=bbox_p.x0 * fx,
        x1=bbox_p.x1 * fx,
    ).clamp(original_shape)

    return b

@dataclass(frozen=True)
class PipelineConfig:
    # Stage 0
    stage0: Stage0Config = Stage0Config()

    # Stage 1 (global threshold proxy)
    stage1: Stage1Config = Stage1Config(method="otsu", source="norm", use_roi_for_threshold=False, air_is_low=True)

    # Stage 2 (proxy morphology cleanup)
    stage2: Stage2Config = Stage2Config(do_open=False, do_close=True, radius=1, fill_holes_2d=True, min_area=0, z_smooth_win=0)

    # Stage 3 (bbox reduction)
    stage3: Stage3Config = Stage3Config(margin_zyx=(0, 0, 0))

    # Stage 3b (largest CC refinement in proxy bbox)
    stage3b: Stage3bConfig = Stage3bConfig(connectivity=26, min_largest_size=0)

    # Parallelism (used where applicable)
    #max_workers: Optional[list] = [None]
    max_workers: Optional[int] = None


@dataclass(frozen=True)
class PipelineOutput:
    # Proxy products
    V_proxy: np.ndarray
    V_norm: np.ndarray
    V_residual: Optional[np.ndarray]
    V_crop: np.ndarray

    thr_proxy: float
    mask_proxy_raw: np.ndarray        # bool
    mask_proxy_clean: np.ndarray      # bool

    bbox_proxy: Optional[BBox3D]
    bbox_proxy_refined: Optional[BBox3D]

    bbox_full_from_proxy: Optional[BBox3D]       # mapped Stage 3 bbox to full
    bbox_full_from_proxy_refined: Optional[BBox3D]  # mapped Stage 3b bbox to full

    meta: Dict[str, Any]


def run_min_bounding_cuboid_pipeline(
    V_full: np.ndarray,                     # (Z,Y,X) full-res volume (denoised recon recommended)
    roi_proxy: Optional[np.ndarray] = None, # (Zp,Yp,Xp) optional proxy ROI mask (e.g., cylinder interior)
    cfg: PipelineConfig = PipelineConfig(),
) -> PipelineOutput:
    """
    End-to-end: Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 3b -> optional Stage 4.
    """

    # -------------------------
    # Stage 0 (shared memory)
    # -------------------------
    s0_cfg = cfg.stage0
    if cfg.max_workers is not None:
        # override Stage0 workers if user provided a global setting
        s0_cfg = Stage0Config(**{**s0_cfg.__dict__, "max_workers": cfg.max_workers})

    out0 = stage0_parallel_preprocess_shared_memory(V_full, s0_cfg)

    # -------------------------
    # Stage 1 (global threshold on proxy; shared memory apply)
    # -------------------------
    s1_cfg = cfg.stage1
    out1 = stage1_parallel_global_threshold_shared_memory(
        V_norm=out0.V_norm,
        V_residual=out0.V_residual,
        cfg=s1_cfg,
        roi_mask=roi_proxy,
        max_workers=cfg.max_workers
    )
    mask_proxy_raw = out1.mask

    # -------------------------
    # Stage 2 (proxy 2D morphology cleanup; shared memory)
    # -------------------------
    s2_cfg = cfg.stage2
    out2 = stage2_parallel_morphology_shared_memory(
        mask_u8_in=out1.mask_u8,
        cfg=s2_cfg,
        roi_mask=roi_proxy,
        max_workers=cfg.max_workers
    )
    mask_proxy_clean = out2.mask

    # -------------------------
    # Stage 3 (proxy bbox reduction)
    # -------------------------
    s3_cfg = cfg.stage3
    out3 = stage3_parallel_bbox_shared_memory(
        mask_u8=out2.mask_u8,
        cfg=s3_cfg,
        max_workers=cfg.max_workers
    )
    bbox_proxy = out3.bbox_proxy

    # map bbox to full-res (coarse)
    bbox_full = None
    if bbox_proxy is not None:
        bbox_full = map_bbox_proxy_to_full(
            bbox_proxy,
            original_shape=out0.original_shape,
            factors=out0.factors
        )

    # -------------------------
    # Stage 3b (largest CC refinement inside proxy bbox)
    # -------------------------
    bbox_proxy_refined = None
    bbox_full_refined = None
    if bbox_proxy is not None:
        out3b = stage3b_refine_bbox_largest_component(
            mask_u8=out2.mask_u8,
            bbox_proxy=bbox_proxy,
            cfg=cfg.stage3b
        )
        bbox_proxy_refined = out3b.bbox_proxy_refined

        if bbox_proxy_refined is not None:
            bbox_full_refined = map_bbox_proxy_to_full(
                bbox_proxy_refined,
                original_shape=out0.original_shape,
                factors=out0.factors
            )

    V_crop = V_full[
        bbox_full_refined.z0:bbox_full_refined.z1,
        bbox_full_refined.y0:bbox_full_refined.y1,
        bbox_full_refined.x0:bbox_full_refined.x1
    ]

    meta = {
        "stage0": {"factors": out0.factors, "original_shape": out0.original_shape},
        "stage1": out1.meta,
        "stage2": out2.meta,
        "stage3": out3.meta,
    }

    return PipelineOutput(
        V_proxy=out0.V_proxy,
        V_norm=out0.V_norm,
        V_residual=out0.V_residual,
        V_crop=V_crop,

        thr_proxy=float(out1.threshold),
        mask_proxy_raw=mask_proxy_raw,
        mask_proxy_clean=mask_proxy_clean,

        bbox_proxy=bbox_proxy,
        bbox_proxy_refined=bbox_proxy_refined,

        bbox_full_from_proxy=bbox_full,
        bbox_full_from_proxy_refined=bbox_full_refined,

        meta=meta
    )
