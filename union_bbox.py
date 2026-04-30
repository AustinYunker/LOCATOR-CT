import os

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import atexit

from shmarray import ShmArray, shm_create_from_array, shm_create_empty, shm_attach


_WORK = {}

def _worker_init_stage3(mask_meta: ShmArray):
    M, shm_m = shm_attach(mask_meta)

    _WORK.update({
        "M": M,
        "_handles": [shm_m],
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
class BBox3D:
    z0: int; z1: int
    y0: int; y1: int
    x0: int; x1: int

    def as_tuple(self) -> Tuple[int,int,int,int,int,int]:
        return (self.z0, self.z1, self.y0, self.y1, self.x0, self.x1)

    def clamp(self, shape_zyx: Tuple[int,int,int]) -> "BBox3D":
        Z, Y, X = shape_zyx
        return BBox3D(
            z0=max(0, min(Z, self.z0)),
            z1=max(0, min(Z, self.z1)),
            y0=max(0, min(Y, self.y0)),
            y1=max(0, min(Y, self.y1)),
            x0=max(0, min(X, self.x0)),
            x1=max(0, min(X, self.x1)),
        )

    def expand(self, margin: Tuple[int,int,int]) -> "BBox3D":
        mz, my, mx = margin
        return BBox3D(
            z0=self.z0 - mz, z1=self.z1 + mz,
            y0=self.y0 - my, y1=self.y1 + my,
            x0=self.x0 - mx, x1=self.x1 + mx,
        )


# Reuse ShmArray + shm_attach from earlier
    
def stage3_worker_slice_bbox_idx(z: int):
    M = _WORK["M"]

    m2d = M[z] != 0

    if not np.any(m2d):
        return (z, False, 0, 0, 0, 0)

    ys, xs = np.where(m2d)

    y0 = int(ys.min())
    y1 = int(ys.max())

    x0 = int(xs.min())
    x1 = int(xs.max())

    return (z, True, y0, y1, x0, x1)


@dataclass(frozen=True)
class Stage3Config:
    # Optional safety margin in proxy voxels
    margin_zyx: Tuple[int,int,int] = (0, 0, 0)

    # If you have an ROI mask and want bbox limited to it, apply ROI in Stage 2 already.
    # Stage 3 assumes mask already reflects desired ROI.

@dataclass(frozen=True)
class Stage3Output:
    bbox_proxy: Optional[BBox3D]
    meta: Dict[str, Any]
    run_time: float = 0.0


def stage3_parallel_bbox_shared_memory(
    mask_u8: np.ndarray,        # (Zp,Yp,Xp) uint8
    cfg: Stage3Config = Stage3Config(),
    max_workers: Optional[int] = None,
) -> Stage3Output:
    mask_u8 = np.asarray(mask_u8, dtype=np.uint8)
    Zp, Yp, Xp = mask_u8.shape

    mask_meta, shm_m = shm_create_from_array(mask_u8)
    max_workers = max_workers or os.cpu_count() or 1

    # Reduction accumulators
    z_present = []
    y0s = []
    y1s = []
    x0s = []
    x1s = []

    try:
        with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_worker_init_stage3,
                initargs=(mask_meta,)
        ) as ex:
            futs = [ex.submit(stage3_worker_slice_bbox_idx, z) for z in range(Zp)]

            for fut in as_completed(futs):
                z, has_fg, y0, y1, x0, x1 = fut.result()
                if not has_fg:
                    continue
                z_present.append(z)
                y0s.append(y0); y1s.append(y1)
                x0s.append(x0); x1s.append(x1)

        if len(z_present) == 0:
            return Stage3Output(
                bbox_proxy=None,
                meta={"reason": "no foreground found in mask", "shape": (Zp, Yp, Xp)}
            )

        z0 = int(min(z_present))
        z1 = int(max(z_present) + 1)  # exclusive
        y0 = int(min(y0s))
        y1 = int(max(y1s) + 1)        # exclusive
        x0 = int(min(x0s))
        x1 = int(max(x1s) + 1)        # exclusive

        bbox = BBox3D(z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1)

        # Expand + clamp
        bbox = bbox.expand(cfg.margin_zyx).clamp((Zp, Yp, Xp))

        meta = {
            "shape": (Zp, Yp, Xp),
            "margin_zyx": cfg.margin_zyx,
            "fg_slices": len(z_present),
        }

        return Stage3Output(bbox_proxy=bbox, meta=meta, run_time=0.)

    finally:
        shm_m.close(); shm_m.unlink()

