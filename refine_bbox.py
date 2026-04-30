import numpy as np
from typing import Tuple, Optional

from dataclasses import dataclass
from typing import Dict, Any, Literal

try:
    import cc3d  # pip install connected-components-3d
except Exception:
    cc3d = None

from scipy.ndimage import label as ndi_label
from union_bbox import BBox3D


def crop_zyx(arr: np.ndarray, bbox: BBox3D) -> np.ndarray:
    """
    Crop a (Z,Y,X) array by bbox (exclusive upper bounds).
    """
    return arr[bbox.z0:bbox.z1, bbox.y0:bbox.y1, bbox.x0:bbox.x1]

def paste_zyx(dst: np.ndarray, src: np.ndarray, bbox: BBox3D) -> None:
    """
    Paste src (cropped) back into dst at bbox location.
    """
    dst[bbox.z0:bbox.z1, bbox.y0:bbox.y1, bbox.x0:bbox.x1] = src

def bbox_from_mask_3d(mask: np.ndarray) -> Optional[BBox3D]:
    """
    Compute tight bbox from a 3D boolean/uint8 mask. Returns None if empty.
    """
    m = mask.astype(bool, copy=False)
    if not np.any(m):
        return None
    zz, yy, xx = np.where(m)
    return BBox3D(
        z0=int(zz.min()), z1=int(zz.max()) + 1,
        y0=int(yy.min()), y1=int(yy.max()) + 1,
        x0=int(xx.min()), x1=int(xx.max()) + 1,
    )

@dataclass(frozen=True)
class Stage3bConfig:
    connectivity: Literal[6, 18, 26] = 26
    # Optional: ignore tiny components (in voxels) even if largest is small
    min_largest_size: int = 0


@dataclass(frozen=True)
class Stage3bOutput:
    bbox_proxy_refined: Optional[BBox3D]
    largest_size: int
    meta: Dict[str, Any]


def _largest_cc_mask_cc3d(m: np.ndarray, connectivity: int) -> Tuple[np.ndarray, int]:
    """
    Return binary mask of largest CC and its size using cc3d.
    """
    # cc3d expects integer input
    m_u8 = np.ascontiguousarray(m, dtype=np.uint8)
    labels = cc3d.connected_components(m_u8, connectivity=connectivity)
    if labels.max() == 0:
        return np.zeros_like(m, dtype=bool), 0
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    k = int(np.argmax(counts))
    return (labels == k), int(counts[k])


def _largest_cc_mask_scipy(m: np.ndarray, connectivity: int) -> Tuple[np.ndarray, int]:
    """
    Return binary mask of largest CC and its size using scipy.ndimage.label.
    connectivity: 6/18/26 mapped to structuring element.
    """
    if connectivity == 6:
        struct = np.zeros((3,3,3), dtype=bool)
        struct[1,1,1] = True
        struct[0,1,1] = struct[2,1,1] = True
        struct[1,0,1] = struct[1,2,1] = True
        struct[1,1,0] = struct[1,1,2] = True
    elif connectivity == 18:
        struct = np.ones((3,3,3), dtype=bool)
        # remove 8 corners to get 18-connectivity
        struct[0,0,0] = struct[0,0,2] = struct[0,2,0] = struct[0,2,2] = False
        struct[2,0,0] = struct[2,0,2] = struct[2,2,0] = struct[2,2,2] = False
    else:  # 26
        struct = np.ones((3,3,3), dtype=bool)

    labels, n = ndi_label(m.astype(bool), structure=struct)
    if n == 0:
        return np.zeros_like(m, dtype=bool), 0

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    k = int(np.argmax(counts))
    return (labels == k), int(counts[k])

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


def stage3b_refine_bbox_largest_component(
    mask_u8: np.ndarray,          # (Zp,Yp,Xp) uint8
    bbox_proxy: BBox3D,           # from Stage 3
    cfg: Stage3bConfig = Stage3bConfig(),
) -> Stage3bOutput:
    """
    Crop mask to bbox_proxy, keep largest 3D CC, compute refined bbox in proxy coords.
    """
    M = np.asarray(mask_u8, dtype=np.uint8)
    bbox_proxy = bbox_proxy.clamp(M.shape)

    Mc = crop_zyx(M, bbox_proxy).astype(bool, copy=False)
    if not np.any(Mc):
        return Stage3bOutput(
            bbox_proxy_refined=None,
            largest_size=0,
            meta={"reason": "empty mask in bbox crop", "connectivity": cfg.connectivity, "backend": None},
        )

    # label & keep largest CC
    if cc3d is not None:
        largest_mask, largest_size = _largest_cc_mask_cc3d(Mc, connectivity=int(cfg.connectivity))
        backend = "cc3d"
    else:
        largest_mask, largest_size = _largest_cc_mask_scipy(Mc, connectivity=int(cfg.connectivity))
        backend = "scipy.ndimage.label"

    if cfg.min_largest_size and largest_size < cfg.min_largest_size:
        return Stage3bOutput(
            bbox_proxy_refined=None,
            largest_size=int(largest_size),
            meta={"reason": "largest component below min_largest_size", "connectivity": cfg.connectivity, "backend": backend},
        )

    # bbox in cropped coords, then map back to proxy coords
    bb_local = bbox_from_mask_3d(largest_mask)
    if bb_local is None:
        return Stage3bOutput(
            bbox_proxy_refined=None,
            largest_size=int(largest_size),
            meta={"reason": "no bbox after labeling", "connectivity": cfg.connectivity, "backend": backend},
        )

    bb_ref = BBox3D(
        z0=bbox_proxy.z0 + bb_local.z0,
        z1=bbox_proxy.z0 + bb_local.z1,
        y0=bbox_proxy.y0 + bb_local.y0,
        y1=bbox_proxy.y0 + bb_local.y1,
        x0=bbox_proxy.x0 + bb_local.x0,
        x1=bbox_proxy.x0 + bb_local.x1,
    ).clamp(M.shape)

    meta = {
        "connectivity": int(cfg.connectivity),
        "backend": backend,
        "largest_size": int(largest_size),
        "crop_shape": tuple(Mc.shape),
    }

    return Stage3bOutput(bbox_proxy_refined=bb_ref, largest_size=int(largest_size), meta=meta)
