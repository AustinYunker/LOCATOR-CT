import tifffile
from tqdm import tqdm
import numpy as np
from pathlib import Path
import re, sys


def natural_sorted(l):
    def key(x):
        return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", str(x))]
    return sorted(l, key=key)

def natural_key(s):
    # simple natural sort (e.g., img_2.tif < img_10.tif)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


# We use the following function to load a stack of images:
def load_stack(paths, binning=1, use_tqdm=True):
    """Load a stack of tiff files.

    :param paths: paths to tiff files
    :param binning: whether angles and projection images should be binned.
    :returns: an np.array containing the values in the tiff files
    :rtype: np.array

    """
    # Read first image for shape and dtype information
    #paths = list(paths)
    #print(paths)

    img0 = tifffile.imread(str(paths[0]))
    img0 = img0[::binning, ::binning]
    dtype = img0.dtype
    # Create empty numpy array to hold result
    imgs = np.empty((len(paths), *img0.shape), dtype=dtype)

    progress = tqdm if use_tqdm else lambda x: x

    for i, p in progress(enumerate(paths)):
        img = tifffile.imread(str(p))[::binning, ::binning]
        img = np.nan_to_num(img)
        imgs[i] = img

    return imgs


def glob(dir_path):
    """Expand path to list of all tiffs in directory

    :param dir_path: directory
    :returns:
    :rtype:

    """
    dir_path = Path(dir_path).expanduser().resolve()
    return natural_sorted(dir_path.glob("*.tif*"))

def list_tiffs_sorted(dir_path):
    p = Path(dir_path)
    files = [str(x) for x in p.iterdir() if x.suffix.lower() in (".tif", ".tiff")]
    if not files:
        raise FileNotFoundError(f"No TIFFs in {dir_path}")
    files.sort(key=natural_key)
    return files

def save_stack(path, stack, prefix="output", exist_ok=True, parents=False):
    path = Path(path).expanduser().resolve()
    path.mkdir(exist_ok=exist_ok, parents=parents)
    for i, s in tqdm(enumerate(stack), mininterval=10.0):
        #opath = path / f"{prefix}_{i:05d}.tif"
        opath = path / f"{i:05d}.tif"
        tifffile.imwrite(str(opath), s)


def load_sino(paths, binning=1, dtype=None, flip_y=False):
    """Load a stack of tiff files into a sinogram

    :param paths: paths to tiff files
    :param binning: whether angles and projection images should be binned.
    :returns: an np.array containing the values in the tiff files
    :rtype: np.array

    """
    # Read first image for shape and dtype information
    paths = list(paths)
    #print(paths[0])
    #sys.exit()
    img0 = tifffile.imread(str(paths[0]))
    img0 = img0[::binning, ::binning]
    if dtype is None:
        dtype = img0.dtype
    # Create empty numpy array to hold result
    imgs = np.empty((img0.shape[0], len(paths), img0.shape[1]), dtype=dtype)
    for i, p in tqdm(enumerate(paths)):
        # Angles in the middle, "up" in front, "right" at the back.
        if flip_y:
            # Flip in the vertical direction
            imgs[:, i, :] = tifffile.imread(str(p))[::-binning, ::binning]
        else:
            imgs[:, i, :] = tifffile.imread(str(p))[::binning, ::binning]
    return imgs