
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from multiprocessing import shared_memory
import numpy as np

@dataclass(frozen=True)
class ShmArray:
    name: str
    shape: Tuple[int, ...]
    dtype: str  # np.dtype.str
    nbytes: int

def shm_create_from_array(arr: np.ndarray) -> Tuple[ShmArray, shared_memory.SharedMemory]:
    arr_c = np.ascontiguousarray(arr)
    shm = shared_memory.SharedMemory(create=True, size=arr_c.nbytes)
    shm_arr = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
    shm_arr[...] = arr_c
    meta = ShmArray(name=shm.name, shape=arr_c.shape, dtype=arr_c.dtype.str, nbytes=arr_c.nbytes)
    return meta, shm

def shm_create_empty(shape: Tuple[int, ...], dtype: np.dtype) -> Tuple[ShmArray, shared_memory.SharedMemory]:
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    meta = ShmArray(name=shm.name, shape=shape, dtype=dtype.str, nbytes=nbytes)
    # do not initialize (faster)
    return meta, shm

def shm_attach(meta: ShmArray) -> Tuple[np.ndarray, shared_memory.SharedMemory]:
    shm = shared_memory.SharedMemory(name=meta.name)
    arr = np.ndarray(meta.shape, dtype=np.dtype(meta.dtype), buffer=shm.buf)
    return arr, shm