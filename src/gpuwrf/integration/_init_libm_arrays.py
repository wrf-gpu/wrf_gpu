"""Init-only scalar C loops around the caller's existing float32 libm pointers."""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

SOURCE = r'''
#include <stddef.h>
typedef float (*unary32)(float);
typedef float (*binary32)(float, float);
void map1(unary32 fn, const float *x, float *out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = fn(x[i]);
}
void map2(binary32 fn, const float *x, float y, float *out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = fn(x[i], y);
}
'''
FLAGS = ('-O2', '-shared', '-fPIC', '-std=c99', '-fno-fast-math',
         '-ffp-contract=off', '-fno-tree-vectorize', '-fno-tree-slp-vectorize',
         '-fno-unroll-loops')
STATUS = {'available': False, 'compile_s': 0.0, 'error': None}


@lru_cache(maxsize=1)
def load():
    """Lazy local build/load; any unavailable compiler/API retains Python fallback.

    No math library is linked here: the calling shim passes its own expf/logf/
    powf addresses. The loop does no arithmetic, vector math or reassociation.
    """
    try:
        compiler = shutil.which('cc')
        root = Path(os.environ.get('GPUWRF_JAX_CACHE_DIR',
                    str(Path.home()/'.cache/gpuwrf'))) / 'init-libm'
        digest = hashlib.sha256(repr((SOURCE, FLAGS, platform.system(),
                                      platform.machine(),ctypes.sizeof(ctypes.c_void_p))).encode()).hexdigest()
        root.mkdir(parents=True,exist_ok=True)
        path = root / (digest+'.so')
        if not path.is_file():
            if compiler is None:
                return None
            with tempfile.TemporaryDirectory(dir=root) as tmp:
                binary = Path(tmp)/'loop.so'
                start = time.perf_counter()
                subprocess.run([compiler,*FLAGS,'-x','c','-','-o',str(binary)],
                    input=SOURCE,text=True,capture_output=True,check=True,timeout=30)
                STATUS['compile_s'] = time.perf_counter()-start
                os.replace(binary,path)
        lib = ctypes.CDLL(str(path))
        ptr = ctypes.POINTER(ctypes.c_float)
        lib.map1.argtypes = [ctypes.c_void_p,ptr,ptr,ctypes.c_size_t]
        lib.map2.argtypes = [ctypes.c_void_p,ptr,ctypes.c_float,ptr,ctypes.c_size_t]
        lib.map1.restype = lib.map2.restype = None
        STATUS.update(available=True,path=str(path),source_sha256=digest,flags=FLAGS)
        return lib
    except Exception as exc:
        STATUS['error'] = f'{type(exc).__name__}: {exc}'
        return None
