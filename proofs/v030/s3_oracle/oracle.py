"""ctypes bridge to the REAL WPS interp_module.F kernels (liboracle.so).

Used by tests/init/test_interp_metgrid.py to grade the JAX port against the
genuine Fortran interp routines (sprint AC: <= 1e-6 rel). Build first:
    bash proofs/v030/s3_oracle/build.sh

The Fortran method-id constants (misc_definitions_module.F) are:
    SIXTEEN_POINT=1, FOUR_POINT=2, N_NEIGHBOR=3, AVERAGE4=4, AVERAGE16=5,
    W_AVERAGE4=6, W_AVERAGE16=7, SEARCH=8
We map the project's interp_metgrid method tags -> these Fortran ids here.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

from gpuwrf.init import interp_metgrid as im

_LIB_PATH = os.path.join(os.path.dirname(__file__), "liboracle.so")

# project tag -> Fortran misc_definitions id
_TAG_TO_FORTRAN = {
    im.SIXTEEN_POINT: 1,
    im.FOUR_POINT: 2,
    im.N_NEIGHBOR: 3,
    im.AVERAGE4: 4,
    im.AVERAGE16: 5,
    im.W_AVERAGE4: 6,
    im.W_AVERAGE16: 7,
    im.SEARCH: 8,
}

_REL_CODE = {" ": 32, "<": 60, ">": 62}


class Oracle:
    def __init__(self, lib_path: str = _LIB_PATH):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"{lib_path} not built; run proofs/v030/s3_oracle/build.sh"
            )
        self.lib = ctypes.CDLL(lib_path)
        self.lib.oracle_interp.restype = None
        self.lib.oracle_interp.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # slab1d
            ctypes.c_int,  # nx
            ctypes.c_int,  # ny
            ctypes.POINTER(ctypes.c_float),  # rx
            ctypes.POINTER(ctypes.c_float),  # ry
            ctypes.c_int,  # npts
            ctypes.POINTER(ctypes.c_int),  # method_ids
            ctypes.c_int,  # nmeth
            ctypes.POINTER(ctypes.c_int),  # opts
            ctypes.c_float,  # msgval
            ctypes.c_int,  # has_mask
            ctypes.POINTER(ctypes.c_float),  # mask1d
            ctypes.c_float,  # maskval
            ctypes.c_int,  # rel_code
            ctypes.POINTER(ctypes.c_float),  # out
        ]
        self.lib.oracle_oned.restype = None
        self.lib.oracle_oned.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.POINTER(ctypes.c_float),  # a
            ctypes.POINTER(ctypes.c_float),  # b
            ctypes.POINTER(ctypes.c_float),  # c
            ctypes.POINTER(ctypes.c_float),  # d
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # out
        ]

    @staticmethod
    def _f32(a):
        return np.ascontiguousarray(a, dtype=np.float32)

    @staticmethod
    def _i32(a):
        return np.ascontiguousarray(a, dtype=np.int32)

    def interp(
        self,
        slab,  # (nx, ny) i=lon (first), j=lat (second)
        rx,  # (npts,) 1-based fractional first-axis index
        ry,  # (npts,) 1-based fractional second-axis index
        chain,  # list of (project_tag, opt)
        msgval=im.DEFAULT_MSGVAL,
        mask_array=None,  # (nx, ny) or None
        maskval=None,
        mask_relational=None,
    ):
        slab = self._f32(slab)
        nx, ny = slab.shape
        rx = self._f32(rx).reshape(-1)
        ry = self._f32(ry).reshape(-1)
        npts = rx.size
        method_ids = self._i32([_TAG_TO_FORTRAN[t] for (t, _o) in chain])
        opts = self._i32([o for (_t, o) in chain])
        nmeth = method_ids.size
        out = np.empty(npts, dtype=np.float32)

        # Fortran reads slab(i,j) as column-major slab1d(i+(j-1)*nx); numpy slab
        # is (nx,ny) C-order so .T.ravel() (order='F') gives that layout.
        slab1d = np.asfortranarray(slab).reshape(-1, order="F")

        if mask_array is not None and maskval is not None:
            has_mask = 1
            mask1d = np.asfortranarray(self._f32(mask_array)).reshape(-1, order="F")
            rel = mask_relational if mask_relational is not None else " "
        else:
            has_mask = 0
            mask1d = np.zeros(1, dtype=np.float32)
            maskval = -1.0
            rel = " "

        self.lib.oracle_interp(
            slab1d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(nx),
            ctypes.c_int(ny),
            rx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ry.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(npts),
            method_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(nmeth),
            opts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_float(msgval),
            ctypes.c_int(has_mask),
            mask1d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(maskval),
            ctypes.c_int(_REL_CODE[rel]),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def oned(self, x, a, b, c, d):
        x = self._f32(x).reshape(-1)
        a = self._f32(a).reshape(-1)
        b = self._f32(b).reshape(-1)
        c = self._f32(c).reshape(-1)
        d = self._f32(d).reshape(-1)
        n = x.size
        out = np.empty(n, dtype=np.float32)
        self.lib.oracle_oned(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out
