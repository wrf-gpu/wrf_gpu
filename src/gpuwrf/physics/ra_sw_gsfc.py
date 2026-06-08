"""JAX port of the WRF GSFC (Chou-Suarez) shortwave scheme (ra_sw_physics=2).

Faithful jit/vmap-traceable single-column port of
``phys/module_ra_gsfcsw.F:GSFCSWRAD`` (the Chou-Suarez delta-Eddington
multi-band broadband shortwave parameterization, "Version Solar-6").  The
spectrum is split into one UV+visible band (8 sub-bands, :func:`_soluv`) and
three near-IR bands (10-point water-vapor k-distribution each, :func:`_solir`),
plus oxygen (Chou 1990) and CO2 (table look-up, :func:`_flxco2`) flux
reductions.  Cloud optical properties are derived from cloud water/ice content
and effective radius; clouds are grouped into high/middle/low layers (maximally
overlapped within a group, randomly overlapped between groups) and the all-sky
flux is the sum over the 8 sky configurations of the two-stream adding solution
(:func:`_cldflx`, Chou 1992).

WRF flips K internally inside ``GSFCSWRAD`` (the public arrays are model order,
``k=kts`` lowest, but the internal 2-D work arrays ``T2D``/``P8W2D``/... are
TOP-DOWN with a phantom layer at index 0).  The operational port supplies
columns in natural model order (``k=0`` lowest layer); this module reproduces
WRF's exact internal index handling: a top-down sweep over ``np = nz + 1``
layers (layer index 1 is the phantom above-top layer, layers 2..nz+1 are the
real model layers from top to bottom) and ``np + 1`` interface levels.  The
per-layer temperature heating rate ``TTEN`` (K s^-1) is mapped back to model
order on output, and the surface net downward SW flux ``GSW`` (W m^-2) and the
TOA upward residual ``RSWTOA`` (W m^-2) are returned, matching the WRF outputs.

This is the *bare column kernel*; the operational coupler
``coupling.physics_couplers.gsfc_sw_theta_tendency`` builds the column view
from operational ``State`` and converts ``TTEN/pi`` to ``RTHRATEN``, exactly as
``GSFCSWRAD`` does (``RTHRATEN += max(TTEN,0)/pi3D``).

Aerosol radiative feedback (the ``WRF_CHEM`` path) is OFF, matching the
operational GPU build: the aerosol optical depth / single-scattering albedo /
asymmetry are all zero (``taual=ssaal=asyal=0``), exactly the ``aer_ra_feedback
/= 1`` branch WRF takes by default.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

# --------------------------------------------------------------------------- #
# GSFC module-level constants (module_ra_gsfcsw.F header).                     #
# --------------------------------------------------------------------------- #
_THRESH = 1.0e-9            # cosz night cutoff / small-number floor
_CO2 = 300.0e-6            # CO2 volume mixing ratio (parts/part)
_IS_SUMMER = 80
_IE_SUMMER = 265

# --------------------------------------------------------------------------- #
# soluv (UV+visible, 8 bands) DATA tables.                                     #
# --------------------------------------------------------------------------- #
_UV_HK = np.array([.00057, .00367, .00083, .00417, .00600, .00556, .05913, .39081], dtype=np.float64)
_UV_XK = np.array([30.47, 187.2, 301.9, 42.83, 7.09, 1.25, 0.0345, 0.0539], dtype=np.float64)
_UV_RY = np.array([.00604, .00170, .00222, .00132, .00107, .00091, .00055, .00012], dtype=np.float64)
_UV_AIG = np.array([.74625000, .00105410, -.00000264], dtype=np.float64)
_UV_AWG = np.array([.82562000, .00529000, -.00014866], dtype=np.float64)

# --------------------------------------------------------------------------- #
# solir (near-IR, 3 bands x 10 k-intervals) DATA tables.                       #
# --------------------------------------------------------------------------- #
_IR_XK = np.array([0.0010, 0.0133, 0.0422, 0.1334, 0.4217, 1.334, 5.623, 31.62, 177.8, 1000.0], dtype=np.float64)
# hk(nband=3, nk=10): Fortran DATA fills column-major (band varies fastest).
_IR_HK = np.array(
    [
        [.20673, .08236, .01074],
        [.03497, .01157, .00360],
        [.03011, .01133, .00411],
        [.02260, .01143, .00421],
        [.01336, .01240, .00389],
        [.00696, .01258, .00326],
        [.00441, .01381, .00499],
        [.00115, .00650, .00465],
        [.00026, .00244, .00245],
        [.00000, .00094, .00145],
    ],
    dtype=np.float64,
).T  # shape (3, 10): _IR_HK[ib, ik] == Fortran hk(ib+1, ik+1)
# aib/awb/aia/awa/aig/awg dimensioned (nband=3, ncoef). Fortran DATA fills
# column-major: first index (band) varies fastest, so each printed 3-tuple is a
# column (one coefficient across the 3 bands).
_IR_AIB = np.array([[.000333, .000333, .000333], [2.52, 2.52, 2.52]], dtype=np.float64).T  # (3,2)
_IR_AWB = np.array([[-0.0101, -0.0166, -0.0339], [1.72, 1.85, 2.16]], dtype=np.float64).T  # (3,2)
_IR_AIA = np.array(
    [[-.00000260, .00215346, .08938331], [.00000746, .00073709, .00299387], [.00000000, -.00000134, -.00001038]],
    dtype=np.float64,
).T  # (3,3)
_IR_AWA = np.array(
    [[.00000007, -.00019934, .01209318], [.00000845, .00088757, .01784739], [-.00000004, -.00000650, -.00036910]],
    dtype=np.float64,
).T  # (3,3)
_IR_AIG = np.array(
    [[.74935228, .76098937, .84090400], [.00119715, .00141864, .00126222], [-.00000367, -.00000396, -.00000385]],
    dtype=np.float64,
).T  # (3,3)
_IR_AWG = np.array(
    [[.79375035, .74513197, .83530748], [.00832441, .01370071, .00257181], [-.00023263, -.00038203, .00005519]],
    dtype=np.float64,
).T  # (3,3)

# --------------------------------------------------------------------------- #
# cldscale maximum-overlap scaling table (caib (nm,nt,na), caif (nt,na)).      #
# Fortran DATA blocks fill caib as ((caib(m,i,j),j=1,11),i=1,9) for m=1..11.   #
# So the source 9 rows x 11 cols per block are caib[m-1, i, j].                #
# --------------------------------------------------------------------------- #
_CAIB_BLOCKS = [
    # m=1
    [[.000, 0.068, 0.140, 0.216, 0.298, 0.385, 0.481, 0.586, 0.705, 0.840, 1.000],
     [.000, 0.052, 0.106, 0.166, 0.230, 0.302, 0.383, 0.478, 0.595, 0.752, 1.000],
     [.000, 0.038, 0.078, 0.120, 0.166, 0.218, 0.276, 0.346, 0.438, 0.582, 1.000],
     [.000, 0.030, 0.060, 0.092, 0.126, 0.164, 0.206, 0.255, 0.322, 0.442, 1.000],
     [.000, 0.025, 0.051, 0.078, 0.106, 0.136, 0.170, 0.209, 0.266, 0.462, 1.000],
     [.000, 0.023, 0.046, 0.070, 0.095, 0.122, 0.150, 0.187, 0.278, 0.577, 1.000],
     [.000, 0.022, 0.043, 0.066, 0.089, 0.114, 0.141, 0.187, 0.354, 0.603, 1.000],
     [.000, 0.021, 0.042, 0.063, 0.086, 0.108, 0.135, 0.214, 0.349, 0.565, 1.000],
     [.000, 0.021, 0.041, 0.062, 0.083, 0.105, 0.134, 0.202, 0.302, 0.479, 1.000]],
    # m=2
    [[.000, 0.088, 0.179, 0.272, 0.367, 0.465, 0.566, 0.669, 0.776, 0.886, 1.000],
     [.000, 0.079, 0.161, 0.247, 0.337, 0.431, 0.531, 0.637, 0.749, 0.870, 1.000],
     [.000, 0.065, 0.134, 0.207, 0.286, 0.372, 0.466, 0.572, 0.692, 0.831, 1.000],
     [.000, 0.049, 0.102, 0.158, 0.221, 0.290, 0.370, 0.465, 0.583, 0.745, 1.000],
     [.000, 0.037, 0.076, 0.118, 0.165, 0.217, 0.278, 0.354, 0.459, 0.638, 1.000],
     [.000, 0.030, 0.061, 0.094, 0.130, 0.171, 0.221, 0.286, 0.398, 0.631, 1.000],
     [.000, 0.026, 0.052, 0.081, 0.111, 0.146, 0.189, 0.259, 0.407, 0.643, 1.000],
     [.000, 0.023, 0.047, 0.072, 0.098, 0.129, 0.170, 0.250, 0.387, 0.598, 1.000],
     [.000, 0.022, 0.044, 0.066, 0.090, 0.118, 0.156, 0.224, 0.328, 0.508, 1.000]],
    # m=3
    [[.000, 0.094, 0.189, 0.285, 0.383, 0.482, 0.582, 0.685, 0.788, 0.894, 1.000],
     [.000, 0.088, 0.178, 0.271, 0.366, 0.465, 0.565, 0.669, 0.776, 0.886, 1.000],
     [.000, 0.079, 0.161, 0.247, 0.337, 0.431, 0.531, 0.637, 0.750, 0.870, 1.000],
     [.000, 0.066, 0.134, 0.209, 0.289, 0.375, 0.470, 0.577, 0.697, 0.835, 1.000],
     [.000, 0.050, 0.104, 0.163, 0.227, 0.300, 0.383, 0.483, 0.606, 0.770, 1.000],
     [.000, 0.038, 0.080, 0.125, 0.175, 0.233, 0.302, 0.391, 0.518, 0.710, 1.000],
     [.000, 0.031, 0.064, 0.100, 0.141, 0.188, 0.249, 0.336, 0.476, 0.689, 1.000],
     [.000, 0.026, 0.054, 0.084, 0.118, 0.158, 0.213, 0.298, 0.433, 0.638, 1.000],
     [.000, 0.023, 0.048, 0.074, 0.102, 0.136, 0.182, 0.254, 0.360, 0.542, 1.000]],
    # m=4
    [[.000, 0.096, 0.193, 0.290, 0.389, 0.488, 0.589, 0.690, 0.792, 0.896, 1.000],
     [.000, 0.092, 0.186, 0.281, 0.378, 0.477, 0.578, 0.680, 0.785, 0.891, 1.000],
     [.000, 0.086, 0.174, 0.264, 0.358, 0.455, 0.556, 0.660, 0.769, 0.882, 1.000],
     [.000, 0.074, 0.153, 0.235, 0.323, 0.416, 0.514, 0.622, 0.737, 0.862, 1.000],
     [.000, 0.061, 0.126, 0.195, 0.271, 0.355, 0.449, 0.555, 0.678, 0.823, 1.000],
     [.000, 0.047, 0.098, 0.153, 0.215, 0.286, 0.370, 0.471, 0.600, 0.770, 1.000],
     [.000, 0.037, 0.077, 0.120, 0.170, 0.230, 0.303, 0.401, 0.537, 0.729, 1.000],
     [.000, 0.030, 0.062, 0.098, 0.138, 0.187, 0.252, 0.343, 0.476, 0.673, 1.000],
     [.000, 0.026, 0.053, 0.082, 0.114, 0.154, 0.207, 0.282, 0.391, 0.574, 1.000]],
    # m=5
    [[.000, 0.097, 0.194, 0.293, 0.392, 0.492, 0.592, 0.693, 0.794, 0.897, 1.000],
     [.000, 0.094, 0.190, 0.286, 0.384, 0.483, 0.584, 0.686, 0.789, 0.894, 1.000],
     [.000, 0.090, 0.181, 0.274, 0.370, 0.468, 0.569, 0.672, 0.778, 0.887, 1.000],
     [.000, 0.081, 0.165, 0.252, 0.343, 0.439, 0.539, 0.645, 0.757, 0.874, 1.000],
     [.000, 0.069, 0.142, 0.218, 0.302, 0.392, 0.490, 0.598, 0.717, 0.850, 1.000],
     [.000, 0.054, 0.114, 0.178, 0.250, 0.330, 0.422, 0.529, 0.656, 0.810, 1.000],
     [.000, 0.042, 0.090, 0.141, 0.200, 0.269, 0.351, 0.455, 0.589, 0.764, 1.000],
     [.000, 0.034, 0.070, 0.112, 0.159, 0.217, 0.289, 0.384, 0.515, 0.703, 1.000],
     [.000, 0.028, 0.058, 0.090, 0.128, 0.174, 0.231, 0.309, 0.420, 0.602, 1.000]],
    # m=6
    [[.000, 0.098, 0.196, 0.295, 0.394, 0.494, 0.594, 0.695, 0.796, 0.898, 1.000],
     [.000, 0.096, 0.193, 0.290, 0.389, 0.488, 0.588, 0.690, 0.792, 0.895, 1.000],
     [.000, 0.092, 0.186, 0.281, 0.378, 0.477, 0.577, 0.680, 0.784, 0.891, 1.000],
     [.000, 0.086, 0.174, 0.264, 0.358, 0.455, 0.556, 0.661, 0.769, 0.882, 1.000],
     [.000, 0.075, 0.154, 0.237, 0.325, 0.419, 0.518, 0.626, 0.741, 0.865, 1.000],
     [.000, 0.062, 0.129, 0.201, 0.279, 0.366, 0.462, 0.571, 0.694, 0.836, 1.000],
     [.000, 0.049, 0.102, 0.162, 0.229, 0.305, 0.394, 0.501, 0.631, 0.793, 1.000],
     [.000, 0.038, 0.080, 0.127, 0.182, 0.245, 0.323, 0.422, 0.550, 0.730, 1.000],
     [.000, 0.030, 0.064, 0.100, 0.142, 0.192, 0.254, 0.334, 0.448, 0.627, 1.000]],
    # m=7
    [[.000, 0.098, 0.198, 0.296, 0.396, 0.496, 0.596, 0.696, 0.797, 0.898, 1.000],
     [.000, 0.097, 0.194, 0.293, 0.392, 0.491, 0.591, 0.693, 0.794, 0.897, 1.000],
     [.000, 0.094, 0.190, 0.286, 0.384, 0.483, 0.583, 0.686, 0.789, 0.894, 1.000],
     [.000, 0.089, 0.180, 0.274, 0.369, 0.467, 0.568, 0.672, 0.778, 0.887, 1.000],
     [.000, 0.081, 0.165, 0.252, 0.344, 0.440, 0.541, 0.646, 0.758, 0.875, 1.000],
     [.000, 0.069, 0.142, 0.221, 0.306, 0.397, 0.496, 0.604, 0.722, 0.854, 1.000],
     [.000, 0.056, 0.116, 0.182, 0.256, 0.338, 0.432, 0.540, 0.666, 0.816, 1.000],
     [.000, 0.043, 0.090, 0.143, 0.203, 0.273, 0.355, 0.455, 0.583, 0.754, 1.000],
     [.000, 0.034, 0.070, 0.111, 0.157, 0.210, 0.276, 0.359, 0.474, 0.650, 1.000]],
    # m=8
    [[.000, 0.099, 0.198, 0.298, 0.398, 0.497, 0.598, 0.698, 0.798, 0.899, 1.000],
     [.000, 0.098, 0.196, 0.295, 0.394, 0.494, 0.594, 0.695, 0.796, 0.898, 1.000],
     [.000, 0.096, 0.193, 0.290, 0.390, 0.489, 0.589, 0.690, 0.793, 0.896, 1.000],
     [.000, 0.093, 0.186, 0.282, 0.379, 0.478, 0.578, 0.681, 0.786, 0.892, 1.000],
     [.000, 0.086, 0.175, 0.266, 0.361, 0.458, 0.558, 0.663, 0.771, 0.883, 1.000],
     [.000, 0.076, 0.156, 0.240, 0.330, 0.423, 0.523, 0.630, 0.744, 0.867, 1.000],
     [.000, 0.063, 0.130, 0.203, 0.282, 0.369, 0.465, 0.572, 0.694, 0.834, 1.000],
     [.000, 0.049, 0.102, 0.161, 0.226, 0.299, 0.385, 0.486, 0.611, 0.774, 1.000],
     [.000, 0.038, 0.078, 0.122, 0.172, 0.229, 0.297, 0.382, 0.498, 0.672, 1.000]],
    # m=9
    [[.000, 0.099, 0.199, 0.298, 0.398, 0.498, 0.598, 0.699, 0.799, 0.899, 1.000],
     [.000, 0.099, 0.198, 0.298, 0.398, 0.497, 0.598, 0.698, 0.798, 0.899, 1.000],
     [.000, 0.098, 0.196, 0.295, 0.394, 0.494, 0.594, 0.695, 0.796, 0.898, 1.000],
     [.000, 0.096, 0.193, 0.290, 0.389, 0.488, 0.588, 0.690, 0.792, 0.895, 1.000],
     [.000, 0.092, 0.185, 0.280, 0.376, 0.474, 0.575, 0.678, 0.782, 0.890, 1.000],
     [.000, 0.084, 0.170, 0.259, 0.351, 0.447, 0.547, 0.652, 0.762, 0.878, 1.000],
     [.000, 0.071, 0.146, 0.224, 0.308, 0.398, 0.494, 0.601, 0.718, 0.850, 1.000],
     [.000, 0.056, 0.114, 0.178, 0.248, 0.325, 0.412, 0.514, 0.638, 0.793, 1.000],
     [.000, 0.042, 0.086, 0.134, 0.186, 0.246, 0.318, 0.405, 0.521, 0.691, 1.000]],
    # m=10
    [[.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000],
     [.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000],
     [.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000],
     [.000, 0.100, 0.199, 0.298, 0.398, 0.498, 0.598, 0.698, 0.798, 0.899, 1.000],
     [.000, 0.098, 0.196, 0.294, 0.392, 0.491, 0.590, 0.691, 0.793, 0.896, 1.000],
     [.000, 0.092, 0.185, 0.278, 0.374, 0.470, 0.570, 0.671, 0.777, 0.886, 1.000],
     [.000, 0.081, 0.162, 0.246, 0.333, 0.424, 0.521, 0.625, 0.738, 0.862, 1.000],
     [.000, 0.063, 0.128, 0.196, 0.270, 0.349, 0.438, 0.540, 0.661, 0.809, 1.000],
     [.000, 0.046, 0.094, 0.146, 0.202, 0.264, 0.337, 0.426, 0.542, 0.710, 1.000]],
    # m=11
    [[.000, 0.101, 0.202, 0.302, 0.402, 0.502, 0.602, 0.702, 0.802, 0.901, 1.000],
     [.000, 0.102, 0.202, 0.303, 0.404, 0.504, 0.604, 0.703, 0.802, 0.902, 1.000],
     [.000, 0.102, 0.205, 0.306, 0.406, 0.506, 0.606, 0.706, 0.804, 0.902, 1.000],
     [.000, 0.104, 0.207, 0.309, 0.410, 0.510, 0.609, 0.707, 0.805, 0.902, 1.000],
     [.000, 0.106, 0.208, 0.309, 0.409, 0.508, 0.606, 0.705, 0.803, 0.902, 1.000],
     [.000, 0.102, 0.202, 0.298, 0.395, 0.493, 0.590, 0.690, 0.790, 0.894, 1.000],
     [.000, 0.091, 0.179, 0.267, 0.357, 0.449, 0.545, 0.647, 0.755, 0.872, 1.000],
     [.000, 0.073, 0.142, 0.214, 0.290, 0.372, 0.462, 0.563, 0.681, 0.822, 1.000],
     [.000, 0.053, 0.104, 0.158, 0.217, 0.281, 0.356, 0.446, 0.562, 0.726, 1.000]],
]
# caib[m-1, i-1, j-1] (nm=11, nt=9, na=11)
_CAIB = np.array(_CAIB_BLOCKS, dtype=np.float64)
_CAIF = np.array(
    [[.000, 0.099, 0.198, 0.297, 0.397, 0.496, 0.597, 0.697, 0.798, 0.899, 1.000],
     [.000, 0.098, 0.196, 0.294, 0.394, 0.494, 0.594, 0.694, 0.796, 0.898, 1.000],
     [.000, 0.096, 0.192, 0.290, 0.388, 0.487, 0.587, 0.689, 0.792, 0.895, 1.000],
     [.000, 0.092, 0.185, 0.280, 0.376, 0.476, 0.576, 0.678, 0.783, 0.890, 1.000],
     [.000, 0.085, 0.173, 0.263, 0.357, 0.454, 0.555, 0.659, 0.768, 0.881, 1.000],
     [.000, 0.076, 0.154, 0.237, 0.324, 0.418, 0.517, 0.624, 0.738, 0.864, 1.000],
     [.000, 0.063, 0.131, 0.203, 0.281, 0.366, 0.461, 0.567, 0.688, 0.830, 1.000],
     [.000, 0.052, 0.107, 0.166, 0.232, 0.305, 0.389, 0.488, 0.610, 0.770, 1.000],
     [.000, 0.043, 0.088, 0.136, 0.189, 0.248, 0.317, 0.400, 0.510, 0.675, 1.000]],
    dtype=np.float64,
)  # caif[i-1, j-1] (nt=9, na=11)
_CLD_DM = 0.1
_CLD_DT = 0.30103
_CLD_DA = 0.1
_CLD_T1 = -0.9031
_NM, _NT, _NA = 11, 9, 11

# --------------------------------------------------------------------------- #
# flxco2 CO2 absorption look-up table cah(22,19).                              #
# Fortran DATA fills ((cah(i,j),i=1,22),j=...): i (1st index) varies fastest,  #
# so each block is read row-by-row j with 22 i-values. cah[i-1, j-1].          #
# --------------------------------------------------------------------------- #
_CAH_FLAT = [
    # j=1..5
    0.9923, 0.9922, 0.9921, 0.9920, 0.9916, 0.9910, 0.9899, 0.9882,
    0.9856, 0.9818, 0.9761, 0.9678, 0.9558, 0.9395, 0.9188, 0.8945,
    0.8675, 0.8376, 0.8029, 0.7621, 0.7154, 0.6647, 0.9876, 0.9876,
    0.9875, 0.9873, 0.9870, 0.9864, 0.9854, 0.9837, 0.9811, 0.9773,
    0.9718, 0.9636, 0.9518, 0.9358, 0.9153, 0.8913, 0.8647, 0.8350,
    0.8005, 0.7599, 0.7133, 0.6627, 0.9808, 0.9807, 0.9806, 0.9805,
    0.9802, 0.9796, 0.9786, 0.9769, 0.9744, 0.9707, 0.9653, 0.9573,
    0.9459, 0.9302, 0.9102, 0.8866, 0.8604, 0.8311, 0.7969, 0.7565,
    0.7101, 0.6596, 0.9708, 0.9708, 0.9707, 0.9705, 0.9702, 0.9697,
    0.9687, 0.9671, 0.9647, 0.9612, 0.9560, 0.9483, 0.9372, 0.9221,
    0.9027, 0.8798, 0.8542, 0.8253, 0.7916, 0.7515, 0.7054, 0.6551,
    0.9568, 0.9568, 0.9567, 0.9565, 0.9562, 0.9557, 0.9548, 0.9533,
    0.9510, 0.9477, 0.9428, 0.9355, 0.9250, 0.9106, 0.8921, 0.8700,
    0.8452, 0.8171, 0.7839, 0.7443, 0.6986, 0.6486,
    # j=6..10
    0.9377, 0.9377, 0.9376, 0.9375, 0.9372, 0.9367, 0.9359, 0.9345,
    0.9324, 0.9294, 0.9248, 0.9181, 0.9083, 0.8948, 0.8774, 0.8565,
    0.8328, 0.8055, 0.7731, 0.7342, 0.6890, 0.6395, 0.9126, 0.9126,
    0.9125, 0.9124, 0.9121, 0.9117, 0.9110, 0.9098, 0.9079, 0.9052,
    0.9012, 0.8951, 0.8862, 0.8739, 0.8579, 0.8385, 0.8161, 0.7900,
    0.7585, 0.7205, 0.6760, 0.6270, 0.8809, 0.8809, 0.8808, 0.8807,
    0.8805, 0.8802, 0.8796, 0.8786, 0.8770, 0.8747, 0.8712, 0.8659,
    0.8582, 0.8473, 0.8329, 0.8153, 0.7945, 0.7697, 0.7394, 0.7024,
    0.6588, 0.6105, 0.8427, 0.8427, 0.8427, 0.8426, 0.8424, 0.8422,
    0.8417, 0.8409, 0.8397, 0.8378, 0.8350, 0.8306, 0.8241, 0.8148,
    0.8023, 0.7866, 0.7676, 0.7444, 0.7154, 0.6796, 0.6370, 0.5897,
    0.7990, 0.7990, 0.7990, 0.7989, 0.7988, 0.7987, 0.7983, 0.7978,
    0.7969, 0.7955, 0.7933, 0.7899, 0.7846, 0.7769, 0.7664, 0.7528,
    0.7357, 0.7141, 0.6866, 0.6520, 0.6108, 0.5646,
    # j=11..15
    0.7515, 0.7515, 0.7515, 0.7515, 0.7514, 0.7513, 0.7511, 0.7507,
    0.7501, 0.7491, 0.7476, 0.7450, 0.7409, 0.7347, 0.7261, 0.7144,
    0.6992, 0.6793, 0.6533, 0.6203, 0.5805, 0.5357, 0.7020, 0.7020,
    0.7020, 0.7019, 0.7019, 0.7018, 0.7017, 0.7015, 0.7011, 0.7005,
    0.6993, 0.6974, 0.6943, 0.6894, 0.6823, 0.6723, 0.6588, 0.6406,
    0.6161, 0.5847, 0.5466, 0.5034, 0.6518, 0.6518, 0.6518, 0.6518,
    0.6518, 0.6517, 0.6517, 0.6515, 0.6513, 0.6508, 0.6500, 0.6485,
    0.6459, 0.6419, 0.6359, 0.6273, 0.6151, 0.5983, 0.5755, 0.5458,
    0.5095, 0.4681, 0.6017, 0.6017, 0.6017, 0.6017, 0.6016, 0.6016,
    0.6016, 0.6015, 0.6013, 0.6009, 0.6002, 0.5989, 0.5967, 0.5932,
    0.5879, 0.5801, 0.5691, 0.5535, 0.5322, 0.5043, 0.4700, 0.4308,
    0.5518, 0.5518, 0.5518, 0.5518, 0.5518, 0.5518, 0.5517, 0.5516,
    0.5514, 0.5511, 0.5505, 0.5493, 0.5473, 0.5441, 0.5393, 0.5322,
    0.5220, 0.5076, 0.4878, 0.4617, 0.4297, 0.3929,
    # j=16..19
    0.5031, 0.5031, 0.5031, 0.5031, 0.5031, 0.5030, 0.5030, 0.5029,
    0.5028, 0.5025, 0.5019, 0.5008, 0.4990, 0.4960, 0.4916, 0.4850,
    0.4757, 0.4624, 0.4441, 0.4201, 0.3904, 0.3564, 0.4565, 0.4565,
    0.4565, 0.4564, 0.4564, 0.4564, 0.4564, 0.4563, 0.4562, 0.4559,
    0.4553, 0.4544, 0.4527, 0.4500, 0.4460, 0.4400, 0.4315, 0.4194,
    0.4028, 0.3809, 0.3538, 0.3227, 0.4122, 0.4122, 0.4122, 0.4122,
    0.4122, 0.4122, 0.4122, 0.4121, 0.4120, 0.4117, 0.4112, 0.4104,
    0.4089, 0.4065, 0.4029, 0.3976, 0.3900, 0.3792, 0.3643, 0.3447,
    0.3203, 0.2923, 0.3696, 0.3696, 0.3696, 0.3696, 0.3696, 0.3696,
    0.3695, 0.3695, 0.3694, 0.3691, 0.3687, 0.3680, 0.3667, 0.3647,
    0.3615, 0.3570, 0.3504, 0.3409, 0.3279, 0.3106, 0.2892, 0.2642,
]
# reshape column-major (i fastest): cah[i-1, j-1]
_CAH = np.array(_CAH_FLAT, dtype=np.float64).reshape(19, 22).T  # (22, 19)


class GsfcSWColumnState(NamedTuple):
    """Single-column inputs for the GSFC shortwave kernel, model order.

    All 3-D fields are (ncol, nz) on mass levels in natural model order
    (index 0 = lowest layer).  ``p8w`` is (ncol, nz+1) interface pressure (Pa)
    in model order (index 0 = surface).  ``coszen``, ``albedo`` are (ncol,).
    """

    T: jnp.ndarray            # layer temperature (K)
    p: jnp.ndarray            # layer pressure (Pa)
    p8w: jnp.ndarray          # interface pressure (Pa), nz+1
    qv: jnp.ndarray           # water vapor mixing ratio (kg/kg)
    qc: jnp.ndarray
    qr: jnp.ndarray
    qi: jnp.ndarray
    qs: jnp.ndarray
    qg: jnp.ndarray
    dz: jnp.ndarray           # layer thickness (m) -- unused by GSFC (kept for symmetry)
    cldfra: jnp.ndarray       # cloud fraction (0..1)
    coszen: jnp.ndarray       # cosine solar zenith angle
    albedo: jnp.ndarray       # surface albedo (0..1)
    solcon: jnp.ndarray       # solar constant at TOA (W/m^2)
    julday: int = 172
    center_lat: float = 0.0   # grid center latitude (deg) for ozone profile
    f_qi: bool = True         # ice is a prognostic species (Thompson-class)
    warm_rain: bool = False
    cp: float = 7.0 * 287.0 / 2.0
    g: float = 9.81


class GsfcSWColumnResult(NamedTuple):
    heating_rate: jnp.ndarray   # (ncol, nz) dT/dt (K/s), model order, >=0
    gsw: jnp.ndarray            # (ncol,) net downward surface SW flux (W/m^2)
    rswtoa: jnp.ndarray         # (ncol,) TOA upward SW residual (W/m^2)


def _select_iprof(center_lat: float, julday: int) -> int:
    """WRF GSFCSWRAD iprof selection (1..5) -- a static Python int."""
    is_summer, ie_summer = _IS_SUMMER, _IE_SUMMER
    clat = float(center_lat)
    jd = int(julday)
    if abs(clat) <= 30.0:
        return 5  # tropical
    if clat > 0.0:
        if clat > 60.0:
            return 3 if (jd > is_summer and jd < ie_summer) else 4
        return 1 if (jd > is_summer and jd < ie_summer) else 2
    if clat < -60.0:
        return 3 if (jd < is_summer or jd > ie_summer) else 4
    return 1 if (jd < is_summer or jd > ie_summer) else 2


# Ozone climatology pressure (mb) / mixing-ratio (g/g) profiles, 75 levels each,
# transcribed from module_ra_gsfcsw.F DATA pres(i,iprof)/ozone(i,iprof).
# Loaded lazily from the pristine source-equivalent constants module.
from gpuwrf.physics._gsfc_ozone import OZONE_PRES_MB, OZONE_GG  # noqa: E402


def _o3prof(p_mb_td, iprof):
    """Replicate o3prof: log-pressure linear interpolation of the iprof ozone
    climatology onto the (top-down, with phantom layer) model levels.

    ``p_mb_td`` is (ncol, npl) layer pressure in mb (top-down incl phantom),
    where npl = nz+1.  Returns the layer ozone (cm-atm STP-equivalent g/g) on
    the same (ncol, npl) grid.  WRF calls o3prof with kts-1..kte (npl entries).
    """
    np_levels = OZONE_PRES_MB.shape[1]
    pres = jnp.log(jnp.asarray(OZONE_PRES_MB[iprof - 1], dtype=p_mb_td.dtype))   # (75,)
    ozone = jnp.asarray(OZONE_GG[iprof - 1], dtype=p_mb_td.dtype)                # (75,)
    lp = jnp.log(p_mb_td)                                                        # (ncol, npl)
    ncol, npl = lp.shape

    # WRF indexes levels k=kts..kte where kts here is 1 (the phantom layer) in
    # the o3prof call (its passed kts-1=0 .. kte). In Fortran the loop is:
    #   interior k = kts+1..kte: find ko (1..np) s.t. p>pres(ko), interpolate.
    #   top k = kts: handle separately (column-mean above pres(ko-1)).
    # We index everything 0-based here; level 0 == phantom top layer.

    # For each level, find ko (1-based, 2..np) = first index with pres(ko) >= lp,
    # i.e. searchsorted on the increasing pres array. WRF advances ko while
    # p>pres(ko); with pres increasing this is searchsorted(pres, lp, 'left')+1
    # giving the first ko with pres(ko) >= lp. Clamp to [2, np].
    ko = jnp.searchsorted(pres, lp.reshape(-1), side="left").reshape(ncol, npl) + 1  # 1-based
    ko = jnp.clip(ko, 2, np_levels)

    def lin(x1, y1, x2, y2, x):
        return (y1 * (x2 - x) + y2 * (x - x1)) / (x2 - x1)

    ko0 = ko - 1   # 0-based upper node
    kom1 = ko - 2  # 0-based lower node
    # WRF: o3 = Linear(pres(ko), ozone(ko), pres(ko-1), ozone(ko-1), p).
    o3 = lin(jnp.take(pres, ko0), jnp.take(ozone, ko0),
             jnp.take(pres, kom1), jnp.take(ozone, kom1), lp)

    # Top layer (level 0): column-mean treatment. Build via cumulative integral.
    # ko_top = first ko with pres(ko) >= lp(top); if ko-1<=1 -> ozone(1) (0-based 0).
    ko_top = ko[:, 0]                              # 1-based
    use_first = (ko_top - 1) <= 1
    # column mean: sum_{kk=ko-2..1} ozone(kk)*(pres(kk+1)-pres(kk)) / (pres(ko-1)-pres(1))
    # 0-based: kk in [0 .. ko-3], weight (pres[kk+1]-pres[kk]), normalize by
    # (pres[ko-2]-pres[0]).
    dpres = pres[1:] - pres[:-1]                   # (74,)
    cum = jnp.concatenate([jnp.zeros((1,), dtype=pres.dtype), jnp.cumsum(ozone[:-1] * dpres)])  # (75,)
    # sum over kk=0..(ko-3) == cum[ko-2] (0-based index ko-2). guard ko_top>=2.
    upper = jnp.clip(ko_top - 2, 0, np_levels - 1)
    num = jnp.take(cum, upper)
    denom = jnp.take(pres, jnp.clip(ko_top - 2, 0, np_levels - 1)) - pres[0]
    mean_o3 = jnp.where(jnp.abs(denom) > 0, num / jnp.where(jnp.abs(denom) > 0, denom, 1.0), ozone[0])
    o3_top = jnp.where(use_first, ozone[0], mean_o3)
    o3 = o3.at[:, 0].set(o3_top)
    return o3


def _deledd(tau, ssc, g0, csm):
    """Delta-Eddington reflect/transmit (King & Harshvardhan 1986). Vectorized."""
    one, three, two, seven, four, fourth, zero = 1.0, 3.0, 2.0, 7.0, 4.0, 0.25, 0.0
    thresh = 1.0e-8
    zth = one / csm
    ff = g0 * g0
    xx = one - ff * ssc
    taup = tau * xx
    sscp = ssc * (one - ff) / xx
    gp = g0 / (one + g0)
    xx3 = three * gp
    gm1 = (seven - sscp * (four + xx3)) * fourth
    gm2 = -(one - sscp * (four - xx3)) * fourth
    akk = jnp.sqrt(jnp.maximum((gm1 + gm2) * (gm1 - gm2), 0.0))
    xx = akk * zth
    st7 = one - xx
    st8 = one + xx
    st3 = st7 * st8
    # WRF: if |st3|<thresh, zth += 0.001 then recompute st7/st8/st3.
    small = jnp.abs(st3) < thresh
    zth2 = zth + 0.001
    xx2 = akk * zth2
    st7b = one - xx2
    st8b = one + xx2
    st3b = st7b * st8b
    zth = jnp.where(small, zth2, zth)
    st7 = jnp.where(small, st7b, st7)
    st8 = jnp.where(small, st8b, st8)
    st3 = jnp.where(small, st3b, st3)

    td = jnp.exp(-taup / zth)
    gm3 = (two - zth * three * gp) * fourth
    xx = gm1 - gm2
    alf1 = gm1 - gm3 * xx
    alf2 = gm2 + gm3 * xx
    xx = akk * two
    all_ = (gm3 - alf2 * zth) * xx * td
    bll = (one - gm3 + alf1 * zth) * xx
    xx = akk * gm3
    cll = (alf2 + xx) * st7
    dll = (alf2 - xx) * st8
    xx = akk * (one - gm3)
    fll = (alf1 + xx) * st8
    ell = (alf1 - xx) * st7
    st2 = jnp.exp(-akk * taup)
    st4 = st2 * st2
    st1 = sscp / ((akk + gm1 + (akk - gm1) * st4) * st3)
    rr = (cll - dll * st4 - all_ * st2) * st1
    tt = -((fll - ell * st4) * td - bll * st2) * st1
    rr = jnp.maximum(rr, zero)
    tt = jnp.maximum(tt, zero)
    return rr, tt, td


def _sagpol(tau, ssc, g0):
    """Diffuse reflect/transmit (Sagan & Pollock 1967). Vectorized."""
    one, three, four = 1.0, 3.0, 4.0
    xx = one - ssc * g0
    uuu = jnp.sqrt(xx / (one - ssc))
    ttt = jnp.sqrt(xx * (one - ssc) * three) * tau
    emt = jnp.exp(-ttt)
    up1 = uuu + one
    um1 = uuu - one
    xx = um1 * emt
    st1 = one / ((up1 + xx) * (up1 - xx))
    rll = up1 * um1 * (one - emt * emt) * st1
    tll = uuu * four * emt * st1
    return rll, tll


def _cldscale(cosz, fcld, taucld, ict, icb, np_layers):
    """Replicate cldscale: high/mid/low cloud cover (cc) + maximum-overlap
    scaled cloud optical thickness (tauclb beam, tauclf diffuse).

    Per-column (vmapped). ``fcld``/``taucld`` are (np_layers,) and (np_layers,2);
    ``ict``/``icb`` are 1-based level indices (scalars). Returns
    cc (3,), tauclb (np,), tauclf (np,).
    """
    caib = jnp.asarray(_CAIB, dtype=fcld.dtype)  # (nm, nt, na)
    caif = jnp.asarray(_CAIF, dtype=fcld.dtype)  # (nt, na)

    k = jnp.arange(np_layers)                    # 0-based layer index
    k1 = k + 1                                   # 1-based (matches Fortran k)
    # group masks: high k1 in [1, ict-1]; mid [ict, icb-1]; low [icb, np]
    is_high = k1 < ict
    is_mid = (k1 >= ict) & (k1 < icb)
    is_low = k1 >= icb
    cc1 = jnp.max(jnp.where(is_high, fcld, 0.0))
    cc2 = jnp.max(jnp.where(is_mid, fcld, 0.0))
    cc3 = jnp.max(jnp.where(is_low, fcld, 0.0))
    cc = jnp.stack([cc1, cc2, cc3])              # (3,)
    cc_layer = jnp.where(is_high, cc1, jnp.where(is_mid, cc2, cc3))  # (np,)

    taux = taucld[:, 0] + taucld[:, 1]
    active = (taux > 0.05) & (fcld > 0.01)
    cc_safe = jnp.where(cc_layer > 0.0, cc_layer, 1.0)
    fa0 = fcld / cc_safe
    tauxc = jnp.minimum(taux, 32.0)

    fm = cosz / _CLD_DM
    ft = (jnp.log10(jnp.maximum(tauxc, 1.0e-30)) - _CLD_T1) / _CLD_DT
    fa = fa0 / _CLD_DA

    im = jnp.clip(jnp.floor(fm + 1.5).astype(jnp.int32), 2, _NM - 1)
    it = jnp.clip(jnp.floor(ft + 1.5).astype(jnp.int32), 2, _NT - 1)
    ia = jnp.clip(jnp.floor(fa + 1.5).astype(jnp.int32), 2, _NA - 1)

    fm = fm - (im - 1).astype(fm.dtype)
    ft = ft - (it - 1).astype(ft.dtype)
    fa = fa - (ia - 1).astype(fa.dtype)

    # 0-based table indices.
    im0, it0, ia0 = im - 1, it - 1, ia - 1

    def cb(mi, ti, ai):  # caib[mi, ti, ai] with 0-based ints (np-vector gathers)
        return caib[mi, ti, ai]

    xai = (-cb(im0 - 1, it0, ia0) * (1.0 - fm) + cb(im0 + 1, it0, ia0) * (1.0 + fm)) * fm * 0.5 \
        + cb(im0, it0, ia0) * (1.0 - fm * fm)
    xai = xai + (-cb(im0, it0 - 1, ia0) * (1.0 - ft) + cb(im0, it0 + 1, ia0) * (1.0 + ft)) * ft * 0.5 \
        + cb(im0, it0, ia0) * (1.0 - ft * ft)
    xai = xai + (-cb(im0, it0, ia0 - 1) * (1.0 - fa) + cb(im0, it0, ia0 + 1) * (1.0 + fa)) * fa * 0.5 \
        + cb(im0, it0, ia0) * (1.0 - fa * fa)
    xai = xai - 2.0 * cb(im0, it0, ia0)
    xai = jnp.maximum(xai, 0.0)
    tauclb = jnp.where(active, tauxc * xai, 0.0)

    def cf(ti, ai):
        return caif[ti, ai]

    xaif = (-cf(it0 - 1, ia0) * (1.0 - ft) + cf(it0 + 1, ia0) * (1.0 + ft)) * ft * 0.5 \
        + cf(it0, ia0) * (1.0 - ft * ft)
    xaif = xaif + (-cf(it0, ia0 - 1) * (1.0 - fa) + cf(it0, ia0 + 1) * (1.0 + fa)) * fa * 0.5 \
        + cf(it0, ia0) * (1.0 - fa * fa)
    xaif = xaif - cf(it0, ia0)
    xaif = jnp.maximum(xaif, 0.0)
    tauclf = jnp.where(active, tauxc * xaif, 0.0)
    return cc, tauclb, tauclf


def _cldflx(rr, tt, td, rs, ts, cc, ict, icb, np_layers):
    """Two-stream adding for high/mid/low randomly-overlapped clouds
    (Chou 1992), faithful per-column reproduction of cldflx.

    Inputs (vmapped, per column):
      rr,tt,td,rs,ts : (np+1, 2) layer properties; index k is 0-based, the
                       np-th entry is the surface (clear/cloudy index 1=clear).
      cc             : (3,) high/mid/low cloud cover.
      ict,icb        : 1-based level indices (scalars).
    Returns fclr,fall,fallu,falld (np+1,), fsdir,fsdif (scalars).
    """
    npl = np_layers
    nlev = npl + 1
    dtype = rr.dtype
    k1 = jnp.arange(npl) + 1                          # 1-based layer index

    # -------- phase 1: top-down composite tda/tta/rsa for high+mid block ------
    # Within high block (k1 < ict), use clear/cloudy index ih; mid block
    # (ict<=k1<icb) uses index im. We build tda[k, ih, im] for k=0..npl-1 and
    # later extend through the low block per sky config. Represent the (ih,im)
    # 2x2 as the last axis pair. tda/tta/rsa shape (nlev, 2, 2).
    tda = jnp.zeros((nlev, 2, 2), dtype)
    tta = jnp.zeros((nlev, 2, 2), dtype)
    rsa = jnp.zeros((nlev, 2, 2), dtype)
    # initialize at k=0 (Fortran k=1) for both ih and both im copies.
    for ih in range(2):
        tda = tda.at[0, ih, :].set(td[0, ih])
        tta = tta.at[0, ih, :].set(tt[0, ih])
        rsa = rsa.at[0, ih, :].set(rs[0, ih])

    is_high = k1 < ict                                # (npl,)
    is_mid = (k1 >= ict) & (k1 < icb)

    def add_down(carry, k):
        tda_c, tta_c, rsa_c = carry            # each (2,2) at level k-1
        # choose layer property index depending on group; high->ih, mid->im
        out_tda = jnp.zeros((2, 2), dtype)
        out_tta = jnp.zeros((2, 2), dtype)
        out_rsa = jnp.zeros((2, 2), dtype)
        for ih in range(2):
            for im in range(2):
                # high block uses cloud-index ih (and copies across im);
                # mid block uses cloud-index im.
                lay = jnp.where(is_high[k], ih, im)
                tdl = td[k, lay]
                ttl = tt[k, lay]
                rrl = rr[k, lay]
                rsl = rs[k, lay]
                tsl = ts[k, lay]
                denm = tsl / (1.0 - rsa_c[ih, im] * rsl)
                ntda = tda_c[ih, im] * tdl
                ntta = tda_c[ih, im] * ttl + (tda_c[ih, im] * rrl * rsa_c[ih, im] + tta_c[ih, im]) * denm
                nrsa = rsl + tsl * rsa_c[ih, im] * denm
                # only update within high+mid block; else carry through.
                upd = is_high[k] | is_mid[k]
                out_tda = out_tda.at[ih, im].set(jnp.where(upd, ntda, tda_c[ih, im]))
                out_tta = out_tta.at[ih, im].set(jnp.where(upd, ntta, tta_c[ih, im]))
                out_rsa = out_rsa.at[ih, im].set(jnp.where(upd, nrsa, rsa_c[ih, im]))
        return (out_tda, out_tta, out_rsa), (out_tda, out_tta, out_rsa)

    init = (tda[0], tta[0], rsa[0])
    ks = jnp.arange(1, npl)
    _, (tda_s, tta_s, rsa_s) = lax.scan(add_down, init, ks)
    tda = tda.at[1:npl].set(tda_s)
    tta = tta.at[1:npl].set(tta_s)
    rsa = rsa.at[1:npl].set(rsa_s)

    # -------- phase 2: bottom-up composite rra/rxa for low+mid block ----------
    rra = jnp.zeros((nlev, 2, 2), dtype)              # (level, im, is)
    rxa = jnp.zeros((nlev, 2, 2), dtype)
    for is_ in range(2):
        rra = rra.at[npl, :, is_].set(rr[npl, is_])
        rxa = rxa.at[npl, :, is_].set(rs[npl, is_])

    def add_up(carry, k):
        rra_c, rxa_c = carry                          # (2,2) at level k+1
        out_rra = jnp.zeros((2, 2), dtype)
        out_rxa = jnp.zeros((2, 2), dtype)
        for im in range(2):
            for is_ in range(2):
                # low block uses cloud-index is_; mid block uses im.
                lay = jnp.where(is_low_arr[k], is_, im)
                tdl = td[k, lay]
                ttl = tt[k, lay]
                rrl = rr[k, lay]
                rsl = rs[k, lay]
                tsl = ts[k, lay]
                denm = tsl / (1.0 - rsl * rxa_c[im, is_])
                nrra = rrl + (tdl * rra_c[im, is_] + ttl * rxa_c[im, is_]) * denm
                nrxa = rsl + tsl * rxa_c[im, is_] * denm
                upd = is_low_arr[k] | is_mid[k]
                out_rra = out_rra.at[im, is_].set(jnp.where(upd, nrra, rra_c[im, is_]))
                out_rxa = out_rxa.at[im, is_].set(jnp.where(upd, nrxa, rxa_c[im, is_]))
        return (out_rra, out_rxa), (out_rra, out_rxa)

    is_low_arr = k1 >= icb
    init_up = (rra[npl], rxa[npl])
    ks_up = jnp.arange(npl - 1, -1, -1)
    _, (rra_s, rxa_s) = lax.scan(add_up, init_up, ks_up)
    # rra_s/rxa_s are in reverse-k order (k = npl-1 .. 0).
    rra = rra.at[0:npl].set(rra_s[::-1])
    rxa = rxa.at[0:npl].set(rxa_s[::-1])

    # -------- phase 3: integrate over 8 sky configs --------------------------
    fclr = jnp.zeros((nlev,), dtype)
    fall = jnp.zeros((nlev,), dtype)
    fallu = jnp.zeros((nlev,), dtype)
    falld = jnp.zeros((nlev,), dtype)
    fsdir = jnp.zeros((), dtype)
    fsdif = jnp.zeros((), dtype)

    is_low_layer = k1 >= icb                          # layers in low block
    is_high_layer = k1 < ict                          # layers in high block

    for ih in range(2):
        ch = (1.0 - cc[0]) if ih == 0 else cc[0]
        for im in range(2):
            cm = ch * ((1.0 - cc[1]) if im == 0 else cc[1])
            for is_ in range(2):
                ct = cm * ((1.0 - cc[2]) if is_ == 0 else cc[2])

                # extend tda/tta/rsa through the low block (going down).
                def ext_down(carry, k):
                    tdac, ttac, rsac = carry
                    lay = is_
                    denm = ts[k, lay] / (1.0 - rsac * rs[k, lay])
                    ntda = tdac * td[k, lay]
                    ntta = tdac * tt[k, lay] + (tdac * rr[k, lay] * rsac + ttac) * denm
                    nrsa = rs[k, lay] + ts[k, lay] * rsac * denm
                    upd = is_low_layer[k]
                    tdac = jnp.where(upd, ntda, tdac)
                    ttac = jnp.where(upd, ntta, ttac)
                    rsac = jnp.where(upd, nrsa, rsac)
                    return (tdac, ttac, rsac), (tdac, ttac, rsac)

                # start at level icb-1 (0-based icb-1 corresponds to Fortran
                # level icb-1, the last mid level). We feed the full top-down
                # arrays and only update the low block.
                tda_cfg = tda[:, ih, im]
                tta_cfg = tta[:, ih, im]
                rsa_cfg = rsa[:, ih, im]
                init_e = (tda_cfg[0] * 0.0, tta_cfg[0] * 0.0, rsa_cfg[0] * 0.0)
                # seed carry with the value at level icb-1 (k loop in Fortran
                # runs k=icb..np updating level k from level k-1). We emulate by
                # scanning all levels k=1..npl and only updating low layers; the
                # carry at the mid/low boundary is rsa/tda at level icb-1.
                # Build a full per-level array by scanning with the existing
                # top-down composite as the seed at the boundary.
                # Simpler & exact: redo the full down-sweep but with low-block
                # cloud index = is_, reusing the high+mid result as the seed.
                def ext_down_full(carry, k):
                    tdac, ttac, rsac = carry
                    # within high+mid block keep the precomputed composite;
                    # at low block, recurse with cloud index is_.
                    denm = ts[k, is_] / (1.0 - rsac * rs[k, is_])
                    ntda = tdac * td[k, is_]
                    ntta = tdac * tt[k, is_] + (tdac * rr[k, is_] * rsac + ttac) * denm
                    nrsa = rs[k, is_] + ts[k, is_] * rsac * denm
                    low = is_low_layer[k]
                    tdac = jnp.where(low, ntda, tda[k, ih, im])
                    ttac = jnp.where(low, ntta, tta[k, ih, im])
                    rsac = jnp.where(low, nrsa, rsa[k, ih, im])
                    return (tdac, ttac, rsac), (tdac, ttac, rsac)

                seed = (tda[0, ih, im], tta[0, ih, im], rsa[0, ih, im])
                _, (tdaf, ttaf, rsaf) = lax.scan(ext_down_full, seed, jnp.arange(1, npl))
                tda_cfg = jnp.concatenate([tda[:1, ih, im], tdaf])
                tta_cfg = jnp.concatenate([tta[:1, ih, im], ttaf])
                rsa_cfg = jnp.concatenate([rsa[:1, ih, im], rsaf])

                # extend rra/rxa through the high block (going up) with index ih.
                def ext_up_full(carry, k):
                    rrac, rxac = carry
                    denm = ts[k, ih] / (1.0 - rs[k, ih] * rxac)
                    nrra = rr[k, ih] + (td[k, ih] * rrac + tt[k, ih] * rxac) * denm
                    nrxa = rs[k, ih] + ts[k, ih] * rxac * denm
                    high = is_high_layer[k]
                    rrac = jnp.where(high, nrra, rra[k, im, is_])
                    rxac = jnp.where(high, nrxa, rxa[k, im, is_])
                    return (rrac, rxac), (rrac, rxac)

                seed_u = (rra[npl, im, is_], rxa[npl, im, is_])
                _, (rraf, rxaf) = lax.scan(ext_up_full, seed_u, jnp.arange(npl - 1, -1, -1))
                rra_cfg = jnp.concatenate([rraf[::-1], rra[npl:nlev, im, is_]])
                rxa_cfg = jnp.concatenate([rxaf[::-1], rxa[npl:nlev, im, is_]])

                # fluxes (Chou 1992 eq 5): for k=2..np+1 (0-based k=1..npl)
                kk = jnp.arange(1, nlev)
                denm = 1.0 / (1.0 - rxa_cfg[kk] * rsa_cfg[kk - 1])
                fdndir = tda_cfg[kk - 1]
                xx = tda_cfg[kk - 1] * rra_cfg[kk]
                fdndif = (xx * rsa_cfg[kk - 1] + tta_cfg[kk - 1]) * denm
                fupdif = (xx + tta_cfg[kk - 1] * rxa_cfg[kk]) * denm
                flxdn_k = fdndir + fdndif - fupdif
                flxdnu_k = -fupdif
                flxdnd_k = fdndir + fdndif
                flxdn = jnp.concatenate([jnp.array([1.0 - rra_cfg[0]], dtype), flxdn_k])
                flxdnu = jnp.concatenate([jnp.array([-rra_cfg[0]], dtype), flxdnu_k])
                flxdnd = jnp.concatenate([jnp.array([1.0], dtype), flxdnd_k])

                if ih == 0 and im == 0 and is_ == 0:
                    fclr = flxdn
                fall = fall + flxdn * ct
                fallu = fallu + flxdnu * ct
                falld = falld + flxdnd * ct
                # surface direct/diffuse (k=np+1 == level npl, i.e. kk last entry)
                fsdir = fsdir + fdndir[-1] * ct
                fsdif = fsdif + fdndif[-1] * ct

    return fclr, fall, fallu, falld, fsdir, fsdif


def _flxco2(swc, swh, csm, df):
    """CO2 flux-reduction table look-up (Chou 1990). Per-column (vmapped).

    ``swc``/``swh`` are (np+1,) scaled CO2 / water amounts, ``csm`` scalar,
    ``df`` (np+1,) the running flux reduction (updated for k=2..np+1).
    """
    cah = jnp.asarray(_CAH, dtype=df.dtype)           # (22, 19)
    xx = 1.0 / 0.3
    nlev = df.shape[0]
    k = jnp.arange(1, nlev)                           # 0-based k=1..np
    clog = jnp.log10(swc[k] * csm)
    wlog = jnp.log10(swh[k] * csm)
    ic = (jnp.floor((clog + 3.15) * xx + 1.0)).astype(jnp.int32)
    iw = (jnp.floor((wlog + 4.15) * xx + 1.0)).astype(jnp.int32)
    ic = jnp.clip(ic, 2, 22)
    iw = jnp.clip(iw, 2, 19)
    dc = clog - (ic - 2).astype(df.dtype) * 0.3 + 3.0
    dw = wlog - (iw - 2).astype(df.dtype) * 0.3 + 4.0
    ic0, iw0 = ic - 1, iw - 1                         # 0-based
    x1 = cah[0, iw0 - 1] + (cah[0, iw0] - cah[0, iw0 - 1]) * xx * dw
    x2 = cah[ic0 - 1, iw0 - 1] + (cah[ic0 - 1, iw0] - cah[ic0 - 1, iw0 - 1]) * xx * dw
    y2 = x2 + (cah[ic0, iw0 - 1] - cah[ic0 - 1, iw0 - 1]) * xx * dc
    x1 = jnp.maximum(x1, y2)
    add = 0.0343 * (x1 - y2)
    df = df.at[k].add(add)
    return df


def _band_layer_rt(tausto, ssatau, asysto, tauclb_k, tauclf_k, ssacl_k, asycl_k,
                   fcld_k, csm, has_ssa_floor):
    """Compute (rr,tt,td,rs,ts) clear (index 0) and cloudy (index 1) for one
    layer/band -- the common soluv/solir inner kernel.

    Mirrors the WRF inner block: clear-sky delta-Eddington + Sagan-Pollock,
    then cloudy if tauclb>=0.01 and fcld>=0.01. ``has_ssa_floor`` toggles the
    solir ssato>0.001 branch (else the soluv path, which always calls deledd).
    Returns arrays of shape (2,) for each of rr,tt,td,rs,ts.
    """
    tauto = tausto
    ssato = ssatau / tauto + 1.0e-8
    if has_ssa_floor:
        ssato_ok = ssato > 0.001
        ssato_c = jnp.minimum(ssato, 0.999999)
        asyto = asysto / (ssato_c * tauto)
        rr1, tt1, td1 = _deledd(tauto, ssato_c, asyto, csm)
        rs1, ts1 = _sagpol(tauto, ssato_c, asyto)
        # fallback branch (pure absorption): no scattering
        td1b = jnp.exp(-tauto * csm)
        ts1b = jnp.exp(-1.66 * tauto)
        rr1 = jnp.where(ssato_ok, rr1, 0.0)
        tt1 = jnp.where(ssato_ok, tt1, 0.0)
        td1 = jnp.where(ssato_ok, td1, td1b)
        rs1 = jnp.where(ssato_ok, rs1, 0.0)
        ts1 = jnp.where(ssato_ok, ts1, ts1b)
    else:
        ssato = jnp.minimum(ssato, 0.999999)
        asyto = asysto / (ssato * tauto)
        rr1, tt1, td1 = _deledd(tauto, ssato, asyto, csm)
        rs1, ts1 = _sagpol(tauto, ssato, asyto)

    cloudy = (tauclb_k >= 0.01) & (fcld_k >= 0.01)
    # direct cloudy
    tauto_b = tausto + tauclb_k
    ssato_b = (ssatau + ssacl_k * tauclb_k) / tauto_b + 1.0e-8
    ssato_b = jnp.minimum(ssato_b, 0.999999)
    asyto_b = (asysto + asycl_k * ssacl_k * tauclb_k) / (ssato_b * tauto_b)
    rr2, tt2, td2 = _deledd(tauto_b, ssato_b, asyto_b, csm)
    # diffuse cloudy
    tauto_f = tausto + tauclf_k
    ssato_f = (ssatau + ssacl_k * tauclf_k) / tauto_f + 1.0e-8
    ssato_f = jnp.minimum(ssato_f, 0.999999)
    asyto_f = (asysto + asycl_k * ssacl_k * tauclf_k) / (ssato_f * tauto_f)
    rs2, ts2 = _sagpol(tauto_f, ssato_f, asyto_f)

    rr2 = jnp.where(cloudy, rr2, rr1)
    tt2 = jnp.where(cloudy, tt2, tt1)
    td2 = jnp.where(cloudy, td2, td1)
    rs2 = jnp.where(cloudy, rs2, rs1)
    ts2 = jnp.where(cloudy, ts2, ts1)
    return (jnp.stack([rr1, rr2], axis=-1), jnp.stack([tt1, tt2], axis=-1),
            jnp.stack([td1, td2], axis=-1), jnp.stack([rs1, rs2], axis=-1),
            jnp.stack([ts1, ts2], axis=-1))


def _soluv_solir_column(oh, dp, wh, cwp, reff, fcld, cosz, csm,
                        rsuvbm, rsuvdf, rsirbm, rsirdf, ict, icb, np_layers):
    """Per-column soluv+solir integration returning flx/flxu/flxd/flc, and the
    surface direct/diffuse IR for the GSW computation.

    Reproduces soluv (8 UV/PAR bands) and solir (3 IR bands x 10 k-intervals).
    Aerosols are zero (taual=ssaal=asyal=0). cwp/reff are (np,2). Returns
    flx,flxu,flxd,flc (np+1,) and fdirir,fdifir scalars (and uv/par sfc fluxes,
    unused downstream but kept for completeness).
    """
    npl = np_layers
    dtype = oh.dtype
    nlev = npl + 1
    flx = jnp.zeros((nlev,), dtype)
    flxu = jnp.zeros((nlev,), dtype)
    flxd = jnp.zeros((nlev,), dtype)
    flc = jnp.zeros((nlev,), dtype)
    fdirir = jnp.zeros((), dtype)
    fdifir = jnp.zeros((), dtype)

    cwp_ice = cwp[:, 0]
    cwp_liq = cwp[:, 1]
    reff_ice = reff[:, 0]
    reff_liq = reff[:, 1]

    # ============================ soluv (UV+PAR) =============================
    uv_hk = jnp.asarray(_UV_HK, dtype)
    uv_xk = jnp.asarray(_UV_XK, dtype)
    uv_ry = jnp.asarray(_UV_RY, dtype)
    aig = jnp.asarray(_UV_AIG, dtype)
    awg = jnp.asarray(_UV_AWG, dtype)

    # cloud optical thickness (cldwater path)
    taucld_ice = cwp_ice * (3.33e-4 + 2.52 / reff_ice)
    taucld_liq = cwp_liq * (-6.59e-3 + 1.65 / reff_liq)
    taucld_uv = jnp.stack([taucld_ice, taucld_liq], axis=-1)        # (np,2)
    cc, tauclb, tauclf = _cldscale(cosz, fcld, taucld_uv, ict, icb, npl)

    # cloud asymmetry (single value across UV bands)
    taux = taucld_ice + taucld_liq
    active = (taux > 0.05) & (fcld > 0.01)
    reff1 = jnp.minimum(reff_ice, 130.0)
    reff2 = jnp.minimum(reff_liq, 20.0)
    g1 = (aig[0] + (aig[1] + aig[2] * reff1) * reff1) * taucld_ice
    g2 = (awg[0] + (awg[1] + awg[2] * reff2) * reff2) * taucld_liq
    asycl_uv = jnp.where(active, (g1 + g2) / jnp.where(taux > 0, taux, 1.0), 1.0)

    for ib in range(8):
        taurs = uv_ry[ib] * dp
        tauoz = uv_xk[ib] * oh
        tausto = taurs + tauoz + 1.0e-8                # taual=0
        ssatau = taurs                                 # ssaal*taual=0 -> +taurs
        asysto = jnp.zeros_like(tausto)                # asyal*ssaal*taual=0
        rr_l, tt_l, td_l, rs_l, ts_l = _band_layer_rt(
            tausto, ssatau, asysto, tauclb, tauclf,
            jnp.ones_like(tauclb), asycl_uv, fcld, csm, has_ssa_floor=False)
        rr, tt, td, rs, ts = _append_surface(rr_l, tt_l, td_l, rs_l, ts_l, rsuvbm, rsuvdf)
        fclr, fall, fallu, falld, fsdir, fsdif = _cldflx(rr, tt, td, rs, ts, cc, ict, icb, npl)
        flx = flx + fall * uv_hk[ib]
        flxu = flxu + fallu * uv_hk[ib]
        flxd = flxd + falld * uv_hk[ib]
        flc = flc + fclr * uv_hk[ib]
        # uv/par surface fluxes are not needed downstream (GSW uses flxd).

    # ============================ solir (near-IR) ============================
    ir_xk = jnp.asarray(_IR_XK, dtype)
    ir_hk = jnp.asarray(_IR_HK, dtype)                 # (3,10)
    ir_aib = jnp.asarray(_IR_AIB, dtype)               # (3,2)
    ir_awb = jnp.asarray(_IR_AWB, dtype)
    ir_aia = jnp.asarray(_IR_AIA, dtype)               # (3,3)
    ir_awa = jnp.asarray(_IR_AWA, dtype)
    ir_aig = jnp.asarray(_IR_AIG, dtype)
    ir_awg = jnp.asarray(_IR_AWG, dtype)

    for ib in range(3):
        taucld_ice = cwp_ice * (ir_aib[ib, 0] + ir_aib[ib, 1] / reff_ice)
        taucld_liq = cwp_liq * (ir_awb[ib, 0] + ir_awb[ib, 1] / reff_liq)
        taucld_ir = jnp.stack([taucld_ice, taucld_liq], axis=-1)
        cc, tauclb, tauclf = _cldscale(cosz, fcld, taucld_ir, ict, icb, npl)

        taux = taucld_ice + taucld_liq
        active = (taux > 0.05) & (fcld > 0.01)
        reff1 = jnp.minimum(reff_ice, 130.0)
        reff2 = jnp.minimum(reff_liq, 20.0)
        w1 = (1.0 - (ir_aia[ib, 0] + (ir_aia[ib, 1] + ir_aia[ib, 2] * reff1) * reff1)) * taucld_ice
        w2 = (1.0 - (ir_awa[ib, 0] + (ir_awa[ib, 1] + ir_awa[ib, 2] * reff2) * reff2)) * taucld_liq
        ssacl = jnp.where(active, (w1 + w2) / jnp.where(taux > 0, taux, 1.0), 1.0)
        g1 = (ir_aig[ib, 0] + (ir_aig[ib, 1] + ir_aig[ib, 2] * reff1) * reff1) * w1
        g2 = (ir_awg[ib, 0] + (ir_awg[ib, 1] + ir_awg[ib, 2] * reff2) * reff2) * w2
        wsum = w1 + w2
        asycl = jnp.where(active, (g1 + g2) / jnp.where(jnp.abs(wsum) > 0, wsum, 1.0), 1.0)

        for ik in range(10):
            tauwv = ir_xk[ik] * wh
            tausto = tauwv + 1.0e-8                     # taual=0
            ssatau = jnp.zeros_like(tausto)            # ssaal*taual=0
            asysto = jnp.zeros_like(tausto)
            rr_l, tt_l, td_l, rs_l, ts_l = _band_layer_rt(
                tausto, ssatau, asysto, tauclb, tauclf,
                ssacl, asycl, fcld, csm, has_ssa_floor=True)
            rr, tt, td, rs, ts = _append_surface(rr_l, tt_l, td_l, rs_l, ts_l, rsirbm, rsirdf)
            fclr, fall, fallu, falld, fsdir, fsdif = _cldflx(rr, tt, td, rs, ts, cc, ict, icb, npl)
            hk = ir_hk[ib, ik]
            flx = flx + fall * hk
            flxu = flxu + fallu * hk
            flxd = flxd + falld * hk
            flc = flc + fclr * hk
            fdirir = fdirir + fsdir * hk
            fdifir = fdifir + fsdif * hk

    return flx, flxu, flxd, flc, fdirir, fdifir


def _append_surface(rr_l, tt_l, td_l, rs_l, ts_l, rbm, rdf):
    """Append the WRF surface boundary row (level np) to the (np,2) layer
    arrays, producing (np+1,2) arrays.

    Surface row (both cloud indices): rr=rdf? No -- WRF sets rr(np+1,*)=rsbm
    (beam reflectance), rs(np+1,*)=rsdf (diffuse reflectance), td=tt=ts=0.
    """
    dtype = rr_l.dtype
    rbm_row = jnp.full((1, 2), rbm, dtype)
    rdf_row = jnp.full((1, 2), rdf, dtype)
    zero_row = jnp.zeros((1, 2), dtype)
    rr = jnp.concatenate([rr_l, rbm_row], axis=0)
    rs = jnp.concatenate([rs_l, rdf_row], axis=0)
    td = jnp.concatenate([td_l, zero_row], axis=0)
    tt = jnp.concatenate([tt_l, zero_row], axis=0)
    ts = jnp.concatenate([ts_l, zero_row], axis=0)
    return rr, tt, td, rs, ts


def _sorad_column(p8w_mb_td, t_td, sh_td, o3_td, cwc_td, reff_td, fcld_td,
                  cosz, alb, ict, icb, np_layers):
    """Per-column sorad: scale gases/clouds, run soluv+solir, apply O2/CO2
    flux reductions, and return the net downward flux flx (np+1,) and the
    direct downward flux flxd (np+1,), as fractions of TOA.

    All inputs are top-down (index 0 = phantom top layer .. index np-1 = surface
    layer); ``p8w_mb_td`` has np+1 interface levels (mb). Mirrors sorad exactly.
    """
    npl = np_layers
    dtype = t_td.dtype
    csm = 35.0 / jnp.sqrt(1224.0 * cosz * cosz + 1.0)

    # layer thickness dp (mb) and pressure-scaling
    pl = p8w_mb_td                                     # (np+1,)
    dp = pl[1:] - pl[:-1]                               # (np,)
    scal = dp * (0.5 * (pl[:-1] + pl[1:]) / 300.0) ** 0.8
    wh = 1.02 * sh_td * scal * (1.0 + 0.00135 * (t_td - 240.0)) + 1.0e-11
    oh = 1.02 * o3_td * dp * 466.7 + 1.0e-11
    cwp_ice = 1.02 * 10000.0 * cwc_td[:, 0] * dp
    cwp_liq = 1.02 * 10000.0 * cwc_td[:, 1] * dp
    cwp = jnp.stack([cwp_ice, cwp_liq], axis=-1)        # (np,2)

    rsuvbm = alb
    rsuvdf = alb
    rsirbm = alb
    rsirdf = alb

    flx, flxu, flxd, flc, fdirir, fdifir = _soluv_solir_column(
        oh, dp, wh, cwp, reff_td, fcld_td, cosz, csm,
        rsuvbm, rsuvdf, rsirbm, rsirdf, ict, icb, npl)

    # O2 flux reduction (Chou 1990). so2(k+1)=so2(k)+165.22*scal(k).
    so2_o2 = jnp.concatenate([jnp.zeros((1,), dtype), jnp.cumsum(165.22 * scal)])  # (np+1,)
    df = jnp.zeros((npl + 1,), dtype)
    k = jnp.arange(1, npl + 1)
    x = so2_o2[k] * csm
    df = df.at[k].add(0.0287 * (1.0 - jnp.exp(-0.00027 * jnp.sqrt(x))))

    # scaled CO2 amount: so2(k+1)=so2(k)+co2*789*scal(k)+1e-11. swh = water swh.
    so2_co2 = jnp.concatenate([jnp.zeros((1,), dtype), jnp.cumsum(_CO2 * 789.0 * scal + 1.0e-11)])
    swh = jnp.concatenate([jnp.zeros((1,), dtype), jnp.cumsum(wh)])
    df = _flxco2(so2_co2, swh, csm, df)

    # adjust clear-sky for o2+co2 (not used downstream for flux, but kept faithful)
    flc = flc.at[1:].add(-df[1:])

    # all-sky adjustment using sclr/sdf cloud-cover accumulation.
    def acc(carry, k):
        sdf_c, sclr_c = carry
        cond = fcld_td[k] > 0.01
        sdf_n = jnp.where(cond, sdf_c + df[k] * sclr_c * fcld_td[k], sdf_c)
        sclr_n = jnp.where(cond, sclr_c * (1.0 - fcld_td[k]), sclr_c)
        # the flux update uses the UPDATED sdf/sclr at level k+1.
        upd = sdf_n + df[k + 1] * sclr_n
        return (sdf_n, sclr_n), upd

    (sdf_final, sclr_final), upd = lax.scan(acc, (jnp.zeros((), dtype), jnp.ones((), dtype)), jnp.arange(npl))
    # flx(k+1) -= upd(k) for k=0..np-1 -> indices 1..np.
    flx = flx.at[1:].add(-upd)
    flxu = flxu.at[1:].add(-upd)
    flxd = flxd.at[1:].add(-upd)

    # direct downward IR adjustment at the surface (np-th level == index npl).
    surf_add = (sdf_final + df[npl] * sclr_final) * rsirbm
    flx = flx.at[npl].add(surf_add)
    flxu = flxu.at[npl].add(surf_add)
    flxd = flxd.at[npl].add(surf_add)
    return flx, flxd


def _gsfc_sw_columns(T, p, p8w, qv, qc, qr, qi, qs, qg, cldfra, coszen, albedo,
                     solcon, iprof, cp, g, f_qi, warm_rain):
    """Vectorized GSFC SW over columns (model order in/out).

    Reproduces GSFCSWRAD: builds the top-down work arrays with a phantom layer
    at the top, calls sorad, scales fluxes to W/m^2, and computes TTEN (K/s),
    GSW (W/m^2), RSWTOA (W/m^2). f_qi / warm_rain are static.
    """
    dtype = jnp.result_type(T, jnp.float64)
    ncol, nz = T.shape
    npl = nz + 1                                       # sorad layers (incl phantom)

    day = coszen > _THRESH

    # ---- WRF reversals: model order (k=0 bottom) -> top-down 2D (index 0 top)
    def flip(x):
        return x[:, ::-1]

    t2d_real = flip(T)                                 # (ncol, nz), index0=top
    p2d_real_mb = flip(p) * 0.01
    sh_real = jnp.maximum(flip(qv) / (1.0 + flip(qv)), 0.0)
    fcld_real = flip(cldfra)
    qc_td = flip(qc)
    qi_td = flip(qi)
    qr_td = flip(qr)
    qs_td = flip(qs)
    qg_td = flip(qg)

    # P8W2D: index K=1..nz+1 -> p8w3d(nk) nk=kme-K+kms; model p8w index 0=surface
    # so top-down interface = flip(p8w). P8W2D(0)=0 (phantom top interface).
    p8w_mb_flipped = flip(p8w) * 0.01                  # (ncol, nz+1), index0=model top iface
    # build (ncol, np+1=nz+2) with index0=0 phantom, then 1..nz+1 = flip(p8w)
    p8w_full = jnp.concatenate([jnp.zeros((ncol, 1), dtype), p8w_mb_flipped], axis=1)  # (ncol, nz+2)

    # cwc(K,1)=ice, cwc(K,2)=liquid. Default predicate=F_QI; if not warm_rain
    # and not F_QI, supercooled qc moves to ice below 273.15. With F_QI present,
    # cwc(:,1)=QI, cwc(:,2)=QC.
    cwc_ice = jnp.zeros_like(qc_td)
    cwc_liq = jnp.maximum(qc_td, 0.0)
    if (not warm_rain) and (not f_qi):
        cold = t2d_real < 273.15
        cwc_ice = jnp.where(cold, cwc_liq, 0.0)
        cwc_liq = jnp.where(cold, 0.0, cwc_liq)
    if f_qi:
        cwc_ice = jnp.maximum(qi_td, 0.0)

    # phantom top layer (index 0 in the np-grid): copy/zero per WRF.
    t_np = jnp.concatenate([t2d_real[:, :1], t2d_real], axis=1)            # T2D(0)=T2D(1)
    sh_np = jnp.concatenate([0.5 * sh_real[:, :1], sh_real], axis=1)
    fcld_np = jnp.concatenate([jnp.zeros((ncol, 1), dtype), fcld_real], axis=1)
    cwc_ice_np = jnp.concatenate([jnp.zeros((ncol, 1), dtype), cwc_ice], axis=1)
    cwc_liq_np = jnp.concatenate([jnp.zeros((ncol, 1), dtype), cwc_liq], axis=1)
    cwc_np = jnp.stack([cwc_ice_np, cwc_liq_np], axis=-1)                  # (ncol, np, 2)
    # P2D(0)=0.5*(P8W2D(0)+P8W2D(1)); interior P2D from real p.
    p2d_np = jnp.concatenate([0.5 * (p8w_full[:, :1] + p8w_full[:, 1:2]), p2d_real_mb], axis=1)

    # ozone profile on the np-grid (top-down incl phantom), via o3prof.
    o3_np = _o3prof(p2d_np, iprof)                                        # (ncol, np)

    # effective radius: reff(:,2)=10 liquid default, reff(:,1)=80 ice (microns)
    reff_ice = jnp.full((ncol, npl), 80.0, dtype)
    reff_liq = jnp.full((ncol, npl), 10.0, dtype)
    reff_np = jnp.stack([reff_ice, reff_liq], axis=-1)                    # (ncol, np, 2)

    # ict/icb: WRF loops k=kts-1..kte+1 over P8W2D (indices 0..np) and records
    # the index k whose P8W2D is closest to 400 / 700 mb (LAST such k wins on a
    # strict `<` comparison, i.e. argmin keeping the FIRST minimum -- ties go to
    # the smaller index because WRF replaces only on strictly-smaller). Our
    # p8w_full[k] == WRF P8W2D[k] exactly, and the resulting index is used
    # directly as the 1-based sorad layer boundary (no offset) -- WRF assigns
    # ict=k and then tests `k < ict` with k the sorad layer index on the SAME
    # base. argmin returns the first (smallest-index) minimum, matching WRF's
    # strict-`<` replacement.
    d400 = jnp.abs(p8w_full - 400.0)
    d700 = jnp.abs(p8w_full - 700.0)
    ict = jnp.argmin(d400, axis=1)
    icb = jnp.argmin(d700, axis=1)

    # vmap sorad across columns (ict/icb per column).
    flx, flxd = _vmap_sorad(p8w_full, t_np, sh_np, o3_np, cwc_np, reff_np, fcld_np,
                            coszen, albedo, ict, icb, npl)

    # scale fluxes to W/m^2: flx(k)=flx(k)*SOLCON*cosz (day only).
    scale = solcon * coszen                                               # (ncol,)
    flx_w = jnp.where(day[:, None], flx * scale[:, None], 0.0)            # (ncol, np+1)

    # heating rate (deg/sec): TTEN2D(k)=-fac*(flx(k)-flx(k+1))/(p8w(k)-p8w(k+1))
    # k=kts..kte -> np-grid indices 1..np (the real layers), interfaces k..k+1.
    fac = 0.01 * g / cp
    # WRF: TTEN2D(i,k) = -fac*(flx(i,k)-flx(i,k+1))/(p8w2d(i,k)-p8w2d(i,k+1))
    # with k=kts..kte (1..nz). In our top-down np-grid, model layer k maps to
    # np-grid level index k (since phantom shifted everything by +0 at the top:
    # real layers occupy np-grid indices 1..np). The flux array flx_w has np+1
    # entries (np-grid interface levels 0..np). For sorad, flx(k) is at the TOP
    # interface of np-grid layer k. WRF heating uses flx(k)-flx(k+1) over the
    # real layers k=1..np (np-grid), i.e. indices 1..np-1 differences? See note.
    # WRF k=kts..kte=1..nz refers to the ORIGINAL model arrays; inside sorad the
    # corresponding flx index is the same k (1..nz) because sorad's level 1 is
    # the TOA and level np+1 the surface. flx(k) here = flx_w[:, k]. So:
    kk = jnp.arange(1, nz + 1)                                            # 1..nz
    dflx = flx_w[:, kk] - flx_w[:, kk + 1]
    dpp = p8w_full[:, kk] - p8w_full[:, kk + 1]
    tten_td = jnp.where(day[:, None], -fac * dflx / dpp, 0.0)             # (ncol, nz) top-down
    tten_td = jnp.maximum(tten_td, 0.0)                                  # GODDARD negative fix

    # GSW = (1-rsuvbm)*flxd(np+1)*SOLCON*cosz (day only). flxd surface=index np.
    gsw = jnp.where(day, (1.0 - albedo) * flxd[:, npl] * scale, 0.0)
    # RSWTOA = flx(kts) - flxd(kts)*SOLCON*cosz. kts (model) -> np-grid index 1.
    rswtoa = jnp.where(day, flx_w[:, 1] - flxd[:, 1] * scale, 0.0)

    # map tten back to model order (flip top-down -> bottom-up).
    tten_model = tten_td[:, ::-1]
    return tten_model, gsw, rswtoa


def _vmap_sorad(p8w_full, t_np, sh_np, o3_np, cwc_np, reff_np, fcld_np,
                coszen, albedo, ict, icb, npl):
    return jax.vmap(
        _sorad_column, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
    )(p8w_full, t_np, sh_np, o3_np, cwc_np, reff_np, fcld_np, coszen, albedo, ict, icb, npl)


def solve_gsfc_sw_column(state: GsfcSWColumnState) -> GsfcSWColumnResult:
    """Run the GSFC shortwave kernel on a batch of model-order columns.

    Returns the per-layer temperature heating rate (K/s, model order, >=0), the
    surface net downward shortwave flux GSW (W/m^2) and the TOA upward residual
    RSWTOA (W/m^2).
    """
    iprof = _select_iprof(state.center_lat, state.julday)   # static int
    tten, gsw, rswtoa = _gsfc_sw_columns(
        state.T, state.p, state.p8w, state.qv, state.qc, state.qr, state.qi,
        state.qs, state.qg, state.cldfra, state.coszen, state.albedo,
        jnp.asarray(state.solcon),
        int(iprof), float(state.cp), float(state.g),
        bool(state.f_qi), bool(state.warm_rain),
    )
    return GsfcSWColumnResult(heating_rate=tten, gsw=gsw, rswtoa=rswtoa)


__all__ = [
    "GsfcSWColumnState",
    "GsfcSWColumnResult",
    "solve_gsfc_sw_column",
]
