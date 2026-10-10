"""JAX port of WRF NSSL 2-moment microphysics (``mp_physics=18``), default configuration.

Port of the UNMODIFIED ``phys/module_mp_nssl_2mom.F`` (sha256 29f42e76...) driver path
``nssl_2mom_driver`` -> calcnfromq -> sediment1d -> nssl_2mom_gs -> NUCOND -> smallvalues
(+ radardd02 / calc_eff_radius diagnostics), for the WRF-default mp=18 configuration
(ipconc=5, hail on, predicted CCN with irenuc=5 / activated CCN, graupel+hail volume,
icdx=icdxhl=6).  Conventions (see ``indices.py``):

* Species indices are the FORTRAN ones (lt=1 ... lccna=18); state stacks ``an`` have a
  leading axis of length ``NA + 1`` whose index 0 is unused, so ``an[LC]`` == Fortran
  ``an(..., lc)``.
* Precision is explicit: ``Prec.R`` = WRF default REAL (float32 in WRF, float64 for the
  ``-fdefault-real-8`` oracle), ``Prec.D`` = DOUBLE PRECISION (always float64).
"""
