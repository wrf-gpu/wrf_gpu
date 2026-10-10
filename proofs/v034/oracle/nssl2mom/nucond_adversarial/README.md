# NUCOND direct-call adversarial oracle (dev provenance, lane o1-nssl)

Pristine `phys/module_mp_nssl_2mom.F` with ONE added line `public nucond` (visibility only; verified by
`diff` against the pristine source) driven by `nucond_direct.f90` on randomized/targeted columns
(`adv.py`, `targeted.py`, `direct.py`), packaged by `make_dataset.py` into
`proofs/v034/f2_oracles/nssl_2mom/nucond_adversarial.npz` (sha256 3fbb88d3...dc0d4a).
The direct-call oracle reproduces the instrumented-driver stage S3->S4 bit-exactly on all 14 cases
(fp32 and fp64). Scripts reference the lane work dir <USER_HOME>/wrf_gpu2_lanes/o1-nssl/nucond_dev.
gcov (fp64) confirms the RK2c halving retry, maxsupersat QVEXCESS cap, near-saturation dt/10 step,
droplet/rain clamps, cx<=cxmin re-seed, full/partial evaporation and nucleation with/without CCN
depletion are exercised.
