# Review: V0.14 Step-1 MYNN Source Coupling

Verdict: `STEP1_STRICT_NOT_CLOSED_AFTER_NOAHMP_ENABLEMENT_SEE_NOAHMP_STEP1_CLOSURE`.

The production adapter fixes are scoped and tested, but strict Step-1 is not closed.
Current after-conv `T_TENDF`: max_abs `1489.5135568470864`, RMSE `13.2001844004901`.

MYNN kernel/source units are not the primary blocker: WRF inputs + WRF QKE raw `RTHBLTEN` max_abs `0.00026978377168347277`, RMSE `2.5913062928007185e-06`.
The narrower blocker is the WRF surface/land flux handoff into MYNN: driver-vs-SFCLAY HFX max_abs `277.80298614281253`.

Next: JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/noahmp_step1_closure.py
