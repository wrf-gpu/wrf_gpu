# Review: V0.14 Step-1 SFCLAY Output Algebra

Verdict: `SFCLAY_OUTPUT_ALGEBRA_BOUNDED_NEXT_BLOCKER_MYNN_SOURCE_COUPLING`.

Surface-layer output algebra is now bounded at the WRF Step-1 boundary after three local fixes: first-step `BR` clamp, `QVSH` virtual-theta, and WRF `phy_prep` density threading.

Residuals: `UST` `0.0007252174862408534`, `HFX` `0.2643125302157898`, `QFX` `6.468560998136325e-08`, `BR` `0.01166976922050278`.

Strict Step-1 remains red: max_abs `847.1446969755725`, RMSE `9.627208432391289`.
Next blocker is later MYNN/PBL source coupling; rerun with a raw MYNNEDMF source hook after the fixed surface outputs.

Proof: `/home/enric/src/wrf_gpu2/proofs/v014/step1_sfclay_output_algebra.md`
WRF hook patch: `/home/enric/src/wrf_gpu2/proofs/v014/step1_sfclay_output_algebra_wrf_patch.diff`
