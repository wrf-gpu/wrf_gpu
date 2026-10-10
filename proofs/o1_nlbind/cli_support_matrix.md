| namelist edit (Swiss RD11 d01, release defaults) | bound | program changes | result | evidence |
|---|---|---|---|---|
| `bl_pbl_physics=1, sf_sfclay_physics=1` | True | True | traces | pre-rebase (main 643416607) |
| `cu_physics=1` | True | True | traces | pre-rebase (main 643416607) |
| `cu_physics=2` | True | True | traces | pre-rebase (main 643416607) |
| `cu_physics=3` | True | True | traces | pre-rebase (main 643416607) |
| `cu_physics=6` | True | True | traces | pre-rebase (main 643416607) |
| `cu_physics=16` | True | True | traces | pre-rebase (main 643416607) |
| `diff_opt=0, km_opt=0` | True | True | traces | pre-rebase (main 643416607) |
| `diff_opt=2, km_opt=1` | True | True | traces | pre-rebase (main 643416607) |
| `diff_opt=2, km_opt=2` | True | True | traces | pre-rebase (main 643416607) |
| `diff_opt=2, km_opt=3` | True | True | traces | pre-rebase (main 643416607) |
| `diff_opt=2, km_opt=5` | True | True | traces | pre-rebase (main 643416607) |
| `damp_opt=0` | True | True | traces | pre-rebase (main 643416607) |
| `w_damping=0` | True | True | traces | pre-rebase (main 643416607) |
| `diff_6th_opt=0` | True | True | traces | pre-rebase (main 643416607) |
| `epssm=0.1` | True | True | traces | pre-rebase (main 643416607) |
| `sf_surface_physics=0` | True | True | traces | pre-rebase (main 643416607) |
| `sf_surface_physics=1` | None | None | refused/crash: ValueError: d01: sf_surface_physics=1 is not wired in the standalone nested pipeline (supported: 0 = prescribed bulk surface, 4 = Noah-MP). Refusing t | pre-rebase (main 643416607) |
| `mp_physics=0` | True | True | traces | pre-rebase (main 643416607) |
| `mp_physics=1` | True | True | traces | final (main dee37ef2a) |
| `mp_physics=2` | True | True | traces | final (main dee37ef2a) |
| `mp_physics=3` | True | True | traces | final (main dee37ef2a) |
| `mp_physics=4` | True | True | traces | final (main dee37ef2a) |
| `mp_physics=6` | True | True | traces | final (main dee37ef2a) |
| `mp_physics=10` | True | True | traces | final (main dee37ef2a) |
| `mp_physics=13` | True | True | traces | final (main dee37ef2a) |
| `mp_physics=14` | True | True | traces | final (main dee37ef2a) |
| `mp_physics=16` | True | True | traces | final (main dee37ef2a) |
| `mp_physics=24` | True | None | refused/crash: NotImplementedError: GPUWRF_ROOT_SCALAR_BDY_RK1 covers ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'Ni', 'Nr'); 'qh' is not represented | pre-rebase (main 643416607) |
| `mp_physics=26` | True | None | refused/crash: NotImplementedError: GPUWRF_ROOT_SCALAR_BDY_RK1 covers ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'Ni', 'Nr'); 'qh' is not represented | pre-rebase (main 643416607) |
| `mp_physics=28` | True | None | refused/crash: NotImplementedError: GPUWRF_ROOT_SCALAR_BDY_RK1 covers ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'Ni', 'Nr'); 'nwfa' is not represented | pre-rebase (main 643416607) |
| `mp_physics=97` | True | True | traces | final (main dee37ef2a) |
| `bl_pbl_physics=0, sf_sfclay_physics=0` | True | True | traces | pre-rebase (main 643416607) |
| `bl_pbl_physics=2, sf_sfclay_physics=2` | True | True | traces | pre-rebase (main 643416607) |
| `bl_pbl_physics=3, sf_sfclay_physics=3` | True | True | traces | pre-rebase (main 643416607) |
| `bl_pbl_physics=7, sf_sfclay_physics=7` | True | None | refused/crash: UnsupportedSchemeSelection: surface-layer/PBL pairing violation: bl_pbl_physics=7 (YSU/ACM2/BouLac/Shin-Hong/GBM/MRF) re-derives its surface-layer for | pre-rebase (main 643416607) |
| `bl_pbl_physics=8, sf_sfclay_physics=1` | True | True | traces | final (main dee37ef2a) |
| `bl_pbl_physics=11, sf_sfclay_physics=1` | True | True | traces | pre-rebase (main 643416607) |
| `bl_pbl_physics=12, sf_sfclay_physics=1` | True | True | traces | final (main dee37ef2a) |
| `bl_pbl_physics=99, sf_sfclay_physics=1` | True | True | traces | pre-rebase (main 643416607) |
| `bl_pbl_physics=1, sf_sfclay_physics=91` | True | None | refused/crash: UnsupportedSchemeSelection: surface-layer/PBL pairing violation: bl_pbl_physics=1 (YSU/ACM2/BouLac/Shin-Hong/GBM/MRF) re-derives its surface-layer for | pre-rebase (main 643416607) |
| `ra_lw_physics=0` | True | True | traces | pre-rebase (main 643416607) |
| `ra_lw_physics=1` | True | True | traces | pre-rebase (main 643416607) |
| `ra_lw_physics=31, ra_sw_physics=0` | True | True | traces | pre-rebase (main 643416607) |
| `ra_sw_physics=0` | True | True | traces | pre-rebase (main 643416607) |
| `ra_sw_physics=1` | True | True | traces | pre-rebase (main 643416607) |
| `ra_sw_physics=2` | True | True | traces | HEAD e90c1d4f4 (main dee37ef2a) |
| `mp_physics=40` | True | None | refused/crash: NotImplementedError: GPUWRF_ROOT_SCALAR_BDY_RK1 covers ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'Ni', 'Nr'); 'Ns' is not represented | final v1 (candidate c82b9b75b; island arms superseded) |
| `bl_pbl_physics=9` | True | True | traces | final v1 (candidate c82b9b75b; island arms superseded) |
| `diff_opt=2, km_opt=1, khdif=100, kvdif=1` | True | True | traces | final v1 (candidate c82b9b75b; island arms superseded) |
| `bl_pbl_physics=7, sf_sfclay_physics=1` | True | True | traces | HEAD e90c1d4f4 (main dee37ef2a) |
| `sf_sfclay_physics=1` | True | True | traces | final (main dee37ef2a) |
| `sf_sfclay_physics=7` | True | True | traces | final (main dee37ef2a) |
| `sf_sfclay_physics=91` | True | True | traces | final (main dee37ef2a) |
| `cu_physics=5` | True | True | traces | rebased 1188aba98 (main b5100f705) |
| `cu_physics=93` | True | True | traces | rebased 1188aba98 (main b5100f705) |
| `cu_physics=4` | True | None | refused pre-JAX: WRF ARW rejects cu_physics=4 (check_a_mundo); scale-aware SAS kernel is available via the Python API only, no CPU-WRF reference exists | o1-sas (tests/test_v034_scalesas_wiring.py) |
| `ra_lw_physics=3, ra_sw_physics=3` | True | True | traces | rebased 2f9004938 (main 14253f8ca, +o1-camrad) |
| `ra_lw_physics=3, ra_sw_physics=3, cam_abs_freq_s=10800` | True | True | traces | rebased 2f9004938 (main 14253f8ca, +o1-camrad) |

| release identity | base sha | candidate sha | identical |
|---|---|---|---|
| pre-rebase (main 643416607) :: cpu | `2bf12cb8374d` | `2bf12cb8374d` | True |
| final v1 (candidate c82b9b75b; island arms superseded) :: cpu | `2bf12cb8374d` | `2bf12cb8374d` | True |
| final (main dee37ef2a) :: cpu | `2bf12cb8374d` | `2bf12cb8374d` | True |
| final PROD d01 (main dee37ef2a) :: s0_case_20260725 | `6fc715a2bd63` | `6fc715a2bd63` | True |
| HEAD e90c1d4f4 (main dee37ef2a) :: cpu | `2bf12cb8374d` | `2bf12cb8374d` | True |
| HEAD e90c1d4f4 PROD d01 (main dee37ef2a) :: s0_case_20260725 | `6fc715a2bd63` | `6fc715a2bd63` | True |
| rebased 1188aba98 (main b5100f705) :: cpu | `2bf12cb8374d` | `2bf12cb8374d` | True |
| rebased 1188aba98 PROD d01 (main b5100f705) :: s0_case_20260725 | `6fc715a2bd63` | `6fc715a2bd63` | True |
| rebased 2f9004938 (main 14253f8ca, +o1-camrad) :: cpu | `2bf12cb8374d` | `2bf12cb8374d` | True |
| rebased 2f9004938 PROD d01 (main 14253f8ca) :: s0_case_20260725 | `6fc715a2bd63` | `6fc715a2bd63` | True |
