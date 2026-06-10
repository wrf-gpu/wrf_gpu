# Review: V0.14 Step-1 SFCLAY Boundary

Verdict: `STEP1_SFCLAY_FIRST_CALL_FIXED_NEXT_BLOCKER_TSK_ZNT_SURFACE_INPUTS`.

Implemented WRF `itimestep<=1` MYNN surface semantics in production and direct Step-1 proof helpers.
Strict Step-1 after-conv metric: max_abs `1497.6112467075195`, rmse `13.296448784742802`.

Primary residual status:
{
  "evidence": {
    "surface_znt_after_mynn_vs_wrf_znt": {
      "first_timestep": {
        "bias": -0.01676578168262908,
        "max_abs": 0.9737602076530456,
        "ref_max_abs": 1.0,
        "rmse": 0.10376647928375139
      },
      "warm": {
        "bias": -0.01676848707923353,
        "max_abs": 0.9737602076530456,
        "ref_max_abs": 1.0,
        "rmse": 0.10376647972218284
      }
    },
    "tskin_vs_wrf_ts": {
      "bias": -0.07544341683712853,
      "max_abs": 8.344940187890643,
      "ref_max_abs": 305.0635376,
      "rmse": 1.1131824623693845
    },
    "znt_input_vs_wrf_znt": {
      "bias": -0.014158440233910828,
      "max_abs": 0.9737602076530456,
      "ref_max_abs": 1.0,
      "rmse": 0.10380183155466993
    }
  },
  "hypothesis": "Skin-temperature and roughness sourcing remain the narrower WRF-anchored blocker.",
  "rank": 2,
  "status": "SURVIVES"
}

Next boundary: Narrow next blocker: WRF-anchored TSK/ZNT surface input sourcing before sfclay_mynn. Fastest next command is a tiny surface-driver hook around module_surface_driver/module_sf_mynn for incoming TSK/ZNT/UST/QSFC/MOL and outgoing UST/HFX/QFX/ZNT on the current d02 Step-1 case, then compare those exact arrays against JAX _surface_column_view inputs and diagnostics.
