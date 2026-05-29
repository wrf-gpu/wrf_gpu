You are GPT-5.5 xhigh, the WRF-domain verifier for the wrf_gpu2 project (JAX GPU port of the WRF v4 dynamical core). After four Opus sprints the dry acoustic core is WRF-cadence-faithful, damped, conservative, circulating, with correct MUT/MUTS mass semantics (your prior F7D verification). But the idealized cases (Skamarock warm bubble, Straka density current) still go non-finite (~80 s / ~40 s) from a **linear w runaway**, and the diagnosis has shifted three times. I need you to settle ONE pivotal fork definitively from WRF source, because it decides whether the next sprint fixes the test setup or the dycore.

This is a VERIFY task. Do NOT edit source. Your ONLY write is the findings file named at the end.

## The observation (quantified, `proofs/f7d/rwtend_check.json`)
A θ′-only warm-bubble IC (θ′max = 2 K, dry-mass perturbation **mu'=0**, base ph, base p) produces a frozen vertical buoyancy tendency `rw_tend` ≈ **0.615 m/s² = 9.4× the expected physical buoyancy** g·θ'/θ0 ≈ 0.065 m/s². This 9.4× over-forcing exactly matches the observed runaway slope. The buoyancy is entirely the `pg_buoy_w` pressure-gradient term `rdn·Δp'`; the mass term `c1f·mu'` is exactly 0 because mu'=0. A θ'=0 base column is exactly stable (max|w|=0 to 100 s); epssm 0.1 vs 0.5 gives identical runaway (constant forcing, not an eigenmode).

## The fork to resolve
- **Hypothesis A (IC bug):** WRF's idealized init for these cases produces a hydrostatically-balanced *perturbed* column with a consistent **mu'≠0** (and consistent ph'/p'), so `pg_buoy_w`'s `rdn·Δp'` and `c1f·mu'` nearly cancel to the small physical buoyancy residual. JAX's IC sets mu'=0, breaking that cancellation → 9.4× over-forcing. Fix = balance the IC (port WRF's iterative column balance).
- **Hypothesis B (dycore bug):** The standard Skamarock/Straka benchmark is a θ-only perturbation (mu'=0 is correct), and `pg_buoy_w` from a θ-only perturbation should already yield ≈ g·θ'/θ0 ≈ 0.065. If JAX yields 0.615 (9.4×), the bug is in the JAX `pg_buoy_w` and/or the perturbation-pressure p′ diagnosis (over-counting), NOT the IC. Fix = the dycore buoyancy/pressure operator.

## What to determine from WRF source (ground truth, `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/`)
1. **WRF idealized init**: read `dyn_em/module_initialize_ideal.F` (and any `test/em_*` namelist/init for a warm-bubble / density-current / squall case if present). Determine EXACTLY how `mu_2` (dry-mass perturbation), `ph_2`, `p`, and `t_2`(θ) are set for the bubble: is mu' iterated/balanced against θ' (an explicit hydrostatic-balance loop), or is the perturbation θ-only with mu'=0 and ph/p derived? Quote the init lines (the frontrunner cited `:1107-1130` — verify and widen as needed).
2. **`pg_buoy_w` semantics**: read `dyn_em/module_big_step_utilities_em.F` `pg_buoy_w` (~2547-2571) and its caller `dyn_em/module_em.F:1361-1368`. For a θ-only perturbation with mu'=0 in a balanced base, what does `rdn·Δp' - c1f·mu'` evaluate to — the physical buoyancy g·θ'/θ0, or something 9.4× larger? Is `p` here the full perturbation pressure (which for a balanced perturbed column is small) such that mu' MUST be nonzero for the term to be physical?
3. **The p′ diagnosis**: does WRF's `calc_p_rho`/`rk_step_prep` p′ for a θ-only-perturbed-but-mu'=0 column give a Δp' that is 9.4× too large (because the column isn't in discrete hydrostatic balance)? I.e., does the imbalance live in p′ (IC) or in how `pg_buoy_w` combines it (dycore)?

## Then read the JAX side
- `src/gpuwrf/ic_generators/idealized.py` (how mu/ph/p/θ and the bubble are built; the comments claim "discretely hydrostatic" base + "θ' bubble breaks balance via buoyancy only").
- `src/gpuwrf/dynamics/core/advance_w.py` (`pg_buoy_w`/`pg_buoy_w_dry`), the p′ path in `calc_p_rho.py`.
- `proofs/f7d/rwtend_check.json`, `proofs/f7d/rk1_source_parity.json`, `proofs/f7d/postfix_runaway_warm_bubble.json`.

## Output
Write to EXACTLY: `/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-29-f7e-ic-vs-dycore-fork/gpt-fork-findings.md`
Structure: (1) WRF ground-truth answers to Q1-Q3 with file:line; (2) **VERDICT: HYPOTHESIS A (IC bug) or B (dycore bug) or BOTH** — be decisive and justify from WRF source; (3) the exact fix spec for the next Opus sprint for whichever hypothesis wins (if A: how WRF balances the perturbed column, which arrays change in idealized.py; if B: exactly where pg_buoy_w/p′ over-counts and the operator fix); (4) a falsifiable check that proves the fix (e.g. "θ'-only bubble yields rw_tend = 0.065 not 0.615" for B, or "balanced IC yields Straka front ≈ 15 km" for A). End with `F7E_FORK_COMPLETE`. Do not modify any other file.
