# FINAL33V release gates — FROZEN by mass-opus (2026-10-07 ~01:10Z, before any final-tree frame; integrate 01:03:06Z list, corrected)
Final arms: V0227 / V0408 / V0115, WN3 3-nest 72 h cold, FAST_PATH_DEFAULTS only (18-key DELTA incl. PLUME_CLOUD_BASE, SHSM_FLOORS,
TKEPROD_UP, ELH_BUDGET, RHOSFC_WRF, CONDENSATION_WRF + D1/D2/D3), full wrfout; truth = CPU-WRF R (wg_<case>_a1). E200 first: finite +
219/219 frames per case or NO verdict. Every gate below is binding unless marked (disclosed).
P0 PRECONDITION (before the GPU claim): step-1 MATCHED + step-2 MATCHED-2 (STEP1 1eee56e4 + STEP2 363639e8) on d01 AND d02 for the
   product key set on the final source (b-core harness, OFF identity included) + mass-opus independent read (mo_matched_read{,2}.py).
G-D6 (wn3 frozen wn3_score/d6_verdict, all 3 cases): d01 and d02 PASS. d03 failing FIELD set ⊆ the wave RC's set on the same case
   (rc33g, 136c6e51c: 0227 {RAINNC, U10, V10}; 0115 {RAINNC, V10}; 0408 {} ); any NEW field must be FLOOR-LIMITED under D6_FLOOR_RULE
   a9e6811d with the case's pair (0227: CPU P0227; 0115: GPU IC pair rc33g_b4 20260115_18z_a1_icp20261005 vs rc33g_b2 twin, rule S/X ≥ .5
   at the majority of failing leads; 0408: no pair → a new d03 field is OPEN = FAIL).
G-INT integrity REGISTRY_ALLFRAMES 2a11c0cb unchanged (wn3 413299de). EXPLAINED-trace ANNEX (post-pass, manager 01:02:58Z) — a flag is
   annex-explained iff ALL: (i) field ∈ {QICE, QSNOW, QGRAUP, QNICE, QNSNOW} ∪ {SNOW, SNOWC,
   SNOWH, TAUSS, QSNOWXY, ACSNOM} ∪ {CANLIQ, CANWAT, CANICE, FWET, ECAN, EVC, QRAINXY}; (ii) GPU value exactly 0.0 at every flagged cell
   (CPU-only presence); (iii) CPU max ≤ cap: mixing ratios 1e-10 kg/kg, QNICE/QNSNOW 10 /kg, SNOW and ACSNOM 1e-3 kg/m², SNOWH 1e-5 m, SNOWC 1e-3,
   TAUSS 1e-2, QSNOWXY 1e-6, CANLIQ/CANWAT/CANICE 1e-4 mm, FWET 0.05, EVC 0.01 W/m², ECAN 1e-6, QRAINXY 1e-7 (observed maxima on
   PLUME72/MYNNSET72 are ≥ 2.4× below every cap: QICE 1.0e-11, QNICE 4.05, SNOW 2.3e-4, CANICE 4.2e-5, FWET 3.9e-3, EVC 1.3e-3);
   (iv) ≤ 200 cells per flag-frame (observed ≤ 94). Anything else (other field, > cap, GPU non-zero, > 200 cells) = real FAIL.
G-R32 (wn3 frozen twin manifest + R32 formal, binding twins): 0227 OUTSIDE_LAG_ENVELOPE ≤ 3 (= the CPU ±1-ulp pair P0227's own count);
   other cases with alisios thresholds: no cell outside that was inside for the wave RC on that case (count disclosed).
G-PD problem-day cells pd_cells_FROZEN.py 8ac08921 on all 3 cases with --base = the wave RC SLIM (rc33g) and the case's IC pair where it
   exists; CASE PASS = 0 OPEN.
G-F1'' final_gate.py d4e5d5b5 UNCHANGED on V0227 τ0–30 frames (subset of the 72 h run; YES to integrate Q1 — the gate reads frames by
   valid time, run length does not enter).
G-OVG 0227 overgrowth: G1–G3 overgrowth.py b25389f05f82 and G4/G5 overgrowth72.py 421240e5, control OFF = retained MYNNSET72OFF frames
   (cbc31d75b, keys off; disclosed: pre-D1/D2/D3 tree) → integrate RETAIN MYNNSET72OFF until the final verdict.
Disclosed (not gates): x1_score 41642771 and xs_cold ad506a8e on V0227; BD92(c) re-measure (b-diff); WAKE H (b-diff, a283d5bc0aef) vs
   MYNNSET72OFF; alisios station re-check = alisios's own frozen rule (they decide DAMAGE/NO-DAMAGE); E41 cubin gate (integrate).
AMENDMENT A1 (pre-data, 2026-10-07 ~01:10Z, before any final frame or GPU claim): ACSNOM added to the snow-companion family with the
   SNOW cap (b-thompson INTEGRITY/README.md lists it in the mechanism-2 family; trace_evidence.py e9e9ae45a41a). Nothing else changed.
AMENDMENT A2 (pre-data, 2026-10-07 ~01:12Z; b-thompson 01:07:06Z data point): ACSNOM and TAUSS are ACCUMULATORS/AGES, so an
   instantaneous cap misfires (MYNNSET72OFF CPU ACSNOM 8.77e-3 kg/m², 2 cells, GPU 0). Replaces their A1/(iii) caps: ACSNOM explained iff
   the CPU per-frame increment at every flagged cell ≤ 1e-3 kg/m² AND the CPU value ≤ 2.5e-2 kg/m²; TAUSS iff CPU value ≤ 5e-2. All other
   (i)–(iv) conditions unchanged (GPU exactly 0, family, ≤ 200 cells).
AMENDMENT A3 (pre-data, 2026-10-07 ~01:20Z; manager DECISION 01:16:03Z): G-D6 is REPLACED by the v0.3.3 72 h release reading =
   frozen wn3 D6 verdict + FLOOR annex. Every 72 h D6 breach (any domain, field, lead) is FLOOR-LIMITED (disclosed, not a GPU weakness)
   iff the case's twin pair breaches the same field at the same lead (C > limit) OR C/X ≥ 0.75 there, with C = the frozen comparator
   (wn3_fast_compare 0fe77c00, D6 manifest 9df4fab8) on the pair; anything else = real FAIL. Pair per case: 0227 = CPU P0227 vs R;
   0115 = GPU IC pair rc33g_b4 20260115_18z_a1_icp20261005 vs rc33g_b2 twin (no CPU pair exists; disclosed proxy — BD101 found the GPU
   pair no more dispersive than the CPU pair, so it is the conservative side); 0408 = no pair → any 72 h breach = real FAIL.
   The 24 h D6 gate (AGENTS §2) stays strict, without annex. The earlier '⊆ wave RC set' clause is withdrawn (superseded).
   A3 practical (pre-data): the 0115 pair's full wrfout is deleted; C for 0115 = plain domain rmse of (member − twin) from the retained
   SLIM d03/d02/d01 (rc33g_b4/post/20260115_18z_a1_icp20261005/slim vs rc33g_b2/post/20260115_18z_a1/slim; Times must be identical) —
   validated identical to the frozen comparator on the 0227 CPU pair (max rel diff 0.0 over RAINNC/V10/W, 4 leads; D6_FLOOR_0115_PRECHECK.txt).
   wn3 RETAINS those two SLIM dirs until the final verdict.
AMENDMENT A4 (pre-data, 2026-10-07 ~01:50Z, before any FINAL33V frame exists; inputs only, rule/thresholds unchanged): G-PD base inputs.
   The rc33g_b1 SLIM .nc files of 0227 and 0408 no longer exist (only slim_manifest.json remains; disk cleanup 10-06, wn3 18:10:40Z).
   0227: --base = rc33g_pair3/b1/20260227_18z_a1/wrfout (<DATA_ROOT>/wrf_gpu2/scratch/wn3/rc33g_pair3/b1/...) = the wave RC 0227 BITWISE
   (wn3 twin_check_0227.json: 219/219 uncompressed-frame sha256 == the original RC run) → same data, new path; --pair = the
   rc33g_pair3 icp member (unchanged). 0408: no RC frames exist anywhere → --base = v0.3.2 OFF SLIM v032_pd1/post/20260408_18z_a1/slim
   (pd_cells_FROZEN's own design base; the wave RC passed it 260/260, 252 cells below base). Disclosed: 0408 G-PD is then a
   no-regression-vs-v0.3.2 check, not vs RC. Optional on the manager's word: wn3 regenerates RC 0408 bitwise (warm rerun of the
   136c6e51c snapshot + cache, sha twin check) after FINAL33V → RC-relative 0408 cells reported as disclosed. 0115 unchanged.
