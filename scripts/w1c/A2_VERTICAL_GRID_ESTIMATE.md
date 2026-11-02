# A2 — vertical-grid variant (60–70 eta levels, densified 300–2000 m): estimate only, nothing built

wrf_gpu2 lane v46-icvariant, 2026-10-31. Labels: [M] measured, [I] inferred, [U] unknown.

## Today
- W1a namelist `e_vert = 45` (44 mass levels), dt 27/9/3, `w_damping 1`, `damp_opt 3`, `epssm 0.5` [M].
- Over the d03 ocean the 300–2000 m band holds 7 w-levels, Δz ≈ 130 → 410 m (305, 438, 603, 807, 1056, 1356, 1713 m) [M];
  an inversion base at 500–900 m is resolved by 2–3 levels.
- 18 h WN3 3-nest member: 213–253 s GPU process wall incl. init/output (10 W1a members) [M]; per-PID VRAM ≈ 6.3 GiB,
  host VmHWM ≈ 13.6 GB (FINDINGS A58/D26) [M].

## What a 60–70-level grid changes
- **Programs/keys: everything.** nz is in every carry aval → `carry_aval` changes for both paid step programs (d01, fused/d02)
  and for the init/radiation/output programs; the eta values themselves are baked as well [M: CPU key probe on the real
  0530 facade, ZNU/ZNW perturbed on d01 → d01 `static_config` differs; HGT +1 m → d01/d02/fused differ] → 2 new
  step-program keys + new init keys. The paid W1a prime (`bff015d2…`, `aedca60f…`) cannot be reused; zero-compile screening
  needs a NEW prime: cold compile + E41 cubin gate + the W1a qualification chain (control / null15 / resume18 / change /
  comparisons; caps 1800+840+600+1080 s) on the new grid [M chain, I cost].
- **Inputs:** `eta_levels`/`e_vert` change the namelist and every real.exe product → ALISIOS re-runs real.exe for each case on
  CPU (met_em unchanged), new facades + admissions per case (not an A1 variant: R6 refuses ZNU/ZNW/PB/PHB/MUB/T_INIT).
- **Qualification (fidelity):** a new vertical grid is a new configuration → it needs its own CPU-WRF reference on the same grid
  (≥ 1 case, 18 h, 3 nests) and the frozen D6 + integrity comparison GPU vs CPU-WRF; the W1a twin evidence on 45 levels does not
  carry over. No engine code change is expected: no kernel hard-codes nz (grep of the bundle source [M]); the GPU engine has
  not been qualified above 45 levels on the WN3 path [U].
- **Numerics risks [I]:** Δz ≈ 50–60 m in the band gives vertical Courant w·dt/Δz ≈ 0.5 (d01, w 1 m/s) to > 1 over Teide waves
  (d01 dt 27 s) — `w_damping 1` limits it, but CFL events/instability must be checked in the qualification run (CPU-WRF sees the
  same). New level count can move XLA fusion/Pallas unrolling (E41 STACK/kInput, ptxas time E103/E157) → CPU deviceless
  prescreen of the two step programs before the GPU slot.

## Cost per 18 h run vs today (~4 min)
Most device work is linear in nz (dycore, acoustic column solves, MYNN, Thompson, RRTMG; A54/A56) [I]:
| levels | factor | GPU process wall / 18 h member | VRAM / case | raw wrfout |
|---|---|---|---|---|
| 45 (today) | 1.00 | 213–253 s [M] | ≈ 6.3 GiB [M] | 1× |
| 60 | ≈ 1.36 | ≈ 4.8–5.7 min [I] | ≈ 8.5 GiB [I] | ≈ 1.35× |
| 65 | ≈ 1.48 | ≈ 5.3–6.3 min [I] | ≈ 9.3 GiB [I] | ≈ 1.45× |
| 70 | ≈ 1.59 | ≈ 5.6–6.7 min [I] | ≈ 10 GiB [I] | ≈ 1.55× |
(4 parallel WN3 cases per 32 GB GPU today → 3 at 65 levels [I]. Host RAM per case ≈ 18–20 GB [I].)

## One-off effort [I]
| item | who | GPU | CPU / wall |
|---|---|---|---|
| eta set design (Δz ≤ 60 m in 300–2000 m, smooth stretching, same p_top) | ALISIOS (+ our review) | – | hours |
| real.exe per case (0219, 0530) | ALISIOS | – | ≈ 10 min each |
| CPU-WRF 18 h reference, ≥ 1 case, 3 nests, 4 ranks | ALISIOS | – | ≈ 18 × 385 s × 1.48 ≈ 2.9 h |
| deviceless prescreen + cold compile + E41 | wrf_gpu2 | ≈ 15 min | ≈ 1 h |
| W1a-style prime qualification chain on the new grid | wrf_gpu2 | ≈ 45–70 min | ≈ 2 h |
| D6 + integrity vs CPU-WRF, critic | wrf_gpu2 | – | ≈ 2 h |
| then each screening member | | ≈ 5.3–6.3 min | |
Total ≈ 1–1.5 GPU-h one-off + ≈ 1 day wall incl. the CPU-WRF reference; no engine rebuild. Decision is yours; we do not start it
before your GO.
