# #149 heavy-rain cell deficit — lane v46-heavyrain (Opus 5.5, 2026-10-31)

Status: FINAL CPU REPORT (one WRF-fidelity fix, see §4.3). CPU only, no GPU, no new model forecast runs, src/ untouched.
Labels: [M] measured, [I] inferred, [U] unknown. Predecessor: lane/o5-heavyrain REPORT fbb99b457d
("not demonstrated as a systematic GPU defect, not excluded"; confirmatory HR07b designed, never run).

## 0. Question and approach

ALISIOS #149: on 0420 (WN3, pure warm rain on d03) the d03 cells with 72 h RAINNC ≥ 10 mm are 12–15 % fewer in all four
GPU runs (v0.4.0/v0.4.1) than in both CPU-WRF members. The one-call pristine oracles (w43-conv: Thompson mp8 on 586 heavy
columns, MYNN, MYNN-SL, diffusion, PD advection, acoustic) pass. This lane looks for the part that one-call oracles cannot
see: a **deterministic** GPU−CPU difference that exists before the runs decorrelate, i.e. a per-step operator/numerics
difference, and asks whether it is large enough to produce the #149 gap.

Data (read-only, no admission needed): WN3 0227 GPU v0.4.0 24 h (`<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/
WN3-0227/gpu`), original CPU-WRF 2×2 C (`<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run`), CPU-WRF 4×1
twin C2 (`<DATA_ROOT>/wrf_gpu2/scratch/w43-twin0227/C2_4x1/cpu`); 0420 derived extracts of o5-heavyrain
(`<USER_HOME>/wrf_gpu2_lanes/o5-heavyrain/x0420/d03_*.npz`, 2-D fields + mean θ profiles, all six runs, 73 leads).
No GPU output of v0.4.4/v0.4.5 at early leads survives on disk (RW3 v0.4.5 only leads 60–72).

## 1. Deterministic early-lead GPU−CPU signals [M] (HV01, `hv01_early_leads.py`, lane dir `HV01_0227_d0{1,2,3}.json`)

0227 day 1 is dry (d03 domain mean ≤ 0.006 mm/h), so it cannot test rain itself, but it shows the deterministic offsets.
Sea interior (≥ 10 cells from the boundary), GPU − C, with C2 − C as the CPU-numerics yardstick:

| domain (steps/h) | θ k0–4 lead 1 | lead 3 | lead 6 | C2 − C | HFX lead 6 |
|---|---:|---:|---:|---:|---:|
| d01 (66.7) | −0.23 mK | −0.91 | −1.91 | ≤ 0.06 mK | +0.037 W/m² |
| d02 (200) | −0.90 | −2.47 | −4.36 | ≤ 0.1 | +0.073 |
| d03 (600) | −2.89 | −3.08 | −7.93 | ≤ 0.5 | +0.109 |

- Vertical structure (all domains, lead 3–6): k0–4 cooler and moister (+1…+8·10⁻⁶ kg/kg by lead 6), k6 warmer and drier,
  k8 cooler — a slightly weaker exchange across the top of the marine boundary layer. HFX/LH rise as a consequence
  (cooler air over prescribed SST).
- Interior-maximal, not nest-ring driven (d03 lead 1: rings 0–1 −0.88 mK, interior ≥ 20 −3.38 mK).
- Rate scales with the number of steps rather than with time: ≈ 2.4·10⁻⁶ (d01), 4.5·10⁻⁶ (d02), 5.6·10⁻⁶ (d03) K per
  step, the same per-step size as the upper-level warm drift o5-heavyrain found (+2.4 mK/h d03, +0.5 mK/h d01).
- 0420 shows the same signature (four GPU runs agree, CPU twin ≈ 100× smaller): sea-interior T2 −2…−3 mK (leads 2–4),
  Q2 −2.4·10⁻⁶ (lead 1) → −6·10⁻⁶ kg/kg (lead 5), LH +0.06 → +0.18 W/m², PBLH +0.6 m.
- d01 lateral relax zone: MU G−C −0.65 Pa in rings 1–7 at lead 1 (≈ 2 × the CPU MU tendency per 54 s step), ring 0 exact,
  ring 1 back to +0.005 Pa exactly at the 06Z boundary record, then spreading inward (−0.4 Pa domain-wide by lead 3).
  This is the signature of the already known released root specified-zone clock (`GPUWRF_ROOT_SPEC_STAGE_CLOCK`,
  o5-coast: ring-0 pins one root step ahead during the acoustic substeps; WRF walk deferred to v0.5) — independent
  confirmation, not a new item.

## 2. Are these offsets big enough for #149? [M inputs, I scaling]

- 0420 near-surface moist static energy over the sea interior (cp·T2 + L·Q2, in K): GPU-mean − CPU-mean −0.011 K at
  leads 1–6 (deterministic, all four GPU runs below both CPU runs at 6/6 leads), then −0.01 … −0.04 K in the 12 h windows
  to lead 72, inside the C2 − C range (−0.10 … +0.02 K).
- Elasticity of the #149 statistic (cells ≥ 10 mm at 72 h) to a uniform rain-intensity scaling, from the fields
  themselves: dlnN/dlnI = 1.4–2.2 (all six runs). Bringing the CPU counts (1454, 1377) down to the GPU mean (1245) needs
  a uniform intensity cut of 7–10 %.
- A −0.04 K θe shift gives at most ≈ 0.6 % intensity even at super-Clausius-Clapeyron 14 %/K, i.e. ≈ 1 % in the count:
  about one tenth of the gap. The deterministic thermodynamic offsets cannot produce #149 by themselves. [I]

## 3. Independence of the members [M] (pattern correlation of accumulated d03 rain)

| window | C–C2 | GPU–GPU | GPU–CPU |
|---|---:|---:|---:|
| 12–18 h | 0.856 | 0.832–0.877 | 0.837–0.881 |
| 36–48 h | 0.621 | 0.621–0.788 | 0.597–0.767 |
| 48–60 h | 0.413 | 0.434–0.636 | 0.333–0.429 |
| 0–72 h | 0.726 | 0.758–0.820 | 0.726–0.781 |

The CPU pair is as decorrelated as any GPU pair from lead 6 on (it is not one realization), but the four GPU runs are
more similar to each other than to either CPU run (0–72 h and 48–60 h): they share the deterministic GPU seed of §1 and
only differ by run-to-run compile noise (B/Bp) or the SWDDIR fix (F/Fp). Their effective sample size is below four, so
the post-hoc p = 1/15 of o5-heavyrain overstates the evidence. A confirmatory test needs independently perturbed
members (HR07b design: 1-ulp θ IC perturbations on both sides).

## 3b. Current release on the second rain day [M]

v0.4.5 RC GPU run of WN3 0227 (RW3 N72 arm, d03 frames 60–72 kept in `<USER_HOME>/wrf_gpu2_lanes/v45-graupel/
RC_N72_20261029/d03`), 72 h RAINNC on d03 (all heavy rain of this case falls in leads 60–72; lead-60 mean 0.09 mm):

| run | mean (mm) | cells ≥ 10 mm | ≥ 25 mm | ≥ 50 mm |
|---|---:|---:|---:|---:|
| GPU v0.4.5 | 2.698 | **659** | 87 | 0 |
| CPU-WRF C (2×2) | 2.623 | 647 | 102 | 0 |
| CPU-WRF C2 (4×1) | 2.678 | 630 | 77 | 0 |

One GPU member: the current release is above both CPU-WRF members on the #149 statistic for this case (+1.9 % / +4.6 %);
≥ 25 mm lies between them.

## 4. Localising the per-step boundary-layer drift on CPU (Swiss RD11, original CPU-WRF reference)

The release program can be stepped on CPU on the shipped Swiss case (real loader, interpret Pallas, harness from
o5-n333). CPU-WRF references: hourly RD11 history and one-/two-step restarts of the identical configuration
(`<DATA_ROOT>/wrf_gpu2/scratch/o5-n333/swiss_1step_tkeF`, history every 18 s step).

### 4.1 The drift is structural, not GPU arithmetic and not REAL precision [M] (HV02, `hv02_step_drift.py`)

Port (CPU) − CPU-WRF, domain-mean θm, mK (400 steps = 2 h; R release, Q `GPUWRF_CARRY_REAL_ALL=0`, P + `GPUWRF_DYN_REAL_ALL=0`):

| arm | k0 1 h | k1 1 h | k2 1 h | k0 2 h | k1 2 h | k7 2 h |
|---|---:|---:|---:|---:|---:|---:|
| R | −2.46 | −3.57 | −2.57 | −3.40 | −3.42 | +0.60 |
| Q | −2.18 | −3.45 | −2.35 | −2.61 | −3.15 | +0.56 |
| P | −2.44 | −3.55 | −2.44 | −3.27 | −3.41 | +0.53 |

The CPU port reproduces the GPU−CPU-WRF signature (cooler low levels, warmer just above). Switching the f64 carry or
the f64 dycore on changes it by ≤ 0.8 mK, so it does not come from REAL dycore arithmetic. (State θ stays REAL32 in
all arms, E47; the full-θ storage idea was overtaken by §4.2.)

### 4.2 Step-level attribution [M] (HV03 `hv03_dump_all.py`, every carry leaf after steps 1–2)

- Initial state: port − WRF θm ≤ 2·10⁻⁷ K; initial w bit-identical.
- After step 1: k0 θm −1.06·10⁻⁴ K (79 % of cells colder), step 2 −2.08·10⁻⁴ K.
- Equal to round-off: HFX (−29.417 W/m² both), UST, MOL, TSK, QSFC, ZNT, QKE, EL_PBL.
- **Held radiative heating differs: RTHRATEN k0 −3.33 % (99 % of cells), k1 −1.29 %, k2 −0.19 %, above +0.5…+1.4 %**
  (clear sky everywhere, so McICA is not involved).

### 4.3 Cause and fix [M] (HV04 causal arm, HV05 fixed source)

The port's Noah-MP initialisation seeded the land emissivity with a uniform 0.97 (`io/noahmp_land_init.py:325`), and
the RRTMG coupler takes the land value when it is valid (`physics_couplers.py:1262–1270`). In WRF the first
radiation call (first_rk_step_part1) precedes the first Noah-MP call and sees `landuse_init`'s EMISS = SFEM(IS,ISN)
(`phys/module_physics_init.F:1970–1971`; Swiss 0.92–0.95, mean 0.930 — also the CPU-WRF lead-zero history EMISS).
The albedo seed had already been moved to `landuse_init` (b-noahmp NF10); the emissivity seed had not.

- HV04 (same harness, only the initial land emissivity replaced by WRF's lead-zero EMISS): **step-1 RTHRATEN equals
  CPU-WRF at every level (relative difference 0.000 %)**; k0 θm gap −1.06·10⁻⁴ → −4.7·10⁻⁵ K (step 1), −2.08·10⁻⁴ →
  −9.3·10⁻⁵ K (step 2).
- Fix (this lane): `land_history.wrf_initial_emissivity` (the existing `_landuse_init` EMISS) seeds `emiss` in
  `noahmp_land_init`. Test `tests/test_v046_noahmp_emiss_seed.py`: seed == CPU-WRF lead-zero EMISS on every land cell
  (Swiss; WN3 0227 d01–d03 when run under an admitted session); deletion mutant (0.97 restored) fails with an
  AssertionError. Consumer suite (18 files touching land init/history): 300 passed, 32 skipped, 0 failed.
- Scope: land cells, the first radiation period only (radt; WN3 30 min, Swiss 10 min) — Noah-MP overwrites EMISS at
  its first call. Sea cells use the 0.98 table value in both models, so this is **not** the marine #149 mechanism.
- Cost: init-only host arrays (one table lookup, already computed for history); no change to the traced program or
  its shapes, no extra kernels, launches or memory; values of the initial land carry change (disclosed identity change
  of every land run; GPU qualification in the v0.4.6 RC, RW1–RW4).

### 4.4 Fixed source on Swiss [M] (HV05 two-step dump, arm F 400 steps)

- HV05 (fixed source, no intervention): initial land EMISS == CPU-WRF lead-zero EMISS on every cell (max |diff| 0);
  **step-1 RTHRATEN == CPU-WRF to round-off: max |diff| 1.2·10⁻⁸ K/s (4·10⁻⁵ of the field maximum), level means
  ≤ 4·10⁻⁶ relative** (before: −3.33 % at k0).
- Arm F vs CPU-WRF, domain-mean θm (mK), release R → fixed F: k0 1 h −2.46 → −2.14, 2 h −3.40 → −2.82; k1 1 h −3.57 →
  −3.43, 2 h −3.42 → −3.20; k5 2 h −0.58 → −0.40; qv k0 2 h −1.45 → −0.27·10⁻⁷. The fix moves the low levels toward
  WRF but removes only ≈ 15–20 % of the Swiss boundary-layer offset: the rest has another, not yet localised source
  (§4.5). My pre-measurement estimate that the first radiation period explains most of it was wrong.

### 4.5 What remains [M/U]

After the fix the k0 residual is noise-like (median −1·10⁻⁵ K, std 1.5·10⁻⁴ K, 53.5 % of cells negative). The surface
w (diagnosed, `set_w_surface`) differs by +5.9·10⁻⁴ m/s mean (rms 1.3·10⁻³) after step 1 because u at k0 differs by
a zero-mean 8.6·10⁻³ m/s rms (≈ 8 % of the one-step change, uncorrelated with RUBLTEN or with the step change) — a
step-level noise-class difference, open lead for the dynamics owners, not a coherent bias. The marine early-lead
cooling on 0227/0420 (§1) is therefore not explained by the EMISS seed; its size is bounded by §2.

## 5. Verdict on #149

- No operator-level defect that can produce the #149 heavy-rain deficit was found. The deterministic GPU−CPU-WRF
  offsets before decorrelation are mK / 10⁻⁶ kg/kg (§1). They shift the inflow moist static energy by ≤ 0.04 K, which
  bounds their effect on the #149 count at about one tenth of the observed gap (§2, [I] scaling).
- The four GPU runs behind #149 are not independent realizations (§3), so the post-hoc p = 1/15 overstates the
  evidence; the current release shows no deficit on the second rain day 0227 (§3b, one member).
- Classification: **consistent with decorrelation (chaotic realization spread) plus at most a small deterministic
  component**; not proven chaotic in the strict sense, which needs independently perturbed members on both sides —
  the pre-registered HR07b test of o5-heavyrain (9 GPU + 3 fresh CPU-WRF members) remains the confirmatory test.
- Side result: one WRF-fidelity bug fixed (land EMISS seed, §4.3): step-1 RTHRATEN now equals CPU-WRF to round-off;
  it explains 15–20 % of the Swiss low-level offset vs CPU-WRF, nothing over the sea.

## 6. Reproduce

Scripts in `proofs/v46_heavyrain/` (`hv01_early_leads.py`, `hv02_step_drift.py`, `hv02_analyze.py`, `hv03_dump_all.py`);
outputs in `<USER_HOME>/wrf_gpu2_lanes/v46-heavyrain/` (HV01_*.json, HV02.json, hv0*_*.npz/log). CPU only,
`JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0`, nice 19, lane cores.
