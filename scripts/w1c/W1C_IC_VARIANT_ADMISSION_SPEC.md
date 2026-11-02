# W1c IC-variant admission package — spec v3 (wrf_gpu2 lane v46-icvariant, 2026-10-31)

v3 = v2 (F1–F5 closed by cr-w44-launchers 01:20:33Z) + host-only residuals R1/R2 fixed. The admission schema the
ALISIOS side writes is unchanged from v2 (`alisios.ic_variant_admission.v2`).
Validator `scripts/w1c/icvariant.py` (rules R1–R10, commands `preflight`, `stage`, `keys`). Schema id
`alisios.ic_variant_admission.v2`, edit policy `A1_THETA_QV_REBALANCE_V1`, bound to your W1c build contract
`TO_WRF_GPU2_FROM_RE_WAVE_W1C_IC_BUILD_CONTRACT_20261030.md` sha256 `c466dd34…87f`.
Template: `scripts/w1c/ic_variant_admission.template.json`.

## 1. What an A1 variant is
An admitted W1a base case (`20260219_18z`, `20260530_18z`; base entry = the `cases` table of the pinned manager
admission, e.g. `W1A_SCREEN6_ADMISSION_v2_20261030.json` sha `37e2de3b…`) in which you replace
`wrfinput_d01`, `wrfinput_d02`, `wrfinput_d03` by copies edited with your A1 operator (contract c466dd34:
MU fixed, θ/qv local, P/PH/AL/P_HYD incremental, THM closed, PSFC diagnosed, all other fields preserved,
land and outer zones bit-equal). Everything else is the base case byte for byte: namelist (eta/grid, dt 27/9/3,
18 h, icloud 1, EDMF 1), `wrfbdy_d01`, the three `wrflowinp`, the 25 tables, engine bundle `32ea753a…`, paid
cache, runtime-parameter JSONs, cold / NO_RESUME / STOP_ON_FIRST_FAIL. gpuwrf never reads met_em.

## 2. Package (one JSON in cross + the files it names)
```
schema            "alisios.ic_variant_admission.v2"      status "ISSUED"      issued_by "alisios_mgr"
policy            "A1_THETA_QV_REBALANCE_V1"             contract_sha256 "c466dd3415c42a1761f77a57b2b30065515fab1b8459e2bebfb4e3a54c17887f"
variant_id        [A-Za-z0-9][A-Za-z0-9_.-]{0,31}
issued_utc / valid_until_utc (date -u) · truth_allowed false · GPU_launch_authorized false
base_case         {case_id, case_admission:{path,sha256}, model_init_utc, model_end_utc}
files[]           per replaced wrfinput: {target_relative_name, path (regular file), sha256, size_bytes,
                   base_sha256 (= admitted base input SHA), changed_variables [sorted, exactly the changed ones]}
provenance        {receipt:{path,sha256}, …}   your build receipt (method, producer code, retrospective sources)
balance           {operator:"assimilation.letkf.state_io.rebalance_increment", receipt:{path,sha256}}
runtime_pins      informative only; the validator re-derives every pin
```
Receipt contents are **your attestation**: we bind them by SHA, we do not prove the producer method or the source chain.
**Variant identity**: case id `<base_case_id>~<variant_id>`; identity SHA = sha256 of canonical JSON {schema, policy,
contract SHA, base case id, base case-admission SHA, base inventory digest, variant_id, sorted (file, SHA), provenance
and balance receipt SHAs}. Any changed byte ⇒ new identity.

## 3. Refusal rules (all before any GPU work, no waiver)
| code | rule |
|---|---|
| R1 | the admission is read **once** as pinned bytes (SHA) and only that parsed object is used; schema v2, ISSUED, issuer, policy, contract SHA, validity window, id syntax, both receipts exist with their SHA, truth/GPU flags false |
| R2 | base authority: base entry from the pinned manager admission; case id, case-admission SHA, model clocks equal; every `base_sha256` equals the inventory; **every base input byte (incl. the replaced ones) is re-hashed against the inventory** before use |
| R3 | A1 replaces only `wrfinput_d01..d03` (no wrfbdy, namelist, wrflowinp, tables, met_em); no duplicates |
| R4 | delivered bytes = declared sha256 + size; a replaced file must differ from its base |
| R5 | dimensions, variable set, per-variable dims/dtype, all global and variable attributes (ordered names; exact text, or dtype + shape + raw bits) and `Times` **bit**-identical to the base file |
| R6 | measured changed set (raw bytes, signed zero/NaN payload visible) == declared set and ⊆ A1 editable set `T QVAPOR THM P PH AL P_HYD PSFC`. MU and every other field must be bit-identical |
| R7 | changed float fields finite; QVAPOR ≥ 0 |
| R8 | every changed variable bit-identical to base on land columns (LANDMASK = 1, staggered faces touching land) and in the outer `spec_bdy_width` (5) rows/columns |
| R9 | native balance, §4 |
| R10 | program-key prescreen with a bound receipt, §5 |

All checks run on a private **snapshot** (verified copies of the delivered files and of the replaced base files);
`stage` publishes exactly that checked snapshot by atomic rename, re-hashes every staged file before publishing, and a
refused build leaves no facade and no partial directory.

Eligibility note: the A1 editable set is a subset of the variables that differ between the 0219 and 0530 real cases,
which loaded the same two paid executables with 0 compiles in 10/10 GPU members (KEY_NEUTRAL sets in the validator).
That set is evidence of key-neutrality only, not a permission: a broader variant policy needs its own contract,
version and review.

## 4. Native balance (your ask 2) — how it is accepted
WRF runs with `rebalance = 0`: start_em takes P/AL/PH as written, so the file must be balanced. We do not use
`io.wrfinput_consistency.validate_fields`; we accept your operator receipt as provenance and **re-check independently
on CPU with WRF's own relations** (f64, every column, every domain, increments against the authenticated base):
- EOS closure `AL = (R_d/p0)(T+300)(1+R_v/R_d·qv)((P+PB)/p0)^cvpm − ALB` (ALB from T_INIT/PB): max relative residual
  ≤ **max(1e-6, 2 × base file)** (real.exe bases 3.4e-7–5.6e-7).
- THM closure `THM = (T+300)(1+R_v/R_d·qv) − 300` (use_theta_m = 1): ≤ **max(1e-4 K, 2 × base file)** (bases ≤ 4.4e-5 K).
- Hydrostatic increments: (variant − base) of P, AL, PH equal the increment of WRF's start_em rebalance relations
  (dyn_em/start_em.F 2368–2460: p from the top with the full moisture load, EOS, hypsometric_opt = 2 PH) for
  (T, moisture, MU) of variant vs base: ≤ **0.5 Pa (P), 0.5 m²/s² (PH), 1e-5 m³/kg (AL)**.
- **P_HYD** increment == P increment (start_em `p_hyd = p + pb`, PB frozen): ≤ **0.5 Pa**.
- MU is not editable (R6), PB/PHB/MUB/T_INIT are not editable; PSFC is editable and not re-derived by us.

Only increments are tested: real.exe files are not start_em-rebalance exact in absolute terms (base P differs up
to ~3e2 Pa), so an absolute test would refuse the unmodified base. Thresholds are pre-registered; no relaxation.
Measured [M] on the real 0530 facade: a synthetic balanced edit (θ +0.5 K, qv ×0.95, levels 4–14, interior water)
passes on d01–d03 with P 1.2e-4 Pa, PH 4.9e-4 m²/s², AL 1.9e-9, EOS ≤ 5.6e-7; the same edit without rebalance is refused.

## 5. Program identity, pins, C reuse (your ask 3)
- A1 adds no engine flag, env variable, namelist, output-stream or shape change; variant handling is launcher work before
  the member process. B41 permits reuse for weather-data-only edits under unchanged static controls, shapes, selected
  operand schema, tables, flags and output configuration — exactly what R2–R8 enforce.
- **R10 is a CPU prescreen**: `keys` loads base and staged variant with the product loader/runtime (as `gpuwrf aot-key`,
  plus the member runtime-parameter JSON, 18 h, 3 domains) and requires the exact program set {d01, d02, d03, fused/d02},
  equal keys and no differing component. CPU-target keys compare CASES; they are not the paid GPU cache keys. The receipt
  binds base authority + inventory digest, variant admission + identity, staged facade inventory + dir, parameter JSON
  SHA + schema, hours, max_dom, dt, engine file + source fingerprint, environment; the launcher recomputes the binding
  (`verify_key_receipt`) and refuses stale, foreign, partial or mismatching receipts. v3 also freezes the execution
  context and refuses a receipt (and refuses to produce one) unless: platform CPU; engine `…/v45-rc/runtime/bundle/src/src/gpuwrf`
  with traced-source fingerprint `dd4b8008…` in every program component; env exactly JAX_PLATFORMS=cpu,
  GPUWRF_FAST_DEFAULTS=1, PYTHONPATH=<bundle src>, GPUWRF_JAX_CACHE=0, JAX_ENABLE_COMPILATION_CACHE=false; dt 27/9/3.
- [M] real 0530 facade (CPU probe, product key functions): θ/qv edit of wrfinput d01–d03 (+ wrfbdy in the probe) → all
  4 programs shared, no differing component; HGT +1 m on d01 → d01, d02 and fused/d02 differ in `static_config`;
  ZNU/ZNW on d01 → d01 differs in `static_config`. Static fields are baked and are refused before any key work.
- The GPU member gates stay mandatory and unchanged: backend compilation refused, loaded executables = prime
  (`bff015d2…` d01, `aedca60f…` fused/d02) with frozen bytes, full-history gate.
- C reuse: same programs, but the old C raw wrfout were released under your dispositions and the old compacts lack the
  cloud fields ⇒ re-run C. Recommended: four cold members in one GO (C0530, V0530, C0219, V0219), ≈ 4 × 213–253 s GPU
  process wall [M, W1a members] ≈ 15–17 min entry, inside the ≤ 1 h cap. The re-run C should reproduce the old C
  history byte for byte (same executable, same inputs [I]); its wrfout SHAs can be compared with the SHAs in the SCREEN4
  member receipts before you call it C.

## 6. History fields (your ask 4)
The paid C histories (SCREEN4 members 1 and 2, 0219 and 0530 C) contain them in every domain and all 54 frames:
378 variables incl. CLDFRA, CLDFRA_BL, QC_BL, QI_BL, SWDOWN, GLW, HFX, LH, QFX, PBLH, T, THM, QVAPOR, QCLOUD, QICE,
PH, PHB, P, PB, HGT (QSQ is not written) [M, header inventory]. The old compacts lack the cloud fields because the
extractor did not take them. If you want them launcher-enforced, the W1c seal adds them to `REQUIRED_HISTORY`
(launcher check only, no program change).

## 7. Launcher integration (W1c seal = SCREEN6 seal + these steps)
`TMPDIR` must be a private existing lane directory on ROOT (not /tmp; the CLI refuses otherwise, S0).
CPU (`JAX_PLATFORMS=cpu`, nice 19, on the lane cores of the current core plan — today SMT 24,25,28; never ALISIOS or v46-9nest cores), per variant, stop at the first refusal (rc 2):
1. `python -P scripts/w1c/icvariant.py stage --base-admission <manager admission> --base-admission-sha <pin> --base-case-id <id> --admission A.json --admission-sha <pin> --facade <DATA>/w1c_facade_<variant_case_id> --out STAGE.json`
   → R1–R9 on the snapshot, atomic publish, `VARIANT_FACADE_RECEIPT.json`; STAGE.json = `cases[<variant_case_id>]`
   (input_dir, inputs[], model clocks, base bundle/metadata SHA, case_admission_sha256 = variant admission SHA,
   `variant{…}`), shape-compatible with the SCREEN6 case table. ≈ 117 MB DATA per staged variant.
2. `python -P scripts/w1c/icvariant.py keys --base-admission … --base-admission-sha … --base-case-id … --entry STAGE.json --params <member params JSON> --out KEYS.json`
   (R10; one native case load ≈ 3 min quiet / 8–11 min on a loaded host, two per variant).
3. `verify_key_receipt(KEYS.json, base, entry, params, not_before_unix=<sequence start>)` + the existing native input
   preflight (`screen4_inputs.validate`) on the staged facade.

GPU: the unchanged SCREEN4/6 member path. Rows: `case_id` = variant case id; registry `issue` = BASE cycle id; tags carry
`variant_id` (≤ 40 chars); REGISTRY_BINDING carries the variant identity SHA. Raw disposition as SCREEN6.
