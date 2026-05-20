# Worker A5 Supplement - Graupel Source-Truth Correction

## Objective

Apply the manager-accepted attempt-5 scope expansion for the confirmed graupel `cge(11)/cgg(11)` literal substitutions, regenerate M5 artifacts, and document the known strict-parity movement as M5-S1.x debt.

## Commit

Graupel source-truth correction commit: `2798b057fdaf475e5a65b631819d40d5c0d57a12`

Commit message: `[M5-S1 attempt-5] fix graupel cge11 source truth (+0.00221K S1.x debt)`

## Fix Details

Added named graupel constants in `src/gpuwrf/physics/thompson_constants.py:71-72`:

- `CGE11 = 2.8204808235`
- `CGG11 = 1.7042533`

Replaced the two graupel scalar coefficient literals with `CGG11` in `src/gpuwrf/physics/thompson_constants.py:92` and `:94`.

Replaced the two graupel `ilamg**CRE11` exponent uses with `ilamg**CGE11` in `src/gpuwrf/physics/thompson_column.py:464` and `:493`. `CRE11` remains imported and used for the rain path at `src/gpuwrf/physics/thompson_column.py:383`.

WRF source citations:

- `module_mp_thompson.F.pre:104`: `mu_g = 0.0`
- `module_mp_thompson.F.pre:156`: `bv_g = 0.640961647` for mp_physics=8
- `module_mp_thompson.F.pre:763`: `cge(11,m) = 0.5*(bv_g(m) + 5. + 2.*mu_g)`
- `module_mp_thompson.F.pre:767`: `cgg(n,m) = WGAMMA(cge(n,m))`
- `module_mp_thompson.F.pre:2761-2764`: graupel sublimation uses `cgg(11)` and `cge(11)`
- `module_mp_thompson.F.pre:2872-2875`: graupel melting uses `cgg(11)` and `cge(11)`

## Parity Movement

Before is A5 lami-only (`bd65be8`) from `worker-a5-report.md`. After is post-graupel artifact `artifacts/m5/tier1_thompson_parity.json:15-35`.

| Field | Before max abs | After max abs | Abs delta | Before max rel | After max rel |
|---|---:|---:|---:|---:|---:|
| `qv` | `1.4304079020558032e-05` | `1.5090182928232613e-05` | `+7.86103907674581e-07` | `0.004086360113325722` | `0.00431093267396623` |
| `qc` | `0.0001517228938283358` | `0.0001519669795683178` | `+2.440857399820169e-07` | `0.9999988999202443` | `0.9999988999202443` |
| `qr` | `4.760876436193939e-06` | `4.760876436193939e-06` | `0` | `45035996.27370496` | `45035996.27370496` |
| `qi` | `0.00013708094759935302` | `0.00013713997976800624` | `+5.903216865322052e-08` | `0.3049187629215758` | `0.30505007231324216` |
| `qs` | `0.0001447943623500526` | `0.0001447943623500526` | `0` | `1249.636404283526` | `1249.636404283526` |
| `qg` | `1.521843532880611e-05` | `1.6510481869645087e-05` | `+1.2920465408389762e-06` | `983428014.0158471` | `986567533.2761194` |
| `Ni` | `126975.12500000041` | `126975.12500000047` | `+5.820766091346741e-11` | `0.3883326751538069` | `0.3883326751538071` |
| `Nr` | `67300.453125` | `67300.453125` | `0` | `45425309332488.81` | `45425309332488.81` |
| `T` | `0.040290844661740266` | `0.042506264947576256` | `+0.00221542028583599` | `0.00015064605175968627` | `0.00015892943032500158` |

Manager accepted the movement as source-truth correctness exposing previously masked table-proxy debt. Tier-1 remains `pass=true` under carry-forward tolerances at `artifacts/m5/tier1_thompson_parity.json:14` and `:38`.

## Proof Status

- Tier-2 remains `pass=true` at `artifacts/m5/tier2_thompson_invariants.json:13`.
- Water residual remains `2.670445271854754e-12` at `artifacts/m5/tier2_thompson_invariants.json:18-21`.
- Positivity violations remain `0` at `artifacts/m5/tier2_thompson_invariants.json:14-16`.
- NaN/Inf violations remain `0` at `artifacts/m5/tier2_thompson_invariants.json:9-11`.
- Profile hard metrics remain stable: `kernel_launches_per_step=1`, `host_to_device_bytes_post_init=0`, and `temporary_bytes_per_step=0` at `artifacts/m5/thompson_profile.json:14-24`.
- HLO diff remains 0 bytes with SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Commands Run

`python scripts/m5_generate_thompson_fixture.py`

Result: completed successfully; fixture sample SHA `a357e2ef8f6e77ede0c6a79debc0dd0d8de1582585743fad3f6e533ba05d7102`; harness SHA `bf9525f9ca68c44c6ba0baafa62bbead439bed14195f5ad3b84a19d400a0b76d`.

`python scripts/m5_run_thompson.py`

Result: completed successfully; Tier-1 `pass=true`; Tier-2 `pass=true`; HLO diff SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; profile `kernel_launches_per_step=1`, `temporary_bytes_per_step=0`, `host_to_device_bytes_post_init=0`.

`python scripts/m5_gate_thompson.py`

Result: `gate_status=GO`, tier1 `true`, tier2 `true`, launches `1`.

`pytest -q`

Result: `398 passed in 253.44s (0:04:13)`.

`python scripts/validate_agentos.py`

Result: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`.

## Next S1.x Scope

M5-S1.x should include graupel-coupling table-proxy residuals alongside the existing Thompson lookup/moment export work. `M5-S1-NEEDS-S1X.md` has been updated accordingly.
