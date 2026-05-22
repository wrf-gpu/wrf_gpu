# c2-A1'' PGF-Fix Re-Reviewer Report

Date: 2026-05-22
Reviewer: Claude Opus 4.7 (1M context, independent re-review)
Branch under review: `reviewer/opus/c2-A1pp-pgf-fix-review` (worker delta `b948fda..c29048d`, merged at `07d21a6`)
Scope: verify c2-A1'' addressed all R-findings (R1–R7) flagged by the prior c2-A1 Opus
review (`.agent/sprints/2026-05-22-c2-A1-architecture-review/reviewer-report.md`, commit
`7cebd16`).

## Verifiability triple

| Proof object | Status | Method |
|---|---|---|
| `proofs/metrics.json` | re-verified | inspected; new `cf1/cf2/cf3` (scalar) + `fnm/fnp` (nz) appear in both analytic-flat and `wrfinput_d02` legs |
| `proofs/hybrid_eta.json` | re-verified | inspected; static PGF interpolation coefficients recorded |
| `proofs/scan_transfer_audit.md` | unchanged | unchanged from c2-A1 — still acceptable as architecture-skeleton claim |
| `proofs/limiter_conservation.json` | unchanged | unchanged from c2-A1 |
| `proofs/integration_warm_bubble.json` | unchanged | still labelled `PARTIAL_SMOKE_NOT_WARM_BUBBLE_PARITY` — honest |
| `pytest tests/test_m6x_c2_*.py tests/test_m6_state_extension.py` | re-ran | `15 passed in 11.63 s` on this worktree (`PYTHONPATH=src`) |

## Per-R-finding verification

### R1 (DycoreMetrics missing cf1/cf2/cf3, fnm, fnp) — **CLOSED**

`src/gpuwrf/contracts/grid.py:84-88` adds `cf1, cf2, cf3, fnm, fnp` as JAX arrays on
`DycoreMetrics`. `_array_names()` (`grid.py:130-134`) lists them in pytree order so
`tree_flatten`/`tree_unflatten` survive JAX transforms. `validate_shapes` requires
`cf1/cf2/cf3` scalar `()` and `fnm/fnp` of shape `(nz,)` (`grid.py:225-229`). The
docstring (`grid.py:62-63`) explicitly labels them as supporting WRF's non-hydrostatic
PGF face-pressure construction.

`DycoreMetrics.flat()` (`grid.py:165-194`) builds the analytic-flat coefficients from
the WRF `module_initialize_real.F:3741-3755` algorithm:

- `fnm[k] = 0.5*dnw[k-1]/dn[k]`, `fnp[k] = 0.5*dnw[k]/dn[k]` for k≥1 — matches WRF
  3744-3745 verbatim (modulo 0/1-based indexing);
- `cof1 = (2*dn[1]+dn[2])/(dn[1]+dn[2])*dnw[0]/dn[1]`, `cof2 = dn[1]/(dn[1]+dn[2])*dnw[0]/dn[2]`
  — matches WRF 3750-3751;
- `cf1 = fnp[1]+cof1`, `cf2 = fnm[1]-cof1-cof2`, `cf3 = cof2` — matches WRF 3753-3755.

For uniform eta (the analytic-flat fixture), this collapses to `cf1=2.0, cf2=-1.5,
cf3=0.5`, which `tests/test_m6x_c2_metrics.py:50` asserts exactly. A minor caveat:
`flat()` sets `dn = dnw` (uniform-spacing simplification) rather than WRF's
`dn(k) = 0.5*(dnw(k)+dnw(k-1))`. For uniform spacing the result is identical, and the
wrfinput loader (below) uses the real DN/DNW directly, so this is acceptable.

`load_wrfinput_metrics` (`src/gpuwrf/dynamics/metrics.py:76-80`) reads `CF1/CF2/CF3/FNM/FNP`
straight from `wrfinput`. `tests/test_m6x_c2_metrics.py:67-95` exercises the WRF-fixture
case and additionally asserts `float(metrics.cfN) == float(wrfinput.CFN[0])` and
`jnp.allclose(metrics.fnm, wrfinput.FNM[0])` — i.e. bit-for-bit equality with
`wrfinput_d02`. `proofs/metrics.json:4-15,116-128,157-167` records the shapes for both
fixtures, matching the c2-A1 d02 dimensions (66×159 with nz=44, cf scalars, fnm/fnp
(44,)).

R1 is closed. Note `flat()` now rejects `nz<3` and validates `eta_levels` shape early
(`grid.py:155-159`); this is a small tightening over the c2-A1 surface but is needed for
`cof1/cof2` (which dereference `dn[2]`) — acceptable, not scope creep.

### R2 (State.replace asymmetry undocumented) — **CLOSED**

`src/gpuwrf/contracts/state.py:516-523` adds a four-line docstring:

> `p_total` is authoritative; `p_perturbation` is a delta the caller maintains explicitly
> against the current `BaseState.pb`. Updating `p_perturbation` alone does not
> auto-recompute `p_total` because `pb` is not visible to `State.replace`.

This is exactly the contract the original R2 asked for. No code change required.

### R3 (al/alt home unassigned) — **CLOSED**

`.agent/decisions/ADR-020-c2-dycore-architecture.md:30-32` adds an explicit
"Intermediate Fields Policy" section: `al`, `alt`, `cqu`, `cqv` are SCAN-CARRIED
intermediates (not `State` leaves, not recomputed per substep), cites
`module_em.F:217,242,1326,1340` for the `calc_p_rho_phi`/`calc_alt` source pattern, and
binds the c2-A2 implementation to build them in
`acoustic_wrf.acoustic_substep_scan`'s carry tuple. `architecture.md:57` propagates the
same policy with the same WRF anchors and same wording. The decision is unambiguous.

### R4 (terrain slope-subtraction WRONG) — **CLOSED**

The original `(g/alpha)*dzdx*dp/deta` clause is **removed** from ADR-020. Line 40 now
mandates the WRF-canonical three-term implicit cancellation:

> `dpxy = M*rdx*(c1h*muu + c2h)*(d(ph)/dx + alt_avg*d(p_perturbation)/dx +
> al_avg*d(pb)/dx)` … At hydrostatic rest with hydrostatic base plus hydrostatic
> perturbation, these three terms cancel exactly without explicit slope subtraction.
> WRF reference: `module_small_step_em.F:828-831` (x) and `:902-905` (y). Per Opus
> review R4, c2-A2 must not add a separate explicit slope-subtraction term. The
> `dzdx`/`dzdy` arrays in `DycoreMetrics` are reserved outside this PGF cancellation
> path for terrain-following advection/diffusion metric corrections.

The ADR-002 amendment patch (`.agent/patches/2026-05-22-c2-adr-002-amendment.md:47-55`)
mirrors the same constraint, with explicit `MUST follow … implicit terrain
cancellation` and `MUST NOT add an explicit hydrostatic slope-subtraction term`.
`architecture.md:64` propagates the WRF anchor and the same constraint.

I cross-checked WRF `module_small_step_em.F:828-831` and `:902-905` against the ADR
formula. The structural mapping holds: `(c1h(k)*muu(i,j)+c2h(k))` mass-coupling, `ph`
horizontal difference summed over (k,k+1), `(alt(i)+alt(i-1))` and `(al(i)+al(i-1))`
averaged-multipliers on the `p`/`pb` horizontal differences, with the `0.5*rdx` outer
prefactor in WRF absorbed into the ADR's `_avg`/horizontal-derivative notation. R4 is
closed.

### R5 (missing 4th non-hydrostatic term) — **CLOSED (with one notation nit, see M1)**

ADR-020:42 now adds the 4th term:

> `dpxy += M*rdx*d(php)/dx * (rdnw*d(dpn)/deta - 0.5*c1h*mu_avg)` for x, with the
> corresponding `M*rdy*d(php)/dy` form for y. Here `php` is hydrostatic perturbation
> pressure at half-levels, computed from `phb + ph_perturbation` at half-faces using
> `fnm/fnp`, and `dpn` is face pressure built with `cf1/cf2/cf3` near boundaries and
> `fnm/fnp` in the interior. … WRF reference: `module_small_step_em.F:854-863` (x) and
> `:928-937` (y), with `dpn` construction at `:836-851` and `:910-925`.

WRF cross-check, line-by-line:

| ADR element | WRF source | Match |
|---|---|---|
| `(msfux/msfuy)` for x, `(msfvy/msfvx)` for y | 861, 935 | ✓ |
| `rdx*(php(i)-php(i-1))` / `rdy*(php(j)-php(j-1))` | 861-862, 935-936 | ✓ |
| `rdnw(k)*(dpn(k+1)-dpn(k))` | 862, 936 | ✓ |
| `dpn` boundary uses `cf1/cf2/cf3` | 836-838 (x bottom), 843-845 (x top), 910-912 (y bottom), 917-919 (y top) | ✓ |
| `dpn` interior uses `fnm/fnp` | 850-851 (x), 924-925 (y) | ✓ |
| cf*/fnm/fnp source-anchored to new R1 metrics | grid.py:84-88 | ✓ |

The 4th term references the new R1 `cf*/fnm/fnp` exactly as required.

M1 (minor — non-blocking notation nit): the literal ADR formula
`-0.5*c1h*mu_avg` is ambiguous. WRF writes
`-0.5*(c1h*mu(i-1) + c1h*mu(i)) = -0.5*c1h*(mu(i-1)+mu(i))`. If a c2-A2 implementer
interprets `mu_avg` as the standard arithmetic mean `0.5*(mu(i-1)+mu(i))` (the natural
reading) and applies the literal ADR formula, the result is `-0.25*c1h*(mu(i-1)+mu(i))`,
i.e. **factor-of-2 light**. The ADR's 3-term formula at line 40 makes the same implicit
convention choice (`alt_avg`/`al_avg` work out only if interpreted as standard averages
with the WRF outer `0.5` absorbed into the `_avg` and `d/dx` notation). Recommend
c2-A2 worker reads the cited WRF line 862/936 verbatim and either:
(a) rewrite the ADR mu term as `-0.5*c1h*(mu_left + mu_right)` (literal-WRF), or
(b) rewrite as `-c1h*mu_avg` with mu_avg explicitly defined as the standard mean.
Because the WRF source is cited and the implementer is required to consult it
(constraint in `AGENTS.md`: "WRF `dyn_em` remains the numerical oracle for formulas"),
M1 does not block c2-A2 dispatch — it is a documentation polish.

### R6 (map-factor ratio naming) — **CLOSED**

ADR-020:41 names the ratios explicitly: "x-PGF uses `msfux/msfuy`; y-PGF uses
`msfvy/msfvx`" and cites `module_small_step_em.F:821-826,886-891`. WRF line 828 has
`(msfux(i,j)/msfuy(i,j))` (x) and line 902 has `(msfvy(i,j)/msfvx(i,j))` (y) — exact
match. `architecture.md:64` and the ADR-002 amendment also repeat the ratios. R6 is
closed.

### R7 (cqu/cqv/mudf/boundary policy) — **CLOSED**

ADR-020:43-44 says cqu/cqv are scan-carried intermediates under the same policy as
al/alt (matching `module_small_step_em.F:868,942`), and `mudf/mudf_xy` is explicitly
flagged as **distinct from `smdiv`** with "exact c2 treatment is TBD pending c2-A2
prototyping". WRF cross-check: line 868 = `u(i,k,j) - dts*cqu(i,k,j)*dpxy(i,k) +
c1h(k)*mudf_xy(i)`, line 942 same form for `v` with `cqv` ✓. The `dzdx_u`/`dzdy_v`
boundary-pad policy aspect of R7 becomes mostly moot under R4: `dzdx/dzdy` are no
longer on the PGF cancellation path (ADR-020:40 explicitly reserves them for
advection/diffusion). R7 is closed.

## Adversarial probes

**P1 — Was new production code introduced beyond R1?**
`git diff b948fda c29048d -- 'src/**/*.py'` is exactly 5 added lines in
`dynamics/metrics.py` (the five `_first_time_variable` calls for `CF1/CF2/CF3/FNM/FNP`)
plus the R1 DycoreMetrics extension in `grid.py` and the R2 docstring in `state.py`.
No new acoustic/dycore production code was introduced. Worker also tightened
`DycoreMetrics.flat()` input validation (`nz>=3`, `eta_levels` shape) — defensible
because the new `cof1/cof2` formulas dereference `dn[2]`. Verdict: **no scope creep**.

**P2 — Does the 4th-term spec match WRF line-by-line?** Yes for the
intermediates and the boundary/interior coefficient policy (see R5 table above). The
`mu_avg` factor is notationally ambiguous (M1) but the WRF source is properly cited so
a careful implementer will not be misled. The ADR is implementable as a c2-A2 spec.

**P3 — Did the worker correctly classify intermediates vs scan-carry vs State leaves?**
ADR-020:30-32 + `architecture.md:57` say `al, alt, cqu, cqv` are SCAN-CARRIED. The
new `cf1/cf2/cf3/fnm/fnp` are STATIC (`DycoreMetrics`). `php` and `dpn` (the new
intermediates introduced by R5) are described as substep-local intermediates in line
42 ("computed from phb + ph_perturbation at half-faces using fnm/fnp", "face pressure
built with cf1/cf2/cf3 near boundaries…") — implicitly per-substep, no explicit class
assignment. This is acceptable because they are derived deterministically from
already-classified state in one substep; recommend c2-A2 explicitly puts them in the
acoustic_substep_scan's carry tuple alongside al/alt (consistent with R3 policy).
**Minor follow-up M2**: ADR could state explicitly that php/dpn are substep-local
intermediates rather than scan-carry; not blocking.

**P4 — Did c2-A1''' need to revisit any of R8–R16 or A1–A8?** No — those were either
flagged in the original review as non-blocking ("Strongly recommended but
non-blocking") or are c2-A2 first-day cleanup items (R8 tautological oracle, R16
hash churn, A1 non_hydrostatic flag, A2 top_lid, A6 theta_perturbation, A7 dt cache
policy). They remain as c2-A2 follow-ups.

**P5 — Did the worker preserve the spike's well-balanced invariant?** ADR-020:16 and
amendment:11 still anchor the spike outcome (flat failure at 150 s, mountain failure
at 70 s, smdiv=0.1 doesn't move it). The "formulation-first" stance is unchanged. The
spike-introduced `_spike_rayleigh_sponge_w` legacy hook is not on this branch (matches
c2-A1 baseline). Damping defaults remain disabled.

## WRF source cross-check on the new R5 4th-term spec

Read `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F`
at lines 820-946 and `module_initialize_real.F:3741-3755`. Findings:

1. **3-term form (R4 closure)** matches WRF 828-831 (x) and 902-905 (y) structurally
   — see R4 table. ✓
2. **4th-term form (R5 closure)** matches WRF 861-862 (x) and 935-936 (y)
   structurally, with the M1 notation nit on the `mu_avg` factor. ✓
3. **dpn boundary kernel** at WRF 836-838 (x bottom), 843-845 (x top), 910-912 (y
   bottom), 917-919 (y top) uses `cf1/cf2/cf3` exactly as the ADR specifies. ✓
4. **dpn interior kernel** at WRF 850-851 (x), 924-925 (y) uses `fnm/fnp` exactly as
   the ADR specifies. ✓
5. **cqu/cqv coupling** at WRF 868 (u), 942 (v) matches the ADR's R7 closure. ✓
6. **cf1/cf2/cf3 construction** at WRF `module_initialize_real.F:3753-3755` matches
   the gpuwrf flat() formula at `grid.py:190-192` exactly under 0/1-based indexing. ✓
7. **fnm/fnp construction** at WRF `module_initialize_real.F:3744-3745` matches the
   gpuwrf flat() formula at `grid.py:167-168` exactly. ✓

## Adversarial: did the worker accidentally hide or paraphrase incorrectly?

- 4th-term sign convention: WRF writes `+ (msfux/msfuy)*rdx*(php(i)-php(i-1)) * (...)`,
  i.e. **additive** to the existing 3-term dpxy. ADR-020:42 says `dpxy +=` (additive).
  ✓ Sign convention preserved.
- WRF `dpn(i,1)` is the *bottom* face (k=1) and `dpn(i,kde) = 0.` is the top face
  (except under `top_lid`). The ADR does not name the top-lid branch (still open as
  A2 from the original review). Recommend c2-A2 add the `top_lid` config; the
  original review already flagged this as a non-blocker.
- `mudf_xy(i)` is computed inline in `module_small_step_em.F:880` and decoupled from
  `cqv` damping — the ADR correctly keeps `mudf` distinct from `smdiv` (R7).

## Net decision

**ACCEPT-WITH-MINOR.**

All 7 R-findings from the original c2-A1 review are addressed in code, ADR, and
architecture.md. The R1 DycoreMetrics extension is correctly typed (fp64), correctly
shaped, correctly constructed from WRF init algorithm, and exercised by both the
analytic-flat and wrfinput_d02 fixtures with bit-equality checks. The R4 slope-
subtraction clause is removed and replaced with the WRF implicit 3-term cancellation
form with proper line cite. The R5 4th term is added with the right intermediates
(`php`, `dpn`), the right coefficient policy (`cf*` at boundary, `fnm/fnp` in
interior), and the right WRF anchors. R2/R3/R6/R7 documentation is exact and
WRF-anchored.

Two minor follow-ups are recommended but **not blocking**:

- **M1** (cosmetic doc fix in ADR-020:42): rewrite the 4th-term `mu` factor as either
  `-0.5*c1h*(mu_left + mu_right)` (literal-WRF) or `-c1h*mu_avg` (with `mu_avg`
  explicitly defined as the standard mean). Either is unambiguous; the current
  `-0.5*c1h*mu_avg` requires the reader to know that `_avg` is shorthand for the
  unweighted sum.

- **M2** (cosmetic doc fix in ADR-020:42 and architecture.md:57-58): explicitly state
  that `php` and `dpn` are substep-local intermediates in the acoustic substep scan
  carry, alongside `al/alt`. The current ADR implies this but doesn't say it.

Both M1 and M2 are documentation-only and can be folded into the c2-A2 first-day work
without blocking dispatch.

## Green-light for c2-A2 implementation dispatch

✅ Dispatch c2-A2 implementation sprint. The architecture skeleton, state taxonomy,
scan-carry policy, static metric set, and PGF specification are now sufficient for the
c2-A2 worker to implement the WRF small-step pressure-gradient operator without
introducing the formulation-error classes the spike already identified. The cf*/fnm/fnp
metrics are loadable, shaped, tested, and proof-anchored. M1/M2 are c2-A2 first-day
cleanup, not c2-A1''' material.

## Hard-rule compliance

- **READ-ONLY**: no code modified during this re-review; only this `reviewer-report.md`
  is added.
- **File:line cites**: every R-finding above cites file:line in either this repo or
  `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/`.
- **WRF source cross-checked**: lines 820-946 of `module_small_step_em.F` and
  3741-3755 of `module_initialize_real.F` read directly and compared to ADR formulas
  and `grid.py` flat() construction.
- **Re-ran proofs**: `pytest -q tests/test_m6x_c2_*.py tests/test_m6_state_extension.py`
  → `15 passed in 11.63 s` on this worktree.
- **Commit + push before /exit**: follows.
