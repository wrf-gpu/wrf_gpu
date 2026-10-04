# S1/S2 — back-channel to manager (0:1)

## 2026-06-21 — S1 finding (heads-up, NOT blocking; proceeding with bit-identical interpretation)

**Brief S1 as written:** "remove hot-carry totals `p_total/ph_total/mu_total`, reconstruct
totals ONLY at I/O (base + perturbation)."

**ACTUAL code state (mapped, not invented):**
- The outer-State hot carry holds THREE full-grid members per family:
  `p` (legacy alias) ≡ `p_total` (authoritative) + `p_perturbation`. `p≡p_total` bitwise at all
  times (kept in sync by `State.replace`; equal at every construction site). Same for `ph`/`mu`.
- The dycore is **total-authoritative**: `small_step_prep`/`small_step_finish` (and
  `operational_mode`) recover the base as `(state.p_total − state.p_perturbation)` **each RK
  stage** and reconstruct `p_total_new = (p_total_old − p_pert_old) + p_pert_new`. The total is
  also consumed **directly** (`diagnose_pressure_al_alt(state, None, …)` → `inverse_density(theta,
  state.p_total)`; `_advance_chunk` reads `state.p_total`/`ph_total`/`mu_total`).
- There is already a `BaseState(pb/phb/mub)` and a `_base_pressure/_base_geopotential/_base_mu`
  contract that take an explicit base OR fall back to total−pert. Restart already derives
  PB/PHB/MUB = base from `(total, perturbation)`.

**Why literal total-removal is NOT bit-identical (proven 3 ways):**
Carrying the perturbation + a STATIC explicit base and reconstructing `total = base + pert`
changes the last-ULP arithmetic vs the current `(total_old − pert_old) + pert_new` round-trip
(`(a−b)+b ≠ a` in IEEE fp64), and `state.p_total` is consumed directly. Any of these → the
bit-identical gate (a) fails. This is exactly the **perturbation-authoritative rewrite = S4**
(roadmap §8.6(3): "explicit base/reference leaves; NO total-minus-perturbation base recovery"),
which is tolerance-gated and lives behind the additive fp32 mode — NOT a bit-identical S1.

**Chosen bit-identical S1 (the maximal precision-invariant liveness reduction):**
Remove the **legacy `p`/`ph`/`mu` duplicate leaves** from the State hot carry (read-only
properties aliasing `p_total`/`ph_total`/`mu_total`). Provably bit-identical (exact duplicates).
Removes 3 full-grid arrays from the carry (the v0.13 VRAM note already flags this exact
redundancy). Satisfies gate (b) (carry no longer holds the duplicate total aliases live).

The brief authorizes this: "If the aliases are named differently or already absent, report the
ACTUAL state — do not invent." Proceeding; will verify bit-identity (gate a) empirically. If you
want the literal total-removal instead, it must move to S4 with a relaxed-tolerance gate — say so
and I'll re-scope.
