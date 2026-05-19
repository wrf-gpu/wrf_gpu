# Manager Closeout

Sprint: `2026-05-19-m2-jax-stencil-column` (M2-S5, jax bakeoff candidate A)
Closed: 2026-05-19
Cycles: 1 worker (codex), 1 tester (Claude Opus xhigh — 4th cross-AI), 1 reviewer (codex). Zero fix cycles.

## Outcome

4/6 M2 candidates satisfied. **JAX is the current ADR-001 frontrunner** by a clear margin on the M1 analytic fixtures.

## Proof Objects

- Implementation: `src/gpuwrf/backends/jax/{stencil,column,bench}.py` (~280 LOC, pure `jax.jit` + `jax.numpy`, no Pallas).
- Runner: `scripts/m2_run_jax.sh` (idempotent venv + `jax[cuda13]==0.10.0` install + bench).
- Profile JSONs (same schema + conventions as prior candidates):
  - **Stencil**: regs=**48**, local=0, occ=83.3%, launches=1 (HLO-verified), wall≈0.08 ms.
  - **Column**: regs=**22**, local=**0** ✅, occ=83.3%, launches=1, wall≈0.23 ms.
- Correctness: both fixtures pass with max_abs_diff=0.
- Tests: 39 new tests by Claude (including HLO/thunk/cubin cross-checks). Pytest 187/187.

## Merge Decision

Merge Decision: **Accept and integrate into main.** Reviewer Accept with zero required fixes. Note for ADR-001: `kernel_launches=1` wording should be carried with a brief explanation that this is the HLO fusion count, independently verified by the thunk_sequence and cubin dump (Claude tester established the chain).

## Scope Changes

None.

## Lessons

1. **XLA fused both problems into single kernels**, with the *lowest* register count of any candidate on both stencil (48 vs cuda_tile 58, cupy 58, kokkos 64) and column (22 vs cuda_tile 24, cupy 24, kokkos 40). The user's "why not just JAX" intuition is empirically supported on these problem shapes.
2. **Zero local memory spill on the column kernel** — the metric I pushed back hardest on earlier as the JAX risk. XLA didn't spill on this analytic surrogate. *Caveat*: this column kernel is far simpler than real Thompson microphysics or MYNN PBL. The actual register-spill risk only resolves at M5 when real physics shapes hit XLA's compiler. ADR-001 must call this out.
3. **Wall-time is noise at this fixture size.** JAX at 0.08 ms vs Kokkos at 0.13 ms is below measurement floor; both effectively the same throughput.
4. **Occupancy 83.3% (JAX) vs 100% (cuda_tile/cupy/kokkos) on column** is interesting. JAX achieves lower register count but slightly lower theoretical occupancy. This is a compiler tradeoff XLA made (likely larger block size). Real-throughput comparison requires bigger fixtures.
5. **Claude tester continues to add critical cross-AI value.** This sprint: verified that `kernel_launches=1` is real (HLO fusion, not "1 jitted function"), confirmed compile time excluded from wall_time, independently reproduced cubin extraction. Codex worker's numbers were honest, but cross-AI verification confirmed it.

## Next Sprint

**M2-S6**: `m2-triton-stencil-column` — fifth candidate (5/6). Per M2-S1 readiness: `triton==3.7.0 + torch==2.12.0` CUDA13. Triton is the natural hybrid partner to JAX: if Triton matches JAX's column numbers, hybrid (JAX dycore + Triton physics) becomes interesting; if Triton clearly beats JAX on column, ADR-001 should select Triton or hybrid. If JAX still wins, the case for pure-JAX is strong.
