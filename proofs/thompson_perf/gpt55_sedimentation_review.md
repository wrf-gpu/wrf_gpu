# GPT-5.5 xhigh review: implicit sedimentation (2026-05-31)

Decision: do **not** treat backward-Euler as a faithful drop-in yet. Treat it as a candidate numerical-scheme change.

1. BE upwind is conservative/positive/stable, but its modified equation differs from WRF’s 64 small-Courant explicit updates. It adds full-step implicit diffusion, smooths fall fronts, and can shift precip timing/intensity and vertical latent heating. Bounded, yes; automatically benign for CPU-WRF skill, no.

2. Freezing `vt` for full `dt` is stable but risky. WRF recomputes `vt` every substep because mass/number sorting changes slopes and fall speeds. Use at least one Picard/corrector recompute before judging; full freeze may bias fast rain/graupel arrival.

3. A 44-level bidiagonal sweep should still beat 64 dependent launch barriers if XLA keeps it as one/few kernels. The vertical serialization is tiny and parallel across columns/species. But verify HLO/nsys: if XLA explodes the scan, the win evaporates.

4. Ranking:
   1. **Adaptive/bucketed WRF explicit NSED**: highest WRF faithfulness, medium launch reduction, medium implementation risk.
   2. **Box/Lagrangian single-step**: good physical fidelity, high launch reduction, high validation risk vs CPU-WRF.
   3. **Picard semi-implicit BE**: high launch reduction, medium-high physics risk.
   4. **Flux-limited one big explicit step / blunt NSED reduction**: cheap, but least defensible; likely precip-skill drift.

5. Bottom line: >1.5x on sedimentation is plausible only with a real scheme change. For strict WRF-faithful explicit sedimentation, expect ~1.1-1.3x. Recommended next gate: prototype adaptive explicit and Picard-BE, compare against a precipitating WRF Thompson oracle plus 6-24h precip/T2/U10/V10 skill.
