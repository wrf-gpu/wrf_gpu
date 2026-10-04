# GPT-5.5 xhigh review: precip-oracle validation + implicit-sed ADOPT/REJECT (2026-05-31)

Verbatim final answer (brief sent in gpt55_precip_sidecar_brief.md):

1. **Do not block the functional “Canary precipitates” gate**, but do not call surface precip parity. This is a diagnosed surface-flux attribution mismatch, not a failed Thompson profile/water-budget validation. It should block any gate that claims WRF-equivalent `RAINNCV`/precip skill.

For #32 I’d set: water closure `<=1e-5 rel`, active hydrometeor profile mass-weighted mean rel `<=10%` with low-mass absolute floors, qv `<=1%`, and surface precip only “order/onset” for now: correct precipitating columns plus total within about `4x` or `<=0.2 mm` absolute for this 18 s oracle. Track adaptive `nstep` + `rr > 1e-9 kg/m3` as required before precip-accumulation parity.

2. **Lean: ADOPT-CONDITIONAL for `nsub=4`, REJECT `nsub=1` as default.** BE is a real numerical-scheme change, but not automatically disqualifying for an operational forecast if coupled T2/U10/V10 and precip scores hold. The caveat: adoption must be via ADR/flag and pass a predeclared multi-case coupled skill gate, including precip bias. Mass closure alone is not evidence of acceptable sedimentation physics.

3. Methodology is credible for this decision, with limits. The single-column oracle and WRF-order monkeypatch are exactly the right local test. The masked-active metric is acceptable only if paired with mass-weighted column/profile metrics and precip accumulators, because it can hide low-mass or attribution errors. I would not distrust the call, but I would not promote BE to default from one 18 s column pack.

Files changed: none. Proof object produced: none; this is reviewer judgment over the supplied oracle evidence.
