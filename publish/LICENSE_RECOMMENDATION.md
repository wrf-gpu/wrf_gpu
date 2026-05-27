# License Recommendation for wrf_gpu Public Release

**Status**: manager recommendation, 2026-05-27. The senior corresponding author (Enric R.G.) chooses; this document explains why each option fits or doesn't fit the stated requirements.

## Your stated requirements (in your own words, my translation)

1. **"I want to give it to anyone"** → permissive in the sense of: anyone can read, run, modify, redistribute
2. **"I don't want a company to take it and make money with it but lock down their port"** → copyleft is required (forks must stay open)
3. **"All forks should need to stay open in my opinion"** → strong copyleft, including network/SaaS distribution
4. **"I don't want liability for any of it"** → liability disclaimer + "AS IS" warranty disclaimer
5. **Hobby project that turned out to be something never achieved before** → license should be defensible if someone files a frivolous DMCA or compliance complaint

## The three real options for a scientific GPU code, ranked against your requirements

### AGPL-3.0-or-later ← RECOMMENDED

The GNU Affero General Public License v3 is the strongest copyleft license in mainstream scientific software use. It satisfies all five of your requirements:

| Requirement | AGPL-3.0 satisfies? |
|---|---|
| Anyone can use, modify, redistribute | ✅ |
| Forks must stay open | ✅ — modified code distributed under AGPL must release source under AGPL |
| Closes the SaaS loophole (companies hosting a service must also release source) | ✅ — this is AGPL's unique feature; GPL v3 does not have this |
| No liability | ✅ — standard AS-IS disclaimer in §15 + §16 |
| Defensible against frivolous claims | ✅ — widely deployed by Mongo, Mastodon, etc.; well-tested |

**Why this is the right fit for you**: a company cannot take wrf_gpu, modify it to forecast for their internal use, run it as a service, and keep the modifications private. They must release their modifications under AGPL too. This is what you described.

**The one caveat to know about**: some large companies (notably Google) forbid their employees from contributing to AGPL projects. If you want corporate research labs to contribute back, AGPL will lose you a small number of contributors. But it gains you the legal guarantee that nobody can build a closed commercial offering on top of your code — which is exactly what you asked for.

### GPL-3.0-or-later

GPL v3 is strong copyleft for distributed software but has the **SaaS loophole**: a company can take wrf_gpu, modify it, run it as an internal weather forecasting service, sell weather forecasts to customers, and never release the modifications — because they're not distributing the binary, only the forecast output.

For something that companies will obviously want to run as a service (NWP for energy traders, agriculture, aviation), this is a real loophole. **AGPL is strictly better than GPL for your stated goal.**

### MIT or Apache-2.0 ← NOT RECOMMENDED for your goal

Both MIT and Apache-2.0 are permissive licenses. They allow anyone to take wrf_gpu, modify it, and ship the modifications as closed-source commercial product. **This is what you said you don't want.** Reject both.

Apache-2.0 does have a strong liability disclaimer + an explicit patent grant, which is nice. But the lack of copyleft makes it wrong for requirement 2.

## Bonus: Software Heritage of the WRF source

WRF Fortran itself is released by NCAR under terms that allow free use ("use freely subject to the [NCAR copyright]"). It is not GPL. This means: you can absolutely AGPL your JAX port. There is no upstream license-poison. The WRF community is permissive about derivative implementations as long as they don't claim to be WRF.

Note for the paper: be careful to say "WRF-compatible port" not "WRF" — your code is a clean-slate JAX implementation that reproduces WRF's small-step structure, not a re-distribution of WRF source.

## Concrete release artifacts

If you choose AGPL-3.0-or-later, the release should include:

1. **`LICENSE`** at repo root: the verbatim AGPL-3.0 text from <https://www.gnu.org/licenses/agpl-3.0.txt>.
2. **`NOTICE`** (recommended): one paragraph explaining "this is a clean-slate JAX implementation inspired by NCAR WRF v4; the WRF source license is preserved at <NCAR URL>, this port is independently AGPL-3.0-or-later".
3. **`AI_USE.md`**: discloses that the code was authored collaboratively by Claude Opus 4.7 + GPT-5.5 Codex under the supervision of the human senior author, with proof-object-driven validation. This satisfies arXiv's recent AI-disclosure expectations.
4. **`CITATION.cff`**: machine-readable citation. References the arXiv preprint DOI + the repo commit hash.
5. **`README.md`**: should include a "**Hobby project disclaimer**" paragraph stating clearly that this is the senior author's research project, not a commercial product, no warranty, no SLA, use at your own risk.
6. **`CONTRIBUTING.md`**: explains the AGPL compliance expectation for forks. Be explicit: "If you fork this project and distribute the modified version, you must release your modifications under AGPL-3.0-or-later. This includes hosted services."

## Liability protection

AGPL §15 + §16 already provides standard liability protection. To strengthen it for a hobby project:

- README: explicit disclaimer paragraph: "wrf_gpu is an experimental research prototype. The authors make no claims of operational suitability for safety-critical or commercial weather forecasting. Forecasts produced by wrf_gpu have known skill regressions vs CPU WRF (see §8 of the preprint). Use at your own risk. The authors accept no liability for any consequences of use, including but not limited to incorrect forecasts, financial loss, or downstream decisions."

- Paper: same disclaimer in the Limitations section + a sentence in Author Contributions noting that the human author's role is research supervision, not operational endorsement.

This combination — AGPL §15-§16 + README disclaimer + paper Limitations + Author Contributions framing — is the maximum liability protection available without a formal indemnification clause (which only makes sense if there's a company behind the release, which there isn't for a hobby project).

## Recommendation summary

**Take AGPL-3.0-or-later.** It is the only license that:
- Lets anyone use the code (✅ your requirement 1)
- Forces forks (including network/SaaS forks) to stay open (✅ your requirements 2-3)
- Provides a strong liability disclaimer (✅ your requirement 4)
- Is well-understood and defensible (✅ your requirement 5)

The trade-off is that AGPL is unwelcoming to some corporate research labs. For a hobby project where your goal is openness over corporate adoption, this trade-off is correct.

The final call is yours. If you want me to write the LICENSE + NOTICE + CONTRIBUTING files for you once you decide, I can do that in a small sprint.
