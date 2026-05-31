# Human Author Notes — source material for Intro / Discussion / Conclusion

Status: author's notes, 2026-05-31. Author/principal/initiator: **Enric R.G.** These
are the human first-author's own words, cleaned up and structured for use in the paper's
Introduction, Discussion, and Conclusion. They should read as *human-written reflection*,
not generated prose. The paper-drafting agent must keep this voice and not over-polish it
into corporate tone. Claims marked **[CITE]** need a reference or a placeholder before
publication; claims marked **[VERIFY]** need a proof object or should be softened.

---

## 1. Origin — the wish that started it

The project began from a concrete personal need, not a research agenda. I run a nightly
WRF v4 forecast for the Canary Islands and wanted it fast enough to be operationally
useful — to compress my runs from roughly **8 h on CPU to ~2 h on GPU** so the local
forecast system works in time, and so I could eventually offer it to everyone on the
islands for free.

I started by giving GPT-5.5 Pro roughly this brief (reconstructed):

> *"I need a WRF GPU port that is at least 3–4× faster on my system, stays true to WRF v4
> solutions, and is built with a modern architecture. This is beyond a single agent, so
> build an agent framework: a manager keeps the project plan and dispatches workers. To
> minimize bias and maximize swarm intelligence, use both GPT-5.5 (xhigh) and Opus 4.7
> (xhigh/max). Phase 0 should explore the optimal kernel architecture and estimate how
> fast each could theoretically be. The remaining phases work toward the goal with
> verifiable milestones. Research what standardized tests should be passed to declare
> success. The mandatory test is comparison against the existing 3 km and 1 km WRF v4
> solutions in [the corpus folder]."*

That single prompt — plus occasional top-level steering — is essentially the entire human
contribution to the engineering.

## 2. The build — agent framework and model-role timeline

- **GPT-5.5 Pro built the foundations**: the skill files, the memory system, the manager
  and frontrunner and verifier roles — the governance scaffolding the rest of the project
  ran on.
- **Manager role**: Opus 4.7 initially; toward the end of the build week, **Opus 4.8**
  took the manager role.
- **Frontrunner (implementer)**: mostly **GPT-5.5** in the early/middle stages; in the
  later stages **Opus 4.8 (max)** became the code frontrunner.
- **Verifier**: tested **every sprint** initially, later **every milestone** (to save
  tokens once the foundations were trustworthy).
- **When stuck**: GPT-5.5 and occasionally **Gemini 3.5 Flash** were dispatched for
  independent angles; the manager collected all the intel and decided.
- **Human (me)**: about **once or twice a day** I asked for a top-level status, sometimes
  ran `/compact`, and gave high-level guidance — usually small course corrections like
  which model to prefer when one was short on tokens, and toward the end asking for
  real-world proof and for guidance on how to style and research the arXiv release, and
  whether this is even a valuable contribution to geoscience or computer science.

## 3. Resourcing and how to count the effort honestly

The wall-clock figure must be stated carefully:

- It was **not 24/7 wall-clock time**. It was **mostly nightly runs**, and **solely
  "free-token" runs** — this is a non-paid, unfunded, open hobby project.
- Because of that, **agent runs / sprints are a more honest unit than wall-clock hours**.
  The paper should count sprints/agent-runs per stage (approximated from the git history;
  an agent is approximating this — see the process-metrics table) rather than imply a
  continuous human-equivalent effort.
- Exclude the **dead earlier attempt** entirely. What counts is the wall-clock span from
  *nothing* to (a) the v0.0.1 working kernel, (b) the v0.1.0 working replacement, and
  (c) the point of publication.
- **Real-world cost**: everything fit within the token limits of a **€200/mo Claude Max
  plan and a €100/mo GPT Pro plan**, used as a side project. With optimization it could
  probably have been done for **€100 or less**. There was **no funding of any kind**.
- A **total-token approximation** would also be informative and is worth including.

## 4. Author's reflection (for Discussion / Conclusion)

*(First person, the author's voice — keep it.)*

A GPU port of WRF — one of the most complex, 20-plus-year-old codebases in the
geosciences — has never really existed as open source. Not because it isn't useful:
GPU compute is now the high-end of most semi-professional and professional workstations,
so a fast GPU WRF would help a lot of people. It hasn't been done because it is **hard**.
It eluded single-agent attempts, and it eluded an earlier attempt with GPT-5.4-era agent
swarms. AI seems to have become *just* capable enough, right now, to actually do it — and
once capable, to do it **fast**. **[VERIFY: characterize the prior abandoned open-source
attempt and the completeness of commercial variants; soften to "to the best of our
knowledge" if not citable.]**

My guess for why this was never successfully ported is that it sits at the intersection of
several fields. To do it you need to be an expert in GPU kernel design, a generally senior
developer, *and* understand the physics and meteorology, *and* be able to read WRF's
Fortran. Each of those is rare; the intersection is extraordinarily rare; and the few
people who hold all of it tend to be in senior positions without the time to write free,
open-source code of this complexity. What struck me most is how well AI performs precisely
**at the intersection of several scientific fields** — exactly where human experts are
scarcest.

The genuinely impressive thing, for me, was the **end-to-end capability**: from a simple
wish — *"I need a faster WRF so my runs fit in time and I can offer a free forecast"* —
all the way through to *"publication written and submitted, code repository published"*,
for code that had never been open-sourced. My own input was so simplistic that it feels
only **one iteration away from being fully replaceable**. By the time someone reads this,
there may already be models that can do this entire project in a single run. The progress
is staggering — predictable if you extrapolate the curves, but still genuinely striking:
one of the hardest, most battle-tested codebases on Earth went from *not doable in March
2026* to *done in roughly a week in May 2026*, on free tokens, with barely any human input.

On the practical side: this will help me try to give the Canary Islands a free, actually
working forecast for the islands' complex microclimates. The Spanish AEMET / HARMONIE-AROME
products are inadequate here, and people on the islands have largely accepted that
forecasts "just don't work" — which is false. They just need a 1 km WRF grid or finer.
**[VERIFY: AEMET/HARMONIE-AROME resolution/adequacy claim — cite or soften.]** But that is
not why I prioritized this on a grand scale. I prioritized it because, with the
mind-boggling capability of systems like the H200, a GPU port like this can probably
genuinely help in some applications — and I hope someone finds it useful and continues it.

**Handoff.** This is a hobby project with no funding and very limited time and tokens. It
now serves my needs, so I will stop actively contributing. I would be grateful to hand full
control to anyone who wants to continue it. Bug reports and requests are welcome — ideally
in a clear form I can feed directly to the agent swarm without having to read or rewrite
them.

---

## Notes for the paper-drafting agent

- Use §1 in the **Introduction** (the "wish → publication" framing is the human anchor for
  the verifiable-AI thesis).
- Use §2 and §3 in the **AI Engineering System / Method** section and the **process-metrics
  table** (cross-reference the git-history accounting worker's output).
- Use §4 in the **Discussion** and **Conclusion** — keep the first-person voice; this is the
  human author's reflection and is part of the paper's point.
- Flag every **[CITE]/[VERIFY]** as a placeholder; feed them back to the manager for
  cross-reference against the v0.1.0/v0.2.0 plan and the proof objects.
- Do not overstate: the human role was real but minimal; the paper's honesty depends on
  representing it accurately (initiator + ~daily top-level steering + the directives), with
  the agent transcripts / git history / proof objects as the evidence.
