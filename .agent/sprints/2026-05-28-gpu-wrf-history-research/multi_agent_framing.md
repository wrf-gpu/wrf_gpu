# Multi-Agent Methodology Framing

## Recommended positioning

The paper should present the AI process as an engineering method under human
responsibility, not as autonomous scientific authority. The strongest framing is
that single-model coding attempts failed to reach a trustworthy WRF GPU port,
while the later frontrunner-critic-feedback system produced better outcomes
because it separated proposal, implementation, evidence collection, and
rejection. That separation matters in numerical weather software because a
plausible forecast field can still be wrong.

The user's stated history is admissible as local evidence: GPT 5.4 alone and
Opus 4.6 alone did not get close in the first attempt. The paper should not
turn that into a universal law about those models. It should say that, in this
project, a single assistant tended to over-commit to a path, lose track of
validation obligations, or accept a plausible numerical explanation without
enough proof. The multi-agent system reduced that failure mode by making every
claim pass through an opposing role before it became project memory or paper
language.

## What the frontrunner-critic-feedback system adds

The frontrunner role keeps momentum. It writes the candidate implementation,
chooses a concrete patch, and produces artifacts. The critic role attacks the
weakest premise: missing source evidence, hidden host/device transfer, wrong
denominator, unvalidated physics, or a claim that outruns the proof object. The
tester role reruns commands and checks that proof objects exist on disk. The
manager role preserves long-horizon project state and decides the next sprint
contract. The feedback loop is valuable because the next contract incorporates
the failed hypothesis rather than hiding it.

That mechanism resembles actor-critic and self-refinement ideas at a high
level, but the paper should avoid overclaiming a formal RL equivalence.
Actor-critic methods separate action-producing and value/evaluation functions
[sources: konda2000actorcritic]. LLM agent systems such as AutoGen show that
multi-agent conversation can coordinate specialized roles [sources: wu2023autogen].
Self-Refine and Reflexion-style work support the idea that generation followed
by explicit feedback and revision can outperform one-shot generation
[sources: madaan2023selfrefine,shinn2023reflexion]. SWE-bench and SWE-agent provide
software-engineering context for repository-level tasks [sources: jimenez2024swebench,yang2024sweagent].
The project-specific novelty is applying these ideas under scientific-model
governance: contracts, proof objects, validation tiers, and reviewer refusal to
accept unsupported claims.

## How to write it in the introduction

Use a concise paragraph like this:

"The implementation was produced through a governed multi-agent workflow rather
than a single assistant session. A manager agent wrote sprint contracts and
preserved repository context; worker agents implemented only scoped tasks;
tester and critic agents reran commands, challenged numerical and performance
claims, and blocked completion when proof objects were missing. This workflow
was not cosmetic. Earlier single-model attempts in this project failed to
produce a trustworthy port, while the later frontrunner-critic-feedback loop
identified an overclaim in the initial performance narrative and forced the
paper to report the slower corrected-physics result."

Then cite the method context, not the local outcome, with
[sources: jimenez2024swebench,yang2024sweagent,wu2023autogen,madaan2023selfrefine,shinn2023reflexion].

## Boundaries and risks

Do not claim that the agents are authors in the human legal sense unless the
target venue allows that. The human principal retains scientific and submission
responsibility. Do not imply that multi-agent review replaces human numerical
methods review; it reduces some AI failure modes but does not certify the
forecast. Do not use vendor blog posts as the only support for agentic software
engineering if the paper makes a scientific-method claim. Use them only for
tool context, and prefer peer-reviewed or stable preprint sources for the
methodology.

The best methodological claim is therefore:

"The work demonstrates a proof-object-driven AI engineering workflow for a
nontrivial regional NWP prototype, including an explicit rejection loop that
changed the published claim when validation contradicted the initial
performance narrative."
