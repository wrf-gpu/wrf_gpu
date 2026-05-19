# Maintainability — gt4py (M2 candidate C, EXCLUDED)

GT4Py + DaCe was not implementable in v0 due to a Python 3.13 + DaCe 0.10.0 + SymPy upstream incompatibility (M2-S1 scout finding). No code was written; no build was attempted past the hello-GPU smoke. Maintainability assessment is therefore based on the published GT4Py + Pace + icon-exclaim evidence cited in PROJECT_PLAN.md, not on hands-on M2 data.

**Build complexity**: GT4Py is pip-installable but DaCe needs CMake + a working C++ toolchain for code generation. Comparable to Kokkos in build friction.

**Error legibility**: stencil DSL errors point at meteorological-domain concepts, which is *good* for atmospheric scientists but *bad* for general-purpose AI agents that don't share the domain vocabulary.

**Debugger story**: limited. GT4Py emits generated C++; debugging means reading the generated source.

**Agent-iteration friction**: unknown; published Pace work suggests it is workable for stencil-heavy code but slow for irregular column physics. The deepthink and GPT-5.5 briefs both note GT4Py's strength is exactly the stencil class M2 already covers, and its weakness is column physics where JAX/Triton both already excel.

ADR-001 does not select GT4Py. If M5 reveals JAX+Triton inadequate, a post-M2 remediation sprint may reopen this candidate under a Python 3.12 venv.
