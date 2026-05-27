# Prior Art Gaps

Cross-reference target: Related Work Sections 2.2, 2.3, and 2.4 against `publication/research_brief/english_brief.txt`.

## GPU NWP And Regional Modeling

- **COSMO/CH GPU precedent is missing.** The brief explicitly says not to claim first GPU-enabled regional NWP because COSMO-CH1-EPS-class systems existed. The paper avoids first-claim language, but Related Work should still mention this precedent to show the authors know the regional GPU history.
- **WRF-specific acceleration history is underdeveloped.** The paper mentions directive ports and AceCAST but does not cite or explain enough WRF-specific OpenACC/CUDA Fortran work. This is the most relevant reviewer comparison.
- **AceCAST is too thinly supported.** The current citation is generic docs while the text makes a numerical speedup claim. Either add exact brief-named sources or remove the number.
- **I/O and workflow acceleration prior art is unused.** `fredj2023adios2wrf` is in the bibliography but not cited. It belongs in a paragraph about what this paper does not solve: I/O, streaming, restart packaging, and live workflow.
- **Kokkos/Triton framework prior art is only indirect.** SCREAM covers Kokkos as a model example, but the Methods/architecture contrast would be stronger with primary framework citations if the paper discusses backend alternatives.

## ML Weather Models

- The draft covers the main ML-weather names from the brief: GraphCast, Pangu-Weather, FourCastNet, GenCast, Aurora, NeuralGCM, Stormer, AIFS.
- The main gap is not names but positioning: explain more explicitly why a 3 km regional, boundary-forced, station-verified replay model is not comparable to global medium-range ML benchmarks.

## AI Agents And Software Engineering

- The brief names SWE-agent, AutoGPT, AutoGen, MetaGPT, Devin, Cline, Cursor, and agent-pattern literature. The draft only cites SWE-bench and Anthropic patterns. That may be enough for concision, but then the paper should not imply it has surveyed agentic software engineering broadly.
- Add SWE-agent or another peer-reviewed/reputable agent-system reference if the paper claims repository-level autonomous coding rather than just "AI-assisted development".
- Avoid relying on tool vendor marketing for claims about agent capabilities. Use it only to describe the local tooling context.

## Verification And Forecast Skill

- The draft includes PyCECT, FSS, and SAL. It should also explain what is actually implemented now versus planned. The current M7 result is station RMSE/BIAS/MAE; precipitation FSS/SAL is future work.
- If METplus or WRF verification tutorials are named in the brief or project plan, mention them in limitations/future validation rather than imply the current verification suite is complete.

## Canary Islands Domain

- The paper's Canary meteorological motivation is plausible but weakly cited. The research brief relies partly on generic/news sources. Add durable sources for trade-wind inversion, Canary orography, island wakes, Saharan Air Layer/calima, and AEMET station data if the meteorological framing remains central.
