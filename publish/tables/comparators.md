# Comparator Speedup Table

Source discipline: prior-art values are taken from `publication/research_brief/english_brief.txt` as directed by the sprint contract; citation keys are from the frozen `publication/draft/references.bib`. These rows are context, not normalized apples-to-apples benchmarks against this repository's 20260521 Canary d02 replay.

| System | Approach | Hardware | Reported Speedup | Source Citation |
|---|---|---|---|---|
| Pace (pyFV3) | GT4Py DSL + DaCe FV3 dynamics port | Heterogeneous CPU/GPU nodes; brief comparison cites Ampere A100-class GPU context | 3.5-4x vs optimized CPU Fortran | \cite{dahm2023pace} |
| ICON-exclaim / ICON GPU | GT4Py DSL and OpenACC production migration | NVIDIA GPU supercomputing nodes | 5.5x socket-to-socket | \cite{fuhrer2026icon,lapillonne2026benchmarking} |
| SCREAM | Clean-slate C++/Kokkos atmosphere model | 27,000 AMD MI250 GPUs on Frontier | 1.26 SYPD at 3.25 km global cloud-permitting resolution; throughput result, not a simple speedup ratio | \cite{bertagna2024scream} |
| NIM | F2C-ACC Fortran-to-CUDA dynamics port | NVIDIA Fermi / Kepler GPUs | Up to 34x, dynamics-only | \cite{govett2017parallelization} |
| AceCAST | CUDA Fortran / OpenACC WRF acceleration line | NVIDIA Tesla / Ampere GPUs | 5-14x reported range | \cite{tempoquest2025acecast} |

Current repository performance is tracked separately in `publish/tables/performance_evolution.md` because its proof object is local measurement evidence, not a bibliographic prior-art citation.
