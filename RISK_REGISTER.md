# Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Architecture lock-in too early | Multi-year wrong turn | M2 bakeoff plus ADR |
| WRF fixture extraction complexity | No trusted oracle | Start with smallest fixtures |
| GPU memory insufficient for 1 km domain | v0 blocked | 3 km first, memory audit before expansion |
| Physics validation false confidence | Wrong forecasts | Four-tier validation and independent tester |
| Skill/memory bloat | Agent drift | Patch protocol and hygiene passes |
| Agent drift | Incoherent implementation | Sprint contracts and proof objects |
| Hidden CPU/GPU transfers | Performance collapse | Transfer audit and profiler gate |
| Public naming/licensing confusion | Release risk | License notes and human legal review |
| Over-parallelization | Merge conflicts | File ownership and branch policy |
