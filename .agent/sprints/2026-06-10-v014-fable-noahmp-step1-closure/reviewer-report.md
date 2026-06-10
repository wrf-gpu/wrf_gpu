# Reviewer Report: V0.14 Fable NoahMP Step-1 Closure

Decision: ACCEPT AS NARROWING AND PRODUCTION FIX.

The result satisfies the contract's fallback endpoint: it either had to make
strict Step-1 green or prove the remaining blocker narrower than
"NoahMP disabled/missing land/static state." It did the latter. The original
configuration gap is gone and independently rerunnable proof artifacts now
identify the leading residual as the NoahMP land-tile energy chain.

Why accept:

- The production edit is small and default-inert for existing callers:
  `first_timestep=False` remains the default, while operational Step-1 now
  forwards the real first-step flag into the NoahMP blend.
- A focused test verifies default equivalence and first-call threading.
- WRF-anchored causal splits eliminate the MYNN kernel, sfclay first-call, land
  input init, and RRTMG surface forcing as the leading cause of the strict
  residual.
- The next proof command is concrete: add a per-column WRF `noahmplsm` energy
  hook at the worst cell and compare against `physics.noahmp`.

Residual risk:

- The strict residual remains large. Do not start TOST, Switzerland, FP32 R1/R2,
  or long GPU validation from this state.
- The pinned WRF truth set lives under `/tmp`; it is reproducible, but the next
  sprint should keep provenance commands and hashes explicit.
