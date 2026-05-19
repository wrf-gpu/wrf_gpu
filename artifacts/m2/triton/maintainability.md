Install complexity: isolated venv under `data/scratch/m2-triton-venv/`; the runner installs `triton==3.7.0` and `torch==2.12.0` only there. Torch is a large runtime dependency for a backend that otherwise wants to be a narrow kernel escape hatch.

Error legibility: Triton compile errors usually point back to the `@triton.jit` source line and are shorter than XLA errors, but type and broadcasting failures still require knowing Triton's block language.

Debugger story: `TRITON_INTERPRET=1` is the source-level option for small kernels. Normal GPU runs are opaque compiled cubins; this sprint preserves cuobjdump output and copied cubins under `data/profiler_artifacts/triton/`.

Agent-iteration friction: moderate. The math expression is direct, but the torch-backed runtime, cache behavior, and resource extraction add more moving parts than pure JAX.
