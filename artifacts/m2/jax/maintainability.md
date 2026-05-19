Install complexity: isolated venv under `data/scratch/m2-jax-venv/`; the runner installs `jax[cuda13]==0.10.0` only there, so the project dependency set stays unchanged. The first run is network and wheel-download heavy; reruns reuse the venv.

Error legibility: the deliberate shape bug in `data/profiler_artifacts/jax/deliberate_jax_bug.txt` gives a useful operation-level broadcast error, but it arrives inside a long JAX/XLA traceback.

Debugger story: `jax.disable_jit` and `jax.debug.print` are the practical source-level tools. HLO text is captured per problem, and cubins are copied from the XLA dump directory when available for `cuobjdump`.

Agent-iteration friction: low for math expression and fixture parity; moderate for profiling because XLA hides CUDA launch/resource details behind dump files and backend-version-specific text formats.
