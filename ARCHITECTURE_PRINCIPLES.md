# Architecture Principles

- Run an architecture bakeoff before locking the stack.
- Candidate backends include JAX, Triton, GT4Py/DaCe, CuPy, Numba, CUDA Tile, and explicit CUDA/CUDA Fortran only if justified.
- Python orchestration is likely, but not frozen.
- Interface contracts precede implementation.
- The high-frequency state is device-resident.
- Public APIs stay minimal until proven by use.
- WRF-compatible I/O and namelist mapping are adopted where they reduce operational friction.
- The system optimizes for Canary v0 value before broad feature parity.
