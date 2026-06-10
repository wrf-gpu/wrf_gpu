# Memory Patch: V0.14 Fable NoahMP Step-1 Closure

Reviewer Status: NO_MEMORY_CHANGE.

This sprint changed physics/coupling semantics, not memory layout. The only
production edit is forwarding `first_timestep` into the NoahMP sfclay blend.
No arrays were added to runtime state, no resident carry shape changed, and no
GPU memory claim is made.

Memory/validation implication:

- Exact-branch memory preflight still needs to be rerun after grid-parity source
  changes stabilize.
- v0.15 kernel-efficiency review remains prepared but must wait until v0.14
  long validation is running or complete and Fable/Mythos is free.
