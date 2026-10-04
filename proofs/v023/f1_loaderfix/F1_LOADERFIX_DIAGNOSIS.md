# F1 Loaderfix Diagnosis

## Verdict

BOUNDED.

The B>1 loader treedef failure is a lane canonicalization bug in the homogeneous
batch check, not evidence that F1 per-lane static aux needs a runtime rewrite.

## Root Cause

`_assert_homogeneous_batch()` compares raw JAX pytree treedefs for each
`OperationalNamelist`:

```python
jax.tree_util.tree_structure(bundles[name].namelist)
```

For the Tenerife fixture the nested pipeline enables Noah-MP.  The
`OperationalNamelist.tree_flatten()` static aux includes identity-based
`_StaticHolder(...)` wrappers for Noah-MP static bundles and parameter bundles.
Two independently loaded identical lanes therefore construct different Python
objects for the same static content.  JAX treedef equality sees those holder
objects as unequal and `_assert_homogeneous_batch()` raises:

```text
batch lane 1 d01: namelist/static treedef differs
```

The first differing aux entry in the CPU probe is index 40, which is
`_StaticHolder(self.noahmp_static)` in `OperationalNamelist.tree_flatten()`.
The same result occurs for `d02`.  Carry treedefs match.

## CPU Probe

Command class:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=0 \
  GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF \
  python <inline loader probe>
```

Probe details:

- Patched only the local probe process so `State.zeros()` allocates on the CPU
  device; no source edit and no forecast run.
- Loaded the real Tenerife fixture:
  `<DATA_ROOT>/wrf_downscale/f1_retest/tenerife_20250121_3km1km/real_run`.
- Called `_load_batched_domains(..., batch_size=2)` with the same path repeated:
  this currently passes because `_load_batched_domains()` deduplicates identical
  `Path.resolve()` values.
- Called `_load_domains()` twice independently on the same fixture to expose the
  underlying non-canonical treedef behavior.

Observed:

```text
BATCH_DEDUP_LOAD_OK
d01 namelist_treedef_equal False
d01 carry_treedef_equal True
d01 first_aux_diff_index 40 _StaticHolder _StaticHolder
d02 namelist_treedef_equal False
d02 carry_treedef_equal True
d02 first_aux_diff_index 40 _StaticHolder _StaticHolder
```

## Interpretation

The current same-path dedupe is not a sufficient F1 loader contract: it only
masks repeated identical paths.  It does not handle separately loaded
same-geometry lanes, which F1 needs for different dates / cases.  The fix should
canonicalize the homogeneous namelist/static comparison for batch eligibility,
using the same content-stable view already used by the AOT cheap-key code for
`_StaticHolder` objects, or an equivalent deterministic static-aux signature.

This is a small bounded fix because:

- the failure happens before the timestep loop;
- carry treedefs already match;
- the per-domain grid/dt/physics signatures already match;
- no dycore, physics, or host/device timestep-loop behavior needs to change.

## Effort Estimate

Expected implementation effort: 1-3 hours for the loader canonicalization plus
the already-required B>1 re-enable and verification.  Risk is moderate only
because the current GATE-2 fix deliberately removed/disabled parts of the F1
runtime path; those must be restored without touching the default B=1/unset path.
