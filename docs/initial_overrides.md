# Initial-field overrides

`gpuwrf override` is merged in the v0.4.4 development build. It edits checked
private copies of WRF inputs on the CPU, preserving the originals. This is a
gpuwrf extension, not a WRF namelist option. Qualification covers host/file
behavior; edited GPU forecasts and the full release gates remain pending.

## Swiss preparation example

Create a JSON spec; YAML is also accepted. This adds 0.1 K to land skin
temperature in the shipped Swiss input:

```bash
mkdir -p runs
cat > runs/swiss-overrides.json <<'EOF'
{"overrides": [{"nest": 1, "field": "TSK", "op": "add", "values": 0.1, "mask": "land"}]}
EOF
JAX_PLATFORMS=cpu gpuwrf override --input-dir examples/switzerland_d01 \
  --output-dir runs/swiss-edited-inputs --spec runs/swiss-overrides.json
```

The output directory must not exist. It is published atomically only after
checks pass and contains copied `wrfinput_dNN`, optional `wrflowinp_dNN`, and
`OVERRIDE_PROTOCOL.json`. **It is not a complete runnable case:** namelist,
wrfbdy, runtime tables and other case files are not copied. Assemble those
from the unchanged original case before forecasting. An empty `{}` spec
copies the selected input files byte for byte.

## Operations and accepted fields

Each edit names an integer `nest` (1 is d01), `field`, and one of `set`
(default), `scale` or `add`. Operations apply in order. `target: wrflowinp`
selects the auxiliary forcing file; SST, VEGFRA and ALBBCK edits there apply
to every supplied record. A later aux4 record may replace an initial-only edit.

Supply exactly one of `values` or `file`. Values may be scalar or exact-shape
arrays; a singleton Time dimension may be omitted. Files may be NPY, NPZ or
NetCDF; `variable` selects the array and defaults to the edited field name.
Paths resolve relative to the spec. `mask` may be `land`, `water`, a boolean/
0–1 array, or `{"file": "cells.npz", "variable": "selected"}`. File masks
default to `mask`; horizontal masks broadcast over Time and levels.

Supported atmospheric storage includes dry potential-temperature perturbation
`T` (K), perturbation pressure `P` (Pa), moisture mass fields and supported
hydrometeor number/volume fields. Surface/land fields include SST, TSK, SMOIS,
TSLB, LAI, VEGFRA, LU_INDEX/IVGTYP, ISLTYP, EMISS, ALBBCK and ZNT. Fields must
already exist. Units, shapes, dtypes and metadata are retained. **VEGFRA is
percent (0–100)**. If LU_INDEX and IVGTYP both exist, edit them consistently.

Masks/values are checked before conversion. Unknown keys, absent fields,
packed NetCDF targets, pickled NumPy objects, integer truncation, nonfinite/
fill values, storage overflow and incompatible shapes refuse.

## Physical checks and explicit waivers

Checks cover nonnegative moisture, liquid-water RH (default bound 1.0), dry
mass, layer thickness, soil porosity, MODIS classes, land/water consistency
and field ranges. Porosity uses the actual STAS `SOILPARM.TBL`, with path and
hash recorded. The shared [explicit-cap helper](soil_porosity.md) can check
already-loaded soil arrays without reading tables.

T/P/QV and other moist-mass edits require an explicit `hydrostatic_consistency`
waiver because real.exe-consistent PH/MU rebalance is not implemented:

```json
{
  "overrides": [{"nest": 1, "field": "QVAPOR", "op": "scale", "values": 0.99}],
  "waivers": {"hydrostatic_consistency": "Controlled unbalanced initialization experiment"}
}
```

Other waivable checks are `moisture_positivity`, `supersaturation_bound`,
`soil_porosity`, `land_water_mask`, `modis_classes` and `field_ranges`.
A waiver needs a nonempty reason and is recorded as `WAIVED`, never `PASS`.
Schema, nonfinite/fill values, positive mass/geometry, source-transport identity
and HGT/base-state editing refusals cannot be waived. No clipping, saturation
adjustment or silent repair is performed. A waiver documents an experiment;
it does not establish physical balance or forecast fidelity.

## What Noah-MP recomputes

`--surface-physics 4` is the default. Under Noah-MP, EMISS/ALBBCK/ZNT storage
edits refuse with `NO EFFECT under Noah-MP; use --surface-controls`.
`--force-no-effect` permits the storage edit with a warning and a protocol
flag; it does not make Noah-MP use the edited value. ALBEDO/SH2O recomputation
is explained in the tool's refusals. Use the existing
[surface-control consumer contract](surface_params.md#what-can-be-controlled)
for persistent parameter changes; this page does not qualify new controls.

Input LAI is used for Noah-MP DVEG 7/8/9; VEGFRA for 1/6/7; DVEG=4 uses
SHDMAX. Soil/snow state preservation belongs to [land cycling](cycling.md).
For an already running model, use the separate
[checkpoint branch manifest](checkpoint-branching.md); it uses carry names
such as `qv`, not this file spec's WRF names such as `QVAPOR`.

## Protocol, qualification and cost

The protocol binds source/output file SHA256, edit-array before/after hashes,
dtype/shape, changed-cell counts, external array/mask hashes, checks, waivers
and no-effect warnings. Failed admission leaves no final output directory.

CPU known-answer, no-op, transaction, nonfinite, array-admission and deletion
tests are recorded in the merged [OVERRIDE report](../proofs/w44_override/OVERRIDE.md).
On original 0227 three-nest files, empty specs preserve all files byte for
byte; explicitly waived QV/T edits change only requested storage. Those host
operations measured 0.421 s empty / 0.914 s edited, with 541 MiB peak process
RSS including imports/guard setup [M]. Later checker measurements are 0.556 s
file validation / 0.072 s in-memory across three domains, excluding payload
loading, with 490 MiB process peak. They are different host scopes, not an
edited forecast benchmark.

The tool adds no numerical timestep operators or device state. Extra host
copies, disk writes and validation have CPU/memory costs. Forecast response,
speed and VRAM for the edited case remain unmeasured [U]; unchanged default
model qualification is carried by source identity. The edited three-nest GPU
smoke and original CPU-WRF follow-ups remain open.

## CPU-WRF comparison history

Compare original CPU-WRF and gpuwrf from the same modified case against their
unchanged controls. Select each edited field and its response diagnostics in
CPU-WRF `iofields_filename`. For a surface experiment, a starting list is:

```text
+:h:0:TSK,TSLB,SMOIS,SH2O,ALBEDO,EMISS,ZNT,LAI,T2,Q2,U10,V10,HFX,LH
```

The exact list depends on the eventual accepted edit and land scheme; keep the
underlying input and manifest, since a history field alone cannot prove which
initial edit reached its intended consumer.

CPU-WRF can use the same edited files for a matching initialization experiment;
the gpuwrf protocol/waivers are not CPU-WRF configuration. Keep both models'
unchanged controls and use identical forcing. A host check does not prove
that either model improves against observations.

To change a land-use class with its dependent fields, flip land and sea, patch
roughness on a mask or change terrain, use [static-field patches](static_patch.md).

Return to [Tune WRF for your region](regional_optimization.md).
