# Why GPU WRF Has Been Hard

## Math

WRF-ARW is not a simple collection of independent column kernels. Its dynamical
core solves compressible non-hydrostatic equations in flux form on a
terrain-following dry hydrostatic pressure coordinate [sources: skamarock2019description].
The split-explicit RK3 method separates slower meteorological modes from fast
acoustic modes, so one apparent timestep contains multiple nested numerical
stages. A GPU port must preserve the recurrence across RK stages and acoustic
substeps; moving only a visually obvious loop body is not enough.

The vertical acoustic/pressure coupling is especially awkward. Horizontally,
many operators look like stencil work. Vertically, the small-step solve contains
implicit or tridiagonal-style dependencies, metric terms, and scratch state.
This is why a single WRF variable cannot be treated as just a dense tensor with
one uniform update. The C-grid staggering also means mass, u, v, and w live at
different logical locations. A fused GPU kernel that ignores these stagger
relationships can be fast and meteorologically wrong.

The lateral boundary and relax zones add more mathematics to the coupling
problem. Regional WRF is not a periodic box. Specified and relaxation boundary
conditions inject external information and smooth it inward at defined stages.
That makes device residency harder because the boundary path is both a
scientific algorithm and an I/O/control interface. Prior OpenCL and OpenACC
studies repeatedly identify host/device transfer as the reason kernel speedups
do not survive whole-model integration [sources: vanderbauwhede2016wrf,nvidia2016wrfgpu].

## Physics

WRF physics is heterogeneous by design. Bulk microphysics, bin microphysics,
PBL, land surface, surface layer, cumulus, and radiation schemes have different
loop shapes, temporary arrays, branching, and required state. Thompson
microphysics, MYNN-style boundary-layer closure, RRTMG radiation, and
Noah/Noah-MP land behavior are not interchangeable stencil kernels
[sources: thompson2008explicit,nakanishi2006numerical,iacono2008radiative,niu2011noah].
They are scheme-specific scientific software components.

The history shows that individual physics schemes can be excellent GPU targets.
Kessler microphysics, WSM5, WDM6, Stony Brook microphysics, Goddard shortwave,
five-layer thermal diffusion, and YSU PBL all have published GPU acceleration
attempts [sources: mielikainen2012wsm5,mielikainen2012sbu,mielikainen2012goddard,wang2013kessler,mielikainen2013wdm6,huang2015thermal,huang2015ysu].
The problem is that a forecast needs these pieces to work together with the
dycore and boundaries, not just run quickly in isolation. A large single-scheme
speedup can disappear if each call requires copying full 3D fields between CPU
and GPU.

Physics also has correctness traps. Many schemes maintain positivity,
saturation, energy, water budget, and empirical limiters through code paths that
are easy to perturb. GPU ports often use fast math, fused multiply-add, altered
loop order, or single precision. Those choices can be acceptable, but only after
savepoint, invariant, or forecast-skill evidence. Without that evidence, a
finite-looking forecast is not automatically a correct forecast.

## Coding

WRF is old in the productive sense: it has accumulated the machinery required
to support many communities. It is also old in the GPU-porting sense: large
Fortran source, registry-generated state, preprocessor macros, nested includes,
MPI decomposition, OpenMP tiling, and scheme-level conventions make automated
offload fragile. The fact that WRF is open source and public-domain-like does
not make it small [sources: powers2017weather].

Compiler directives are attractive because they preserve source familiarity.
The 2016 NVIDIA/NCAR slides explicitly pursued that route, moving many routines
to OpenACC while trying to keep code readable to scientists [sources: nvidia2016wrfgpu].
The tradeoff is that directive ports often inherit host-resident control flow.
If a GPU directive marks a loop but the next scheme call, boundary update, or
diagnostic still requires host data, the performance model collapses into
PCIe/NVLink traffic and synchronization.

Manual CUDA rewrites have the opposite tradeoff. They can be fast but fork the
model into a second implementation that must be validated and maintained. The
Mielikainen/Huang family of papers succeeded on many individual schemes, but
that does not automatically create a coherent WRF release. OpenMP offload work
in 2024 still needed profilers, static analysis, loop fission, memory analysis,
and `diffwrf` validation for just parts of FSBM [sources: wichitrnithed2024openmp].
That is strong evidence that the current hard part is integration discipline,
not finding a GPU syntax.

## Organisational

The prior attempts deserve respectful framing. Michalakes and Vachharajani
showed early that WRF had real fine-grain GPU parallelism [sources: michalakes2008gpu].
The SSEC/NOAA work mapped a large set of important physics schemes to GPUs and
showed the community where the parallelism lives. WRFg and AceCAST show that
industrial groups took WRF acceleration seriously enough to build product-like
systems [sources: adie2018wrfg,nvidia2023acecast]. These were not naive efforts.

The organisational difficulty is that WRF is a community model, not a single
researcher's benchmark. A full GPU port has to satisfy scientists who want
source readability, operations teams who need namelist and workflow continuity,
HPC teams who need compiler and architecture support, and reviewers who need
validation. This creates a tension between "change little, preserve WRF" and
"change enough to be GPU-native." WRFg's restricted-source/object release and
AceCAST's proprietary route are understandable responses to that tension, but
they do not provide the same public audit surface as a source-open research
artifact.

There is also an incentive problem. A paper can publish a 100x kernel speedup;
an operationally useful WRF port must publish a smaller, messier story:
denominator definition, physics options, transfer audits, boundary behavior,
I/O, restart, validation tolerances, and failed cases. The `wrf_gpu` paper
should lean into that mess. Its strongest contribution is not pretending that
the previous groups failed. It is using proof objects, rejection loops, and
careful claim bounds to make a GPU-native path auditable.
