# GPU WRF Port History

## Narrative judgement

The short version is that WRF has had a long GPU history, but not a clean public
history of a fully open, full-physics, device-resident WRF port. The evidence
does not support saying that no commercial GPU WRF exists: TempoQuest AceCAST is
a commercial WRF acceleration product and NVIDIA/TempoQuest materials describe
it as proprietary OpenACC/CUDA WRF with broad dynamics, physics, and namelist
support [sources: nvidia2023acecast,tempoquest2025acecast]. The evidence does
support a narrower claim: prior public WRF GPU efforts were usually single
kernels, selected physics schemes, hybrid CPU/GPU runs, binary/object community
releases with restricted source, or proprietary products. A first-claim for
`wrf_gpu` must therefore be framed around the exact architecture: source-open,
Python/JAX, WRF-compatible, whole-state-resident in the high-frequency loop, and
validated by proof objects. It should not be framed as the first GPU regional
NWP system or as the first commercial WRF acceleration.

## WRF origins and why WRF became the target

WRF emerged from the NCAR/NCEP/community effort to build a next-generation
research and operational mesoscale model, replacing older limited-area systems
with a shared framework useful to universities, national laboratories, and
forecast centers [sources: powers2017weather]. The Advanced Research WRF (ARW)
solver is specified in the WRF technical note as a compressible,
non-hydrostatic, flux-form model using a terrain-following dry hydrostatic
pressure coordinate [sources: skamarock2019description]. Its operational value comes
from both the dynamical core and the large catalogue of interchangeable physics:
microphysics, boundary-layer, surface, land, radiation, cumulus, chemistry, and
coupled extensions.

That maturity is also why WRF is difficult to port. ARW uses an Arakawa C-grid,
staggered velocity components, split-explicit Runge-Kutta time integration, and
small acoustic substeps. The slow meteorological modes and fast acoustic modes
have different numerical character; the vertical part of the acoustic step uses
implicit or tridiagonal-style structure; pressure, geopotential, dry-air mass,
moisture, map factors, and terrain-following coordinates must remain mutually
consistent. The lateral boundary zone is not a trivial halo: relaxation and
specified-boundary behavior interact with time levels and with the RK stages.
These features are normal atmospheric-model mathematics, not bugs, but they
make "wrap the loops with GPU pragmas" a poor route to a trustworthy full model
[sources: skamarock2019description,vanderbauwhede2016wrf].

The software structure adds another layer. WRF is large Fortran infrastructure,
with MPI domain decomposition, OpenMP tiling, registry-generated state, scheme
interfaces designed for host execution, and many arrays whose memory layout and
loop order differ between dynamics and physics. Physics schemes such as
Thompson microphysics, MYNN PBL, RRTMG radiation, and Noah/Noah-MP land state
have column-like or branch-heavy data flow and often expect host-side control
logic [sources: thompson2008explicit,nakanishi2006numerical,iacono2008radiative,niu2011noah].
The project problem is therefore not just CUDA syntax. It is keeping enough
state resident on the device while preserving WRF's coupled numerical behavior.

## Early CUDA work: impressive kernels, small whole-model gains

The canonical starting point is Michalakes and Vachharajani's 2008 WRF GPU
paper. They showed that a computationally intensive WRF portion could run nearly
10x faster on NVIDIA GPUs, but that this translated into only about 1.23x for
the whole weather model [sources: michalakes2008gpu]. That pattern appears again and
again: a kernel can be spectacular, while the full model remains dominated by
unported work, data transfers, communication, or control flow.

The Wisconsin/SSEC and NOAA/NESDIS line of WRF physics GPU papers is the best
documented example of successful kernel acceleration. WSM5, Stony Brook
University 5-class microphysics, Goddard shortwave radiation, Kessler
microphysics, WDM6, WSM6, five-layer thermal diffusion, Eta/Ferrier, and YSU PBL
all had GPU/CUDA implementations or related GPU studies across 2012-2015
[sources: mielikainen2012wsm5,mielikainen2012sbu,mielikainen2012goddard,wang2013kessler,mielikainen2013wdm6,huang2015wsm6,huang2015thermal,huang2015ysu].
Their headline per-scheme speedups are high, often tens to hundreds of times
against a single CPU core. These papers are important because they prove that
many WRF physics kernels contain exploitable data parallelism. They are not,
however, evidence of a full open WRF port: each is a scheme or module effort,
with results often measured against serial or small CPU baselines.

Other academic work targeted dynamics or integration rather than cloud physics.
Vanderbauwhede and Takemi accelerated WRF scalar advection through an OpenCL
integration path and found up to 7x for the kernel but about 2x once integrated,
with data transfer cost dominating; they concluded that roughly 5x whole-model
acceleration was achievable only if a larger fraction of WRF moved to the GPU
[sources: vanderbauwhede2016wrf]. Gualan-Saavedra and coauthors accelerated the
WRF horizontal diffusion method with CUDA and reported 19x for that method, not
the whole forecast [sources: gualan2015horizontal]. Silva and coauthors explicitly
called their work "another step" toward full GPU WRF, which is exactly how it
should be read: a significant partial step rather than a completed open port
[sources: silva2014fullgpu]. Ridwan et al. combined CUDA and OpenMP around WSM5 and
reported strong kernel speedups and a meaningful whole-WRF improvement against
their chosen baseline, but still through a hybrid partial-physics route
[sources: ridwan2015hybrid].

## Directive and hybrid lines: OpenACC, WRFg, OpenMP offload

The directive path tried to preserve more of WRF's Fortran source. NVIDIA/NCAR
slides from 2016 describe hybrid CUDA and OpenACC WRF 3.6.1 work: 20+ CUDA
physics modules from SSEC, 35+ OpenACC routines across dynamics and physics,
and an explicit warning that the full-model hybrid speedup was modest because
of excessive CPU-GPU data transfer [sources: nvidia2016wrfgpu]. This is the core
lesson for the paper introduction: the blocker was not lack of GPU kernels, but
whole-forecast residency and coupling.

WRFg is the most important non-commercial prior-art boundary. By 2018 NVIDIA
slides describe WRFg as based on ARW 3.7.1, with enough limited physics options
for a full model on GPU, about 4x full-model CPU:GPU performance, freely
available objects/executables, and restricted source availability
[sources: adie2018wrfg]. A 2019 Summit study describes a WRFg/OpenACC port of WRF
3.7.1 to POWER9, with preliminary conclusions of 7.5x on the dycore and 5x on
the full model with physics, tested up to 512 nodes and 3072 GPUs but with
scaling and communication caveats [sources: sever2019wrfsummit]. WRFg prevents a
simple "no full GPU WRF has ever existed" sentence. It does not, on the public
evidence available here, defeat a claim about a fully source-open, modern
WRF-compatible, device-resident implementation, because the source was
restricted and the physics/options were bounded.

The newest open academic route is OpenMP device offload. Wichitrnithed et al.
ported expensive WRF Fast Spectral Bin Microphysics routines to NVIDIA GPUs on
Perlmutter, using profiler and static-analysis workflow. The paper reports 2.08x
overall speedup for the CONUS-12km test at one configuration and explicitly
documents validation through `diffwrf` and NVHPC comparison tools
[sources: wichitrnithed2024openmp]. This is useful because it shows contemporary
HPC groups still treat WRF GPU work as a hard modernization problem, even in
2024, and still target selected expensive routines first.

A 2026 GitHub repository, `FahrenheitResearch/wrf-gpu-port`, is a live
open-source counterexample to any lazy claim that no public WRF GPU work exists.
It is MIT-licensed and patches WRF 4.7.1 with NVHPC OpenACC directives, but its
own README says that physics parameterizations, boundary conditions, I/O, and
advection remain on CPU or disabled, and that single-GPU execution is the target
[sources: fahrenheit2026wrfgpuport]. It should be catalogued as a partial open
directive port, not as a full physics-resident port.

## Commercial route: AceCAST

TempoQuest AceCAST is the commercial prior art that the paper must handle
directly. NVIDIA's technical blog describes AceCAST as a proprietary OpenACC and
CUDA implementation of WRF, scaled on multi-GPU and multi-node systems, with
claims of broad dynamics, physics, and namelist support, plus 5x-9x-class
performance claims depending on benchmark framing [sources: nvidia2023acecast].
TempoQuest's current site describes AceCAST as GPU-accelerated WRF for
operational workflows and advertises a 9x acceleration story
[sources: tempoquest2025acecast]. This is not a peer-reviewed end-to-end WRF GPU
paper, and vendor claims need cautious language, but it absolutely means the
paper should not say "no commercial GPU WRF exists."

The correct contrast is that AceCAST is proprietary and not a source-open,
auditable research artifact. It is also unclear from public material whether it
has the same "zero host/device transfer inside the high-frequency loop" property
or whether it should be compared to a clean-slate JAX reimplementation. The
paper can say AceCAST proves WRF GPU acceleration has serious commercial value,
while the public literature still lacks a peer-reviewed, source-open,
full-physics, device-resident WRF v4 port with open validation artifacts.

## Parallel track: ML weather models are not WRF ports

The ML-weather literature is a separate comparator class. GraphCast,
Pangu-Weather, FourCastNet, GenCast, Aurora, NeuralGCM, Stormer, and AIFS show
that data-driven global forecasting can be extremely fast and increasingly
skillful [sources: lam2023graphcast,bi2022pangu,pathak2022fourcastnet,price2023gencast,bodnar2024aurora,kochkov2023neuralgcm,nguyen2023stormer,lang2024aifs].
They do not port WRF. They bypass the traditional regional physics/dynamics
software problem by learning forecast operators from reanalysis and model data,
or by combining learned components with different dynamical frameworks. That
distinction matters: an ML emulator can be a forecast product, a boundary source,
or a downstream hybrid component, but it is not evidence that the WRF-ARW
split-explicit, terrain-following, physics-coupled codebase has been made
device-resident and source-open on GPUs.

## Pattern that emerges

The historical pattern is stable. First, WRF has many kernels that GPUs can
accelerate dramatically. Second, early papers often compared against one CPU
core, making large numbers useful for kernel insight but weak as whole-forecast
claims. Third, integrated full-model speedups fall sharply when unported
physics, lateral boundaries, I/O, or MPI/control infrastructure force host/device
movement. Fourth, WRFg and AceCAST show that serious full-model or near-full
model GPU WRF efforts did exist, but public source availability and peer-reviewed
evidence are limited. Fifth, modern work continues to target selected expensive
schemes, which reinforces the claim that a full source-open, validated,
device-resident WRF-compatible path is hard enough to be publishable if the
claim is carefully bounded.
