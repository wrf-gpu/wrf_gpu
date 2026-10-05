# BP80: MYNN surface-layer snow roughness

WRF selects Andreas_2002 on land where SNOWH >= 0.1 m. The port always used Zilitinkevich, so Swiss snow columns had excessive thermal roughness and weak drag. This patch implements WRF module_sf_mynn.F:1559–1608, preserving separate heat/moisture lengths and the unchanged momentum ZNT. The existing Noah-MP coupler supplies its entry snow depth to the shared legacy/native surface-layer call. No State or restart field is added.

CPU gates on immutable source 13fdd001b [M]:

- 19 focused tests pass in 31.20 s, including a fresh byte-identical pristine MYNN-SL build.
- 48 real Swiss h1 columns: 24 SNOWH >= 0.1 and 24 bare/thin. Both paths pass all 16 inherited bounds and a Q2 bound of 1%, registered before execution. The original implementation failed 13 fields.
- MOL max relative residual 2.60e-5 legacy / 1.14e-5 native; UST 1.57e-5 / 8.75e-7. Q2 retains a 0.613% residual; its existing diagnostic conversion is unchanged.
- Original real PROD d01/d02 (48 columns each, original snow depth asserted zero): all 27 output leaves are byte identical to f807f59e1, for both kinds and eager/JIT calls (216/216 leaf comparisons).
- Native preoptimized HLO f64-compute census 0 before / 0 with active snow. Legacy remains its existing f64 reference mode.
- Three source deletions killed by unchanged assertions: snow selector 2/2, entry-Noah snow producer 1/1, native forwarding 1/2 (legacy passes). Zero errors. Initial CPU-guard/importlib setup refusals were discarded and are not kills.

Independent corroboration: mass-opus MO10b disables only the Andreas branch in pristine WRF. Swiss h1 wind excess +0.168 m/s / MOL +34.6% matches the original port +0.160 / +33.2%; MO10c also sees the omission in native mode. Receipt path and SHA are in review_packet.json.

Scope: paired CPU component fidelity and byte/kind regression. The coupled Swiss GPU/24h drift and RC2 cubin/restart gates belong to the manager/integrate chain. No GPU forecast of this patch is claimed. CPU cores 24,25,28,29, nice19, JAX_PLATFORMS=cpu, private scratch and QUIET guards.
