#!/usr/bin/env python3
"""Build a REAL production-state snapshot for the M0 cancellation map (contract §10).

§10 is explicit: "Use real ``20260725_18z`` production-state snapshots, not
synthetic arrays." This module is the only sanctioned state source for the A5
harness. It reads a completed CPU-WRF FAST-v025 arm directory -- whose
``wrfinput_d01`` / ``wrfbdy_d01`` / ``namelist.input`` are byte-identical (or
derived, with a recorded derivation) from the real ALISIOS ``20260725_18z``
production bundle -- and assembles the production ``State`` / ``BaseState`` /
``DycoreMetrics`` triple that the dycore operators actually consume.

Conventions, taken from the production writer (``io/wrfout_writer.py``) rather
than from WRF folklore:

- ``State.theta`` is MOIST ``theta_m`` (``use_theta_m=1``); the writer emits
  ``THM = theta - 300``. The inverse used here is ``theta = THM + 300``.
  Reading ``T`` instead would silently hand every operator a DRY theta and move
  the whole moisture-coupling cancellation analysis onto the wrong branch.
- ``p_total = P + PB``, ``p_perturbation = P``; likewise ``ph``/``PH``/``PHB``
  and ``mu``/``MU``/``MUB``. The perturbation and the total are BOTH carried,
  because the ``total - perturbation`` recovery is precisely the algebra §10
  asks to be inventoried.
- Every metric in ``DycoreMetrics`` maps to a wrfout variable one-for-one; none
  is invented. ``dzdx``/``dzdy`` are the only derived entries and come from real
  ``HGT`` through the production helper ``dynamics.metrics.terrain_slope_metrics``.

The snapshot is loaded in fp64 (the production storage dtype) from the fp32
netCDF file. That is not a precision claim: it is the widest lossless container
for the fp32 values WRF wrote, which is exactly what an fp64-vs-fp32 comparison
needs as its common baseline.

CPU-ONLY. Importing a GPU backend is a GPU touch under contract §13; this
module asserts the CPU platform at import so an accidental GPU initialisation
fails loudly instead of silently violating the coordination rule.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# Contract §13: no GPU import/query/compile/run without dual-manager coordination.
# The platform pin lives in ONE place so it cannot drift between generators; it
# must be imported before jax is imported anywhere in this process.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_guard  # noqa: E402,F401  MUST precede jax/gpuwrf imports

os.environ.setdefault("JAX_ENABLE_X64", "true")
# Radiation and land tables live in the pristine WRF tree, not in this repo.
import wrf_source_authority as _wsa  # noqa: E402

_wsa.apply_default_root(os.environ)
# XLA:CPU's multi-threaded Eigen backend deadlocks on this box under heavy
# external load -- a known failure mode on this project (v0.23.4 sprint,
# 2026-07-17). It presents as a live process at ~8% CPU sleeping in
# futex_do_wait with no progress, which is easy to mistake for "just slow".
# The A5/A6 harnesses are correctness/static instruments, not timing runs, so
# single-threaded XLA:CPU costs nothing they measure and removes the hazard.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("XLA_CPU_MULTI_THREAD_EIGEN", "false")
# Do not contend with another agent for the shared persistent compile cache.
os.environ.setdefault("GPUWRF_JAX_CACHE", "0")
os.environ.setdefault("GPUWRF_JAX_CACHE_LOCK", "0")

THETA_OFFSET_K = 300.0


def assert_cpu_only() -> dict[str, Any]:
    """Fail loudly if this process ever acquired a GPU backend."""

    import jax

    devices = jax.devices()
    platforms = sorted({device.platform for device in devices})
    if platforms != ["cpu"]:
        raise RuntimeError(
            f"contract §13 violation: non-CPU JAX platform(s) {platforms} initialised. "
            "A GPU window requires affirmative coordination from BOTH 0:2 and 0:3."
        )
    return {"platforms": platforms, "device_count": len(devices)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RealSnapshot:
    """A real production-state snapshot plus the provenance that proves it real."""

    state: Any
    base_state: Any
    metrics: Any
    config: dict[str, Any]
    raw: dict[str, np.ndarray]
    provenance: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            int(self.config["nz"]),
            int(self.config["ny"]),
            int(self.config["nx"]),
        )


def _read(dataset, name: str) -> np.ndarray | None:
    """Read one wrfout variable as fp64, with the leading Time axis squeezed."""

    if name not in dataset.variables:
        return None
    value = np.asarray(dataset.variables[name][:])
    if value.ndim >= 1 and value.shape[0] == 1:
        value = value[0]
    return np.asarray(value, dtype=np.float64)


def _require(raw: dict[str, np.ndarray], name: str) -> np.ndarray:
    if raw.get(name) is None:
        raise KeyError(f"required wrfout variable missing: {name}")
    return raw[name]


# Variables the snapshot pulls. Anything absent is reported in the provenance
# rather than silently defaulted, so a critic can see exactly what was real.
WRFOUT_VARIABLES = (
    # prognostic
    "U", "V", "W", "THM", "T", "PH", "PHB", "P", "PB", "MU", "MUB",
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNICE", "QNRAIN",
    "QKE", "TKE_PBL",
    # metrics / vertical coordinate
    "MAPFAC_M", "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_U", "MAPFAC_UX", "MAPFAC_UY",
    "MAPFAC_V", "MAPFAC_VX", "MAPFAC_VY",
    "C1H", "C2H", "C3H", "C4H", "C1F", "C2F", "C3F", "C4F",
    "DN", "DNW", "RDN", "RDNW", "CF1", "CF2", "CF3", "FNM", "FNP",
    "ZNU", "ZNW", "RDX", "RDY", "F", "E", "SINALPHA", "COSALPHA", "P_TOP",
    "HGT", "T00", "P00", "TLP", "TISO",
    # surface / land handles
    "UST", "TSK", "XLAND", "LAKEMASK", "LU_INDEX", "SMOIS", "HFX", "QFX",
    "PSFC", "T2", "Q2", "RAINNC", "SNOWNC", "GRAUPELNC", "RAINC", "HAILNC",
    "CLDFRA",
    # sub-grid orographic statistics consumed by the GWD scheme (gwd_opt=1)
    "VAR", "CON", "OA1", "OA2", "OA3", "OA4", "OL1", "OL2", "OL3", "OL4",
    "VAR_SSO", "XLAT", "XLONG",
)


def load_real_snapshot(
    run_dir: str | Path,
    *,
    wrfout_name: str | None = None,
    domain: str = "d01",
) -> RealSnapshot:
    """Assemble the production State/BaseState/DycoreMetrics from a real wrfout.

    ``run_dir`` is a completed CPU-WRF arm directory (a valid Gen2 run layout).
    ``wrfout_name`` defaults to the LAST history file, i.e. the spun-up
    forecast-hour-1 state rather than the t=0 initial condition. That choice is
    deliberate: hour 0 has no cloud, no turbulence and a balanced mass field, so
    it would understate every cancellation the map exists to find.
    """

    from netCDF4 import Dataset

    import jax.numpy as jnp

    from gpuwrf.contracts.grid import DycoreMetrics
    from gpuwrf.contracts.state import BaseState, State
    from gpuwrf.dynamics.metrics import terrain_slope_metrics
    from gpuwrf.io.gen2_accessor import Gen2Run, parse_namelist

    assert_cpu_only()

    run_path = Path(run_dir).expanduser().resolve()
    histories = sorted(run_path.glob(f"wrfout_{domain}_*"))
    if not histories:
        raise FileNotFoundError(f"no wrfout_{domain}_* history in {run_path}")
    wrfout = run_path / wrfout_name if wrfout_name else histories[-1]
    if not wrfout.is_file():
        raise FileNotFoundError(wrfout)

    namelist = parse_namelist(run_path / "namelist.input")

    raw: dict[str, np.ndarray | None] = {}
    with Dataset(str(wrfout), "r") as dataset:
        dims = {key: len(value) for key, value in dataset.dimensions.items()}
        times = [b"".join(row).decode().strip() for row in dataset.variables["Times"][:]]
        for name in WRFOUT_VARIABLES:
            raw[name] = _read(dataset, name)

    nx = int(dims["west_east"])
    ny = int(dims["south_north"])
    nz = int(dims["bottom_top"])

    dx = float(np.atleast_1d(namelist["domains"]["dx"])[0])
    dy = float(np.atleast_1d(namelist["domains"]["dy"])[0])

    # --- metrics -----------------------------------------------------------
    def m2d(name: str, fallback: str | None = None) -> np.ndarray:
        value = raw.get(name)
        if value is None and fallback is not None:
            value = raw.get(fallback)
        if value is None:
            raise KeyError(f"missing map factor {name}")
        return np.asarray(value, dtype=np.float64)

    def m1d(name: str) -> np.ndarray:
        return np.asarray(_require(raw, name), dtype=np.float64).reshape(-1)

    def scalar(name: str) -> float:
        return float(np.asarray(_require(raw, name)).reshape(-1)[0])

    hgt = np.asarray(_require(raw, "HGT"), dtype=np.float64)
    dzdx, dzdy, dzdx_u, dzdy_v = terrain_slope_metrics(jnp.asarray(hgt), dx, dy)

    metrics = DycoreMetrics(
        msftx=jnp.asarray(m2d("MAPFAC_MX", "MAPFAC_M")),
        msfty=jnp.asarray(m2d("MAPFAC_MY", "MAPFAC_M")),
        msfux=jnp.asarray(m2d("MAPFAC_UX", "MAPFAC_U")),
        msfuy=jnp.asarray(m2d("MAPFAC_UY", "MAPFAC_U")),
        msfvx=jnp.asarray(m2d("MAPFAC_VX", "MAPFAC_V")),
        msfvy=jnp.asarray(m2d("MAPFAC_VY", "MAPFAC_V")),
        c1h=jnp.asarray(m1d("C1H")),
        c2h=jnp.asarray(m1d("C2H")),
        c3h=jnp.asarray(m1d("C3H")),
        c4h=jnp.asarray(m1d("C4H")),
        c1f=jnp.asarray(m1d("C1F")),
        c2f=jnp.asarray(m1d("C2F")),
        c3f=jnp.asarray(m1d("C3F")),
        c4f=jnp.asarray(m1d("C4F")),
        dn=jnp.asarray(m1d("DN")),
        dnw=jnp.asarray(m1d("DNW")),
        rdn=jnp.asarray(m1d("RDN")),
        rdnw=jnp.asarray(m1d("RDNW")),
        cf1=jnp.asarray(scalar("CF1")),
        cf2=jnp.asarray(scalar("CF2")),
        cf3=jnp.asarray(scalar("CF3")),
        fnm=jnp.asarray(m1d("FNM")),
        fnp=jnp.asarray(m1d("FNP")),
        dzdx=dzdx,
        dzdy=dzdy,
        dzdx_u=dzdx_u,
        dzdy_v=dzdy_v,
        f=jnp.asarray(m2d("F")),
        e=jnp.asarray(m2d("E")),
        sina=jnp.asarray(m2d("SINALPHA")),
        cosa=jnp.asarray(m2d("COSALPHA")),
        p_top=jnp.asarray(scalar("P_TOP")),
        provenance=f"real-wrfout:{wrfout.name}",
    )

    # --- base state --------------------------------------------------------
    pb = jnp.asarray(_require(raw, "PB"))
    phb = jnp.asarray(_require(raw, "PHB"))
    mub = jnp.asarray(_require(raw, "MUB"))
    t00 = float(np.asarray(_require(raw, "T00")).reshape(-1)[0])
    theta_base = jnp.full((nz, ny, nx), t00, dtype=jnp.float64)
    base_state = BaseState(
        pb=pb,
        phb=phb,
        mub=mub,
        t0=jnp.asarray(t00, dtype=jnp.float64),
        theta_base=theta_base,
    )

    # --- prognostic state --------------------------------------------------
    # THM is the moist theta_m perturbation the production writer emits; T is
    # the DRY theta perturbation. Using T here would be a silent branch change.
    thm = raw.get("THM")
    theta_source = "THM"
    if thm is None:  # pragma: no cover - every WRF v4 wrfout carries THM
        thm = _require(raw, "T")
        theta_source = "T (THM absent -- DRY theta fallback, recorded)"
    theta = jnp.asarray(np.asarray(thm, dtype=np.float64) + THETA_OFFSET_K)

    p_pert = jnp.asarray(_require(raw, "P"))
    ph_pert = jnp.asarray(_require(raw, "PH"))
    mu_pert = jnp.asarray(_require(raw, "MU"))

    zeros3 = jnp.zeros((nz, ny, nx), dtype=jnp.float64)
    zeros2 = jnp.zeros((ny, nx), dtype=jnp.float64)

    def opt3(name: str) -> Any:
        value = raw.get(name)
        return zeros3 if value is None else jnp.asarray(value)

    def opt2(name: str) -> Any:
        value = raw.get(name)
        return zeros2 if value is None else jnp.asarray(value)

    # Surface-layer handles: derived from the REAL wrfout surface fluxes rather
    # than zeroed, so the surface/PBL coupling operators see production scales.
    # theta_flux = HFX / (rho_sfc * cp); qv_flux = QFX / rho_sfc.
    r_d, c_p = 287.0, 1004.5
    psfc = np.asarray(raw.get("PSFC") if raw.get("PSFC") is not None else np.full((ny, nx), 1.0e5))
    t2 = np.asarray(raw.get("T2") if raw.get("T2") is not None else np.full((ny, nx), 288.0))
    rhosfc_np = psfc / (r_d * np.maximum(t2, 1.0))
    hfx = np.asarray(raw.get("HFX") if raw.get("HFX") is not None else np.zeros((ny, nx)))
    qfx = np.asarray(raw.get("QFX") if raw.get("QFX") is not None else np.zeros((ny, nx)))
    theta_flux_np = hfx / (rhosfc_np * c_p)
    qv_flux_np = qfx / rhosfc_np
    ustar = opt2("UST")

    # Land handles through the production loader when wrfinput is present; it
    # carries the LANDUSE.TBL cold-start rules for ZNT/MAVAIL that the wrfout
    # itself does not store.
    land_meta: dict[str, Any] = {}
    roughness_m = jnp.full((ny, nx), 0.1, dtype=jnp.float64)
    mavail = jnp.ones((ny, nx), dtype=jnp.float64)
    try:
        from gpuwrf.io.land_state import load_prescribed_land_state

        land = load_prescribed_land_state(Gen2Run(run_path), domain=domain)
        roughness_m = jnp.asarray(np.asarray(land.roughness_m, dtype=np.float64))
        mavail = jnp.asarray(np.asarray(land.mavail, dtype=np.float64))
        land_meta = {
            "source": "gpuwrf.io.land_state.load_prescribed_land_state",
            "source_file": land.source.get("source_file"),
            "roughness_note": land.source.get("roughness_note"),
            "mavail_note": land.source.get("mavail_note"),
        }
    except Exception as exc:  # noqa: BLE001 - provenance records the degradation
        land_meta = {
            "source": "FALLBACK",
            "reason": f"{type(exc).__name__}: {exc}",
            "note": "roughness_m/mavail are constants; every operator consuming them is flagged",
        }

    # Boundary leaves. The dycore operators inventoried here are interior
    # operators; the lateral-boundary blend is inventoried separately and reads
    # these arrays. They are shaped to the State contract
    # (time, side=W/E/S/N, z-like, padded side index) and zero-filled, which is
    # recorded honestly: no boundary-blend operator is classified from them.
    side = max(nx, ny)
    bdy3 = jnp.zeros((1, 4, nz, side), dtype=jnp.float64)
    bdy3f = jnp.zeros((1, 4, nz + 1, side), dtype=jnp.float64)
    bdy2 = jnp.zeros((1, 4, side), dtype=jnp.float64)

    state = State(
        u=jnp.asarray(_require(raw, "U")),
        v=jnp.asarray(_require(raw, "V")),
        w=jnp.asarray(_require(raw, "W")),
        theta=theta,
        qv=jnp.asarray(_require(raw, "QVAPOR")),
        p_total=jnp.asarray(_require(raw, "P")) + pb,
        p_perturbation=p_pert,
        ph_total=jnp.asarray(_require(raw, "PH")) + phb,
        ph_perturbation=ph_pert,
        mu_total=jnp.asarray(_require(raw, "MU")) + mub,
        mu_perturbation=mu_pert,
        qc=opt3("QCLOUD"),
        qr=opt3("QRAIN"),
        qi=opt3("QICE"),
        qs=opt3("QSNOW"),
        qg=opt3("QGRAUP"),
        Ni=opt3("QNICE"),
        Nr=opt3("QNRAIN"),
        Ns=zeros3,
        Ng=zeros3,
        qke=opt3("QKE"),
        ustar=ustar,
        theta_flux=jnp.asarray(theta_flux_np),
        qv_flux=jnp.asarray(qv_flux_np),
        tau_u=zeros2,
        tau_v=zeros2,
        rhosfc=jnp.asarray(rhosfc_np),
        fltv=jnp.asarray(theta_flux_np),
        t_skin=opt2("TSK"),
        soil_moisture=(
            jnp.asarray(np.asarray(raw["SMOIS"], dtype=np.float64)[0])
            if raw.get("SMOIS") is not None
            else zeros2
        ),
        xland=opt2("XLAND"),
        lakemask=opt2("LAKEMASK"),
        mavail=mavail,
        roughness_m=roughness_m,
        rain_acc=opt2("RAINNC"),
        snow_acc=opt2("SNOWNC"),
        graupel_acc=opt2("GRAUPELNC"),
        ice_acc=opt2("HAILNC"),
        u_bdy=bdy3,
        v_bdy=bdy3,
        theta_bdy=bdy3,
        qv_bdy=bdy3,
        ph_bdy=bdy3f,
        mu_bdy=bdy2,
        lu_index=(
            jnp.asarray(np.asarray(raw["LU_INDEX"]).astype(np.int32))
            if raw.get("LU_INDEX") is not None
            else None
        ),
        rainc_acc=opt2("RAINC"),
    )

    physics = namelist.get("physics", {})
    dynamics = namelist.get("dynamics", {})
    domains = namelist.get("domains", {})
    bdy = namelist.get("bdy_control", {})

    def first(section: dict[str, Any], key: str, default: Any = None) -> Any:
        value = section.get(key, default)
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value

    config = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": dx,
        "dy": dy,
        "dt": float(first(domains, "time_step", 54)),
        "epssm": float(first(dynamics, "epssm", 0.1)),
        "mp_physics": int(first(physics, "mp_physics", 0)),
        "ra_lw_physics": int(first(physics, "ra_lw_physics", 0)),
        "ra_sw_physics": int(first(physics, "ra_sw_physics", 0)),
        "sf_sfclay_physics": int(first(physics, "sf_sfclay_physics", 0)),
        "sf_surface_physics": int(first(physics, "sf_surface_physics", 0)),
        "bl_pbl_physics": int(first(physics, "bl_pbl_physics", 0)),
        "cu_physics": int(first(physics, "cu_physics", 0)),
        "sf_urban_physics": int(first(physics, "sf_urban_physics", 0)),
        "radt_min": float(first(physics, "radt", 30)),
        "cudt_min": float(first(physics, "cudt", 0)),
        "diff_opt": int(first(dynamics, "diff_opt", 0)),
        "km_opt": int(first(dynamics, "km_opt", 0)),
        "diff_6th_opt": int(first(dynamics, "diff_6th_opt", 0)),
        "diff_6th_factor": float(first(dynamics, "diff_6th_factor", 0.0)),
        "damp_opt": int(first(dynamics, "damp_opt", 0)),
        "w_damping": int(first(dynamics, "w_damping", 0)),
        "zdamp": float(first(dynamics, "zdamp", 5000.0)),
        "dampcoef": float(first(dynamics, "dampcoef", 0.2)),
        "gwd_opt": int(first(dynamics, "gwd_opt", 0)),
        "moist_adv_opt": int(first(dynamics, "moist_adv_opt", 0)),
        "scalar_adv_opt": int(first(dynamics, "scalar_adv_opt", 0)),
        "hybrid_opt": int(first(domains, "hybrid_opt", first(dynamics, "hybrid_opt", 2))),
        "non_hydrostatic": bool(first(dynamics, "non_hydrostatic", True)),
        "specified": bool(first(bdy, "specified", False)),
        "nested": bool(first(bdy, "nested", False)),
        "spec_bdy_width": int(bdy.get("spec_bdy_width", 5)),
        "max_dom": int(domains.get("max_dom", 1)),
    }

    provenance = {
        "run_dir": str(run_path),
        "wrfout": str(wrfout),
        "wrfout_sha256": sha256_file(wrfout),
        "valid_times": times,
        "namelist": str(run_path / "namelist.input"),
        "namelist_sha256": sha256_file(run_path / "namelist.input"),
        "wrfinput_sha256": (
            sha256_file(run_path / f"wrfinput_{domain}")
            if (run_path / f"wrfinput_{domain}").is_file()
            else None
        ),
        "dims": dims,
        "theta_source": theta_source,
        "theta_convention": "State.theta = THM + 300 (moist theta_m, use_theta_m=1)",
        "land_state": land_meta,
        "boundary_leaves": "zero-filled; no lateral-boundary operator is classified from them",
        "surface_flux_handles": (
            "theta_flux = HFX/(rho_sfc*cp), qv_flux = QFX/rho_sfc from real wrfout HFX/QFX/PSFC/T2"
        ),
        "absent_variables": sorted(name for name in WRFOUT_VARIABLES if raw.get(name) is None),
        "state_source_is_real": True,
    }

    numpy_raw = {name: value for name, value in raw.items() if value is not None}
    return RealSnapshot(
        state=state,
        base_state=base_state,
        metrics=metrics,
        config=config,
        raw=numpy_raw,
        provenance=provenance,
    )


STAGE_PAIR_FIELDS = (
    "u", "v", "w", "theta", "qv",
    "p_total", "p_perturbation", "ph_total", "ph_perturbation",
    "mu_total", "mu_perturbation",
    "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "qke",
)


def select_state_pair(
    run_dir: Path,
    domain: str,
    *,
    previous: str | None = None,
    snapshot: str | None = None,
    expect_sha256: list[str] | None = None,
) -> dict[str, Any]:
    """Select the EXACT stage pair, fail-closed. No implicit first/last.

    The drivers used to take ``sorted(glob(...))[0]`` and ``[-1]``. That is
    silently wrong for every case with more than two history files:

    * d02 has 163 files, so first/last spans **6.75 days** (2026-07-26 00:00 to
      2026-08-01 18:00);
    * d03/d06 have 55 files, so first/last spans **18 hours**.

    Combined with a scaler that assumed an hourly gap, that would have inflated
    the stage increment by orders of magnitude and corrupted every work prime
    built from it -- while looking like a clean run.

    Passing both filenames explicitly is now required whenever the directory
    holds more than two histories, and any supplied SHA-256 must match byte for
    byte.
    """
    histories = sorted(run_dir.glob(f"wrfout_{domain}_*"))
    if len(histories) < 2:
        raise SystemExit(
            f"{run_dir} has {len(histories)} history file(s) for {domain}; the stage pair "
            "needs two real states"
        )
    if previous is None or snapshot is None:
        if len(histories) > 2:
            raise SystemExit(
                f"{run_dir} holds {len(histories)} {domain} histories spanning "
                f"{histories[0].name} .. {histories[-1].name}. Implicit first/last selection "
                "is refused: pass --previous and --snapshot explicitly."
            )
        previous_path, snapshot_path = histories[0], histories[1]
    else:
        previous_path = run_dir / previous
        snapshot_path = run_dir / snapshot
    for path in (previous_path, snapshot_path):
        if not path.is_file():
            raise SystemExit(f"state file does not exist: {path}")
    if previous_path == snapshot_path:
        raise SystemExit("previous and snapshot are the same file; the pair would be degenerate")

    digests = [sha256_file(previous_path), sha256_file(snapshot_path)]
    sizes = [previous_path.stat().st_size, snapshot_path.stat().st_size]
    if expect_sha256:
        if len(expect_sha256) != 2:
            raise SystemExit(
                f"--expect-sha256 must be given exactly twice (previous, snapshot); "
                f"got {len(expect_sha256)}"
            )
        for role, want, got, path in zip(
            ("previous", "snapshot"), expect_sha256, digests,
            (previous_path, snapshot_path),
        ):
            if want != got:
                raise SystemExit(
                    f"{role} sha256 mismatch for {path}\n  expected {want}\n  actual   {got}"
                )
    return {
        "previous": previous_path, "snapshot": snapshot_path,
        "previous_sha256": digests[0], "snapshot_sha256": digests[1],
        "previous_bytes": sizes[0], "snapshot_bytes": sizes[1],
        "verified": bool(expect_sha256),
        "candidates_in_dir": len(histories),
    }


def snapshot_interval_seconds(previous: RealSnapshot, current: RealSnapshot) -> float:
    """Real model-time gap between two snapshots, from their own stamps.

    The scaler used to divide by a hardcoded 3600 s. That silently assumed the
    pair was one hour apart, which is true for the d01 FAST-v025 arm and FALSE
    for every other candidate: d03/d06 history is 20 min apart, and an unguarded
    first/last selection on d02 spans 6.75 DAYS. Scaling a 6.75-day difference as
    if it were an hour would inflate the stage increment by ~160x and quietly
    corrupt every work prime built from it.
    """
    from datetime import datetime

    def stamp(snapshot: RealSnapshot) -> datetime:
        raw = snapshot.provenance["valid_times"][0]
        return datetime.strptime(raw, "%Y-%m-%d_%H:%M:%S")

    delta = (stamp(current) - stamp(previous)).total_seconds()
    if delta <= 0:
        raise ValueError(
            f"snapshots are not in forward order: previous={stamp(previous)} "
            f"current={stamp(current)}"
        )
    return delta


def stage_advanced_state(
    previous: RealSnapshot, current: RealSnapshot, dt_s: float,
    interval_s: float | None = None,
):
    """Build the *second* member of a realistic RK stage pair.

    WRF's small-step work primes are ``reference - current`` differences taken
    between two states one sub-advance apart. If both members are the same
    array the primes are IDENTICALLY ZERO, every operator downstream of them
    returns zero, and a cancellation map built on that pair is vacuous -- it
    would report perfect fp32 safety for precisely the subtractions the v0.25
    rewrite is most worried about.

    There is no stored sub-step savepoint for this case, so the increment is
    taken from the two REAL history states an hour apart and rescaled to one
    timestep: ``current + (current - previous) * dt / 3600``. The direction,
    the spatial structure and the relative magnitudes across fields are all
    real production tendency; only the amplitude is rescaled. This is recorded
    verbatim in the map's provenance so a critic can weigh it, and it is a
    LOWER bound on the true acoustic-substep increment (an hourly mean tendency
    is smoother than an instantaneous one), so it understates rather than
    inflates the cancellation it measures.
    """

    import jax.numpy as jnp

    interval = float(interval_s) if interval_s else snapshot_interval_seconds(previous, current)
    scale = float(dt_s) / interval
    updates = {}
    for name in STAGE_PAIR_FIELDS:
        now = getattr(current.state, name, None)
        before = getattr(previous.state, name, None)
        if now is None or before is None:
            continue
        now = jnp.asarray(now)
        before = jnp.asarray(before)
        if now.shape != before.shape:
            continue
        updates[name] = now + (now - before) * scale
    return current.state.replace(**updates), {
        "construction": (
            f"current + (current - previous) * dt/{interval:.0f}s over "
            f"{len(updates)} prognostic fields"
        ),
        "snapshot_interval_seconds": interval,
        "interval_source": "derived from the snapshots' own Times stamps" if not interval_s
                           else "supplied by caller",
        "previous_wrfout": previous.provenance["wrfout"],
        "current_wrfout": current.provenance["wrfout"],
        "dt_s": float(dt_s),
        "fields": sorted(updates),
        "why": (
            "a stage pair whose two members are the same array makes every WRF small-step "
            "work prime identically zero and the map vacuous"
        ),
    }


def default_run_dir() -> Path:
    """The canonical FAST-v025 CPU arm directory used by the M0 harnesses."""

    root = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms")
    for name in ("fastbind_r1", "fast_r1"):
        candidate = root / name
        if (candidate / "namelist.input").is_file():
            return candidate
    raise FileNotFoundError(f"no FAST-v025 CPU arm directory under {root}")


if __name__ == "__main__":  # pragma: no cover - smoke entry point
    import json

    snapshot = load_real_snapshot(default_run_dir())
    print(json.dumps({
        "shape": list(snapshot.shape),
        "config": snapshot.config,
        "provenance": snapshot.provenance,
    }, indent=2, sort_keys=True, default=str))
