#!/usr/bin/env python3
"""Offline source/carry audit for corrected-d03 step 9314.

The audit is intentionally non-observing: it imports neither JAX nor gpuwrf,
does not execute model code, and reads the retained carries through an inert
pickle decoder.  It checks the live child-boundary clock and relaxation-band
coverage against pristine WRF v4.7.1, then binds those source discrepancies to
the authenticated step-9313/9314 ordinary and gain-1 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-13-v0234-corrected-ni-dry-sequence-audit"

BASE_COMMIT = "dbad590b93f807966583a5ba0332dd3ae9483a89"
ORDINARY_LAUNCH_COMMIT = "818975804c34edb5989969513ee07f7cc7fc392e"
WRF_REPO = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"

WORK_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a"
)
CHECKPOINT_DIR = WORK_ROOT / "ordinary_bisection_effe1f43/checkpoints"
STEP_9313 = CHECKPOINT_DIR / "last-finite-input-to-first-bad-d03-step-9313.pkl"
STEP_9314_A = CHECKPOINT_DIR / "first-bad-output-d03-step-9314.pkl"
STEP_9314_B = WORK_ROOT / "gain1_ab_b82c3e5e/gain1-output-d03-step-9314.pkl"
STEP_9313_SHA256 = "f82a25c35e6bd738cd248d759c291d825ca0cfc4e08753dc5082741a78b23fc7"
STEP_9314_A_SHA256 = "633d9b09b8d19084ab69b3d260abab7f5bffed1a22fb5fba4f51da583e47298f"
STEP_9314_B_SHA256 = "8e472c10af1726d53444f33021fdd8e0117cec345fb2ea73f9a13b35db7ed25f"

GAIN1_PROOF = ROOT / ".agent/sprints/2026-07-13-v0234-corrected-ni-gain1-ab/gain1-ab-proof.json"
GAIN1_PROOF_SHA256 = "cb8643d7e24b139331a3fad780ccb35564809cbfc1ffb44c37fd981372486ebb"
IDENTITY_PROOF = ROOT / ".agent/sprints/2026-07-13-v0234-corrected-ni-phase-tap/identity-redesign-proof.json"
IDENTITY_PROOF_SHA256 = "49a7bbe94b71e4bacdf4904d14c6f47cf9a3120ed5494fcde99d394bdeae32a1"
V10_PROOF = ROOT / ".agent/sprints/2026-07-13-v0234-corrected-ni-rca-max/v10-spatial-causal-proof.json"
V10_PROOF_SHA256 = "d269513748c85006a7a95b87f778e4c442041c812b2d76dee85ff20309599e68"

WRFINPUT_D03 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf/wrfinput_d03"
)
WRFINPUT_D03_SHA256 = "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a"

LIVE_SOURCES = {
    "boundary_apply": (
        "src/gpuwrf/coupling/boundary_apply.py",
        "9f3515aa598f0d0bf389b45139124802db5efc942b401928743285010ca28230",
    ),
    "boundary_construction": (
        "src/gpuwrf/nesting/boundary_construction.py",
        "013a917dcf0fd42a046b4cc38d18163bf1bddbdb09deb63cc559e722bd149d23",
    ),
    "operational": (
        "src/gpuwrf/runtime/operational_mode.py",
        "f27dcd04f284b1b6056ac12de50ddf9c1ef5c9da3da9ca16ca246707677c5140",
    ),
    "domain_tree": (
        "src/gpuwrf/runtime/domain_tree.py",
        "79f141e06d87e38473222364ec13d5c274a0d42f5d81b00aa44dc42c01678b85",
    ),
    "acoustic": (
        "src/gpuwrf/dynamics/core/acoustic.py",
        "9d4aff41560857140661d44d873c8d958ad254660d0caceac0f4724ed1084b8e",
    ),
    "mu": (
        "src/gpuwrf/dynamics/mu_t_advance.py",
        "8146f69d75ba6e7d3612dff0be7306379f242f6a7e99a523e5acfa9f079658f0",
    ),
    "prep": (
        "src/gpuwrf/dynamics/core/small_step_prep.py",
        "76a82b377d61817f72ca5663cbb12081009096fff8c74f204f285eddbdb5e191",
    ),
}

WRF_SOURCES = {
    "solve_em": ("dyn_em/solve_em.F", "3322d34e0ad070cf4d891aa6dbc93be70ef8240b84eb5f366a149487550aef89"),
    "module_bc_em": ("dyn_em/module_bc_em.F", "6cfb52b849e3dd0b24769709cf502faa75886da454d9b668af725cccdc56d47f"),
    "module_em": ("dyn_em/module_em.F", "11105cbf8255f30ca6a44cd7429a92cedce1fb91db6ce90fd7217002a72fb7fa"),
    "small_step": ("dyn_em/module_small_step_em.F", "cabf1a177d50fb0096db79644af20cfe6d75217dbe63ab406a7e29bb54c17634"),
    "big_step": ("dyn_em/module_big_step_utilities_em.F", "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"),
    "module_bc": ("share/module_bc.F", "61b9235004b2a7799faabaa928276af8a7ef2e4672619c8ad120c857461301ad"),
    "interp": ("share/interp_fcn.F", "680881db48904396f38856ca1dd2578a84e8eb0a9cdd05d49f4c63cd8ce9bdaa"),
    "nest_force": ("share/mediation_force_domain.F", "aaef43f69eb810809eb890688b840ce894aea9ae1ae99f8266e4fa8c3b9f5518"),
    "couple": ("dyn_em/couple_or_uncouple_em.F", "a716939b0bd4d78bb694cb66d278914264cfb34fa8eb04681b4ce2a5c4f26cdf"),
}

DT_CHILD_S = 6.0
DT_PARENT_S = 18.0
PARENT_RATIO = 3
SPEC_ZONE = 1
RELAX_ZONE = 4
SIDES = ("W", "E", "S", "N")
SIDE_INDEX = {side: index for index, side in enumerate(SIDES)}
FINAL_APICES = ((1, 109), (91, 1))
FOUR_INNER_CORNERS = ((1, 1), (1, 109), (91, 1), (91, 109))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def verify_embedded_digest(payload: dict[str, Any]) -> bool:
    expected = payload.get("proof_sha256")
    if not expected:
        return False
    unsigned = dict(payload)
    del unsigned["proof_sha256"]
    return canonical_digest(unsigned) == expected


def git_bytes(repo: Path, rev: str, path: str) -> bytes:
    return subprocess.check_output(("git", "-C", str(repo), "show", f"{rev}:{path}"))


def git_text(repo: Path, rev: str, path: str) -> str:
    return git_bytes(repo, rev, path).decode()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def source_anchor(text: str, needle: str) -> dict[str, Any]:
    lines = text.splitlines()
    matches = [index + 1 for index, line in enumerate(lines) if needle in line]
    if not matches:
        raise AssertionError(f"source anchor absent: {needle!r}")
    line = matches[0]
    return {"line": line, "needle": needle, "text": lines[line - 1].strip()}


class _OpaqueGpuwrfObject:
    """Inert target for gpuwrf classes found in a retained pickle."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "_OpaqueGpuwrfObject":
        del args, kwargs
        return object.__new__(cls)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._opaque_state = state


class _CarryUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("gpuwrf"):
            return _OpaqueGpuwrfObject
        if module.startswith("numpy") or module in {"builtins", "copyreg", "collections"}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked class {module}.{name}")


def load_carry(path: Path) -> _OpaqueGpuwrfObject:
    with path.open("rb") as handle:
        value = _CarryUnpickler(handle).load()
    if not isinstance(value, _OpaqueGpuwrfObject):
        raise AssertionError("checkpoint root is not an inert carry")
    return value


def state_arrays(carry: _OpaqueGpuwrfObject) -> dict[str, np.ndarray]:
    raw = getattr(carry.state, "_opaque_state", None)
    if isinstance(raw, tuple) and raw and isinstance(raw[-1], dict):
        values = raw[-1]
    elif isinstance(getattr(carry.state, "__dict__", None), dict):
        values = carry.state.__dict__
    else:
        raise AssertionError("cannot decode inert State fields")
    return {name: value for name, value in values.items() if isinstance(value, np.ndarray)}


def live_boundary_alpha(completed_step: int, dt_s: float, cadence_s: float) -> float:
    """Exact scalar analogue of the two-record production interpolation."""

    lead_index = float(completed_step) * float(dt_s) / float(cadence_s)
    lower = min(max(np.floor(lead_index), 0.0), 1.0)
    alpha = min(max(lead_index - lower, 0.0), 1.0)
    return float(alpha)


def wrf_child_phase(completed_step: int, ratio: int = PARENT_RATIO) -> float:
    """Endpoint phase of a refreshed two-record package for a child subcycle."""

    return float(((int(completed_step) - 1) % int(ratio)) + 1) / float(ratio)


def local_boundary_lead_seconds(
    completed_step: int, dt_s: float = DT_CHILD_S, cadence_s: float = DT_PARENT_S
) -> float:
    lead = float(completed_step) * float(dt_s)
    return float(np.mod(lead - float(dt_s), float(cadence_s)) + float(dt_s))


def wrf_rk_spec_endpoint_phases(
    completed_step: int,
    dt_s: float = DT_CHILD_S,
    cadence_s: float = DT_PARENT_S,
) -> tuple[float, float, float]:
    """Boundary phases reached by WRF's three RK stages for one child step."""

    end_lead = local_boundary_lead_seconds(completed_step, dt_s, cadence_s)
    start_lead = end_lead - float(dt_s)
    return tuple(
        float((start_lead + dt_rk) / float(cadence_s))
        for dt_rk in (float(dt_s) / 3.0, float(dt_s) / 2.0, float(dt_s))
    )


def naive_localized_live_stage_phases(
    completed_step: int,
    dt_s: float = DT_CHILD_S,
    cadence_s: float = DT_PARENT_S,
) -> tuple[float, float, float]:
    """What the current ``lead_stage=lead+dt_rk`` would do after only a modulo fix."""

    end_lead = local_boundary_lead_seconds(completed_step, dt_s, cadence_s)
    return tuple(
        float(min(max((end_lead + dt_rk) / float(cadence_s), 0.0), 1.0))
        for dt_rk in (float(dt_s) / 3.0, float(dt_s) / 2.0, float(dt_s))
    )


def field_sides_3d(field: np.ndarray, width: int = 5, side_len: int = 112) -> np.ndarray:
    z_len, y_len, x_len = field.shape
    del z_len
    strips = (
        np.moveaxis(field[:, :, :width], 2, 0),
        np.moveaxis(field[:, :, x_len - width :][:, :, ::-1], 2, 0),
        np.moveaxis(field[:, :width, :], 1, 0),
        np.moveaxis(field[:, y_len - width :, :][:, ::-1, :], 1, 0),
    )
    output = []
    for strip in strips:
        padded = np.zeros((width, strip.shape[1], side_len), dtype=field.dtype)
        padded[: strip.shape[0], :, : strip.shape[-1]] = strip
        output.append(padded)
    return np.stack(output)


def field_sides_2d(field: np.ndarray, width: int = 5, side_len: int = 112) -> np.ndarray:
    return field_sides_3d(field[None, :, :], width=width, side_len=side_len)


def full_ring_target(leaf: np.ndarray, field_shape: tuple[int, int, int], alpha: float) -> np.ndarray:
    forcing = leaf[0] + float(alpha) * (leaf[1] - leaf[0])
    z_len, y_len, x_len = field_shape
    output = np.zeros(field_shape, dtype=forcing.dtype)
    for b_dist in range(int(forcing.shape[1])):
        output[:, :, b_dist] = forcing[SIDE_INDEX["W"], b_dist, :z_len, :y_len]
        output[:, :, x_len - 1 - b_dist] = forcing[SIDE_INDEX["E"], b_dist, :z_len, :y_len]
        # WRF Y sides own the diagonals: write them after the X sides.
        output[:, b_dist, :] = forcing[SIDE_INDEX["S"], b_dist, :z_len, :x_len]
        output[:, y_len - 1 - b_dist, :] = forcing[SIDE_INDEX["N"], b_dist, :z_len, :x_len]
    return output


def boundary_package_stats(
    state: dict[str, np.ndarray], field_name: str, leaf_name: str, *, is_2d: bool = False
) -> dict[str, Any]:
    field = np.asarray(state[field_name])
    leaf = np.asarray(state[leaf_name])
    old, new = leaf
    valid_lengths = (field.shape[-2], field.shape[-2], field.shape[-1], field.shape[-1])
    current = field_sides_2d(field) if is_2d else field_sides_3d(field)
    by_side: dict[str, Any] = {}
    increments: list[np.ndarray] = []
    exact_spec_matches = []
    for index, side in enumerate(SIDES):
        valid = int(valid_lengths[index])
        delta = new[index, :, :, :valid] - old[index, :, :, :valid]
        increments.append(delta.ravel())
        spec_residual = current[index, 0, :, :valid] - new[index, 0, :, :valid]
        exact = bool(np.array_equal(current[index, 0, :, :valid], new[index, 0, :, :valid]))
        exact_spec_matches.append(exact)
        by_side[side] = {
            "max_abs_interval_increment": float(np.max(np.abs(delta))),
            "p99_abs_interval_increment": float(np.percentile(np.abs(delta), 99.0)),
            "max_abs_spec_increment": float(np.max(np.abs(delta[0]))),
            "completed_step_9313_spec_equals_record_1_exact": exact,
            "completed_step_9313_spec_residual_to_record_1_max_abs": float(
                np.max(np.abs(spec_residual))
            ),
        }
    joined = np.concatenate(increments)
    maximum = float(np.max(np.abs(joined)))
    if not all(exact_spec_matches):
        raise AssertionError(f"{field_name}: retained spec ring did not consume record 1")
    return {
        "field_shape": list(field.shape),
        "leaf_shape": list(leaf.shape),
        "all_completed_step_9313_spec_sides_equal_record_1_exact": True,
        "max_abs_parent_interval_increment": maximum,
        "max_abs_live_minus_wrf_target_step_9313": maximum * (2.0 / 3.0),
        "max_abs_live_minus_wrf_target_step_9314": maximum * (1.0 / 3.0),
        "by_side": by_side,
    }


def wrf_relax_coverage(
    ny: int, nx: int, spec_zone: int = SPEC_ZONE, relax_zone: int = RELAX_ZONE
) -> np.ndarray:
    """WRF relax_bdytend_core side ownership count for one horizontal field."""

    count = np.zeros((int(ny), int(nx)), dtype=np.int16)
    for b_dist in range(int(spec_zone), int(relax_zone)):
        # Y/SN is full across the diagonal endpoints and therefore owns corners.
        count[b_dist, b_dist : nx - b_dist] += 1
        count[ny - 1 - b_dist, b_dist : nx - b_dist] += 1
        # X/WE excludes both diagonal endpoints already owned by Y/SN.
        count[b_dist + 1 : ny - b_dist - 1, b_dist] += 1
        count[b_dist + 1 : ny - b_dist - 1, nx - 1 - b_dist] += 1
    return count


def live_normal_relax_coverage(
    ny: int,
    nx: int,
    component: str,
    spec_zone: int = SPEC_ZONE,
    relax_zone: int = RELAX_ZONE,
) -> np.ndarray:
    """Coverage of the released moving normal-only work-array relaxation."""

    count = np.zeros((int(ny), int(nx)), dtype=np.int16)
    for b_dist in range(int(spec_zone), int(relax_zone)):
        if component == "u":
            count[b_dist + 1 : ny - b_dist - 1, b_dist] = 1
            count[b_dist + 1 : ny - b_dist - 1, nx - 1 - b_dist] = 1
        elif component == "v":
            count[b_dist, b_dist : nx - b_dist] = 1
            count[ny - 1 - b_dist, b_dist : nx - b_dist] = 1
        else:
            raise ValueError(f"unknown component {component!r}")
    return count


def relax_tendency_numpy(
    field: np.ndarray,
    target: np.ndarray,
    *,
    dt_s: float = DT_CHILD_S,
    spec_zone: int = SPEC_ZONE,
    relax_zone: int = RELAX_ZONE,
) -> np.ndarray:
    """Independent 2-D NumPy transcription of WRF relax_bdytend_core."""

    field = np.asarray(field, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if field.shape != target.shape or field.ndim != 2:
        raise ValueError("field and target must be same-shape 2-D arrays")
    ny, nx = field.shape
    residual = target - field
    tendency = np.zeros_like(field)
    for b_dist in range(int(spec_zone), int(relax_zone)):
        linear = (spec_zone + relax_zone - (b_dist + 1)) / float(relax_zone - 1)
        fcx = 0.1 / float(dt_s) * linear
        gcx = 1.0 / float(dt_s) / 50.0 * linear

        for y in (b_dist, ny - 1 - b_dist):
            normal_edge = y - 1 if y == b_dist else y + 1
            normal_inner = y + 1 if y == b_dist else y - 1
            for x in range(b_dist, nx - b_dist):
                xm = max(x - 1, 0)
                xp = min(x + 1, nx - 1)
                lap = (
                    residual[y, xm]
                    + residual[y, xp]
                    + residual[normal_edge, x]
                    + residual[normal_inner, x]
                    - 4.0 * residual[y, x]
                )
                tendency[y, x] += fcx * residual[y, x] - gcx * lap

        for x in (b_dist, nx - 1 - b_dist):
            normal_edge = x - 1 if x == b_dist else x + 1
            normal_inner = x + 1 if x == b_dist else x - 1
            for y in range(b_dist + 1, ny - b_dist - 1):
                lap = (
                    residual[y - 1, x]
                    + residual[y + 1, x]
                    + residual[y, normal_edge]
                    + residual[y, normal_inner]
                    - 4.0 * residual[y, x]
                )
                tendency[y, x] += fcx * residual[y, x] - gcx * lap
    return tendency


def relax_tendency_numpy_3d(field: np.ndarray, target: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if field.shape != target.shape or field.ndim != 3:
        raise ValueError("field and target must be same-shape 3-D arrays")
    return np.stack(
        [relax_tendency_numpy(level, target_level) for level, target_level in zip(field, target)]
    )


def mass_coupled_apex_oracle(
    state: dict[str, np.ndarray], metrics: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Velocity-equivalent WRF relax response on the four apex flux faces.

    This uses the existing leaf contract's decoupled parent target and couples it
    with the child step-start mass.  It is exact for the child-side
    ``specified_relax_dry_tendencies`` algebra, but not a substitute for WRF's
    coupling-before-parent-interpolation construction; that distinction is
    recorded as a prelaunch requirement.
    """

    u = np.asarray(state["u"], dtype=np.float64)
    v = np.asarray(state["v"], dtype=np.float64)
    mu = np.asarray(state["mu_total"], dtype=np.float64)
    c1h = np.asarray(metrics["c1h"], dtype=np.float64)
    c2h = np.asarray(metrics["c2h"], dtype=np.float64)
    msfuy = np.asarray(metrics["msfuy"], dtype=np.float64)
    msfvx = np.asarray(metrics["msfvx"], dtype=np.float64)

    muu = 0.5 * (
        np.concatenate([mu[:, :1], mu], axis=1)
        + np.concatenate([mu, mu[:, -1:]], axis=1)
    )
    muv = 0.5 * (
        np.concatenate([mu[:1, :], mu], axis=0)
        + np.concatenate([mu, mu[-1:, :]], axis=0)
    )
    mass_u = c1h[:, None, None] * muu[None, :, :] + c2h[:, None, None]
    mass_v = c1h[:, None, None] * muv[None, :, :] + c2h[:, None, None]
    u_target = full_ring_target(np.asarray(state["u_bdy"]), u.shape, 2.0 / 3.0)
    v_target = full_ring_target(np.asarray(state["v_bdy"]), v.shape, 2.0 / 3.0)
    ru = mass_u * u / msfuy[None, :, :]
    rv = mass_v * v / msfvx[None, :, :]
    ru_target = mass_u * u_target / msfuy[None, :, :]
    rv_target = mass_v * v_target / msfvx[None, :, :]
    ru_tend = relax_tendency_numpy_3d(ru, ru_target)
    rv_tend = relax_tendency_numpy_3d(rv, rv_target)

    wrf_u = wrf_relax_coverage(u.shape[1], u.shape[2])
    wrf_v = wrf_relax_coverage(v.shape[1], v.shape[2])
    live_u = live_normal_relax_coverage(u.shape[1], u.shape[2], "u")
    live_v = live_normal_relax_coverage(v.shape[1], v.shape[2], "v")
    rows: dict[str, Any] = {}
    for y, x in FINAL_APICES:
        faces = (
            ("u_west", ru_tend, mass_u / msfuy[None, :, :], wrf_u, live_u, y, x),
            ("u_east", ru_tend, mass_u / msfuy[None, :, :], wrf_u, live_u, y, x + 1),
            ("v_south", rv_tend, mass_v / msfvx[None, :, :], wrf_v, live_v, y, x),
            ("v_north", rv_tend, mass_v / msfvx[None, :, :], wrf_v, live_v, y + 1, x),
        )
        face_rows = {}
        for name, tendency, coupled_mass, wrf_cov, live_cov, yy, xx in faces:
            equivalent = DT_CHILD_S * tendency[:, yy, xx] / coupled_mass[:, yy, xx]
            index = int(np.argmax(np.abs(equivalent)))
            face_rows[name] = {
                "wrf_relaxed": bool(wrf_cov[yy, xx]),
                "released_live_normal_relaxed": bool(live_cov[yy, xx]),
                "max_abs_six_second_velocity_equivalent_m_s": float(
                    np.max(np.abs(equivalent))
                ),
                "signed_velocity_equivalent_at_max_m_s": float(equivalent[index]),
                "vertical_index_at_max": index,
            }
        missing_rows = [
            row
            for row in face_rows.values()
            if row["wrf_relaxed"] and not row["released_live_normal_relaxed"]
        ]
        if len(missing_rows) != 3 or any(
            row["max_abs_six_second_velocity_equivalent_m_s"] <= 1.0 for row in missing_rows
        ):
            raise AssertionError("mass-coupled apex response no longer discriminates the live gap")
        rows[f"y{y}_x{x}"] = {"faces": face_rows}

    return {
        "target_phase": 2.0 / 3.0,
        "apices": rows,
        "result": (
            "Every missing final-apex flux face has a source-backed six-second frozen-relax "
            "velocity-equivalent response above 1 m/s (up to about 7.4 m/s); the released path applies zero."
        ),
        "limitation": (
            "Targets are coupled after interpolation with child mass because retained leaves are decoupled. "
            "Pristine WRF couples parent and child prognostics before nest interpolation; an exact candidate "
            "must close that construction algebra without adding carry leaves."
        ),
    }


def algebraic_oracles() -> dict[str, Any]:
    shape = (17, 19)
    coverage = wrf_relax_coverage(*shape)
    if int(coverage.max()) != 1:
        raise AssertionError("WRF corner ownership is not single-valued")

    field = np.zeros(shape, dtype=np.float64)
    zero = relax_tendency_numpy(field, field)
    if not np.array_equal(zero, np.zeros_like(zero)):
        raise AssertionError("relax operator violates zero-residual null space")
    uniform = relax_tendency_numpy(field, np.ones(shape, dtype=np.float64))
    responses = {}
    for b_dist, expected in ((1, 0.1), (2, 0.1 * 2.0 / 3.0), (3, 0.1 / 3.0)):
        actual = float(DT_CHILD_S * uniform[b_dist, shape[1] // 2])
        if not np.isclose(actual, expected, rtol=0.0, atol=1.0e-15):
            raise AssertionError("uniform-residual Davies response changed")
        responses[str(b_dist)] = {"actual": actual, "expected": expected}

    old = np.asarray([-4.0, 1.5, 9.0], dtype=np.float64)
    new = np.asarray([8.0, -1.5, 3.0], dtype=np.float64)
    trajectory = [old + wrf_child_phase(step) * (new - old) for step in (1, 2, 3)]
    if not np.array_equal(trajectory[-1], new):
        raise AssertionError("three-child-step boundary endpoint did not close")
    slopes = [trajectory[0] - old, trajectory[1] - trajectory[0], trajectory[2] - trajectory[1]]
    if not all(np.allclose(slope, (new - old) / 3.0, rtol=0.0, atol=1.0e-15) for slope in slopes):
        raise AssertionError("boundary endpoint increments are not conserved")

    rng = np.random.default_rng(234)
    u_flux = rng.standard_normal((shape[0], shape[1] + 1))
    v_flux = rng.standard_normal((shape[0] + 1, shape[1]))
    divergence = (u_flux[:, 1:] - u_flux[:, :-1]) + (v_flux[1:, :] - v_flux[:-1, :])
    domain_sum = float(np.sum(divergence))
    boundary_flux = float(
        np.sum(u_flux[:, -1] - u_flux[:, 0]) + np.sum(v_flux[-1, :] - v_flux[0, :])
    )
    if not np.isclose(domain_sum, boundary_flux, rtol=0.0, atol=2.0e-14):
        raise AssertionError("discrete continuity telescope failed")

    return {
        "zero_residual_null_space_exact": True,
        "uniform_residual_integrated_response_by_b_dist": responses,
        "corner_ownership": {
            "max_owner_count": int(coverage.max()),
            "min_active_owner_count": int(coverage[coverage > 0].min()),
            "duplicate_cells": int(np.count_nonzero(coverage > 1)),
        },
        "parent_interval_endpoint_conservation": {
            "old": old.tolist(),
            "trajectory": [row.tolist() for row in trajectory],
            "new": new.tolist(),
            "equal_thirds": True,
            "final_exact": True,
        },
        "continuity_flux_telescope": {
            "domain_divergence_sum": domain_sum,
            "net_boundary_flux": boundary_flux,
            "absolute_closure": abs(domain_sum - boundary_flux),
        },
    }


def projected_nonfinite_mask(value: np.ndarray) -> np.ndarray:
    mask = ~np.isfinite(np.asarray(value))
    while mask.ndim > 2:
        mask = np.any(mask, axis=0)
    return mask


def directed_cone_mask(ny: int = 93, nx: int = 111) -> np.ndarray:
    mask = np.zeros((ny, nx), dtype=bool)
    for y in range(ny):
        for x in range(nx):
            southeast = y >= 1 and x <= nx - 2 and (y - 1) + ((nx - 2) - x) <= 9
            northwest = y <= ny - 2 and x >= 1 and ((ny - 2) - y) + (x - 1) <= 9
            mask[y, x] = southeast or northwest
    return mask


def corner_maxima(value: np.ndarray, radius: int = 12) -> dict[str, float]:
    array = np.abs(np.asarray(value))
    if array.ndim == 3:
        array = np.max(array, axis=0)
    return {
        "southwest": float(np.max(array[:radius, :radius])),
        "southeast": float(np.max(array[:radius, -radius:])),
        "northwest": float(np.max(array[-radius:, :radius])),
        "northeast": float(np.max(array[-radius:, -radius:])),
    }


def corner_coverage_and_residuals(state: dict[str, np.ndarray]) -> dict[str, Any]:
    u = np.asarray(state["u"])
    v = np.asarray(state["v"])
    u_target = full_ring_target(np.asarray(state["u_bdy"]), u.shape, 2.0 / 3.0)
    v_target = full_ring_target(np.asarray(state["v_bdy"]), v.shape, 2.0 / 3.0)
    wrf_u = wrf_relax_coverage(u.shape[1], u.shape[2])
    wrf_v = wrf_relax_coverage(v.shape[1], v.shape[2])
    live_u = live_normal_relax_coverage(u.shape[1], u.shape[2], "u")
    live_v = live_normal_relax_coverage(v.shape[1], v.shape[2], "v")
    if int(wrf_u.max()) != 1 or int(wrf_v.max()) != 1:
        raise AssertionError("WRF staggered corner ownership changed")

    apices: dict[str, Any] = {}
    for y, x in FINAL_APICES:
        faces = (
            ("u_west", u, u_target, wrf_u, live_u, y, x),
            ("u_east", u, u_target, wrf_u, live_u, y, x + 1),
            ("v_south", v, v_target, wrf_v, live_v, y, x),
            ("v_north", v, v_target, wrf_v, live_v, y + 1, x),
        )
        rows: dict[str, Any] = {}
        for name, field, target, wrf_cov, live_cov, yy, xx in faces:
            residual = field[:, yy, xx] - target[:, yy, xx]
            index = int(np.argmax(np.abs(residual)))
            rows[name] = {
                "wrf_relaxed": bool(wrf_cov[yy, xx]),
                "released_live_normal_relaxed": bool(live_cov[yy, xx]),
                "max_abs_state_minus_wrf_phase_target": float(np.max(np.abs(residual))),
                "signed_residual_at_max": float(residual[index]),
                "vertical_index_at_max": index,
            }
        missing = sum(row["wrf_relaxed"] and not row["released_live_normal_relaxed"] for row in rows.values())
        if missing != 3:
            raise AssertionError("final-cone apex no longer has three missing flux-face relax paths")
        apices[f"y{y}_x{x}"] = {"missing_live_faces_of_four": missing, "faces": rows}

    return {
        "u": {
            "wrf_relax_cells": int(np.count_nonzero(wrf_u)),
            "released_live_relax_cells": int(np.count_nonzero(live_u)),
            "missing_tangential_or_corner_cells": int(np.count_nonzero((wrf_u > 0) & (live_u == 0))),
            "duplicate_wrf_owner_cells": int(np.count_nonzero(wrf_u > 1)),
        },
        "v": {
            "wrf_relax_cells": int(np.count_nonzero(wrf_v)),
            "released_live_relax_cells": int(np.count_nonzero(live_v)),
            "missing_tangential_or_corner_cells": int(np.count_nonzero((wrf_v > 0) & (live_v == 0))),
            "duplicate_wrf_owner_cells": int(np.count_nonzero(wrf_v > 1)),
        },
        "final_nonfinite_cone_apices": apices,
        "interpretation": (
            "WRF assigns every active relax cell to exactly one side, with S/N owning diagonals. "
            "The released normal-only path omits 648 U and 528 V relax cells; at each retained "
            "final-cone apex, three of the four continuity flux faces lack the WRF relaxation path."
        ),
    }


def authenticate_json(path: Path, expected_file_sha: str, *, embedded: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    actual = sha256_file(path)
    if actual != expected_file_sha:
        raise AssertionError(f"proof file hash changed: {path}: {actual}")
    payload = json.loads(path.read_text())
    if embedded and not verify_embedded_digest(payload):
        raise AssertionError(f"embedded proof digest invalid: {path}")
    return payload, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": actual,
        "payload_sha256": payload.get("proof_sha256"),
        "embedded_digest_valid": bool(verify_embedded_digest(payload)) if embedded else None,
    }


def source_map(live: dict[str, str], wrf: dict[str, str]) -> dict[str, Any]:
    return {
        "package_and_clock": {
            "live_two_record_contract": source_anchor(
                live["boundary_construction"], "return jnp.stack([old, new], axis=0)"
            ),
            "live_global_step_clock": source_anchor(
                live["operational"],
                "lead_seconds = step_index.astype(jnp.float64) * float(namelist.dt_s)",
            ),
            "live_global_clock_explicit": source_anchor(
                live["domain_tree"],
                "del reset_clock  # step clocks are per-domain global clocks, not subcycle-local",
            ),
            "live_two_record_clip": source_anchor(
                live["boundary_apply"],
                "lower = jnp.clip(jnp.floor(lead_index).astype(jnp.int32), 0, max_index)",
            ),
            "wrf_parent_interval_reciprocal": source_anchor(wrf["interp"], "rdt = 1.D0/cdt"),
            "wrf_package_refresh_resets_phase": source_anchor(
                wrf["nest_force"], "nested_grid%dtbc = 0."
            ),
            "wrf_completed_step_advances_phase": source_anchor(
                wrf["solve_em"], "grid%dtbc = grid%dtbc + grid%dt"
            ),
            "wrf_boundary_slope": source_anchor(
                wrf["interp"], "bdy_txs( nj,k,ni ) = rdt*(psca"
            ),
        },
        "frozen_rk1_bundle": {
            "wrf_rk1_gate": source_anchor(wrf["solve_em"], "rk_step == 1"),
            "wrf_relax_call": source_anchor(wrf["solve_em"], "CALL relax_bdy_dry"),
            "wrf_ru_full_four_side_relax": source_anchor(
                wrf["module_bc_em"], "CALL relax_bdytend ( ru, ru_tendf"
            ),
            "wrf_nested_w_relax": source_anchor(
                wrf["module_bc_em"], "CALL relax_bdytend_tile ( rfield, rw_tendf"
            ),
            "live_nested_excluded_from_frozen_bundle": source_anchor(
                live["operational"],
                "return bool(_spec) and bool(namelist.boundary_config.force_geopotential)",
            ),
            "live_nested_ph_recomputed_in_stage": source_anchor(
                live["operational"], "ph_relax = nested_ph_relax_tendency("
            ),
            "live_normal_moving_residual": source_anchor(
                live["boundary_apply"], "u = u_work + wu * (u_target - u_work)"
            ),
            "live_released_gain": source_anchor(
                live["boundary_apply"],
                'NORMAL_BDY_RELAX_STRENGTH = float(os.environ.get("GPUWRF_NORMAL_BDY_RELAX_STRENGTH", "20.0"))',
            ),
        },
        "mass_coupled_boundary_construction": {
            "wrf_parent_coupled_before_force": source_anchor(
                wrf["nest_force"], "! couple parent domain"
            ),
            "wrf_child_coupled_before_force": source_anchor(
                wrf["nest_force"], "! couple nested domain"
            ),
            "wrf_theta_coupling": source_anchor(
                wrf["couple"], "grid%t_2(i,k,j)  =  grid%t_2(i,k,j)*muth_2(i,k,j)"
            ),
            "wrf_u_coupling": source_anchor(
                wrf["couple"], "grid%u_2(i,k,j)  =  grid%u_2(i,k,j)*muut_2(i,k,j)"
            ),
            "live_declared_decoupled_approximation": source_anchor(
                live["boundary_construction"], "we interpolate the"
            ),
            "verdict": (
                "The retained leaf shapes can stay unchanged, but an exact WRF implementation must define dry leaf "
                "contents/consumers in coupled space or prove an equivalent product interpolation. Coupling a "
                "decoupled interpolated target afterward is only the current O(mu-gradient) approximation."
            ),
        },
        "corner_and_spec_semantics": {
            "wrf_y_side_full_diagonal": source_anchor(
                wrf["module_bc"], "DO i = max(its,b_limit+ibs), min(itf,ibe-b_limit)"
            ),
            "wrf_x_side_trim": source_anchor(
                wrf["module_bc"], "DO j = max(jts,b_dist+jbs+1), min(jtf,jbe-b_dist-1)"
            ),
            "live_tangential_spec_condition": source_anchor(
                live["acoustic"],
                "if state.u_spec_tan_target is not None and state.v_spec_tan_target is not None:",
            ),
            "live_tangential_targets_only_under_spec_cadence": source_anchor(
                live["operational"], "u_spec_tan_target = tangential_bdy_work_target_u("
            ),
            "wrf_nested_w_spec_update": source_anchor(
                wrf["solve_em"], "CALL spec_bdyupdate ( grid%w_2, rw_tend, dts_rk"
            ),
        },
        "dry_sequence": {
            "wrf_advance_uv": source_anchor(wrf["solve_em"], "CALL advance_uv"),
            "wrf_advance_mu_t": source_anchor(wrf["solve_em"], "CALL advance_mu_t"),
            "wrf_advance_w": source_anchor(wrf["solve_em"], "CALL advance_w"),
            "wrf_calc_p_rho": source_anchor(wrf["solve_em"], "CALL calc_p_rho"),
            "live_sequence_doc": source_anchor(
                live["acoustic"],
                "WRF cadence (``solve_em.F:3065-4206``): ``advance_uv`` -> ``advance_mu_t``",
            ),
            "live_terrain_w_deviation": source_anchor(
                live["acoustic"], "u=state_for_w.u_1,"
            ),
        },
        "operator_contracts": {
            "advance_uv_and_large_step_momentum": {
                "staggering": {
                    "u": "(k,y,x_face) = (44,93,112)",
                    "v": "(k,y_face,x) = (44,94,111)",
                },
                "wrf_nested_spec_bounds": (
                    "u/v tendency and PGF loops exclude the outer spec_zone on both axes; "
                    "u extends to ide-spec_zone and v to jde-spec_zone."
                ),
                "wrf_bound_anchor": source_anchor(
                    wrf["small_step"], "i_start = max( its,ids+spec_zone )"
                ),
                "wrf_large_tendency_consumption": source_anchor(
                    wrf["small_step"], "u(i,k,j) = u(i,k,j) + dts*ru_tend(i,k,j)"
                ),
                "live_all_face_tendency_before_boundary_adapter": source_anchor(
                    live["acoustic"], "u = state.u + dts * u_tend_value"
                ),
                "live_stage_recompute": source_anchor(
                    live["operational"],
                    "tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)",
                ),
                "live_frozen_physics_merge": source_anchor(
                    live["operational"], "merged = rk_addtend_dry("
                ),
                "wrf_frozen_ru_fold": source_anchor(
                    wrf["module_em"],
                    "ru_tend(i,k,j) = ru_tend(i,k,j) + ru_tendf(i,k,j)/msfuy(i,j)",
                ),
                "verdict": (
                    "Interior PGF/tendency ordering is source-aligned. At the lateral band, live computes all "
                    "faces and repairs only moving normal faces; WRF excludes the spec ring and supplies full "
                    "four-side spec/relax tendencies. Treat PGF as amplifier until a separate equation mismatch is shown."
                ),
            },
            "advance_mu_t_and_continuity": {
                "staggering": (
                    "mu/mudf/muave/muts (y,x); theta/dvdxi (k,y,x); ww (k_face,y,x); "
                    "continuity consumes west/east U faces and south/north V faces."
                ),
                "wrf_bounds": "nonperiodic nested mass cells y=1..ny-2 and x=1..nx-2 (zero based)",
                "wrf_bound_anchor": source_anchor(
                    wrf["small_step"], "i_start = max(its,ids+1)"
                ),
                "live_bound_anchor": source_anchor(live["mu"], "y0, y1 = (1, ny - 1)"),
                "wrf_vertical_integral": source_anchor(
                    wrf["small_step"], "DMDT(i)    = DMDT(i) + dnw(k)*dvdxi(i,k)"
                ),
                "live_flux_divergence": source_anchor(
                    live["mu"],
                    "dvdxi = inputs.msftx[ys, xs] * inputs.msfty[ys, xs] * (",
                ),
                "wrf_mass_update": source_anchor(
                    wrf["small_step"], "MU(i,j) = MU(i,j)+dts*(DMDT(i)+MU_TEND(i,j))"
                ),
                "carried_outputs": "mudf=dmdt+mu_tend; muts=mut+mu_work; muave is epssm time average; ww is the vertical closure",
                "verdict": (
                    "Bounds and conservative divergence algebra match. The retained apices expose missing WRF "
                    "boundary control on three of the four face inputs, so continuity is the first many-to-one "
                    "amplification/propagation operator, not independently identified as the source discrepancy."
                ),
            },
            "advance_w_ph_pressure_feedback": {
                "staggering": "w/ph on k faces (45,93,111); pressure/theta on k mass levels (44,93,111)",
                "order": (
                    "advance_w -> sumflux -> spec_bdyupdate_ph -> nested spec_bdyupdate(w) -> calc_p_rho; "
                    "the diagnosed pressure feeds the next acoustic substep and refreshed grid pressure feeds the next RK stage."
                ),
                "wrf_ph_boundary_anchor": source_anchor(
                    wrf["solve_em"], "CALL spec_bdyupdate_ph( ph_save, grid%ph_2, ph_tend"
                ),
                "wrf_nested_w_boundary_anchor": source_anchor(
                    wrf["solve_em"], "CALL spec_bdyupdate ( grid%w_2, rw_tend, dts_rk"
                ),
                "live_w_solve_anchor": source_anchor(
                    live["acoustic"], "w_solved, ph_next, t_2ave_next = advance_w_wrf("
                ),
                "live_pressure_anchor": source_anchor(
                    live["acoustic"], "p_rho = calc_p_rho_step("
                ),
                "verdict": (
                    "The live numerical order is aligned, but nested ph relax is rebuilt per RK stage, ph is "
                    "hard-pinned, nested w relax/spec is incomplete, and none is driven by an exact package slope. "
                    "These are rank-1 boundary-bundle members and a strong coupled amplification path."
                ),
            },
            "halo_and_end_step": {
                "live_stage_halo": source_anchor(
                    live["operational"], "haloed = apply_halo(stage_carry.state, halo_spec(namelist.grid))"
                ),
                "live_end_step_dry_gate": source_anchor(
                    live["operational"], "dry_spec_only=_specified_bdy_cadence_active(namelist)"
                ),
                "verdict": (
                    "Halo refresh occurs at every RK stage. Because the nested child is excluded from specified cadence, "
                    "its end-step pass still performs the legacy dry value/relax rewrite. That rewrite polluted the "
                    "authenticated incoming carry over prior steps but occurs after the within-step dry runaway and is "
                    "not the step9314 first-bad operator."
                ),
            },
        },
    }


def build_proof() -> dict[str, Any]:
    if git_value(ROOT, "rev-parse", BASE_COMMIT) != BASE_COMMIT:
        raise AssertionError("base commit unavailable")
    if git_value(ROOT, "rev-parse", ORDINARY_LAUNCH_COMMIT) != ORDINARY_LAUNCH_COMMIT:
        raise AssertionError("ordinary launch commit unavailable")
    if git_value(WRF_REPO, "rev-parse", WRF_COMMIT) != WRF_COMMIT:
        raise AssertionError("pristine WRF commit unavailable")

    live_text: dict[str, str] = {}
    live_auth: dict[str, Any] = {}
    for name, (path, expected) in LIVE_SOURCES.items():
        raw = git_bytes(ROOT, BASE_COMMIT, path)
        actual = sha256_bytes(raw)
        if actual != expected:
            raise AssertionError(f"live source object changed: {path}: {actual}")
        launch_raw = git_bytes(ROOT, ORDINARY_LAUNCH_COMMIT, path)
        if launch_raw != raw:
            raise AssertionError(f"relevant source changed after ordinary launch: {path}")
        live_text[name] = raw.decode()
        live_auth[name] = {
            "path": path,
            "sha256": actual,
            "bytes": len(raw),
            "byte_identical_to_ordinary_launch": True,
        }

    wrf_text: dict[str, str] = {}
    wrf_auth: dict[str, Any] = {}
    for name, (path, expected) in WRF_SOURCES.items():
        raw = git_bytes(WRF_REPO, WRF_COMMIT, path)
        actual = sha256_bytes(raw)
        if actual != expected:
            raise AssertionError(f"WRF source object changed: {path}: {actual}")
        wrf_text[name] = raw.decode()
        wrf_auth[name] = {"path": path, "sha256": actual, "bytes": len(raw)}

    gain, gain_auth = authenticate_json(GAIN1_PROOF, GAIN1_PROOF_SHA256, embedded=True)
    identity, identity_auth = authenticate_json(
        IDENTITY_PROOF, IDENTITY_PROOF_SHA256, embedded=True
    )
    v10, v10_auth = authenticate_json(V10_PROOF, V10_PROOF_SHA256, embedded=False)
    if gain["verdict"] != "GAIN1_CANDIDATE_FALSIFIED":
        raise AssertionError("gain-1 verdict changed")
    if identity["verdict"] != "NO_FIX_LOCALIZED":
        raise AssertionError("identity redesign verdict changed")

    carries = ((STEP_9313, STEP_9313_SHA256), (STEP_9314_A, STEP_9314_A_SHA256), (STEP_9314_B, STEP_9314_B_SHA256))
    carry_auth = {}
    for path, expected in carries:
        actual = sha256_file(path)
        if actual != expected:
            raise AssertionError(f"retained carry changed: {path}: {actual}")
        carry_auth[path.name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}

    input_carry = load_carry(STEP_9313)
    ordinary = load_carry(STEP_9314_A)
    gain1 = load_carry(STEP_9314_B)
    state = state_arrays(input_carry)
    required_finite = ("u", "v", "w", "theta", "ph_perturbation", "mu_perturbation")
    if any(not np.isfinite(state[name]).all() for name in required_finite):
        raise AssertionError("step-9313 dry input is no longer finite")

    fields = (
        ("u", "u_bdy", False),
        ("v", "v_bdy", False),
        ("w", "w_bdy", False),
        ("theta", "theta_bdy", False),
        ("ph_perturbation", "ph_bdy", False),
        ("mu_perturbation", "mu_bdy", True),
    )
    package = {
        field: boundary_package_stats(state, field, leaf, is_2d=is_2d)
        for field, leaf, is_2d in fields
    }

    clock_rows = []
    for step in range(9313, 9319):
        live_alpha = live_boundary_alpha(step, DT_CHILD_S, DT_PARENT_S)
        wrf_alpha = wrf_child_phase(step)
        local_lead = local_boundary_lead_seconds(step)
        if live_alpha != 1.0 or not np.isclose(local_lead / DT_PARENT_S, wrf_alpha):
            raise AssertionError("boundary clock oracle changed")
        clock_rows.append(
            {
                "completed_step": step,
                "global_lead_seconds": step * DT_CHILD_S,
                "production_two_record_alpha": live_alpha,
                "wrf_package_alpha": wrf_alpha,
                "package_local_lead_seconds": local_lead,
            }
        )

    coverage = corner_coverage_and_residuals(state)
    oracle = algebraic_oracles()

    expected_a = directed_cone_mask()
    masks = {}
    for label, carry, expected_cells in (("gain20_A", ordinary, 110), ("gain1_B", gain1, 29)):
        reference_mask = None
        rows = {}
        for field in ("t_2ave", "ww", "mudf", "muave", "muts"):
            mask = projected_nonfinite_mask(np.asarray(getattr(carry, field)))
            if reference_mask is None:
                reference_mask = mask
            elif not np.array_equal(mask, reference_mask):
                raise AssertionError(f"{label} scratch masks differ")
            rows[field] = int(np.count_nonzero(~np.isfinite(np.asarray(getattr(carry, field)))))
        assert reference_mask is not None
        if int(reference_mask.sum()) != expected_cells:
            raise AssertionError(f"{label} scratch footprint changed")
        if label == "gain20_A" and not np.array_equal(reference_mask, expected_a):
            raise AssertionError("ordinary exact directed cones changed")
        if label == "gain1_B" and np.any(reference_mask & ~expected_a):
            raise AssertionError("gain1 footprint is no longer a subset of ordinary cones")
        masks[label] = {
            "projected_cells": int(reference_mask.sum()),
            "contains_both_final_apices": all(bool(reference_mask[y, x]) for y, x in FINAL_APICES),
            "nonfinite_counts": rows,
        }

    stage_saves = {}
    for field in ("u_save", "v_save", "w_save", "t_save", "ph_save", "mu_save", "ww_save"):
        a = np.asarray(getattr(ordinary, field))
        b = np.asarray(getattr(gain1, field))
        a_corners = corner_maxima(a)
        b_corners = corner_maxima(b)
        if not all(value > 1.0e6 for value in a_corners.values()):
            raise AssertionError(f"ordinary four-corner RK3-entry signature changed: {field}")
        if not all(value > 1.0e6 for value in b_corners.values()):
            raise AssertionError(f"gain1 four-corner RK3-entry signature changed: {field}")
        stage_saves[field] = {
            "gain20_A_corner_max_abs": a_corners,
            "gain1_B_corner_max_abs": b_corners,
            "gain20_A_global_max_abs": float(np.max(np.abs(a))),
            "gain1_B_global_max_abs": float(np.max(np.abs(b))),
        }

    wrfinput_actual = sha256_file(WRFINPUT_D03)
    if wrfinput_actual != WRFINPUT_D03_SHA256:
        raise AssertionError("wrfinput_d03 changed")
    from netCDF4 import Dataset  # local import: no model/JAX dependency

    geometry = {}
    metric_arrays: dict[str, np.ndarray] = {}
    with Dataset(WRFINPUT_D03) as dataset:
        metric_arrays = {
            "c1h": np.asarray(dataset.variables["C1H"][0]),
            "c2h": np.asarray(dataset.variables["C2H"][0]),
            "c1f": np.asarray(dataset.variables["C1F"][0]),
            "c2f": np.asarray(dataset.variables["C2F"][0]),
            "msfuy": np.asarray(dataset.variables["MAPFAC_UY"][0]),
            "msfvx": np.asarray(dataset.variables["MAPFAC_VX"][0]),
            "msfty": np.asarray(dataset.variables["MAPFAC_MY"][0]),
        }
        for y, x in (*FOUR_INNER_CORNERS, (48, 78)):
            geometry[f"y{y}_x{x}"] = {
                "landmask": float(dataset.variables["LANDMASK"][0, y, x]),
                "hgt_m": float(dataset.variables["HGT"][0, y, x]),
                "lat": float(dataset.variables["XLAT"][0, y, x]),
                "lon": float(dataset.variables["XLONG"][0, y, x]),
            }
    if any(geometry[f"y{y}_x{x}"]["landmask"] != 0.0 or geometry[f"y{y}_x{x}"]["hgt_m"] != 0.0 for y, x in FOUR_INNER_CORNERS):
        raise AssertionError("four inner corners are no longer flat ocean")
    coupled_apex = mass_coupled_apex_oracle(state, metric_arrays)

    proof: dict[str, Any] = {
        "schema": "gpuwrf.v0234.corrected-ni-dry-sequence-offline-audit.v1",
        "status": "OFFLINE_COMPLETE_GPU_HELD",
        "verdict": "COMPLETE_NESTED_FROZEN_BUNDLE_CANDIDATE_PENDING_FABLE",
        "scope": {
            "gpu_calls": 0,
            "model_calls": 0,
            "jax_imported": False,
            "gpu_lock_acquired": False,
            "model_or_numerical_edits": 0,
            "agents_or_models_launched": 0,
            "observer_summaries_read": False,
            "fable_broad_review_duplicated": False,
        },
        "authority": {
            "base_commit": BASE_COMMIT,
            "ordinary_launch_commit": ORDINARY_LAUNCH_COMMIT,
            "live_git_objects": live_auth,
            "pristine_wrf": {
                "repo": str(WRF_REPO),
                "commit": WRF_COMMIT,
                "working_tree_used": False,
                "git_objects": wrf_auth,
            },
            "retained_carries": carry_auth,
            "gain1_proof": gain_auth,
            "identity_redesign_proof": identity_auth,
            "v10_separation_proof": v10_auth,
            "wrfinput_d03": {
                "path": str(WRFINPUT_D03),
                "bytes": WRFINPUT_D03.stat().st_size,
                "sha256": wrfinput_actual,
            },
        },
        "source_equation_map": source_map(live_text, wrf_text),
        "clock_discrepancy": {
            "rows": clock_rows,
            "step_9314": {
                "production_alpha": live_boundary_alpha(9314, DT_CHILD_S, DT_PARENT_S),
                "wrf_alpha": wrf_child_phase(9314),
                "production_minus_wrf_alpha": live_boundary_alpha(9314, DT_CHILD_S, DT_PARENT_S) - wrf_child_phase(9314),
                "wrf_rk_spec_endpoint_phases": list(wrf_rk_spec_endpoint_phases(9314)),
                "production_rk_spec_endpoint_phases": [1.0, 1.0, 1.0],
                "naive_modulo_only_rk_spec_endpoint_phases": list(
                    naive_localized_live_stage_phases(9314)
                ),
            },
            "causal_class": "PROVED_GENERAL_SOURCE_DISCREPANCY",
            "statement": (
                "A freshly rebuilt [old child ring, new parent target] leaf is consumed with the "
                "global forecast clock. Both record indices clip to 1 after the first 18 seconds, "
                "so every later package is immediately forced to its final record instead of "
                "traversing 1/3, 2/3, 1 over each three-child-step parent interval."
            ),
            "stage_cadence_consequence": (
                "At step 9314 WRF's frozen relax target uses the completed-step dtbc endpoint 2/3, while "
                "the three direct spec-zone RK endpoints are 4/9, 1/2, and 2/3 from the prior 1/3 state. "
                "Simply applying modulo to the existing lead_stage=lead+dt_rk expression would instead use "
                "7/9, 5/6, and 1, so a clock-only edit is algebraically incomplete."
            ),
        },
        "retained_step9313_package": package,
        "boundary_coverage_discrepancy": coverage,
        "independent_numpy_oracles": {
            **oracle,
            "retained_mass_coupled_apex_response": coupled_apex,
        },
        "retained_dynamic_compatibility": {
            "strongest_temporal_interval": identity["ordinary_localization"]["strongest_temporal_interval"],
            "gain_discriminator": masks,
            "rk3_entry_four_corner_amplitudes": stage_saves,
            "geometry": geometry,
            "terrain_w_deviation_falsifier": (
                "All four RK3-entry runaway corners and both final cone apices are LANDMASK=0/HGT=0. "
                "The known terrain-following lower-wind deviation cannot seed this flat-ocean corner mode."
            ),
        },
        "candidate_ranking": [
            {
                "rank": 1,
                "candidate": "complete_nested_frozen_RK1_boundary_bundle_with_package_local_phase",
                "classification": "SPECIFIC_GENERAL_SOURCE_DISCREPANCY; DYNAMIC_CAUSALITY_UNTESTED",
                "survives": [
                    "WRF source requires a parent-interval boundary slope and a full four-side frozen RK1 relax bundle",
                    "production source saturates the two-record package and excludes nested children from the frozen bundle",
                    "released normal-only coverage omits three of four flux faces at each final-cone apex",
                    "ordinary and gain1 retained RK3 entries both show the predicted four-corner runaway",
                    "gain1 changes scale/footprint but does not restore the missing fields, sides, cadence, or frozen semantics",
                ],
                "remaining_falsifier": (
                    "No observer-free same-carry dynamic arm has yet shown that the complete bundle keeps step 9314 healthy."
                ),
            },
            {
                "rank": 2,
                "candidate": "package_local_phase_only",
                "classification": "PROVED_SOURCE_BUG; INSUFFICIENT_STANDALONE_INCIDENT_CANDIDATE",
                "falsifier": (
                    "It corrects a one-third step-9314 target error but leaves 648 U and 528 V WRF relax cells absent, "
                    "including three of four flux faces at both final-cone apices, and leaves t/mu/w/ph cadence incomplete. "
                    "Naively reusing lead_stage=localized_endpoint+dt_rk also produces 7/9,5/6,1 instead of "
                    "WRF's 4/9,1/2,2/3 spec-stage endpoints."
                ),
            },
            {
                "rank": 3,
                "candidate": "moving_normal_gain_20",
                "classification": "INCIDENT_AMPLITUDE_OR_FOOTPRINT_MODULATOR_ONLY",
                "falsifier": (
                    "The authenticated gain1 B remains nonfinite at both same apices; its RK3-entry global maxima are "
                    "larger for every saved dry family despite the smaller final nonfinite footprint."
                ),
            },
            {
                "rank": 4,
                "candidate": "advance_uv_PGF_or_continuity_algebra",
                "classification": "DOWNSTREAM_AMPLIFIER_NOT_INDEPENDENT_ROOT",
                "falsifier": (
                    "The NumPy continuity divergence telescopes to boundary flux. The established discrepancy is the "
                    "unfaithful boundary flux/tendency ownership entering advance_mu_t, not a failed divergence identity."
                ),
            },
            {
                "rank": 5,
                "candidate": "large_step_momentum_mass_or_w_ph_feedback",
                "classification": "COUPLED_AMPLIFICATION_PATH; NO_SEPARATE_SOURCE_CAUSE_CLOSED",
                "falsifier": (
                    "Retained data place the runaway before RK3 and the five scratch families after continuity, but do "
                    "not invert their many-to-one source terms. WRF-incomplete ph/w boundary handling belongs in rank 1."
                ),
            },
            {
                "rank": 6,
                "candidate": "terrain_lower_w_boundary_or_halo_end_step",
                "classification": "PROXIMATE_SEED_FALSIFIED",
                "falsifier": (
                    "The four corner seeds are flat ocean. End-step dry nudging can pollute the incoming carry but cannot "
                    "be the within-step9314 first nonfinite operation; it is removed as part of a faithful bundle."
                ),
            },
        ],
        "minimum_source_backed_candidate_design": {
            "boundary_clock": (
                "Derive a package-local completed-step dtbc ((global_lead-dt_child) mod parent_dt)+dt_child "
                "only for refreshed child leaves; retain the global clock for radiation, physics, and outputs. "
                "Use that endpoint for frozen RK1 relax, but use endpoint-dt_child+dt_rk for each direct spec "
                "RK target. This needs no new carry leaf and preserves the 106-leaf interface."
            ),
            "rk1_relax": (
                "Build exactly once from the step-start/RK1 reference and proper package phase: coupled ru/rv, "
                "mass-weighted theta/ph/w, and plain mu with WRF fcx/gcx and S/N corner ownership. Reuse frozen "
                "ru/rv/theta/ph/w through all RK stages; apply mu only at RK1 as WRF does."
            ),
            "coupled_leaf_algebra": (
                "Pristine WRF couples both parent and child prognostics before forcedown interpolation. Preserve "
                "the existing boundary-leaf shapes/tree, but either store the dry two-record package in coupled "
                "space and decouple at value consumers using the synchronized mu leaf, or prove an algebraically "
                "equivalent product interpolation. Child-mass coupling after interpolation is not exact enough "
                "to claim the final general fix."
            ),
            "spec_zone": (
                "Replace the moving normal convex blend and nested hard ph pin with WRF boundary-slope updates for "
                "normal and tangential u/v, theta, mu/muts, mass-reweighted ph, and nested w in their normal order: "
                "post-advance_uv; post-advance_mu_t; post-advance_w before calc_p_rho."
            ),
            "end_of_step": (
                "Remove the legacy nested dry relax/value rewrite from the dynamical ownership path; retain only "
                "source-backed moisture handling and an idempotent endpoint sync if bit/prognostic algebra proves it."
            ),
            "required_prelaunch_prediction": (
                "Step 9314 must contain zero nonfinite scratch values, remain physically bounded against the finite "
                "step9313 baseline, preserve both cone apices, and run step9315 only if step9314 passes. A partial "
                "clock-only or coefficient arm is not admissible."
            ),
            "current_design_gate": (
                "The retained coupled-response oracle establishes a discriminating correction on every missing "
                "apex face, but exact parent-before-interpolation coupling and substep spec-slope CPU oracles are "
                "not implemented yet. The candidate therefore remains a redesign, not a launchable patch."
            ),
        },
        "gpu_gate": {
            "gpu_eligible": False,
            "hard_hold": True,
            "reason": (
                "Owner-authorized Fable broad review through dbad590b is still outstanding and must be authenticated "
                "and incorporated before any conditional B arm. No model implementation or HLO/interface proof exists yet."
            ),
            "gpu_commands_run": 0,
        },
        "mechanism_separation": {
            "ni": (
                "This audit addresses the upstream dry corner runaway in the carry immediately before Ni onset; "
                "the canonical Ni cell y48,x78 is interior offshore and is not reclassified as a boundary seed."
            ),
            "v10": v10["verdict"],
            "verdict": "V10_REMAINS_SEPARATE; NO_TEMPORAL_CAUSAL_LINK_ADDED",
        },
    }
    proof["proof_sha256"] = canonical_digest(proof)
    return proof


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SPRINT / "offline-dry-sequence-proof.json",
        help="proof path",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    proof = build_proof()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(
        "OFFLINE_DRY_SEQUENCE_AUDIT "
        f"verdict={proof['verdict']} proof_sha256={proof['proof_sha256']} "
        f"gpu_eligible={proof['gpu_gate']['gpu_eligible']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
