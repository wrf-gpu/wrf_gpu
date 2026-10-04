#!/usr/bin/env python3
"""Deterministic CPU/source proof for NESTED_FROZEN_WRF_BOUNDARY_BUNDLE_V1.

The default mode authenticates pristine WRF and retained incident evidence,
then runs a matched tiny nested-step probe against both this worktree and an
immutable dbad590b worktree.  Probe mode imports only the explicitly selected
source root.  CUDA is hidden and no production/GPU artifact is read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-13-v0234-nested-frozen-wrf-bundle-v1"
BASE_COMMIT = "dbad590b93f807966583a5ba0332dd3ae9483a89"
WRF_REPO = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"
FABLE_ROOT = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-corrected-ni-fable5")

OWNED_SOURCES = (
    "src/gpuwrf/coupling/boundary_apply.py",
    "src/gpuwrf/dynamics/core/acoustic.py",
    "src/gpuwrf/integration/nested_pipeline.py",
    "src/gpuwrf/runtime/domain_tree.py",
    "src/gpuwrf/runtime/operational_mode.py",
)

WRF_SOURCES = {
    "share/module_bc.F": "61b9235004b2a7799faabaa928276af8a7ef2e4672619c8ad120c857461301ad",
    "dyn_em/module_bc_em.F": "6cfb52b849e3dd0b24769709cf502faa75886da454d9b668af725cccdc56d47f",
    "dyn_em/module_small_step_em.F": "cabf1a177d50fb0096db79644af20cfe6d75217dbe63ab406a7e29bb54c17634",
    "dyn_em/solve_em.F": "3322d34e0ad070cf4d891aa6dbc93be70ef8240b84eb5f366a149487550aef89",
    "dyn_em/module_em.F": "11105cbf8255f30ca6a44cd7429a92cedce1fb91db6ce90fd7217002a72fb7fa",
}

EVIDENCE = {
    ".agent/sprints/2026-07-13-v0234-corrected-ni-gain1-ab/gain1-ab-proof.json":
        "cb8643d7e24b139331a3fad780ccb35564809cbfc1ffb44c37fd981372486ebb",
    ".agent/sprints/2026-07-13-v0234-corrected-ni-phase-tap/identity-redesign-proof.json":
        "49a7bbe94b71e4bacdf4904d14c6f47cf9a3120ed5494fcde99d394bdeae32a1",
    ".agent/sprints/2026-07-13-v0234-corrected-ni-rca-max/v10-spatial-causal-proof.json":
        "d269513748c85006a7a95b87f778e4c442041c812b2d76dee85ff20309599e68",
    ".agent/sprints/2026-07-13-v0234-corrected-ni-dry-sequence-audit/offline-dry-sequence-proof.json":
        "5c32d4d372e4184de7b78a498ae65a113159f76a37af073c1598c6855229b0eb",
}

FABLE_EVIDENCE = {
    ".agent/sprints/2026-07-13-v0234-corrected-ni-fable5-review/fable-review.md":
        "8ca702798a6b6c530bf2563ee6d0f6085326425fd4b46ef3127b1def06b333bf",
    ".agent/sprints/2026-07-13-v0234-corrected-ni-fable5-review/fable-review-findings.json":
        "f5fd486282693734e23090a735d29abb3ca09ce474eca95089a9214bda7a9f4a",
}

FORBIDDEN_HLO = (
    "xla_python_cpu_callback",
    "host_callback",
    "io_callback",
    "outside_compilation",
    "custom_call",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def git_bytes(repo: Path, revision: str, path: str) -> bytes:
    return subprocess.check_output(
        ("git", "-C", str(repo), "show", f"{revision}:{path}")
    )


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(repo), *args), text=True
    ).strip()


def source_anchor(text: str, needle: str) -> dict[str, Any]:
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines, 1) if needle in line]
    if not matches:
        raise AssertionError(f"missing source anchor {needle!r}")
    line = matches[0]
    return {"line": line, "needle": needle, "text": lines[line - 1].strip()}


def tree_digest(tree: object) -> tuple[str, list[dict[str, Any]]]:
    import jax
    import numpy as np

    digest = hashlib.sha256()
    signature: list[dict[str, Any]] = []
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        item = {"dtype": str(array.dtype), "shape": list(array.shape)}
        signature.append(item)
        digest.update(item["dtype"].encode())
        digest.update(json.dumps(item["shape"], separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest(), signature


def run_probe(probe_root: Path, *, candidate_on: bool) -> dict[str, Any]:
    """Build/lower/execute one ordinary CPU nested child step."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("probe requires CUDA_VISIBLE_DEVICES='' before import")
    sys.path.insert(0, str(probe_root / "src"))

    import dataclasses
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    from gpuwrf.runtime.aot_cheap_key import static_config_hash
    from gpuwrf.runtime.domain_tree import _operational_force
    from gpuwrf.runtime import operational_mode as operational
    from gpuwrf.validation.moving_nest_testbed import build_nested_pair

    tree = build_nested_pair(
        parent_nx=18,
        parent_ny=18,
        child_nx=14,
        child_ny=14,
        nz=4,
        ratio=3,
        i_start=3,
        j_start=3,
        dt_s=0.6,
    )
    parent_bundle = tree.domains["d01"]
    child_bundle = tree.domains["d02"]
    child_grid = dataclasses.replace(
        child_bundle.grid,
        bc=dataclasses.replace(child_bundle.grid.bc, source="AIFS"),
    )
    boundary_config = child_bundle.namelist.boundary_config
    if hasattr(boundary_config, "nested_frozen_wrf_boundary_bundle"):
        boundary_config = dataclasses.replace(
            boundary_config,
            nested_frozen_wrf_boundary_bundle=bool(candidate_on),
        )
    elif candidate_on:
        raise AssertionError("candidate-on probe selected a source tree without the feature")
    child_namelist = dataclasses.replace(
        child_bundle.namelist,
        grid=child_grid,
        acoustic_substeps=1,
        boundary_config=boundary_config,
    )
    parent = operational._initial_carry_for_run(
        parent_bundle.state, parent_bundle.namelist
    )
    child = operational._initial_carry_for_run(child_bundle.state, child_namelist)
    forced = _operational_force(tree.edges["d01"][0], parent, child)

    def step(value):
        return operational._physics_boundary_step(
            value,
            child_namelist,
            jnp.asarray(1, dtype=jnp.int32),
            run_radiation=False,
        )

    input_hash, input_signature = tree_digest(forced)
    treedef = str(jax.tree_util.tree_structure(forced))
    jaxpr = str(jax.make_jaxpr(step)(forced))
    lowered = jax.jit(step).lower(forced)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    # Module symbol names are closure-name metadata, not generated equations.
    stablehlo = re.sub(
        r"module @jit_[^ ]+", "module @jit_STEP", stablehlo, count=1
    )
    output = step(forced)
    jax.block_until_ready(output.state.theta)
    output_hash, output_signature = tree_digest(output)

    finite = True
    for leaf in jax.tree_util.tree_leaves(output):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.floating):
            finite = finite and bool(np.all(np.isfinite(array)))
    if not finite:
        raise AssertionError("tiny candidate probe produced a nonfinite leaf")
    if input_signature != output_signature:
        raise AssertionError("ordinary carry signature changed across one step")

    return {
        "probe_root": str(probe_root),
        "candidate_on": bool(candidate_on),
        "feature_gate_active": bool(
            getattr(
                operational,
                "_nested_frozen_wrf_boundary_active",
                lambda _namelist: False,
            )(child_namelist)
        ),
        "static_config_hash": static_config_hash(child_namelist),
        "leaf_count": len(input_signature),
        "input_value_sha256": input_hash,
        "output_value_sha256": output_hash,
        "signature_sha256": canonical_digest(input_signature),
        "treedef_sha256": sha256_bytes(treedef.encode()),
        "jaxpr_sha256": sha256_bytes(jaxpr.encode()),
        "jaxpr_bytes": len(jaxpr.encode()),
        "stablehlo_sha256": sha256_bytes(stablehlo.encode()),
        "stablehlo_bytes": len(stablehlo.encode()),
        "forbidden_hlo_tokens": {
            token: stablehlo.count(token) for token in FORBIDDEN_HLO
        },
        "output_finite": finite,
        "carry_signature_unchanged": input_signature == output_signature,
    }


def subprocess_probe(
    script: Path, probe_root: Path, *, candidate_on: bool
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "JAX_ENABLE_X64": "1",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
            "PYTHONPATH": str(probe_root / "src"),
            "GPUWRF_PROBE_ROOT": str(probe_root),
        }
    )
    output = subprocess.check_output(
        (
            sys.executable,
            str(script),
            "--probe",
            "on" if candidate_on else "off",
        ),
        cwd=probe_root,
        env=environment,
        text=True,
    )
    return json.loads(output)


def authenticate_files(
    root: Path, expected: dict[str, str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative, digest in expected.items():
        path = root / relative
        actual = sha256_file(path)
        if actual != digest:
            raise AssertionError(f"authority mismatch {path}: {actual} != {digest}")
        result[relative] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return result


def ownership_counts(ny: int, nx: int) -> dict[str, int]:
    import numpy as np

    spec = np.zeros((ny, nx), dtype=np.int32)
    relax = np.zeros((ny, nx), dtype=np.int32)
    spec[0, :] += 1
    spec[-1, :] += 1
    spec[1:-1, 0] += 1
    spec[1:-1, -1] += 1
    for b in (1, 2, 3):
        relax[b, b : nx - b] += 1
        relax[ny - 1 - b, b : nx - b] += 1
        relax[b + 1 : ny - 1 - b, b] += 1
        relax[b + 1 : ny - 1 - b, nx - 1 - b] += 1
    return {
        "spec_owned_cells": int((spec == 1).sum()),
        "relax_owned_cells": int((relax == 1).sum()),
        "duplicate_spec_cells": int((spec > 1).sum()),
        "duplicate_relax_cells": int((relax > 1).sum()),
        "spec_relax_overlap_cells": int(((spec + relax) > 1).sum()),
    }


def find_numeric_key(value: Any, keys: Iterable[str], expected: int) -> bool:
    key_set = set(keys)
    if isinstance(value, dict):
        if any(value.get(key) == expected for key in key_set):
            return True
        return any(find_numeric_key(item, key_set, expected) for item in value.values())
    if isinstance(value, list):
        return any(find_numeric_key(item, key_set, expected) for item in value)
    return False


def added_source_lines(root: Path) -> list[str]:
    output = subprocess.check_output(
        (
            "git",
            "-C",
            str(root),
            "diff",
            "--unified=0",
            BASE_COMMIT,
            "--",
            *OWNED_SOURCES,
        ),
        text=True,
    )
    return [
        line[1:]
        for line in output.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def build_proof(base_worktree: Path) -> dict[str, Any]:
    if git_text(ROOT, "rev-parse", "HEAD") != BASE_COMMIT:
        raise AssertionError("proof must be generated before the enclosing candidate commit")
    if git_text(base_worktree, "rev-parse", "HEAD") != BASE_COMMIT:
        raise AssertionError("immutable base worktree is not dbad590b")
    if git_text(WRF_REPO, "rev-parse", f"{WRF_COMMIT}^{{commit}}") != WRF_COMMIT:
        raise AssertionError("pristine WRF authority commit is unavailable")

    wrf_objects: dict[str, Any] = {}
    wrf_texts: dict[str, str] = {}
    for path, expected in WRF_SOURCES.items():
        value = git_bytes(WRF_REPO, WRF_COMMIT, path)
        actual = sha256_bytes(value)
        if actual != expected:
            raise AssertionError(f"pristine WRF object mismatch {path}")
        wrf_objects[path] = {"bytes": len(value), "sha256": actual}
        wrf_texts[path] = value.decode()

    retained = authenticate_files(ROOT, EVIDENCE)
    fable = authenticate_files(FABLE_ROOT, FABLE_EVIDENCE)
    gain_payload = json.loads(
        (ROOT / next(iter(EVIDENCE))).read_text()
    )
    production_106_bound = find_numeric_key(
        gain_payload,
        ("carry_leaf_count", "input_leaf_count", "output_leaf_count", "leaf_count"),
        106,
    )
    if not production_106_bound:
        raise AssertionError("retained gain-1 proof no longer binds 106 leaves")

    script = Path(__file__).resolve()
    base_off = subprocess_probe(script, base_worktree, candidate_on=False)
    candidate_off = subprocess_probe(script, ROOT, candidate_on=False)
    candidate_on = subprocess_probe(script, ROOT, candidate_on=True)
    compared = (
        "leaf_count",
        "input_value_sha256",
        "output_value_sha256",
        "signature_sha256",
        "treedef_sha256",
        "jaxpr_sha256",
        "jaxpr_bytes",
        "stablehlo_sha256",
        "stablehlo_bytes",
        "forbidden_hlo_tokens",
        "output_finite",
        "carry_signature_unchanged",
    )
    mismatches = {
        key: {"base": base_off[key], "candidate_off": candidate_off[key]}
        for key in compared
        if base_off[key] != candidate_off[key]
    }
    if mismatches:
        raise AssertionError(f"candidate-off exact-base mismatch: {mismatches}")
    if candidate_off["feature_gate_active"]:
        raise AssertionError("candidate-off probe unexpectedly active")
    if not candidate_on["feature_gate_active"]:
        raise AssertionError("candidate-on probe failed to activate")
    if any(candidate_on["forbidden_hlo_tokens"].values()):
        raise AssertionError("candidate HLO introduced a forbidden callback/custom-call")

    source_hashes = {
        path: {
            "bytes": (ROOT / path).stat().st_size,
            "sha256": sha256_file(ROOT / path),
        }
        for path in OWNED_SOURCES
    }
    source_texts = {path: (ROOT / path).read_text() for path in OWNED_SOURCES}
    live_anchors = {
        "single_feature_surface": source_anchor(
            source_texts["src/gpuwrf/coupling/boundary_apply.py"],
            "nested_frozen_wrf_boundary_bundle: bool = False",
        ),
        "wrf_single_owner_sync": source_anchor(
            source_texts["src/gpuwrf/coupling/boundary_apply.py"],
            "def _apply_3d_spec_only_wrf_owned",
        ),
        "nested_w_relax_lane": source_anchor(
            source_texts["src/gpuwrf/coupling/boundary_apply.py"],
            "include_nested_w: bool = False",
        ),
        "child_static_gate": source_anchor(
            source_texts["src/gpuwrf/runtime/operational_mode.py"],
            "def _nested_frozen_wrf_boundary_active",
        ),
        "integer_subcycle_clock": source_anchor(
            source_texts["src/gpuwrf/runtime/operational_mode.py"],
            "def nested_boundary_package_endpoint_seconds",
        ),
        "stage_clock": source_anchor(
            source_texts["src/gpuwrf/runtime/operational_mode.py"],
            "def nested_boundary_stage_seconds",
        ),
        "rk1_mudf": source_anchor(
            source_texts["src/gpuwrf/runtime/operational_mode.py"],
            "def _stage_entry_mudf",
        ),
        "frozen_bundle_hoist": source_anchor(
            source_texts["src/gpuwrf/runtime/operational_mode.py"],
            "nested_frozen_relax = (",
        ),
        "production_exact_env": source_anchor(
            source_texts["src/gpuwrf/integration/nested_pipeline.py"],
            'os.environ.get("GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE", "0") == "1"',
        ),
    }
    wrf_anchors = {
        "dtbc_increment": source_anchor(
            wrf_texts["dyn_em/solve_em.F"],
            "grid%dtbc = grid%dtbc + grid%dt",
        ),
        "relax_rk1": source_anchor(
            wrf_texts["dyn_em/solve_em.F"], "CALL relax_bdy_dry"
        ),
        "u_spec_walk": source_anchor(
            wrf_texts["dyn_em/solve_em.F"],
            "CALL spec_bdyupdate(grid%u_2",
        ),
        "theta_spec_walk": source_anchor(
            wrf_texts["dyn_em/solve_em.F"],
            "CALL spec_bdyupdate(grid%t_2",
        ),
        "ph_spec_walk": source_anchor(
            wrf_texts["dyn_em/solve_em.F"], "CALL spec_bdyupdate_ph"
        ),
        "w_spec_walk": source_anchor(
            wrf_texts["dyn_em/solve_em.F"],
            "CALL spec_bdyupdate ( grid%w_2",
        ),
        "mudf_zero": source_anchor(
            wrf_texts["dyn_em/module_small_step_em.F"], "MUDF(i,j) = 0."
        ),
        "finish_u_stage_face_mass": source_anchor(
            wrf_texts["dyn_em/module_small_step_em.F"],
            "/(c1h(k)*muus(i,j)+c2h(k))",
        ),
        "nested_w_relax": source_anchor(
            wrf_texts["dyn_em/module_bc_em.F"],
            "CALL relax_bdytend_tile ( rfield, rw_tendf",
        ),
    }

    added = added_source_lines(ROOT)
    forbidden_source = {
        token: [line for line in added if token in line]
        for token in (
            "jax.device_get",
            "jax.pure_callback",
            "io_callback",
            "host_callback",
            "debug.callback",
        )
    }
    if any(forbidden_source.values()):
        raise AssertionError(f"new in-loop transfer/callback surface: {forbidden_source}")

    clocks = {}
    for child_dt, parent_dt in ((6.0, 18.0), (2.0, 6.0), (1.5, 7.5)):
        ratio = round(parent_dt / child_dt)
        clocks[f"{child_dt:g}/{parent_dt:g}"] = [
            child_dt * (((step - 1) % ratio) + 1)
            for step in range(1, 2 * ratio + 1)
        ]
    strength20_response = 1.0 - (1.0 - 0.2) ** 10
    if strength20_response != 0.8926258175999999:
        raise AssertionError("negative-control response changed")

    return {
        "schema": "gpuwrf.v0234.nested-frozen-wrf-boundary-bundle-proof.v1",
        "verdict": "READY_FOR_GPT_CRITIC",
        "authority": {
            "base_commit": BASE_COMMIT,
            "candidate_head_before_enclosing_commit": git_text(
                ROOT, "rev-parse", "HEAD"
            ),
            "pristine_wrf_commit": WRF_COMMIT,
            "pristine_wrf_objects": wrf_objects,
            "retained_evidence": retained,
            "fable_independent_review": fable,
            "fable_treated_as_claims_rechecked_against_pristine_source": True,
            "production_carry_leaf_count": 106,
            "production_106_bound_by_authenticated_gain1_proof": production_106_bound,
            "retained_step9313_input_sha256":
                "f82a25c35e6bd738cd248d759c291d825ca0cfc4e08753dc5082741a78b23fc7",
            "retained_ordinary_step9314_manifest":
                "60bba36797c09100bd86d09f885930cdb0f72d5be587d05692a998ef9080bfc7",
            "frozen_terminal_manifest":
                "2dcfc195baaf3e59ba539f6701fdb82b9a134a0461cfc68dec3a429e38433565",
        },
        "candidate_sources": source_hashes,
        "source_mapping": {
            "live": live_anchors,
            "pristine_wrf": wrf_anchors,
        },
        "cpu_program_proof": {
            "base_candidate_off": base_off,
            "candidate_candidate_off": candidate_off,
            "candidate_on": candidate_on,
            "candidate_off_exact_base_fields_compared": list(compared),
            "candidate_off_exact_base_mismatches": mismatches,
            "candidate_off_exact_base": not mismatches,
            "candidate_off_static_config_hash_intentionally_differs": (
                base_off["static_config_hash"]
                != candidate_off["static_config_hash"]
            ),
            "static_key_note": (
                "The added default-off static flag intentionally changes the cheap-key "
                "serialization while the traced values/JAXPR/StableHLO remain exact."
            ),
            "candidate_on_differs_from_off": {
                "static_config_hash": (
                    candidate_on["static_config_hash"]
                    != candidate_off["static_config_hash"]
                ),
                "jaxpr_sha256": (
                    candidate_on["jaxpr_sha256"]
                    != candidate_off["jaxpr_sha256"]
                ),
                "stablehlo_sha256": (
                    candidate_on["stablehlo_sha256"]
                    != candidate_off["stablehlo_sha256"]
                ),
            },
            "fixture_leaf_count_note": (
                "The bounded tiny CPU fixture has 71 leaves; the unchanged 106-leaf "
                "production interface is bound by retained authenticated proof."
            ),
        },
        "analytic_oracles": {
            "two_leaf_released_saturation_steps_1_to_6": [
                1.0 / 3.0,
                2.0 / 3.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "integer_subcycle_endpoint_seconds_two_cycles": clocks,
            "d03_stage_endpoint_seconds_at_package_endpoint_6": [
                14.0 / 3.0,
                5.0,
                6.0,
            ],
            "released_strength20_ten_substep_response": strength20_response,
            "candidate_moving_relax_row_response": 0.0,
            "wrf_full_step_uniform_residual_taper": {
                "ring1": 0.1,
                "ring2": 0.1 * 2.0 / 3.0,
                "ring3": 0.1 / 3.0,
                "interior": 0.0,
            },
            "ownership": {
                "mass_14x13": ownership_counts(14, 13),
                "u_14x14": ownership_counts(14, 14),
                "v_15x13": ownership_counts(15, 13),
            },
            "independent_numpy_test_module":
                "tests/test_v0234_nested_frozen_wrf_boundary_bundle.py",
        },
        "transfer_audit": {
            "added_source_lines_scanned": len(added),
            "forbidden_added_source_tokens": forbidden_source,
            "candidate_stablehlo_forbidden_token_counts": candidate_on[
                "forbidden_hlo_tokens"
            ],
            "new_state_or_carry_fields": 0,
            "new_callback_or_observer_outputs": 0,
        },
        "causal_scope": {
            "candidate_class": (
                "source-proven complete nested lateral-boundary discrepancy bundle; "
                "dynamic incident causality remains untested until post-critic GPU replay"
            ),
            "gain1_strength_only_candidate": "FALSIFIED",
            "v10": (
                "SEPARATE: the 00:00 surface-algebra component and broad later V10 "
                "drift are not attributed to this bundle without new causal evidence"
            ),
            "known_decoupled_leaf_limit": (
                "The retained two-record leaves remain decoupled linear-SINT values. "
                "Pristine WRF couples parent and child prognostics before forcedown/SINT; "
                "this candidate therefore does not claim exact coupled forcedown parity."
            ),
            "stage_pin_limit": (
                "Ring-0 work targets land exactly on each RK-stage endpoint but are "
                "held across that stage's acoustic substeps; this preserves the existing "
                "v0.14 architecture and is not a claim of byte-identical WRF intra-stage history."
            ),
            "gpu_evidence": "NONE; prohibited pending independent GPT acceptance",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=("off", "on"))
    parser.add_argument("--base-worktree", type=Path)
    parser.add_argument(
        "--output", type=Path, default=SPRINT / "bundle-proof.json"
    )
    args = parser.parse_args()

    if args.probe is not None:
        probe_root = Path(os.environ["GPUWRF_PROBE_ROOT"]).resolve()
        print(
            json.dumps(
                run_probe(probe_root, candidate_on=args.probe == "on"),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    if args.base_worktree is None:
        parser.error("--base-worktree is required outside --probe mode")

    proof = build_proof(args.base_worktree.resolve())
    proof["proof_sha256"] = canonical_digest(proof)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "proof_sha256": proof["proof_sha256"],
                "verdict": proof["verdict"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
