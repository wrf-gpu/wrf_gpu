#!/usr/bin/env python3
"""V0.14 Step-1 QVAPOR pre-call truth schema inventory."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "proofs/v014/step1_qvapor_precall_truth_schema.json"
OUT_MD = REPO / "proofs/v014/step1_qvapor_precall_truth_schema.md"
OUT_REVIEW = REPO / ".agent/reviews/2026-06-09-v014-step1-qvapor-precall-truth-schema.md"
OUT_SAVEPOINT = (
    REPO
    / ".agent/sprints/2026-06-09-v014-step1-qvapor-precall-truth-schema/artifacts/proposed_wrf_savepoint.md"
)

TARGET_SURFACE = "before_first_rk_step_part1_call"
TARGET_DOMAIN = 2
TARGET_STEP = 1
TARGET_RK = 1
ANCESTOR = "5b1f6b10"

PREPART_ROOT = Path("/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth")
SAME_INPUT_NPZ = Path(
    "/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/"
    "same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz"
)
SAME_INPUT_RAW_ROOT = Path("/mnt/data/wrf_gpu2/v014_step1_same_input_truth/raw_truth")
MNT_ROOT = Path("/mnt/data/wrf_gpu2")

PRISTINE_MEDIATION = Path("/home/user/src/wrf_pristine/WRF/share/mediation_integrate.F")
PRISTINE_NEST_INIT = Path("/home/user/src/wrf_pristine/WRF/dyn_em/nest_init_utils.F")
PRISTINE_SOLVE = Path("/home/user/src/wrf_pristine/WRF/dyn_em/solve_em.F")
INSTRUMENTED_SOLVE = Path("/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF/dyn_em/solve_em.F")


def run(cmd: list[str], cwd: Path = REPO) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def text_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def source_line(path: Path, line_no: int) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "line": line_no, "text": None, "exists": False}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            if idx == line_no:
                return {"path": str(path), "line": line_no, "text": line.rstrip("\n"), "exists": True}
    return {"path": str(path), "line": line_no, "text": None, "exists": True}


def parse_header(path: Path) -> dict[str, Any]:
    header: dict[str, Any] = {"path": str(path), "record_schemas": []}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("record_schema "):
                parts = line.split()
                if len(parts) >= 3:
                    header["record_schemas"].append({"record": parts[1], "fields": parts[2:]})
                continue
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]
            if key in {"MASS_PREPART", "WPH_PREPART", "MASS_FULL", "U_FULL", "V_FULL", "WPH_FULL"}:
                break
            header[key] = parts[1:] if len(parts) > 2 else (parts[1] if len(parts) == 2 else "")
    return header


def int_header(header: dict[str, Any], key: str) -> int | None:
    value = header.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def list_files(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.glob(pattern))


def rg_qvapor_files(roots: list[Path]) -> list[Path]:
    existing = [str(p) for p in roots if p.exists()]
    if not existing:
        return []
    proc = subprocess.run(
        ["rg", "-l", "QVAPOR", *existing],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode not in {0, 1}:
        raise RuntimeError(f"rg failed while searching QVAPOR: {proc.stderr}")
    return sorted(Path(line) for line in proc.stdout.splitlines() if line.strip())


def infer_surface_from_npz(path: Path) -> str | None:
    name = path.name
    if "post_after_all_rk_steps_pre_halo" in name:
        return "post_after_all_rk_steps_pre_halo"
    if TARGET_SURFACE in name:
        return TARGET_SURFACE
    return None


def inspect_npz(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "artifact_type": "npz",
        "path": str(path),
        "exists": path.is_file(),
        "surface": infer_surface_from_npz(path),
        "domain_id": 2 if "_d02_" in path.name or "_d2_" in path.name else None,
        "step": 1 if "_step_1" in path.name else None,
        "rk_step": None,
        "contains_qvapor": False,
        "fields": [],
        "qvapor": None,
    }
    if not path.is_file():
        item["classification"] = "missing"
        return item
    with np.load(path, allow_pickle=False) as z:
        item["fields"] = sorted(z.files)
        item["contains_qvapor"] = "QVAPOR" in z.files
        if "QVAPOR" in z.files:
            qv = z["QVAPOR"]
            item["qvapor"] = {
                "shape": list(qv.shape),
                "dtype": str(qv.dtype),
                "min": float(np.nanmin(qv)) if qv.size else None,
                "max": float(np.nanmax(qv)) if qv.size else None,
            }
    item["same_boundary_as_precall"] = (
        item["surface"] == TARGET_SURFACE
        and item["domain_id"] == TARGET_DOMAIN
        and item["step"] == TARGET_STEP
        and item["contains_qvapor"]
    )
    item["classification"] = (
        "same_boundary_precall_candidate"
        if item["same_boundary_as_precall"]
        else "post_rk_or_different_boundary"
    )
    return item


def group_text_qvapor_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str | None], list[Path]] = defaultdict(list)
    headers: dict[tuple[str, str | None], dict[str, Any]] = {}
    for path in paths:
        header = parse_header(path)
        surface = header.get("surface")
        if isinstance(surface, list):
            surface = " ".join(surface)
        key = (str(path.parent), surface)
        groups[key].append(path)
        headers.setdefault(key, header)

    artifacts: list[dict[str, Any]] = []
    for (root, surface), files in sorted(groups.items()):
        header = headers[(root, surface)]
        mass_fields: list[str] = []
        for schema in header.get("record_schemas", []):
            if schema.get("record") in {"MASS_FULL", "MASS_PREPART"}:
                mass_fields = schema.get("fields", [])
                break
        domain = int_header(header, "domain_id")
        step = int_header(header, "grid_itimestep_after_increment")
        rk = int_header(header, "rk_step")
        shape = infer_mass_shape(header)
        artifacts.append(
            {
                "artifact_type": "text_tile_group",
                "root": root,
                "surface": surface,
                "domain_id": domain,
                "step": step,
                "rk_step": rk,
                "tile_file_count": len(files),
                "files": [str(p) for p in files],
                "sample_file": str(files[0]) if files else None,
                "mass_schema_fields": mass_fields,
                "qvapor_shape_zyx": shape,
                "contains_qvapor": "QVAPOR" in mass_fields,
                "same_boundary_as_precall": (
                    surface == TARGET_SURFACE
                    and domain == TARGET_DOMAIN
                    and step == TARGET_STEP
                    and rk == TARGET_RK
                    and "QVAPOR" in mass_fields
                ),
                "classification": (
                    "same_boundary_precall_candidate"
                    if surface == TARGET_SURFACE
                    and domain == TARGET_DOMAIN
                    and step == TARGET_STEP
                    and rk == TARGET_RK
                    and "QVAPOR" in mass_fields
                    else "post_rk_or_different_boundary"
                ),
            }
        )
    return artifacts


def infer_mass_shape(header: dict[str, Any]) -> list[int] | None:
    vert = header.get("mass_vertical_fortran_k_start_k_end_inclusive")
    bounds = header.get("global_mass_i_j_end_exclusive_fortran")
    if not isinstance(vert, list) or not isinstance(bounds, list):
        return None
    try:
        k0, k1 = (int(vert[0]), int(vert[1]))
        ids, ide, jds, jde = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
    except (ValueError, IndexError):
        return None
    return [k1 - k0 + 1, jde - jds, ide - ids]


def accepted_precall_schema() -> dict[str, Any]:
    files = list_files(PREPART_ROOT, f"{TARGET_SURFACE}_d2_step_1_rk_1_*.txt")
    headers = [parse_header(path) for path in files]
    schema_records: dict[str, list[str]] = {}
    for schema in headers[0].get("record_schemas", []) if headers else []:
        schema_records[schema["record"]] = schema["fields"]
    qvapor_files = rg_qvapor_files([PREPART_ROOT])
    qvapor_precall_files = [p for p in qvapor_files if p.name.startswith(TARGET_SURFACE)]
    return {
        "root": str(PREPART_ROOT),
        "surface": TARGET_SURFACE,
        "file_count": len(files),
        "sample_file": str(files[0]) if files else None,
        "sample_header": headers[0] if headers else None,
        "record_schemas": schema_records,
        "mass_shape_zyx": infer_mass_shape(headers[0]) if headers else None,
        "contains_qvapor_anywhere_in_root": bool(qvapor_files),
        "contains_qvapor_at_precall_surface": bool(qvapor_precall_files),
        "qvapor_files_in_root": [str(p) for p in qvapor_files],
        "qvapor_files_at_precall_surface": [str(p) for p in qvapor_precall_files],
    }


def candidate_text_roots() -> list[Path]:
    roots: set[Path] = set()
    for d in MNT_ROOT.glob("v014_step1*"):
        if not d.is_dir():
            continue
        for child in d.glob("*truth*"):
            if child.is_dir():
                roots.add(child)
    roots.add(PREPART_ROOT)
    roots.add(SAME_INPUT_RAW_ROOT)
    return sorted(roots)


def candidate_npz_files() -> list[Path]:
    files = set()
    for root in [MNT_ROOT / "v014_same_input_contract_builder/wrf_truth", *candidate_text_roots()]:
        if root.is_dir():
            files.update(root.glob("*.npz"))
    files.add(SAME_INPUT_NPZ)
    return sorted(files)


def source_evidence() -> dict[str, Any]:
    return {
        "live_nest_order": [
            source_line(PRISTINE_MEDIATION, 678),
            source_line(PRISTINE_MEDIATION, 716),
            source_line(PRISTINE_MEDIATION, 737),
            source_line(PRISTINE_MEDIATION, 756),
            source_line(PRISTINE_MEDIATION, 757),
            source_line(PRISTINE_MEDIATION, 758),
            source_line(PRISTINE_MEDIATION, 759),
            source_line(PRISTINE_MEDIATION, 762),
        ],
        "adjust_tempqv_semantics": [
            source_line(PRISTINE_NEST_INIT, 812),
            source_line(PRISTINE_NEST_INIT, 813),
            source_line(PRISTINE_NEST_INIT, 836),
            source_line(PRISTINE_NEST_INIT, 846),
            source_line(PRISTINE_NEST_INIT, 852),
            source_line(PRISTINE_NEST_INIT, 853),
            source_line(PRISTINE_NEST_INIT, 869),
            source_line(PRISTINE_NEST_INIT, 870),
            source_line(PRISTINE_NEST_INIT, 877),
            source_line(PRISTINE_NEST_INIT, 884),
        ],
        "use_theta_m_solve_em_comment": [
            source_line(PRISTINE_SOLVE, 515),
            source_line(PRISTINE_SOLVE, 516),
            source_line(PRISTINE_SOLVE, 517),
            source_line(PRISTINE_SOLVE, 521),
        ],
        "first_rk_entry_pristine": [
            source_line(PRISTINE_SOLVE, 806),
            source_line(PRISTINE_SOLVE, 808),
        ],
        "accepted_precall_hook_instrumented": [
            source_line(INSTRUMENTED_SOLVE, 1052),
            source_line(INSTRUMENTED_SOLVE, 1054),
            source_line(INSTRUMENTED_SOLVE, 7017),
            source_line(INSTRUMENTED_SOLVE, 7021),
            source_line(INSTRUMENTED_SOLVE, 7034),
        ],
    }


def savepoint_spec() -> dict[str, Any]:
    return {
        "needed": True,
        "reason": (
            "No existing QVAPOR artifact is at surface before_first_rk_step_part1_call, "
            "domain 2, step 1, rk_step 1."
        ),
        "file": "dyn_em/solve_em.F",
        "function": "solve_em",
        "location": {
            "boundary_name": TARGET_SURFACE,
            "insert_or_extend": "extend existing wrfgpu2_dump_pre_part1_surface output",
            "before": "CALL first_rk_step_part1",
            "after": "IF (coupler_on) CALL cpl_settime( curr_secs2 )",
            "instrumented_line_evidence": [
                "v014_step1_pre_part1_handoff/WRF/dyn_em/solve_em.F:1052",
                "v014_step1_pre_part1_handoff/WRF/dyn_em/solve_em.F:1054",
            ],
        },
        "fields": {
            "mass_existing": ["T_STATE", "P_STATE", "PB", "MU_STATE", "MUB", "MUT"],
            "mass_add": ["QVAPOR"],
            "mass_expression": "REAL(moist(i,k,j,P_QV),KIND=8)",
            "optional_header": "moist_index_qv P_QV",
            "wph_existing_unchanged": ["W_STATE", "PH_STATE", "PHB"],
        },
        "shape_contract": {
            "fortran_mass_loops": "i=ids..ide-1, j=jds..jde-1, k=kds..kde-1 over owned tiles",
            "d02_mass_shape_zyx": [44, 66, 159],
            "d02_wph_shape_zyx": [45, 66, 159],
            "tile_policy": "mass_owned_single_owner_no_overlap, same as accepted pre-call text",
        },
        "acceptance_checks": [
            "All emitted files have surface before_first_rk_step_part1_call, domain_id 2, step 1, rk_step 1.",
            "MASS_PREPART schema includes QVAPOR after MUT or otherwise with unambiguous field name.",
            "Tile count and tile bounds match the accepted pre-call T/P/PB/MU/MUB/PH/PHB/W text set.",
            "Existing T_STATE/P_STATE/PB/MU_STATE/MUB/MUT/W_STATE/PH_STATE/PHB values remain numerically identical to the accepted pre-call dump.",
            "QVAPOR has shape [44,66,159] after assembly, finite values, and no reuse of post-RK/pre-halo artifacts.",
            "Validator reports gpu_used=false and same_boundary_qvapor_truth_exists=true only for this boundary.",
        ],
    }


def theta_contract() -> dict[str, Any]:
    return {
        "wrf_t_state_contract": (
            "WRF grid%t_2/T is perturbation theta; with use_theta_m=1, "
            "adjust_tempqv treats it as moist-theta perturbation theta_m - 300."
        ),
        "evidence": [
            "nest_init_utils.F:852-853 divides th+300 by (1+R_v/R_d*qv) to recover dry temperature input when use_theta_m=1.",
            "nest_init_utils.F:869-877 converts dry thloc+dth back to (theta_m)-300 when use_theta_m=1.",
            "solve_em.F:515-521 comments that use_theta_m selects moist theta and needs old Qv.",
        ],
        "next_proof_compare": {
            "direct_wrf_field": "WRF T_STATE = grid%t_2 = theta_m - 300 at this boundary when use_theta_m=1",
            "current_jax_dry_state_compare": (
                "If JAX State.theta remains dry full theta, compare State.theta to "
                "(WRF T_STATE + 300)/(1 + R_v/R_d * WRF QVAPOR), and separately compare "
                "State.theta * (1 + R_v/R_d * State.qv) - 300 to WRF T_STATE."
            ),
            "do_not_claim_yet": (
                "Do not decide a production State.theta semantic change from post-RK QVAPOR; "
                "same-boundary pre-call QVAPOR is required."
            ),
        },
    }


def build_payload() -> dict[str, Any]:
    git_head = run(["git", "rev-parse", "HEAD"])
    git_branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    ancestor = run(["git", "merge-base", "--is-ancestor", ANCESTOR, "HEAD"])

    accepted = accepted_precall_schema()
    text_qvapor_files = rg_qvapor_files(candidate_text_roots())
    text_artifacts = group_text_qvapor_artifacts(text_qvapor_files)
    npz_artifacts = [inspect_npz(path) for path in candidate_npz_files()]
    qvapor_artifacts = [
        *[a for a in text_artifacts if a.get("contains_qvapor")],
        *[a for a in npz_artifacts if a.get("contains_qvapor")],
    ]
    same_boundary = [a for a in qvapor_artifacts if a.get("same_boundary_as_precall")]
    verdict = (
        "STEP1_QVAPOR_PRECALL_TRUTH_EXISTS"
        if same_boundary
        else "STEP1_QVAPOR_PRECALL_TRUTH_MISSING_SAVEPOINT_SPEC_READY"
    )

    payload = {
        "verdict": verdict,
        "gpu_used": False,
        "environment": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "script_imported_jax": False,
            "script_used_gpu": False,
        },
        "target": {
            "surface": TARGET_SURFACE,
            "domain_id": TARGET_DOMAIN,
            "step": TARGET_STEP,
            "rk_step": TARGET_RK,
        },
        "git": {
            "head": git_head["stdout"],
            "branch": git_branch["stdout"],
            "ancestor_required": ANCESTOR,
            "ancestor_check_returncode": ancestor["returncode"],
            "ancestor_is_present": ancestor["returncode"] == 0,
        },
        "accepted_precall_schema": accepted,
        "truth_inventory": {
            "text_roots_scanned": [str(p) for p in candidate_text_roots()],
            "text_qvapor_file_count": len(text_qvapor_files),
            "qvapor_artifacts": qvapor_artifacts,
            "same_boundary_qvapor_truth_exists": bool(same_boundary),
            "same_boundary_candidates": same_boundary,
        },
        "source_evidence": source_evidence(),
        "theta_contract": theta_contract(),
        "proposed_wrf_savepoint": savepoint_spec() if not same_boundary else None,
        "classification": {
            "accepted_pre_call_text_has_qvapor": accepted["contains_qvapor_at_precall_surface"],
            "existing_qvapor_truth_is_only_post_rk_or_different_boundary": not bool(same_boundary)
            and bool(qvapor_artifacts),
            "production_fix_supported": False,
            "reason_production_fix_not_supported": (
                "Same-boundary QVAPOR is missing; existing QVAPOR truth is post-RK/pre-halo."
            ),
        },
        "validation_commands": [
            "python -m py_compile proofs/v014/step1_qvapor_precall_truth_schema.py",
            "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_qvapor_precall_truth_schema.py",
            "python -m json.tool proofs/v014/step1_qvapor_precall_truth_schema.json >/tmp/step1_qvapor_precall_truth_schema.validated.json",
            "git diff -- src/gpuwrf",
        ],
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
            "proposed_wrf_savepoint": str(OUT_SAVEPOINT) if not same_boundary else None,
        },
    }
    return payload


def render_inventory_table(artifacts: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Artifact | Boundary | Step/RK | Shape | Same-boundary? | Classification |",
        "|---|---|---:|---|---:|---|",
    ]
    if not artifacts:
        lines.append("| none | none | none | none | no | no QVAPOR artifacts found |")
        return lines
    for artifact in artifacts:
        if artifact["artifact_type"] == "npz":
            label = artifact["path"]
            shape = artifact.get("qvapor", {}).get("shape") if artifact.get("qvapor") else None
            count = "1 npz"
        else:
            label = artifact["root"]
            shape = artifact.get("qvapor_shape_zyx")
            count = f"{artifact.get('tile_file_count')} text tiles"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{count}: `{label}`",
                    f"`{artifact.get('surface')}`",
                    f"{artifact.get('step')}/{artifact.get('rk_step')}",
                    f"`{shape}`",
                    "`yes`" if artifact.get("same_boundary_as_precall") else "`no`",
                    f"`{artifact.get('classification')}`",
                ]
            )
            + " |"
        )
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    accepted = payload["accepted_precall_schema"]
    inv = payload["truth_inventory"]
    lines = [
        "# V0.14 Step-1 QVAPOR Pre-Call Truth Schema",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- GPU used: `{str(payload['gpu_used']).lower()}`.",
        f"- Target WRF boundary: `{TARGET_SURFACE}`, domain `{TARGET_DOMAIN}`, step `{TARGET_STEP}`, rk `{TARGET_RK}`.",
        f"- Accepted pre-call text files: `{accepted['file_count']}` under `{accepted['root']}`.",
        f"- Accepted pre-call mass schema: `{accepted['record_schemas'].get('MASS_PREPART')}`.",
        f"- Accepted pre-call W/PH schema: `{accepted['record_schemas'].get('WPH_PREPART')}`.",
        f"- QVAPOR at accepted pre-call boundary: `{accepted['contains_qvapor_at_precall_surface']}`.",
        f"- Existing QVAPOR truth artifact groups: `{len(inv['qvapor_artifacts'])}`.",
        "",
        "## QVAPOR Inventory",
        "",
        *render_inventory_table(inv["qvapor_artifacts"]),
        "",
        "## Boundary Evidence",
        "",
        "- The accepted pre-call hook is immediately before `CALL first_rk_step_part1` in the instrumented Step-1 WRF tree.",
        "- The accepted dump schema writes `T_STATE/P_STATE/PB/MU_STATE/MUB/MUT` and `W_STATE/PH_STATE/PHB`, but not `QVAPOR`.",
        "- The only QVAPOR-bearing Step-1 truth found is `post_after_all_rk_steps_pre_halo` with `rk_step 4`, plus the promoted NPZ from that same boundary.",
        "",
        "## Theta Contract",
        "",
        "- WRF `grid%t_2` is perturbation theta; with `use_theta_m=1`, `adjust_tempqv` stores moist-theta perturbation (`theta_m - 300`).",
        "- For the next proof, same-boundary WRF `QVAPOR` is required to convert between WRF moist theta and any dry `State.theta` convention.",
        "- If JAX `State.theta` remains dry full theta, compare it to `(WRF T_STATE + 300)/(1 + R_v/R_d * WRF QVAPOR)` and compare derived JAX `theta_m - 300` to WRF `T_STATE`.",
        "",
        "## Next Action",
        "",
        "Add a minimal CPU-WRF savepoint extension at the existing `before_first_rk_step_part1_call` hook to emit `QVAPOR` from `moist(i,k,j,P_QV)` on the mass grid, preserving all existing pre-call fields for identity checks. Do not reuse post-RK/pre-halo QVAPOR for the live-nest theta/debug proof.",
    ]
    return "\n".join(lines) + "\n"


def render_review(payload: dict[str, Any]) -> str:
    return (
        "# Review: V0.14 Step-1 QVAPOR Pre-Call Truth Schema\n\n"
        f"- objective: establish whether same-boundary WRF pre-call `QVAPOR` truth exists for `{TARGET_SURFACE}`.\n"
        "- files changed: `proofs/v014/step1_qvapor_precall_truth_schema.py`, `.json`, `.md`, "
        "`.agent/reviews/2026-06-09-v014-step1-qvapor-precall-truth-schema.md`, and proposed savepoint spec.\n"
        "- commands run: see validation section in the JSON/Markdown proof object.\n"
        f"- proof objects produced: `{OUT_JSON}`, `{OUT_MD}`, `{OUT_REVIEW}`, `{OUT_SAVEPOINT}`.\n"
        f"- verdict: `{payload['verdict']}`.\n"
        "- unresolved risks: no production theta fix is justified until the minimal WRF savepoint emits same-boundary `QVAPOR`.\n"
        "- next decision needed: run the proposed CPU-WRF savepoint emitter, then rerun the theta/debug proof against same-boundary `T_STATE` and `QVAPOR`.\n"
    )


def render_savepoint_markdown(spec: dict[str, Any]) -> str:
    fields = spec["fields"]
    checks = "\n".join(f"- {item}" for item in spec["acceptance_checks"])
    return (
        "# Proposed WRF Savepoint: Step-1 Pre-Call QVAPOR\n\n"
        f"Boundary: `{TARGET_SURFACE}` in `{spec['file']}::{spec['function']}`.\n\n"
        "Place the capture at the existing pre-call hook, after `IF (coupler_on) CALL cpl_settime( curr_secs2 )` "
        "and before `CALL first_rk_step_part1`.\n\n"
        "Minimal schema change:\n\n"
        f"- Keep mass fields: `{fields['mass_existing']}`.\n"
        f"- Add mass field: `{fields['mass_add'][0]}` from `{fields['mass_expression']}`.\n"
        f"- Keep W/PH fields unchanged: `{fields['wph_existing_unchanged']}`.\n"
        f"- Add optional header: `{fields['optional_header']}`.\n\n"
        "Shape contract:\n\n"
        f"- Mass `QVAPOR`: `{spec['shape_contract']['d02_mass_shape_zyx']}` in assembled `(z,y,x)` order.\n"
        f"- W/PH remains `{spec['shape_contract']['d02_wph_shape_zyx']}`.\n"
        f"- Tile policy: `{spec['shape_contract']['tile_policy']}`.\n\n"
        "Acceptance checks:\n\n"
        f"{checks}\n"
    )


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    if payload.get("proposed_wrf_savepoint"):
        OUT_SAVEPOINT.parent.mkdir(parents=True, exist_ok=True)
        OUT_SAVEPOINT.write_text(render_savepoint_markdown(payload["proposed_wrf_savepoint"]), encoding="utf-8")


def main() -> int:
    payload = build_payload()
    write_outputs(payload)
    print(f"verdict={payload['verdict']}")
    print(f"same_boundary_qvapor_truth_exists={payload['truth_inventory']['same_boundary_qvapor_truth_exists']}")
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
