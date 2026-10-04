#!/usr/bin/env python3
"""Re-measure the SP2 five-category partition on the POST-fix authority.

Phase 1 of sprint 2026-07-19-v0234-fable-sp2-residual-closure.

Applies the exact frozen comparator method of
``scripts/v0234_gpt_operand_attribution.py`` (whose pure functions are
imported, not reimplemented) to the sealed POST-QKE-fix 111-array
single-authority CPU capture, against the sealed 462-file pristine WRF dump
tree.  Pure NumPy comparator: no JAX, no GPU, no WRF/MPI, no production
adapter invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GPT_COMPARATOR = REPO / "scripts/v0234_gpt_operand_attribution.py"
POSTFIX_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_gpt_postfix_sp2_a73aef522fbdca9c/"
    "cpu-adapter/single-authority-v1"
)
POSTFIX_ARCHIVE = POSTFIX_ROOT / "single-authority-capture.npz"
POSTFIX_MANIFEST = POSTFIX_ROOT / "manifest.json"
EXPECTED_POSTFIX_ARCHIVE = (
    "0f778d51658148fa2d482493fd8fae8a15c5862c2052e6d5942e839e16a850e0"
)
EXPECTED_WRF_TREE = (
    "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b"
)


class Failure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gpt_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "v0234_gpt_operand_attribution", GPT_COMPARATOR
    )
    if spec is None or spec.loader is None:
        raise Failure("GPT_COMPARATOR_IMPORT")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        print(f"REFUSE:OUTPUT_NOT_FRESH:{output}", file=sys.stderr)
        return 74

    try:
        gpt = load_gpt_module()
        git_head = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
            text=True, stdout=subprocess.PIPE,
        ).stdout.strip()

        # --- validate post-fix capture authority (fail closed) ---
        actual_archive = sha256_file(POSTFIX_ARCHIVE)
        if actual_archive != EXPECTED_POSTFIX_ARCHIVE:
            raise Failure(f"POSTFIX_ARCHIVE_HASH:{actual_archive}")
        manifest = json.loads(POSTFIX_MANIFEST.read_text(encoding="utf-8"))
        if gpt.canonical_without_self(manifest) != manifest.get(
            "canonical_payload_sha256"
        ):
            raise Failure("POSTFIX_MANIFEST_CANONICAL")
        authority = manifest.get("authority", {})
        if not (
            manifest.get("passed") is True
            and manifest.get("single_adapter_invocation") is True
            and authority.get("adapter_invocations") == 1
            and authority.get("gpu_actions") == 0
            and authority.get("wrf_or_mpi_executions") == 0
            and "restart=False" in authority.get("adapter_entry", "")
            and "first_timestep=True" in authority.get("adapter_entry", "")
        ):
            raise Failure("POSTFIX_AUTHORITY")
        declared = manifest["archive"]["arrays"]
        order = manifest["archive"]["array_order"]

        capture: dict[str, np.ndarray] = {}
        with np.load(POSTFIX_ARCHIVE, allow_pickle=False) as archive:
            if archive.files != order or set(archive.files) != set(declared):
                raise Failure("POSTFIX_ARCHIVE_INVENTORY")
            for name in archive.files:
                value = gpt.finite(name, archive[name])
                observed = {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "nbytes": value.nbytes,
                    "logical_c_bitpayload_sha256": gpt.array_sha(value),
                }
                if observed != declared[name]:
                    raise Failure(f"POSTFIX_ARRAY_DRIFT:{name}")
                capture[name] = np.ascontiguousarray(value)

        # --- validate WRF dump tree (fail closed) ---
        tree: dict[str, str] = {}
        for path in sorted(gpt.WRF_ROOT.rglob("*")):
            if path.is_symlink():
                raise Failure(f"WRF_TREE_SYMLINK:{path}")
            if path.is_file():
                tree[path.relative_to(gpt.WRF_ROOT).as_posix()] = sha256_file(path)
        tree_sha = hashlib.sha256(json.dumps(
            tree, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        if len(tree) != 462 or tree_sha != EXPECTED_WRF_TREE:
            raise Failure(f"WRF_TREE_DRIFT:{len(tree)}:{tree_sha}")
        reader, reader_record = gpt.import_dump_reader()

        column_tags = (
            "bc_lower_operands", "bc_rho", "bc_dfm", "bc_u", "bc_v",
            "bc_dtz", "bc_rhoz", "bc_kmdz", "bc_s_aw",
            "bc_s_awu", "bc_s_awv",
            "driver_qke_entry", "driver_rho", "driver_dz",
            "mix_qke_initialized", "mix_qke_turbulence_input",
            "mix_qke_after_predict", "mix_el_for_dfm", "mix_dfm",
            "mix_sm_for_dfm", "driver_km_exit",
            *(f"solve_{component}_{term}" for component in ("u", "v")
              for term in ("a", "b", "c", "d", "x")),
        )
        wrf = {tag: reader.columns(tag) for tag in column_tags}
        for tag in ("ust", "hfx", "qfx", "wspd"):
            wrf[f"surface_{tag}"] = gpt.load_outer2(reader, tag)
        wrf["rublten"] = reader.outer3("rublten_exit")
        wrf["rvblten"] = reader.outer3("rvblten_exit")
        lower = wrf["bc_lower_operands"]

        proof: dict[str, Any] = {
            "schema": "wrfgpu2-v0234-fable-sp2-postfix-partition-v1",
            "generated_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "backend": "numpy-cpu",
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "production_adapter_invocations_in_this_comparator": 0,
            "method": (
                "Frozen comparator functions imported from "
                "scripts/v0234_gpt_operand_attribution.py, applied to the "
                "sealed POST-QKE-fix single-authority capture."
            ),
            "input_evidence": {
                "git_head": git_head,
                "gpt_comparator_sha256": sha256_file(GPT_COMPARATOR),
                "postfix_archive": {
                    "path": str(POSTFIX_ARCHIVE),
                    "sha256": actual_archive,
                },
                "postfix_manifest_canonical_sha256": manifest[
                    "canonical_payload_sha256"
                ],
                "wrf_dump_tree": {"file_count": len(tree), "sha256": tree_sha},
                "reader": reader_record,
            },
            "components": {},
        }

        combined_error = []
        combined_vectors: dict[str, list[np.ndarray]] = {
            name: [] for name in gpt.CATEGORIES
        }
        for component, field in (("u", "rublten"), ("v", "rvblten")):
            port, pristine = gpt.make_operands(capture, wrf, component)
            port_eval = gpt.evaluate(port)
            authentic = capture[f"adapter_r{component}blten"] - wrf[field]
            capture_closure = gpt.metrics(
                port_eval["tendency"], capture[f"adapter_r{component}blten"]
            )
            if capture_closure["rms"] > 5.0e-13:
                raise Failure(f"CAPTURE_FORMULA_CLOSURE:{component}")
            partition, vectors = gpt.coefficient_partition(
                port, pristine, authentic
            )
            proof["components"][component] = {
                "authentic_postfix_adapter_vs_pristine_wrf": (
                    gpt.vertical_metrics(
                        capture[f"adapter_r{component}blten"], wrf[field]
                    )
                ),
                "capture_formula_closure": capture_closure,
                "five_category_partition": partition,
                "mass_flux_partition": gpt.mass_flux_detail(
                    port, pristine, authentic
                ),
                "mixing_partition": gpt.mixing_detail(
                    port, pristine, authentic
                ),
            }
            combined_error.append(authentic)
            for name in gpt.CATEGORIES:
                combined_vectors[name].append(vectors[name])

        error_both = np.stack(combined_error, axis=0)
        proof["combined_vector_shapley"] = {
            name: gpt.projection(
                np.stack(combined_vectors[name], axis=0), error_both
            )
            for name in gpt.CATEGORIES
        }
        proof["combined_vector_shapley_ranking_by_absolute_projection"] = sorted(
            gpt.CATEGORIES,
            key=lambda name: abs(proof["combined_vector_shapley"][name][
                "signed_projection_fraction_of_authentic_error_sse"
            ]),
            reverse=True,
        )

        proof["stage_differences_postfix"] = {
            "pbl_entry": {
                "rho": gpt.vertical_metrics(
                    capture["mean_tendencies_rho"], wrf["driver_rho"]
                ),
                "dz": gpt.vertical_metrics(
                    capture["mean_tendencies_dz"], wrf["driver_dz"]
                ),
                "u": gpt.vertical_metrics(
                    capture["mean_tendencies_state_u"], wrf["bc_u"]
                ),
                "v": gpt.vertical_metrics(
                    capture["mean_tendencies_state_v"], wrf["bc_v"]
                ),
                "qke_entry": gpt.vertical_metrics(
                    capture["entry_state_qke"], wrf["driver_qke_entry"]
                ),
            },
            "qke_chain": {
                "initialized_qke": gpt.vertical_metrics(
                    capture["initialized_state_qke"], wrf["mix_qke_initialized"]
                ),
                "turbulence_input_qke": gpt.vertical_metrics(
                    capture["turbulence_qke_input"],
                    wrf["mix_qke_turbulence_input"],
                ),
                "predicted_qke": gpt.vertical_metrics(
                    capture["qke_predict_qke_after"],
                    wrf["mix_qke_after_predict"],
                ),
            },
            "mixing_span": {
                "el": gpt.vertical_metrics(
                    capture["turbulence_el"], wrf["mix_el_for_dfm"]
                ),
                "dfm": gpt.vertical_metrics(
                    capture["turbulence_dfm"], wrf["mix_dfm"]
                ),
                "rhoz": gpt.vertical_metrics(
                    capture["mean_tendencies_rhoz"], wrf["bc_rhoz"]
                ),
                "kmdz_operational_floored": gpt.vertical_metrics(
                    capture["mean_tendencies_kmdz_floored"], wrf["bc_kmdz"]
                ),
                "kmdz_raw": gpt.vertical_metrics(
                    capture["mean_tendencies_kmdz_raw"], wrf["bc_kmdz"]
                ),
            },
            "mass_flux_span": {
                "s_aw": gpt.vertical_metrics(
                    capture["mass_flux_s_aw"], wrf["bc_s_aw"]
                ),
                "s_awu": gpt.vertical_metrics(
                    capture["mass_flux_s_awu"], wrf["bc_s_awu"]
                ),
                "s_awv": gpt.vertical_metrics(
                    capture["mass_flux_s_awv"], wrf["bc_s_awv"]
                ),
            },
            "surface_span": {
                "ustar": gpt.metrics(
                    capture["surface_terms_flux_ustar"], wrf["surface_ust"]
                ),
                "rhosfc": gpt.metrics(
                    capture["surface_terms_rhosfc"], lower[10]
                ),
                "wind": gpt.metrics(
                    capture["surface_terms_wind"], wrf["surface_wspd"]
                ),
            },
        }

        # Descriptive activity statistics for the mass-flux span: where is
        # each side's s_aw nonzero, and how large is it there?
        activity: dict[str, Any] = {}
        for name, port_field, wrf_field in (
            ("s_aw", "mass_flux_s_aw", "bc_s_aw"),
            ("s_awu", "mass_flux_s_awu", "bc_s_awu"),
            ("s_awv", "mass_flux_s_awv", "bc_s_awv"),
        ):
            port_arr = np.asarray(capture[port_field], dtype=np.float64)
            wrf_arr = np.asarray(wrf[wrf_field], dtype=np.float64)
            port_cols = np.any(port_arr != 0.0, axis=0)
            wrf_cols = np.any(wrf_arr != 0.0, axis=0)
            activity[name] = {
                "port_nonzero_fraction": float(np.mean(port_arr != 0.0)),
                "wrf_nonzero_fraction": float(np.mean(wrf_arr != 0.0)),
                "port_active_columns": int(np.sum(port_cols)),
                "wrf_active_columns": int(np.sum(wrf_cols)),
                "columns_port_only": int(np.sum(port_cols & ~wrf_cols)),
                "columns_wrf_only": int(np.sum(wrf_cols & ~port_cols)),
                "columns_both": int(np.sum(port_cols & wrf_cols)),
                "port_rms": gpt.rms(port_arr),
                "wrf_rms": gpt.rms(wrf_arr),
                "port_max_abs": float(np.max(np.abs(port_arr))),
                "wrf_max_abs": float(np.max(np.abs(wrf_arr))),
                "rms_on_both_active_columns": {
                    "port": gpt.rms(port_arr[:, port_cols & wrf_cols])
                    if np.any(port_cols & wrf_cols) else None,
                    "wrf": gpt.rms(wrf_arr[:, port_cols & wrf_cols])
                    if np.any(port_cols & wrf_cols) else None,
                },
            }
        proof["mass_flux_activity"] = activity

        gpt.atomic_json(output, proof)
        print(json.dumps({
            "passed": True,
            "output": str(output),
            "authentic_rms": {
                component: proof["components"][component][
                    "authentic_postfix_adapter_vs_pristine_wrf"
                ]["rms"] for component in ("u", "v")
            },
            "combined_ranking": proof[
                "combined_vector_shapley_ranking_by_absolute_projection"
            ],
            "combined_fractions": {
                name: proof["combined_vector_shapley"][name][
                    "signed_projection_fraction_of_authentic_error_sse"
                ] for name in gpt.CATEGORIES
            },
            "canonical_payload_sha256": gpt.canonical_without_self(proof),
        }, sort_keys=True, indent=1))
        return 0
    except Exception as exc:  # fail closed, print reason
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    sys.exit(main())
