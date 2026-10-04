"""Lower-only debug probe: why did arm A's StableHLO differ from production?

Compares jit avals (shape/dtype/weak_type) between the initial d03 carry from
the production loader and the unpickled authenticated 8800 carry, and lowers
the production d03 one-step on both to find the exact HLO divergence cause.
Locked-GPU only (domain load requires a GPU device); no compile, no dispatch.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import v0234_nested_frozen_wrf_boundary_window as runner  # noqa: E402
from scripts import v0234_step9000_autotune_discriminator as disc  # noqa: E402

runner.CANDIDATE_COMMIT = disc.PRODUCTION_CANDIDATE_COMMIT
runner.CANDIDATE_TREE = disc.PRODUCTION_CANDIDATE_TREE

OUT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_step9000_autotune_discriminator_kimi1"
    "/aval-probe.json"
)


def _avals(runtime, carry):
    jax = runtime.jax
    rows = []
    for path, leaf in zip(
        [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(carry)[0]],
        jax.tree_util.tree_leaves(carry),
    ):
        aval = jax.typeof(leaf)
        rows.append({
            "path": path,
            "shape": list(aval.shape),
            "dtype": str(aval.dtype),
            "weak_type": bool(getattr(aval, "weak_type", False)),
        })
    return rows


def main() -> int:
    runtime = runner._import_runtime()
    jax = runtime.jax
    import time as _time
    work = OUT.parent / f"aval-probe-load-{int(_time.time())}"
    work.mkdir(parents=True, exist_ok=True)
    tree, names, carries0, dt_by_domain, _ = runtime.ordinary.load_corrected_tree(work)
    ic = carries0["d03"]
    host = pickle.loads(disc.CARRY_8800_PATH.read_bytes())
    rc = jax.device_put(host)

    avals_ic = _avals(runtime, ic)
    avals_rc = _avals(runtime, rc)
    mismatches = [
        {"index": i, "initial": a, "resumed": b}
        for i, (a, b) in enumerate(zip(avals_ic, avals_rc))
        if a != b
    ]

    namelist = tree.domains["d03"].namelist
    clock = runtime.build_clock_base(namelist)
    cadence = int(namelist.radiation_cadence_steps)
    jnp = runtime.jnp

    def lower_sha(carry, start_step):
        lowered = runtime._advance_chunk_fori.lower(
            carry,
            namelist,
            jnp.asarray(int(start_step), dtype=jnp.int32),
            clock,
            n_steps=1,
            cadence=cadence,
        )
        text = str(lowered.compiler_ir(dialect="stablehlo"))
        return hashlib.sha256(text.encode()).hexdigest(), len(text)

    sha_initial_9199, bytes_initial = lower_sha(ic, 9199)
    sha_resumed_9199, bytes_resumed = lower_sha(rc, 9199)

    # weak_type repair attempt: rebuild the resumed carry with the initial
    # carry's exact avals (dtype + weak_type) leaf by leaf.
    repaired = None
    sha_repaired_9199 = None
    repair_mismatches: list[dict] = []
    repair_note = "no mismatches"
    if mismatches:
        target_by_path = {row["path"]: row for row in avals_ic}

        def _repair(path, leaf):
            key = jax.tree_util.keystr(path)
            target = target_by_path[key]
            if bool(target["weak_type"]) and not bool(
                jax.typeof(leaf).weak_type
            ):
                try:
                    import jax.numpy as _jnp

                    return _jnp.asarray(
                        leaf, dtype=getattr(_jnp, target["dtype"])
                    ) + _jnp.zeros(
                        (), dtype=getattr(_jnp, target["dtype"])
                    )
                except Exception:  # noqa: BLE001 - probe reports, never masks
                    return leaf
            return leaf

        try:
            repaired = jax.tree_util.tree_map_with_path(_repair, rc)
            avals_repaired = _avals(runtime, repaired)
            repair_mismatches = [
                {"index": i, "initial": a, "repaired": b}
                for i, (a, b) in enumerate(zip(avals_ic, avals_repaired))
                if a != b
            ]
            sha_repaired_9199, _ = lower_sha(repaired, 9199)
            repair_note = "attempted"
        except Exception as exc:  # noqa: BLE001
            repair_note = f"repair-failed:{type(exc).__name__}:{exc}"

    payload = {
        "schema": "gpuwrf.v0234.step9000-aval-probe.v1",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:40],
        "repair_mismatches": repair_mismatches[:40],
        "repair_note": repair_note,
        "stablehlo_sha256_initial_carry_start9199": sha_initial_9199,
        "stablehlo_sha256_resumed_carry_start9199": sha_resumed_9199,
        "stablehlo_sha256_repaired_resumed_start9199": sha_repaired_9199,
        "stablehlo_bytes": bytes_initial,
        "stablehlo_bytes_resumed": bytes_resumed,
        "expected_production_stablehlo_sha256": disc.EXPECTED_STABLEHLO_SHA256,
        "initial_matches_production": sha_initial_9199 == disc.EXPECTED_STABLEHLO_SHA256,
        "resumed_matches_production": sha_resumed_9199 == disc.EXPECTED_STABLEHLO_SHA256,
        "repaired_matches_production": sha_repaired_9199 == disc.EXPECTED_STABLEHLO_SHA256,
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
