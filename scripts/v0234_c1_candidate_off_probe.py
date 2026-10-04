"""Deterministic CPU probe for the v0.23.4 scalar candidate-off program.

Run this exact script against the pre-candidate and repaired source trees.  It
seeds active Thompson Ni/Nr gradients in a live frozen-boundary child, selects
the released scalar options 0/0, executes one ordinary operational step, and
hashes every carry leaf.  Equality is the binding C1 candidate-off gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gpuwrf.runtime.operational_mode as operational
from gpuwrf.runtime.domain_tree import DomainTree, _operational_force
from gpuwrf.validation.moving_nest_testbed import build_nested_pair


RELEASED_REFERENCE = {
    "source_commit": "21d3cdc93080c8b0a247577b69b096282dd5d42f",
    "operational_source_sha256": (
        "68efe76b9d1f91860e9a49a6e6c9ab9e573986dd11a47677fa161badb3a79282"
    ),
    "input_tree_sha256": (
        "d4d5d1c9f676ef26940d0f2c3a38a624266fb380259623ad00f1645cee029c45"
    ),
    "output_tree_sha256": (
        "f6bd47b87c10866cbd6daba41069fc9e00c4e34a573a7d78d92d5bfbe90df7bb"
    ),
    "Ni_sha256": "8623838c2cb5085fb02c5f426137200a939e02c42b745acbfbc67fe04e33fd00",
    "Nr_sha256": "3aeae3b05adf156470552133374cb6009e98e38ea9cc807b8f195f05bc07524e",
}

BROKEN_CANDIDATE = {
    "source_commit": "b8d5484b5d6497db585abf211324caf44b3968d3",
    "operational_source_sha256": (
        "698478eab5b279793e21a41c3f84b183c5a3d52f8a5ec2bf068878a9a2b0e33d"
    ),
    "input_tree_sha256": (
        "d4d5d1c9f676ef26940d0f2c3a38a624266fb380259623ad00f1645cee029c45"
    ),
    "output_tree_sha256": (
        "c45d300b332b83a959634be36c715c1bc329e14a4aed26ba23649a2030a1cee0"
    ),
    "Ni_sha256": "e021454bc9dbcbec6e3d2a1870f9076f498cbe8e4c663c88c2a22e79837f69e9",
    "Nr_sha256": "6b9796b2f6941040b53b2881095e0265c6f7ed52231ac79ea7bea26e01882825",
}


def _array_row(value) -> dict[str, object]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "finite": bool(
            np.all(np.isfinite(array))
            if np.issubdtype(array.dtype, np.floating)
            else True
        ),
    }


def _tree_sha256(value) -> tuple[str, list[dict[str, object]]]:
    rows = []
    digest = hashlib.sha256()
    for index, leaf in enumerate(jax.tree_util.tree_leaves(value)):
        array = np.asarray(leaf)
        row = _array_row(array)
        row["index"] = index
        rows.append(row)
        digest.update(str(index).encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest(), rows


def run_probe() -> dict[str, object]:
    if jax.default_backend() != "cpu":
        raise RuntimeError(
            f"CPU-only probe resolved backend {jax.default_backend()!r}"
        )
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
    state = child_bundle.state
    nz, ny, nx = state.Ni.shape
    zz, yy, xx = jnp.meshgrid(
        jnp.arange(nz, dtype=state.Ni.dtype),
        jnp.arange(ny, dtype=state.Ni.dtype),
        jnp.arange(nx, dtype=state.Ni.dtype),
        indexing="ij",
    )
    ni = (1.0e4 + 7.0 * xx + 11.0 * yy + 13.0 * zz).astype(state.Ni.dtype)
    nr = (2.0e4 + 17.0 * xx + 19.0 * yy + 23.0 * zz).astype(state.Nr.dtype)
    state = state.replace(Ni=ni, Nr=nr)
    namelist = dataclasses.replace(
        child_bundle.namelist,
        grid=child_grid,
        acoustic_substeps=1,
        moist_adv_opt=0,
        scalar_adv_opt=0,
        use_flux_advection=True,
        run_physics=False,
        boundary_config=dataclasses.replace(
            child_bundle.namelist.boundary_config,
            nested_frozen_wrf_boundary_bundle=True,
        ),
    )
    candidate_bundle = dataclasses.replace(
        child_bundle,
        grid=child_grid,
        state=state,
        namelist=namelist,
    )
    candidate_tree = DomainTree.from_domains(
        tree.hierarchy,
        {"d01": parent_bundle, "d02": candidate_bundle},
        feedback_enabled=False,
    )
    parent = operational._initial_carry_for_run(
        parent_bundle.state, parent_bundle.namelist
    )
    child = operational._initial_carry_for_run(state, namelist)
    forced = _operational_force(candidate_tree.edges["d01"][0], parent, child)
    input_sha, input_rows = _tree_sha256(forced)
    step = jax.jit(
        lambda value: operational._physics_boundary_step(
            value,
            namelist,
            jnp.asarray(1, dtype=jnp.int32),
            run_radiation=False,
        )
    )
    output = step(forced)
    jax.block_until_ready(output.state.theta)
    output_sha, output_rows = _tree_sha256(output)
    source = Path(operational.__file__).resolve()
    payload = {
        "schema": "gpuwrf.v0234.c1-candidate-off-cpu-probe.v1",
        "backend": jax.default_backend(),
        "operational_source": str(source),
        "operational_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "options": {"moist_adv_opt": 0, "scalar_adv_opt": 0},
        "nested_frozen_wrf_boundary_bundle": True,
        "seeded_fields": {"Ni": _array_row(ni), "Nr": _array_row(nr)},
        "input_tree_sha256": input_sha,
        "output_tree_sha256": output_sha,
        "input_leaf_count": len(input_rows),
        "output_leaf_count": len(output_rows),
        "output_all_finite": all(bool(row["finite"]) for row in output_rows),
        "output_fields": {
            "Ni": _array_row(output.state.Ni),
            "Nr": _array_row(output.state.Nr),
        },
        "released_reference": RELEASED_REFERENCE,
        "broken_candidate": BROKEN_CANDIDATE,
    }
    payload["candidate_off_identity_pass"] = bool(
        payload["input_tree_sha256"] == RELEASED_REFERENCE["input_tree_sha256"]
        and payload["output_tree_sha256"]
        == RELEASED_REFERENCE["output_tree_sha256"]
        and payload["output_fields"]["Ni"]["sha256"]
        == RELEASED_REFERENCE["Ni_sha256"]
        and payload["output_fields"]["Nr"]["sha256"]
        == RELEASED_REFERENCE["Nr_sha256"]
        and payload["output_all_finite"] is True
    )
    payload["broken_control_reproduced_as_nonidentity"] = bool(
        BROKEN_CANDIDATE["input_tree_sha256"]
        == RELEASED_REFERENCE["input_tree_sha256"]
        and BROKEN_CANDIDATE["output_tree_sha256"]
        != RELEASED_REFERENCE["output_tree_sha256"]
        and BROKEN_CANDIDATE["Ni_sha256"] != RELEASED_REFERENCE["Ni_sha256"]
        and BROKEN_CANDIDATE["Nr_sha256"] != RELEASED_REFERENCE["Nr_sha256"]
    )
    payload["verdict"] = (
        "C1_REPAIRED__CANDIDATE_OFF_MATCHES_RELEASED_FULL_CARRY_BYTES"
        if payload["candidate_off_identity_pass"]
        and payload["broken_control_reproduced_as_nonidentity"]
        else "C1_REPAIR_RED"
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["canonical_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_probe()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)
    return 0 if payload["verdict"].startswith("C1_REPAIRED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
