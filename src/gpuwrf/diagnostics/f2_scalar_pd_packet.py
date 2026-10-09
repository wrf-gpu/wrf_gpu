"""One qc post-RK extremum accounting tuple; no operator recomputation.

This diagnostic deliberately has no PD stencils, parent-SINT or population
proof. An ON program requires its own admission. None is the OFF bypass.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

HEADERS = tuple("version present own_step flat_index nz ny nx rk_stage horizontal_ring0 spec_zone nested root_flow_qc native_seen pd_preload physics_sc_present other_sc_present seam_complete".split())
VALUES = tuple("post_rk_qc physical_origin_qc rk1_origin_qc pd_preloaded_qc native_stage_qc physical_origin_mu rk1_origin_mu native_old_mu native_current_mu update_new_mu native_mub c1h c2h msftx msfty rdnw fnm fnp frozen_physics_sc frozen_other_sc native_pd_tendency merged_update_tendency before_flow_qc post_nondry_qc pre_mp_qc dt_rk pd_dt pd_rdx pd_rdy actual_mass_old actual_mass_new actual_inverse_mass_new actual_numerator actual_q_update".split())
PACKET_BYTES = 4 * (len(HEADERS) + len(VALUES) + 4)


class QcPacket(NamedTuple):
    header: object
    values: object
    clocks: object


def enabled():
    return os.environ.get("GPUWRF_F2_QC_PACKET", "0") == "1"


def initial_packet():
    import jax.numpy as jnp
    return QcPacket(jnp.zeros(len(HEADERS), jnp.int32),
                    jnp.zeros(len(VALUES), jnp.float32),
                    jnp.zeros(4, jnp.uint32))


def capture(packet, smaller, context, step, index, post_rk):
    """Reuse census's exact strict-smaller predicate/index. Gather on device."""
    if packet is None:
        return None
    if context is None or "native_args" not in context or "update" not in context:
        raise ValueError("qc packet needs actual native PD and final-RK update context")
    if not context.get("pd_preload"):
        raise ValueError("qc packet is restricted to the admitted final-RK PD-preload path")
    import jax
    import jax.numpy as jnp
    args = context["native_args"]
    species = context["species"].index("qc")
    update = context["update"]
    nz, ny, nx = post_rk.shape
    if any(a.dtype != jnp.float32 for a in args):
        raise TypeError("actual native PD arguments must be REAL32")

    def gather(_):
        k, j, i = index // (ny * nx), index // nx % ny, index % nx

        def point(a):
            if a.dtype != jnp.float32:
                raise TypeError("thin qc accounting input is not REAL32")
            return a[k, j, i] if a.ndim == 3 else a[j, i]

        def source(name):
            a = context.get(name)
            return jnp.float32(0) if a is None else point(a)

        physical, origin = context["physical_origin"], context["rk1_origin"]
        data = (point(post_rk), point(physical.qc), point(origin.qc),
                point(args[1][species]), point(args[0][species]), point(physical.mu_total), point(origin.mu_total),
                point(args[6]), point(args[5]), point(context["mu_update"]),
                jnp.float32(0), args[7][k], args[8][k], point(args[9]), point(args[10]),
                args[11][k], args[12][k], args[13][k], source("physics_sc"), source("other_sc"),
                point(context["native_tendency"][species]), point(context["merged_tendency"]),
                point(context["before_flow"]), jnp.float32(0), jnp.float32(0),
                jnp.float32(context["dt_rk"]), jnp.float32(context["native_params"]["dt"]),
                jnp.float32(context["native_params"]["rdx"]), jnp.float32(context["native_params"]["rdy"]),
                point(update["mass_old"]), point(update["mass_new"]),
                point(update["inverse_mass_new"]), point(update["numerator"]), point(update["result"]))
        header = jnp.stack(tuple(jnp.asarray(x, jnp.int32) for x in (
            1, 1, step, index, nz, ny, nx, 3,
            (j == 0) | (j == ny - 1) | (i == 0) | (i == nx - 1),
            context["spec_zone"], context["nested"], context["root_flow_qc"], 1,
            context["pd_preload"], context.get("physics_sc") is not None,
            context.get("other_sc") is not None, 0)))
        clock_arrays = (context["boundary_lead_seconds"], context["physics_lead_seconds"])
        if any(a.dtype != jnp.float64 for a in clock_arrays):
            raise TypeError("clock provenance must use existing f64 clocks")
        clocks = jnp.concatenate(tuple(jax.lax.bitcast_convert_type(a, jnp.uint32).reshape(2)
                                       for a in clock_arrays))
        return QcPacket(header, jnp.stack(data), clocks)

    return jax.lax.cond(smaller, gather, lambda _: packet, None)


def finish_seam(packet, step, post_nondry, pre_mp):
    """Finish only the SAME selected call, before MP; never touch State."""
    if packet is None:
        return None
    import jax
    import jax.numpy as jnp
    if post_nondry.dtype != jnp.float32 or pre_mp.dtype != jnp.float32:
        raise TypeError("qc seam arrays must preserve REAL32")
    valid = (packet.header[HEADERS.index("present")] == 1) & (packet.header[HEADERS.index("own_step")] == step)

    def finish(_):
        flat = packet.header[HEADERS.index("flat_index")]
        values = packet.values.at[VALUES.index("post_nondry_qc")].set(post_nondry.reshape(-1)[flat])
        values = values.at[VALUES.index("pre_mp_qc")].set(pre_mp.reshape(-1)[flat])
        return QcPacket(packet.header.at[HEADERS.index("seam_complete")].set(jnp.int32(1)), values, packet.clocks)

    return jax.lax.cond(valid, finish, lambda _: packet, None)


# References only at the existing segment boundary. No materialization here.
_PENDING = {}


def retain_terminal(output_dir, carries, bundles, own_steps):
    packets = {n: c.census.f2_qc_packet for n, c in carries.items()}
    _PENDING[str(Path(output_dir).resolve())] = (packets, dict(own_steps), {
        n: {"dt_s": float(b.namelist.dt_s), "dx": float(b.namelist.grid.projection.dx_m),
            "dy": float(b.namelist.grid.projection.dy_m)} for n, b in bundles.items()})


def export_after_normal_return(output_dir, normal_receipt, bindings):
    """Producer calls once AFTER normal pipeline return, which follows joins.

    Receipts remain quarantined until actual-source/program/reader admission.
    Caller placement is a required source-review item, never inferred from time.
    """
    import hashlib
    import jax
    import numpy as np
    root = Path(output_dir).resolve()
    if not isinstance(normal_receipt, dict) or not normal_receipt:
        raise ValueError("normal joined pipeline receipt required")
    for key in ("runtime_source", "numeric_source", "input_identity", "actual_pid"):
        if not bindings.get(key):
            raise ValueError("missing actual binding: " + key)
    packets, steps, metrics = _PENDING.pop(str(root))
    if steps != {"d01": 200, "d02": 600, "d03": 1800}:
        raise ValueError("packet export is only the admitted ONE final 3h work")
    host = jax.device_get(packets)
    arrays, candidates = {}, []
    for domain, packet in sorted(host.items()):
        if packet is None:
            raise ValueError("missing qc packet slot: " + domain)
        if int(packet.header[HEADERS.index("present")]):
            if int(packet.header[HEADERS.index("seam_complete")]) != 1:
                raise ValueError("selected packet has unfinished same-call seam")
            candidates.append((float(packet.values[VALUES.index("post_rk_qc")]), domain))
        for field, a in zip(QcPacket._fields, packet):
            arrays[domain + "." + field] = np.asarray(a)
    path = root / "f2_qc_packet.npz"
    np.savez(path, **arrays)
    if path.stat().st_size > 1048576:
        raise ValueError("terminal packet exceeds 1MiB")
    manifest = {"schema_version": 1, "status": "QUARANTINED_ACTUAL_POSTGATES_PENDING",
                "bindings": bindings, "own_steps": steps, "metrics": metrics,
                "header_names": HEADERS, "value_names": VALUES, "packet_bytes": PACKET_BYTES,
                "selected_domain": min(candidates)[1] if candidates else None,
                "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "coverage": "qc postRK accounting only; PD stencils/parent SINT/population unsupported",
                "production_equivalent": False}
    (root / "f2_qc_packet.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
