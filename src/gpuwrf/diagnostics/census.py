"""Resident pre-repair event and actual-work counters.

None is the disabled mode: no device leaves or operations. Only
``write_segment`` materializes counters, at an existing segment boundary.
Counts are cumulative uint64 to cover multi-day PROD runs without overflow.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

MOISTURE_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
BOUNDARY_FINITE_FIELDS = (
    "u", "v", "w", "theta", "p", "ph", "p_total", "ph_total",
    "p_perturbation", "ph_perturbation",
)
GUARDS = (
    "dynamics.theta", "dynamics.mu_total", "dynamics.mu_perturbation",
    *(f"dynamics.{name}" for name in MOISTURE_FIELDS),
    *(f"boundary.{name}" for name in BOUNDARY_FINITE_FIELDS),
    "boundary.qv", "boundary.mu_total", "boundary.mu_perturbation",
)
EVENTS = ("nonfinite", "out_of_range", "repaired")
WORK = ("steps", "acoustic_trips", "kf_calls", "radiation_tendency_calls",
        "radiation_surface_calls", "native_mass_guard_events",
        "radiation_init_calls", "radiation_output_calls", "output_writes")
WRITER_IO = ("radiation_output_calls", "output_writes")
F2_PHASES = ("step_entry", "post_rk", "pre_dynamics_guard", "pre_boundary_guard")
F2_FIELDS = MOISTURE_FIELDS + ("Ni", "Nr", "Ns", "Ng", "Nc", "Nn", "Nh", "nwfa", "nifa")
F2_EVENTS = ("negative", "nonfinite", "above_moisture_cap")
F2_REPAIRS = ("dynamics", "boundary")


class WriterIoLedger:
    """Host counts for successful writer persistence, including async callbacks.

    Attach as ``writer.census_io_ledger`` when census is enabled. Call
    ``record_persist(domain, radiation_output_calls=N)`` only after a file was
    actually persisted; N is the number of output radiation solves for that file.
    Final segment readout must follow the writer join when writes are async.
    """

    def __init__(self):
        self._counts = {}
        self._lock = Lock()

    def record_persist(self, domain: str, *, radiation_output_calls: int = 0):
        if not isinstance(radiation_output_calls, int) or radiation_output_calls < 0:
            raise ValueError("radiation_output_calls must be a nonnegative integer")
        with self._lock:
            counts = self._counts.setdefault(domain, dict.fromkeys(WRITER_IO, 0))
            counts["radiation_output_calls"] += radiation_output_calls
            counts["output_writes"] += 1

    def snapshot(self):
        with self._lock:
            return {domain: dict(counts) for domain, counts in self._counts.items()}


class DeviceCensus(NamedTuple):
    guards: jax.Array
    work: jax.Array
    f2_counts: jax.Array | None = None
    f2_first_step: jax.Array | None = None
    f2_first_index: jax.Array | None = None
    f2_first_value: jax.Array | None = None
    water_changes: jax.Array | None = None
    unknown_water_changes: jax.Array | None = None
    f2_min_negative: jax.Array | None = None
    f2_min_step: jax.Array | None = None
    f2_min_index: jax.Array | None = None
    f2_negative_location: jax.Array | None = None
    f2_field_observations: jax.Array | None = None
    f2_qc_packet: object = None


def enabled() -> bool:
    return os.environ.get("GPUWRF_CENSUS", "0").lower() in {"1", "true", "yes", "on"}


def initial_census() -> DeviceCensus:
    from gpuwrf.diagnostics.f2_scalar_pd_packet import enabled as f2_packet_enabled, initial_packet
    f2_packet = initial_packet() if f2_packet_enabled() else None
    shape = (len(F2_PHASES), len(F2_FIELDS), len(F2_EVENTS))
    return DeviceCensus(jnp.zeros((len(GUARDS), len(EVENTS)), jnp.uint64),
                        jnp.zeros((len(WORK),), jnp.uint64),
                        jnp.zeros(shape, jnp.uint64),
                        jnp.full(shape, -1, jnp.int32),
                        jnp.full(shape, -1, jnp.int32),
                        jnp.zeros(shape, jnp.float32),
                        jnp.zeros((len(F2_REPAIRS), len(MOISTURE_FIELDS), 2), jnp.float32),
                        jnp.zeros((len(F2_REPAIRS), len(MOISTURE_FIELDS)), jnp.uint64),
                        jnp.zeros(shape[:2], jnp.float32),
                        jnp.full(shape[:2], -1, jnp.int32),
                        jnp.full(shape[:2], -1, jnp.int32),
                        jnp.zeros((*shape[:2], 2), jnp.uint64),
                        jnp.zeros((*shape[:2], 3), jnp.uint64),
                        f2_packet)


def observe_f2_state(census, phase, state, step, *, pd_context=None):
    """Resident classification before any policy repairs; does not alter state."""
    if census is None or census.f2_counts is None:
        return census
    p = F2_PHASES.index(phase)
    f2_packet = census.f2_qc_packet
    counts, steps, indices, values = (census.f2_counts, census.f2_first_step,
                                    census.f2_first_index, census.f2_first_value)
    minima, min_steps, min_indices = (census.f2_min_negative, census.f2_min_step,
                                      census.f2_min_index)
    locations, observations = census.f2_negative_location, census.f2_field_observations
    for f, name in enumerate(F2_FIELDS):
        value = getattr(state, name, None)
        if observations is not None:
            # present, absent, present-but-not-REAL32; no unobserved zero is coverage.
            observations = observations.at[p, f, int(value is None)].add(jnp.uint64(1))
        if value is None:
            continue
        value = jnp.asarray(value)
        finite = jnp.isfinite(value)
        masks = (finite & (value < 0), ~finite,
                 finite & (value > 0.05) if name in MOISTURE_FIELDS else jnp.zeros_like(finite))
        for e, mask in enumerate(masks):
            count = jnp.sum(mask, dtype=jnp.uint64)
            index = jnp.argmax(mask.reshape(-1)).astype(jnp.int32)
            first = (count > 0) & (steps[p, f, e] < 0)
            counts = counts.at[p, f, e].add(count)
            steps = steps.at[p, f, e].set(jnp.where(first, jnp.asarray(step, jnp.int32), steps[p, f, e]))
            indices = indices.at[p, f, e].set(jnp.where(first, index, indices[p, f, e]))
            sample = value.reshape(-1)[index].astype(jnp.float32)
            values = values.at[p, f, e].set(jnp.where(first, sample, values[p, f, e]))
        if minima is not None:
            # REAL32 diagnostic only. Non-REAL32 inputs are explicitly marked
            # unsupported for an exact magnitude bound in the host summary.
            observations = observations.at[p, f, 2].add(jnp.uint64(value.dtype != jnp.float32))
            negative = masks[0]
            candidates = jnp.where(negative, value.astype(jnp.float32), jnp.float32(jnp.inf))
            index = jnp.argmin(candidates.reshape(-1)).astype(jnp.int32)
            minimum = candidates.reshape(-1)[index]
            smaller = minimum < minima[p, f]
            if f2_packet is not None and phase == "post_rk" and name == "qc":
                from gpuwrf.diagnostics.f2_scalar_pd_packet import capture
                f2_packet = capture(f2_packet, smaller, pd_context, step, index, value)
            minima = minima.at[p, f].set(jnp.where(smaller, minimum, minima[p, f]))
            min_steps = min_steps.at[p, f].set(jnp.where(smaller, jnp.asarray(step, jnp.int32), min_steps[p, f]))
            min_indices = min_indices.at[p, f].set(jnp.where(smaller, index, min_indices[p, f]))
            # Ring 0 is the lateral outer row/column at every level, not the
            # top/bottom vertical levels. The remaining horizontal cells are interior.
            ny, nx = value.shape[-2:]
            j, i = jnp.arange(ny)[:, None], jnp.arange(nx)[None, :]
            ring0 = (j == 0) | (j == ny - 1) | (i == 0) | (i == nx - 1)
            locations = locations.at[p, f].add(jnp.stack((
                jnp.sum(negative & ring0, dtype=jnp.uint64),
                jnp.sum(negative & ~ring0, dtype=jnp.uint64))))
    return census._replace(f2_counts=counts, f2_first_step=steps,
                           f2_first_index=indices, f2_first_value=values,
                           f2_min_negative=minima, f2_min_step=min_steps,
                           f2_min_index=min_indices, f2_negative_location=locations,
                           f2_field_observations=observations, f2_qc_packet=f2_packet)


def count_water_repair(census, phase, before, after, dry_mass_kg):
    """Water a guard adds/removes, (q_after - q_before) * dry-air mass, in kg.

    REAL32 diagnostic arithmetic. Cells whose old/new value or weight is
    nonfinite have no defined budget and are counted separately.
    """
    if census is None or census.water_changes is None:
        return census
    p = F2_REPAIRS.index(phase)
    amounts, unknown = census.water_changes, census.unknown_water_changes
    weight = jnp.asarray(dry_mass_kg, jnp.float32)
    for f, name in enumerate(MOISTURE_FIELDS):
        old = jnp.asarray(getattr(before, name), jnp.float32)
        new = jnp.asarray(getattr(after, name), jnp.float32)
        defined = jnp.isfinite(old) & jnp.isfinite(new) & jnp.isfinite(weight)
        delta = jnp.where(defined, (new - old) * weight, jnp.float32(0))
        unknown = unknown.at[p, f].add(jnp.sum(~defined & (new != old), dtype=jnp.uint64))
        added = jnp.sum(jnp.maximum(delta, jnp.float32(0)), dtype=jnp.float32)
        removed = jnp.sum(jnp.maximum(-delta, jnp.float32(0)), dtype=jnp.float32)
        amounts = amounts.at[p, f].add(jnp.stack((added, removed)))
    return census._replace(water_changes=amounts, unknown_water_changes=unknown)


def guard_dry_mass_kg(state, namelist):
    """Dry-air mass per cell (kg): (c1h*MUT + c2h) * -dnw / g * dx*dy / (msftx*msfty)."""
    metrics = namelist.metrics
    mu = jnp.asarray(state.mu_total, jnp.float32)
    c1 = jnp.asarray(metrics.c1h, jnp.float32)
    c2 = jnp.asarray(metrics.c2h, jnp.float32)
    dnw = jnp.asarray(metrics.dnw, jnp.float32)
    column = (c1[:, None, None] * mu[None] + c2[:, None, None]) * (-dnw[:, None, None])
    area = jnp.float32(namelist.grid.projection.dx_m * namelist.grid.projection.dy_m)
    area = area / (jnp.asarray(metrics.msftx, jnp.float32) * jnp.asarray(metrics.msfty, jnp.float32))
    return column * area[None] / jnp.float32(9.81)  # WRF module_model_constants g


def count_work(census: DeviceCensus | None, name: str, amount=1):
    if census is None:
        return None
    return census._replace(work=census.work.at[WORK.index(name)].add(jnp.asarray(amount, jnp.uint64)))


def count_native_mass_guard_events(census: DeviceCensus | None, events):
    """Accumulate native mass-scale events on active columns, once per trip.

    The kernel's int32 mask combines nonfinite and finite mass-floor triggers;
    this total makes no claim about their individual classification.
    """
    if census is None:
        return None
    total = jnp.sum(events != 0, dtype=jnp.uint64)
    return count_work(census, "native_mass_guard_events", total)


def count_guard(census, name, candidate, *, lower=None, upper=None, rejected=None):
    if census is None:
        return None
    finite = jnp.isfinite(candidate)
    outside = jnp.zeros_like(finite)
    if lower is not None:
        outside = outside | (candidate < lower)
    if upper is not None:
        outside = outside | (candidate > upper)
    outside = outside & finite
    repair = (~finite | outside) if rejected is None else rejected
    counts = jnp.stack(tuple(jnp.sum(mask, dtype=jnp.uint64)
                             for mask in (~finite, outside, repair)))
    return census._replace(guards=census.guards.at[GUARDS.index(name)].add(counts))


def count_mass(census, phase, state):
    if census is None:
        return None
    rejected = (~jnp.isfinite(state.mu_total) | ~jnp.isfinite(state.mu_perturbation)
                | (state.mu_total < 1.0))
    census = count_guard(census, f"{phase}.mu_total", state.mu_total, lower=1.0, rejected=rejected)
    return count_guard(census, f"{phase}.mu_perturbation", state.mu_perturbation, rejected=rejected)


def count_dynamics_guards(census, state, *, theta_min, theta_max):
    if census is None:
        return None
    census = count_guard(census, "dynamics.theta", state.theta, lower=theta_min, upper=theta_max)
    census = count_mass(census, "dynamics", state)
    for field in MOISTURE_FIELDS:
        census = count_guard(census, f"dynamics.{field}", getattr(state, field), lower=0.0, upper=0.05)
    return census


def count_boundary_guards(census, state):
    if census is None:
        return None
    for field in BOUNDARY_FINITE_FIELDS:
        census = count_guard(census, f"boundary.{field}", getattr(state, field))
    census = count_guard(census, "boundary.qv", state.qv, lower=0.0, upper=0.05)
    return count_mass(census, "boundary", state)


def _json_float(value):
    value = float(value)
    return value if value == value and abs(value) != float("inf") else repr(value)


def f2_record(census, field_shape):
    """Host F2 summary: nonzero pre-repair events and signed guard water budgets."""
    events = []
    for p, f, e in zip(*(axis.tolist() for axis in census.f2_counts.nonzero())):
        flat = int(census.f2_first_index[p, f, e])
        events.append({
            "phase": F2_PHASES[p], "field": F2_FIELDS[f], "event": F2_EVENTS[e],
            "count": int(census.f2_counts[p, f, e]),
            "first_step": int(census.f2_first_step[p, f, e]),
            "first_flat_index": flat,
            "first_value": _json_float(census.f2_first_value[p, f, e]),
        })
        if field_shape is not None:
            events[-1]["first_index_kji"] = [int(i) for i in np.unravel_index(flat, field_shape)]
    water = {repair: {field: {"added_kg": _json_float(census.water_changes[r, f, 0]),
                              "removed_kg": _json_float(census.water_changes[r, f, 1]),
                              "undefined_cells": int(census.unknown_water_changes[r, f])}
                      for f, field in enumerate(MOISTURE_FIELDS)}
             for r, repair in enumerate(F2_REPAIRS)}
    return {"phases": list(F2_PHASES), "moisture_cap_kg_kg": 0.05,
            "events": events, "guard_water": water}


def f2_negative_summary(census, field_shape):
    """Segment-boundary readout; evidence only, never a justification verdict.

    Separate sidecar preserves the existing census.json schema and readers.
    Legacy/resumed carries without the new leaves supply no summary evidence.
    """
    if census.f2_min_negative is None:
        return {"status": "UNAVAILABLE"}
    records = []
    for p, phase in enumerate(F2_PHASES):
        for f, field in enumerate(F2_FIELDS):
            present, absent, non_real32 = map(int, census.f2_field_observations[p, f])
            coverage = ("UNOBSERVED" if not present and not absent else
                        "N_A" if not present else "PARTIAL" if absent else "OBSERVED")
            count = int(census.f2_counts[p, f, 0])
            minimum = None
            if count:
                flat = int(census.f2_min_index[p, f])
                minimum = {"value": _json_float(census.f2_min_negative[p, f]),
                           "step": int(census.f2_min_step[p, f]), "flat_index": flat}
                if field_shape is not None and flat >= 0:
                    minimum["index_kji"] = [int(i) for i in np.unravel_index(flat, field_shape)]
            ring0, interior = map(int, census.f2_negative_location[p, f])
            records.append({"phase": phase, "field": field, "coverage": coverage,
                            "present_observations": present, "absent_observations": absent,
                            "non_real32_observations": non_real32,
                            "exact_real32_extremum": bool(present and not non_real32),
                            "negative_count": count, "minimum_negative": minimum,
                            "ring0_count": ring0, "interior_count": interior,
                            "nonfinite_count": int(census.f2_counts[p, f, 1]),
                            "moisture_cap_applicability": "APPLICABLE" if field in MOISTURE_FIELDS else "N_A"})
    return {"status": "RECORDED", "summary_dtype": "float32",
            "ring0_definition": "j=0 or j=ny-1 or i=0 or i=nx-1, all k",
            "minimum_ties": "first observation; lowest flat index within that observation",
            "records": records}


def resolved_namelist(namelist, *, output_cadence_steps, output_set):
    """Serialize resolved runtime values, never derive executed counts from them."""
    names = ("dt_s", "acoustic_substeps", "rk_order", "radiation_cadence_steps",
             "run_physics", "run_boundary", "disable_guards", "mp_physics",
             "bl_pbl_physics", "cu_physics", "ra_sw_physics", "ra_lw_physics",
             "sf_surface_physics", "use_noahmp", "force_fp64", "rad_rk_tendf")
    result = {name: getattr(namelist, name) for name in names if hasattr(namelist, name)}
    result["stepcu"] = int(getattr(namelist, "cumulus_cadence_steps", 1))
    result["radt_minutes"] = float(namelist.dt_s) * int(namelist.radiation_cadence_steps) / 60
    result["output_cadence_steps"] = int(output_cadence_steps)
    result["output_set"] = list(output_set) if output_set is not None else "default"
    return result


def write_segment(output_dir, carries, bundles, own_steps, writer, output_cadence):
    """Read device counters once per completed segment; atomically replace JSON.

    Timestep call/trip counts come from device. Writer IO slots come from the
    host persistence ledger when attached; submissions remain separately counted.
    Disabled runs perform no transfers or file IO here.
    """
    active = [getattr(carry, "census", None) is not None for carry in carries.values()]
    if not any(active):
        return
    if not all(active):
        raise ValueError("census carry missing from one or more domains")
    records, summaries = {}, {}
    resident = {name: carry.census for name, carry in carries.items()}
    if any(c.f2_qc_packet is not None for c in resident.values()):
        from gpuwrf.diagnostics.f2_scalar_pd_packet import retain_terminal
        retain_terminal(output_dir, carries, bundles, own_steps)
        resident = {name: c._replace(f2_qc_packet=None) for name, c in resident.items()}
    host = jax.device_get(resident)
    ledger = getattr(writer, "census_io_ledger", None)
    persisted = ledger.snapshot() if ledger is not None else {}
    for domain, census in host.items():
        if census is None:
            continue
        actual_work = dict(zip(WORK, map(int, census.work)))
        if ledger is not None:
            actual_work.update(persisted.get(domain, dict.fromkeys(WRITER_IO, 0)))
        records[domain] = {
            "own_steps": int(own_steps[domain]),
            "resolved_namelist": resolved_namelist(bundles[domain].namelist,
                output_cadence_steps=output_cadence[domain],
                output_set=getattr(writer, "_variable_subset", None)),
            "guards": {name: dict(zip(EVENTS, map(int, counts)))
                       for name, counts in zip(GUARDS, census.guards)},
            "actual_work": actual_work,
            "output_submissions": len(writer.written.get(domain, ())),
            "full_output_set": bool(getattr(writer, "_full_variable_set", False)),
        }
        if census.f2_counts is not None:
            state = getattr(carries[domain], "state", None)
            shape = None if state is None else state.qv.shape
            records[domain]["f2"] = f2_record(census, shape)
            summaries[domain] = f2_negative_summary(census, shape)
    payload = {"schema_version": 1, "counter_dtype": "uint64", "domains": records}
    path = Path(output_dir) / "census.json"
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)
    summary_path = Path(output_dir) / "f2_negative_summary.json"
    summary_pending = summary_path.with_suffix(".json.tmp")
    summary_pending.write_text(json.dumps({"schema_version": 1, "domains": summaries},
                                         indent=2, sort_keys=True, allow_nan=False) + "\n")
    summary_pending.replace(summary_path)
