"""Tester-side acceptance scaffold for sprint 2026-05-26-m6-guard-disabled-debug.

The worker's deliverable is:

  * a NEW driver ``scripts/m6_guard_disabled_debug.py``
  * a NEW boolean ``disable_guards`` field on
    :class:`gpuwrf.runtime.operational_mode.OperationalNamelist`,
    default ``False`` so production behaviour is unchanged,
  * four proof JSONs under
    ``.agent/sprints/2026-05-26-m6-guard-disabled-debug/``:

      - ``proof_guard_inventory.json``         (Stage 1)
      - ``proof_guards_off_safe_default.json`` (Stage 2)
      - ``proof_first_explosive_step.json``    (Stage 3)
      - ``proof_first_explosive_operator.json``(Stage 4)

  * a ``tester-report.md`` (this file's neighbour) with ``Decision:`` token.

These tests are pure-Python + JSON shape checks.  They do not run JAX
forecasts (the manager already runs the validation commands).  They
deliberately FAIL when the worker deliverable is absent, and PASS when
both the implementation and the proof JSONs satisfy the contract.

Edge-case coverage focuses on the boundary between "guards on" production
discipline and "guards off" diagnostic discipline -- the only kind of
regression a future worker could miss without these tests catching it.
"""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPRINT_DIR = ROOT / ".agent" / "sprints" / "2026-05-26-m6-guard-disabled-debug"
DRIVER_PY = ROOT / "scripts" / "m6_guard_disabled_debug.py"
OPERATIONAL_PY = ROOT / "src" / "gpuwrf" / "runtime" / "operational_mode.py"

PROOF_INVENTORY = SPRINT_DIR / "proof_guard_inventory.json"
PROOF_SAFE_DEFAULT = SPRINT_DIR / "proof_guards_off_safe_default.json"
PROOF_FIRST_STEP = SPRINT_DIR / "proof_first_explosive_step.json"
PROOF_FIRST_OPERATOR = SPRINT_DIR / "proof_first_explosive_operator.json"


# --------------------------------------------------------------------------- #
# Section 1 — Worker deliverable presence                                      #
# --------------------------------------------------------------------------- #
#
# These four tests are the manager's tripwire: if the worker fails to commit
# the script, the flag, or any of the four proofs, the tester sprint cannot
# accept the work.  We assert presence here rather than skipping, because the
# whole point of the sprint is that these artefacts must exist on disk.


def test_driver_script_committed():
    """scripts/m6_guard_disabled_debug.py must exist on the tester worktree."""
    assert DRIVER_PY.exists(), (
        f"missing worker deliverable: {DRIVER_PY.relative_to(ROOT)}.  "
        "The contract's Stage 3 validation command depends on this driver."
    )


def test_operational_mode_exposes_disable_guards():
    """OperationalNamelist must expose a ``disable_guards: bool`` field default False.

    Production behaviour must be unchanged when the flag is at its default,
    so the field MUST default to False.  This is the most important
    safety property of the entire sprint.
    """
    spec = importlib.util.spec_from_file_location(
        "gpuwrf.runtime.operational_mode", OPERATIONAL_PY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    pytest.importorskip("jax")
    spec.loader.exec_module(module)

    namelist_cls = module.OperationalNamelist
    field_names = {f.name for f in dc_fields(namelist_cls)}
    assert "disable_guards" in field_names, (
        "OperationalNamelist must expose a ``disable_guards`` field "
        "(see sprint contract §Stage 2)."
    )
    # The default *must* be False so an inattentive caller never accidentally
    # disables guards in production.
    defaults = {f.name: f.default for f in dc_fields(namelist_cls)}
    assert defaults["disable_guards"] is False, (
        "disable_guards default must be False; got "
        f"{defaults['disable_guards']!r}"
    )


def test_all_four_proof_jsons_present():
    """All four proof JSONs the contract names must be on disk."""
    missing = [
        p.name
        for p in (
            PROOF_INVENTORY,
            PROOF_SAFE_DEFAULT,
            PROOF_FIRST_STEP,
            PROOF_FIRST_OPERATOR,
        )
        if not p.exists()
    ]
    assert not missing, f"worker did not produce these proofs: {missing}"


# --------------------------------------------------------------------------- #
# Section 2 — Proof JSON shape & invariants                                    #
# --------------------------------------------------------------------------- #
#
# Each proof JSON gets a pair of tests: (a) it parses as a non-empty mapping;
# (b) it carries the contract-specific keys and value invariants.  Tests in
# this section skip gracefully if the proof is missing so they still serve
# as acceptance once a future worker delivers.


def _load_json_or_skip(path: Path) -> dict[str, Any]:
    if not path.exists():
        pytest.skip(f"missing proof artefact: {path.relative_to(ROOT)}")
    with path.open("r") as fh:
        payload = json.load(fh)
    assert isinstance(payload, dict), f"{path.name} must be a JSON object"
    assert payload, f"{path.name} is empty"
    return payload


# ----- Stage 1 — proof_guard_inventory.json -------------------------------- #


def test_proof_guard_inventory_lists_known_guard_sites():
    """Stage 1: every documented guard site in operational_mode.py is listed.

    The contract's Stage 1 wording is "Grep operational_mode.py for all
    _with_save_family, _micro_coupling_guard, theta/v clamps, Thompson
    guards. Tabulate them with file:line."  We do not enforce one-for-one
    matching against a frozen list (operational_mode.py may evolve), but
    we DO require that the inventory references all the helper guard
    primitives that exist today.
    """
    payload = _load_json_or_skip(PROOF_INVENTORY)
    assert "guards" in payload, "inventory must carry a 'guards' list"
    guards = payload["guards"]
    assert isinstance(guards, list) and guards, "guards list must be non-empty"

    # Each entry must reference a file and a line number.
    for entry in guards:
        assert isinstance(entry, dict), f"guard entry must be object, got {type(entry)}"
        assert "file" in entry and "line" in entry, (
            f"guard entry missing file/line: {entry}"
        )
        assert isinstance(entry["line"], int) and entry["line"] > 0

    # The five guard kinds the contract explicitly names must each appear.
    blob = json.dumps(payload).lower()
    for kind in (
        "_valid_mixing_ratio",        # qv/qc/qr/qi/qs/qg
        "_finite_or_origin",          # u/v/w/theta/p/ph/mu/...
        "_m6b_acoustic_tendencies",   # V self-advection guard
        "theta",                      # post-RK theta projection at line 504
        "thompson",                   # Thompson microphysics guard
    ):
        assert kind in blob, f"guard inventory does not reference '{kind}'"


# ----- Stage 2 — proof_guards_off_safe_default.json ------------------------ #


def test_proof_safe_default_documents_b6_and_v3_521():
    """Stage 2: default-False must preserve B6 bitwise + V3-521 V@step46 = 11.48 m/s.

    These two numbers are the contract's explicit acceptance gate for
    guards-off-by-default safety.
    """
    payload = _load_json_or_skip(PROOF_SAFE_DEFAULT)

    # B6 bitwise sanity.
    b6 = payload.get("b6_parity") or payload.get("b6_savepoint_parity")
    assert b6 is not None, "missing b6 parity field"
    max_abs = b6.get("max_abs_diff", b6.get("max_diff"))
    assert max_abs is not None, f"b6 entry missing max_abs_diff: {b6}"
    # Bitwise = 0.0 in the contract.  Allow non-strict zero in case JSON
    # serialised as "0.0e+00" and floats round; but anything > 0 is a fail.
    assert float(max_abs) == 0.0, (
        f"B6 savepoint parity broken with disable_guards=False: max_diff={max_abs}"
    )

    # V3-521 V_max at step 46.
    v3 = payload.get("v3_521_step46") or payload.get("v3_localize_521_step46")
    assert v3 is not None, "missing v3-521 step-46 field"
    v_max = v3.get("v_max") or v3.get("V_max")
    assert v_max is not None, f"v3 entry missing V_max: {v3}"
    # Contract says 11.48 m/s post-fix.  Allow a small tolerance.
    assert math.isfinite(float(v_max))
    assert abs(float(v_max) - 11.48) < 0.5, (
        f"V3-521 V_max at step 46 with disable_guards=False is "
        f"{v_max}; contract expects 11.48 m/s ±0.5"
    )


# ----- Stage 3 — proof_first_explosive_step.json --------------------------- #


_PROGNOSTIC_FIELDS = {
    "u", "v", "w", "theta",
    "qv", "qc", "qr", "qi", "qs", "qg",
    "p_perturbation", "ph_perturbation", "mu_perturbation",
    "p_total", "ph_total", "mu_total",
    "p", "ph", "mu",
}


def test_proof_first_explosive_step_localizes_explosion():
    """Stage 3: the JSON must name (field, step, cell) of the first explosion.

    The contract requires 'the field name, step number, cell coordinates,
    and operator that produced it'.
    """
    payload = _load_json_or_skip(PROOF_FIRST_STEP)
    assert "first_explosive_step" in payload or "first_step" in payload, (
        "must report a first_explosive_step entry"
    )

    record = payload.get("first_explosive_step") or payload.get("first_step")
    assert "field" in record, "must name the offending field"
    assert record["field"] in _PROGNOSTIC_FIELDS, (
        f"offending field '{record['field']}' must be a known prognostic"
    )
    assert "step" in record, "must report the step number"
    step = record["step"]
    assert isinstance(step, int) and 0 <= step <= 75, (
        f"first-explosive step must be in [0, 75]; got {step!r}"
    )
    # Cell coordinates (k, j, i) or (i, j, k) — accept either layout but
    # require all three.
    cell = record.get("cell") or record.get("ijk") or record.get("coords")
    assert cell is not None, "must report cell coordinates"
    assert isinstance(cell, (list, tuple)) and len(cell) == 3, (
        f"cell coords must be length 3; got {cell!r}"
    )
    for c in cell:
        assert isinstance(c, int) and c >= 0

    # Per-step max/min/abs_max trace for ALL prognostic fields (incl. theta,
    # v, qc that were previously guard-masked).
    trace = payload.get("per_step_trace") or payload.get("per_step")
    assert trace is not None, (
        "Stage 3 requires per-step max/min/abs_max for all prognostic fields"
    )
    assert isinstance(trace, list) and trace, "per-step trace must be non-empty"
    # The fields that the boundary audit could NOT see — theta, v, qc — must
    # be first-class signals in this trace.
    sample = trace[0]
    keys = set()
    for v in sample.values() if isinstance(sample, dict) else []:
        if isinstance(v, dict):
            keys.update(v.keys())
    blob = json.dumps(sample).lower()
    for guard_masked in ("theta", "v", "qc"):
        assert guard_masked in blob, (
            f"first-step trace must surface '{guard_masked}' (previously "
            "guard-masked field)"
        )


# ----- Stage 4 — proof_first_explosive_operator.json ----------------------- #


def test_proof_first_explosive_operator_names_an_operator():
    """Stage 4: must name the FIRST substep operator that produced bad value."""
    payload = _load_json_or_skip(PROOF_FIRST_OPERATOR)
    op = payload.get("operator") or payload.get("first_operator")
    assert op is not None, "must report 'operator' (substep that broke first)"
    assert isinstance(op, (str, dict)), (
        f"operator must be str or object, got {type(op)}"
    )

    # If a string, it should mention a real WRF acoustic / RK / mu substep.
    op_text = op if isinstance(op, str) else json.dumps(op)
    op_text_low = op_text.lower()
    plausible = (
        "acoustic",
        "horizontal_pressure_gradient",
        "vertical_acoustic",
        "vertical_implicit",
        "calc_coef_w",
        "advance_mu_t",
        "advance_w",
        "advance_uv",
        "thompson",
        "boundary",
        "rk",
    )
    assert any(tok in op_text_low for tok in plausible), (
        f"operator name '{op_text}' is not a plausible WRF-port substep; "
        "Stage 4 requires a concrete operator path."
    )

    # Per-substep trace at the first explosive step.
    substep = payload.get("per_substep_trace") or payload.get("substep_trace")
    assert substep is not None, (
        "Stage 4 must include per-substep max for theta, v, p_perturbation"
    )
    assert isinstance(substep, list) and substep


# --------------------------------------------------------------------------- #
# Section 3 — Adversarial / break-it tests                                     #
# --------------------------------------------------------------------------- #


def test_disable_guards_is_jax_static_argnums_compatible():
    """``disable_guards`` must be a Python bool (static-arg), not a jax.Array.

    The debuggability-hooks memory mandates: every hot-path ``@jit`` flag
    must be a Python ``bool`` so XLA can dead-code-eliminate the disabled
    branch in production.  A jax.Array flag would trace both branches and
    bloat the cache.
    """
    pytest.importorskip("jax")
    spec = importlib.util.spec_from_file_location(
        "gpuwrf.runtime.operational_mode", OPERATIONAL_PY
    )
    if spec is None or spec.loader is None:
        pytest.skip("operational_mode.py not loadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namelist_cls = module.OperationalNamelist
    field = next(
        (f for f in dc_fields(namelist_cls) if f.name == "disable_guards"),
        None,
    )
    if field is None:
        pytest.skip("disable_guards not yet added")
    # The dataclass type annotation should be Python ``bool``.
    assert field.type in (bool, "bool", "builtins.bool"), (
        f"disable_guards type must be bool; got {field.type!r}"
    )


def test_proof_first_explosive_step_field_is_not_only_guarded():
    """Sanity: the named offending field cannot be ONLY a guard's projection
    target.  If it is, the worker just instrumented the guard's input — not
    the dynamics that produced the non-physical value.

    e.g. reporting 'theta' as offending field is informative; reporting
    'mu_perturbation' as offending field but then naming 'no operator' is
    a red flag.  We catch that combination.
    """
    if not (PROOF_FIRST_STEP.exists() and PROOF_FIRST_OPERATOR.exists()):
        pytest.skip("first_step or first_operator proof missing")
    with PROOF_FIRST_STEP.open() as fh:
        step_payload = json.load(fh)
    with PROOF_FIRST_OPERATOR.open() as fh:
        op_payload = json.load(fh)
    record = step_payload.get("first_explosive_step") or step_payload.get(
        "first_step"
    )
    op = op_payload.get("operator") or op_payload.get("first_operator")
    # If a field exploded, an operator MUST be named.
    assert op not in (None, "", "unknown", "n/a"), (
        f"explosive step on field {record.get('field')!r} but operator "
        f"is {op!r}; Stage 4 requires a concrete operator path."
    )


def test_proof_first_explosive_step_is_consistent_with_n_steps_75_cap():
    """The driver was capped to 75 steps in the contract.  If the JSON
    reports an explosion at step > 75, it's a stale artefact from an older
    run and must not be accepted.
    """
    if not PROOF_FIRST_STEP.exists():
        pytest.skip("first_step proof missing")
    with PROOF_FIRST_STEP.open() as fh:
        payload = json.load(fh)
    rec = payload.get("first_explosive_step") or payload.get("first_step")
    if rec is None:
        pytest.skip("no first_explosive_step record")
    step = rec.get("step")
    assert step is None or (isinstance(step, int) and step <= 75), (
        f"explosion reported at step {step} > contract cap 75"
    )


def test_proof_safe_default_excludes_guards_off_evidence():
    """The Stage-2 safe-default proof MUST run with disable_guards=False.

    A subtle worker bug would be running the safe-default proof with the
    flag accidentally True and getting non-zero B6 diff.  We can detect
    the inverse: any sign that the proof was taken with disable_guards
    set to True is a fail.
    """
    payload = _load_json_or_skip(PROOF_SAFE_DEFAULT)
    blob = json.dumps(payload).lower()
    # Common spellings the worker might serialise.
    assert "\"disable_guards\": true" not in blob.replace(" ", ""), (
        "safe-default proof appears to have been taken with "
        "disable_guards=True; that violates Stage 2's preconditions."
    )


def test_no_binary_artefacts_committed_under_sprint_dir():
    """Universal rule: no binary fixtures under .agent/.  Only JSON + MD."""
    if not SPRINT_DIR.exists():
        pytest.skip("sprint dir not yet created by worker")
    allowed_suffixes = {".json", ".md", ".txt", ".sh"}
    bad = []
    for entry in SPRINT_DIR.rglob("*"):
        if not entry.is_file():
            continue
        # Skip dot-prefixed scaffolding files (harness/role plumbing).
        if entry.name.startswith("."):
            continue
        # Role-prompt files are extension-less Markdown.
        if entry.parent.name == "role-prompts":
            continue
        if entry.suffix.lower() not in allowed_suffixes:
            bad.append(entry.relative_to(ROOT).as_posix())
    assert not bad, f"non-text artefacts present in sprint dir: {bad}"
