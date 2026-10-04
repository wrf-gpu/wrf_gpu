"""The shape sweep must be a harness, not prose (manager review of `dd5ed7b8`).

Everything here runs with a STUB launcher: no census process starts, no GPU, no
measurement. What is under test is the harness's judgement — whether it refuses
to issue a verdict when the invariants that make the comparison meaningful do
not hold.

The invariants exist because of specific ways this comparison could be wrong:

* if d06 quietly ran fewer operators than d01, the sweep would measure inventory
  rather than shape;
* the smaller domains run `cu=0, gwd=0` in production, so the diagnostic
  inventory FORCES cumulus and GWD in — and membership in a list is not evidence
  that they lowered;
* implicit first/last pair selection spans 6.75 days on d02.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import run_shape_sweep as sweep  # noqa: E402


_A6 = REPO / "proofs/v025/m0/hlo_dtype_transfer_census.json"


def frozen_operator_names() -> list[str]:
    """The REAL 49 A6 operator names.

    The stub census has to emit the genuine inventory now, otherwise the
    positive-control tests would be asserting that a clean sweep produces a
    verdict while feeding it an inventory the harness is required to reject.
    """
    if not _A6.exists():
        pytest.skip("A6 census not on this machine")
    return [op["name"] for op in json.loads(_A6.read_text())["operators"]]


def _census(shape: str, *, operators=None, lowered_forced=True, compile_s=10.0, lower_s=1.0):
    names = operators if operators is not None else frozen_operator_names()
    ops = []
    for name in names:
        forced = name in sweep.FORCED_OPERATORS
        ops.append({
            "name": name,
            "status": "LOWERED" if (lowered_forced or not forced) else "NOT_LOWERED",
            "compile_seconds": compile_s / len(names),
            "lower_seconds": lower_s / len(names),
        })
    return {
        "operators": ops,
        "case_config": {"nx": 120, "ny": 70, "nz": 44, "cu_physics": 1, "gwd_opt": 1,
                        "mp_physics": 8, "ra_lw_physics": 4, "bl_pbl_physics": 5},
    }


def _harness(tmp_path, per_shape_census, *, repeats=1, fail_shapes=()):
    launched = []

    def launch(command, *, cwd, env):
        shape = command[command.index("--domain") + 1]
        launched.append(shape)
        if shape in fail_shapes:
            return 1, "boom"
        Path(command[command.index("--out") + 1]).write_text("{}")
        return 0, ""

    def read(path: Path):
        shape = path.name.split("_")[1]
        return per_shape_census[shape]

    obj = sweep.run_sweep(out=tmp_path / "sweep.json", raw_dir=tmp_path / "raw",
                          repeats=repeats, launch=launch, read_census=read)
    return obj, launched


# --------------------------------------------------------------------------- #
# registered plan                                                              #
# --------------------------------------------------------------------------- #
def test_every_shape_names_both_state_files_explicitly():
    for shape, spec in sweep.SHAPES.items():
        assert spec["previous"].startswith(f"wrfout_{shape}_")
        assert spec["snapshot"].startswith(f"wrfout_{shape}_")
        assert spec["previous"] != spec["snapshot"]


def test_no_hash_literal_is_stored_in_the_registered_table():
    """A literal here is a digest nobody verified; a truncated one matches nothing."""
    for spec in sweep.SHAPES.values():
        assert "sha256" not in spec


def test_the_command_passes_both_hashes_for_fail_closed_verification():
    shapes = sweep.resolve_hashes()
    if not shapes["d01"]["exists"]:
        pytest.skip("state files not on this machine")
    command = sweep.census_command("d01", shapes["d01"], 1, Path("/tmp/x.json"))
    assert command.count("--expect-sha256") == 2
    for digest in shapes["d01"]["sha256"]:
        assert len(digest) == 64
        assert digest in command


@pytest.mark.parametrize("digests", [
    [None, None],                     # file missing
    ["a20c48601be9ea85", "b" * 64],   # truncated — reads like a hash, matches nothing
    ["a" * 64],                       # only one of the pair
])
def test_a_missing_or_truncated_digest_refuses_to_build_a_command(digests):
    spec = dict(sweep.SHAPES["d01"], run_dir="/nope", sha256=digests)
    with pytest.raises(SystemExit, match="64-char"):
        sweep.census_command("d01", spec, 1, Path("/tmp/x.json"))


def test_every_generated_command_is_accepted_by_the_real_census_parser():
    """Parse-check with the census's OWN parser, not a hand-written mirror.

    W1 attempt 1 died inside a coordinated GPU window on `--hours 1.0` against an
    int option, while 27 tests passed because they inspected the command list
    instead of parsing it. A generated command is trustworthy only if the actual
    parser accepts it.
    """
    # Parser acceptance needs valid digest spellings, not external forecast
    # files. The census's import-time CPU guard must precede JAX, which other
    # test modules may already have imported into the pytest process.
    shapes = {
        shape: dict(spec, sha256=["a" * 64, "b" * 64])
        for shape, spec in sweep.SHAPES.items()
    }
    commands = []
    for repeat in range(1, 4):
        for shape in sweep.ROTATIONS[(repeat - 1) % len(sweep.ROTATIONS)]:
            commands.append(sweep.census_command(
                shape, shapes[shape], repeat, Path(f"/tmp/c_{shape}_{repeat}.json")))
    child = subprocess.run(
        [sys.executable, "-c", """
import json, sys
sys.path.insert(0, sys.argv[1])
import build_hlo_census
parser = build_hlo_census.build_parser()
fields = ('domain', 'previous', 'snapshot', 'expect_sha256', 'repeat_tag')
parsed = []
for command in json.load(sys.stdin):
    args = parser.parse_args(command[5:])
    parsed.append({name: getattr(args, name) for name in fields})
print(json.dumps(parsed))
""", str(REPO / "scripts/v025")],
        input=json.dumps(commands), text=True, capture_output=True, check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
        timeout=30,
    )
    parsed = json.loads(child.stdout)
    assert len(parsed) == 12
    for command, args in zip(commands, parsed):
        shape = command[command.index("--domain") + 1]
        assert args["domain"] == shape
        assert args["previous"] == sweep.SHAPES[shape]["previous"]
        assert args["snapshot"] == sweep.SHAPES[shape]["snapshot"]
        assert args["expect_sha256"] == shapes[shape]["sha256"]
        assert args["repeat_tag"] == command[command.index("--repeat-tag") + 1]


def test_the_rotation_covers_every_shape_in_a_different_position():
    for index, shape in enumerate(sweep.ROTATIONS[0]):
        positions = {rotation.index(shape) for rotation in sweep.ROTATIONS}
        assert len(positions) > 1, f"{shape} sits in the same slot in every rotation"


def test_the_environment_is_pinned_and_cpu_only():
    assert sweep.SWEEP_ENV["JAX_PLATFORMS"] == "cpu"
    assert sweep.SWEEP_ENV["CUDA_VISIBLE_DEVICES"] == ""
    assert sweep.SWEEP_ENV["GPUWRF_JAX_CACHE"] == "0"
    assert "multi_thread_eigen=false" in sweep.SWEEP_ENV["XLA_FLAGS"]


def test_twelve_runs_are_launched_in_the_registered_rotation(tmp_path):
    census = {s: _census(s) for s in sweep.SHAPES}
    obj, launched = _harness(tmp_path, census, repeats=3)
    assert len(obj["runs"]) == 12
    assert launched[:4] == sweep.ROTATIONS[0]
    assert launched[4:8] == sweep.ROTATIONS[1]


# --------------------------------------------------------------------------- #
# invariants gate the verdict                                                  #
# --------------------------------------------------------------------------- #
def test_positive_control_clean_runs_do_produce_a_real_verdict(tmp_path):
    """Without this, every BLOCKED assertion below could be passing vacuously."""
    census = {s: _census(s) for s in sweep.SHAPES}
    obj, _ = _harness(tmp_path, census, repeats=3)
    assert obj["invariants"]["all_invariants_hold"] is True
    assert obj["verdict_compile"]["band"] != "BLOCKED"
    assert obj["verdict_lower"]["band"] != "BLOCKED"


def test_the_band_tracks_the_measured_spread_end_to_end(tmp_path):
    """Drive real per-shape times through the whole harness, not just `verdict`."""
    census = {s: _census(s, compile_s=10.0) for s in sweep.SHAPES}
    census["d06"] = _census("d06", compile_s=90.0)
    obj, _ = _harness(tmp_path, census, repeats=3)
    assert obj["invariants"]["all_invariants_hold"] is True
    assert obj["verdict_compile"]["band"] == "FALSIFIED"
    assert obj["verdict_compile"]["spread_ratio"] == pytest.approx(9.0)
    assert obj["per_shape"]["d06"]["compile_seconds"]["median"] == pytest.approx(90.0)


# --------------------------------------------------------------------------- #
# the frozen A6 inventory (manager adjudication, main 6a497c0e)                 #
# --------------------------------------------------------------------------- #
def test_the_frozen_digest_is_the_one_the_manager_adjudicated():
    assert sweep.FROZEN_OPERATOR_COUNT == 49
    assert sweep.FROZEN_OPERATOR_DIGEST == (
        "abd39cd2a8cce2876cf4d824658ce08840fa4d93ab4d20846250306df980fd9e")


def test_the_frozen_digest_reproduces_from_the_committed_a6_census():
    """The constant must be derivable from evidence, not just asserted."""
    census_path = REPO / "proofs/v025/m0/hlo_dtype_transfer_census.json"
    if not census_path.exists():
        pytest.skip("A6 census not on this machine")
    census = json.loads(census_path.read_text())
    assert len(census["operators"]) == sweep.FROZEN_OPERATOR_COUNT
    assert sweep.operator_digest(census) == sweep.FROZEN_OPERATOR_DIGEST


def test_all_shapes_agreeing_on_a_WRONG_inventory_still_blocks(tmp_path):
    """THE mutation test the manager asked for.

    This is the case cross-shape equality cannot catch. Every shape emits the
    SAME inventory -- so `identical_operator_digest` is True and the old check
    was satisfied -- but it is not the frozen A6 inventory. An operator silently
    dropped from the harness fails exactly this way: dropped identically
    everywhere, invisible to any comparison the shapes make among themselves.
    """
    dropped = [f"op{i}" for i in range(45)] + list(sweep.FORCED_OPERATORS)  # 47, not 49
    census = {s: _census(s, operators=dropped) for s in sweep.SHAPES}
    obj, _ = _harness(tmp_path, census, repeats=3)

    assert obj["invariants"]["identical_operator_digest"] is True, (
        "the shapes DO agree with each other -- that is the point of this test"
    )
    assert obj["invariants"]["frozen_a6_inventory"]["matches_frozen"] is False
    assert obj["invariants"]["all_invariants_hold"] is False
    assert obj["verdict_compile"]["band"] == "BLOCKED"
    assert obj["verdict_lower"]["band"] == "BLOCKED"


def test_the_right_count_with_the_wrong_names_still_blocks(tmp_path):
    """49 operators is not the same claim as the 49 frozen operators."""
    renamed = [f"wrong{i}" for i in range(47)] + list(sweep.FORCED_OPERATORS)
    assert len(renamed) == sweep.FROZEN_OPERATOR_COUNT
    census = {s: _census(s, operators=renamed) for s in sweep.SHAPES}
    obj, _ = _harness(tmp_path, census)
    assert obj["invariants"]["frozen_a6_inventory"]["matches_frozen"] is False
    assert obj["verdict_compile"]["band"] == "BLOCKED"


def test_the_mismatch_is_reported_per_run_not_just_as_a_flag(tmp_path):
    """A blocked sweep must say which shape and repeat diverged."""
    census = {s: _census(s, operators=[f"op{i}" for i in range(30)]) for s in sweep.SHAPES}
    obj, _ = _harness(tmp_path, census, repeats=3)
    mismatches = obj["invariants"]["frozen_a6_inventory"]["mismatches"]
    assert len(mismatches) == 12
    assert {m["shape"] for m in mismatches} == set(sweep.SHAPES)
    assert all(m["operators_total"] == 30 for m in mismatches)


def test_a_differing_operator_inventory_blocks_the_verdict(tmp_path):
    """Otherwise the sweep measures inventory, not shape."""
    census = {s: _census(s) for s in sweep.SHAPES}
    census["d06"] = _census("d06", operators=[f"op{i}" for i in range(20)])
    obj, _ = _harness(tmp_path, census)
    assert obj["invariants"]["identical_operator_digest"] is False
    assert obj["verdict_compile"]["band"] == "BLOCKED"


def test_a_forced_operator_that_did_not_lower_blocks_the_verdict(tmp_path):
    """cu=0/gwd=0 domains must still LOWER cumulus and GWD, not just list them."""
    census = {s: _census(s) for s in sweep.SHAPES}
    census["d06"] = _census("d06", lowered_forced=False)
    obj, _ = _harness(tmp_path, census)
    assert obj["invariants"]["forced_operator_failures"]
    assert obj["verdict_compile"]["band"] == "BLOCKED"


def test_a_failed_run_blocks_the_verdict(tmp_path):
    census = {s: _census(s) for s in sweep.SHAPES}
    obj, _ = _harness(tmp_path, census, fail_shapes=("d03",))
    assert obj["invariants"]["all_runs_succeeded"] is False
    assert obj["verdict_compile"]["band"] == "BLOCKED"


def test_snapshot_config_is_recorded_separately_from_the_forced_inventory(tmp_path):
    """A reader must see d06's namelist cu value alongside the forced run."""
    census = {s: _census(s) for s in sweep.SHAPES}
    obj, _ = _harness(tmp_path, census)
    assert obj["per_shape"]["d06"]["snapshot_config"] is not None
    assert "cu_physics" in obj["per_shape"]["d06"]["snapshot_config"]
    assert "FORCES cumulus and GWD" in obj["invariants"]["note"]


# --------------------------------------------------------------------------- #
# the three-band rule                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("medians,band", [
    ({"a": 100.0, "b": 110.0}, "GRID_INDEPENDENT"),
    ({"a": 100.0, "b": 150.0}, "PARTIAL_SENSITIVITY"),
    ({"a": 100.0, "b": 400.0}, "FALSIFIED"),
    ({"a": 100.0}, "INSUFFICIENT"),
])
def test_the_three_band_rule_is_applied_mechanically(medians, band):
    assert sweep.verdict(medians)["band"] == band


def test_no_band_permits_case_substitution():
    for medians in ({"a": 1.0, "b": 1.1}, {"a": 1.0, "b": 1.5}, {"a": 1.0, "b": 9.0}):
        action = sweep.verdict(medians)["next_action"]
        assert "no case substitution" in action.lower() or "NO cheaper case" in action


def test_lower_and_compile_get_separate_verdicts(tmp_path):
    census = {s: _census(s) for s in sweep.SHAPES}
    obj, _ = _harness(tmp_path, census)
    assert "verdict_compile" in obj and "verdict_lower" in obj


def test_the_object_states_it_qualifies_nothing(tmp_path):
    census = {s: _census(s) for s in sweep.SHAPES}
    obj, _ = _harness(tmp_path, census)
    assert "nominates no case" in obj["purpose"]


def test_summaries_carry_median_cv_and_bootstrap_ci(tmp_path):
    census = {s: _census(s) for s in sweep.SHAPES}
    obj, _ = _harness(tmp_path, census, repeats=3)
    entry = obj["per_shape"]["d01"]["compile_seconds"]
    assert entry["n"] == 3
    assert entry["median"] is not None
    assert entry["cv_percent"] is not None
    assert "low" in entry["bootstrap_ci95"] and "high" in entry["bootstrap_ci95"]
