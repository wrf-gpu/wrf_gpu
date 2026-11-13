"""Compare two gpuwrf wrfout streams frame by frame (checkpoint-branch gates; regression evidence, not fidelity).

    python scripts/compare_branch_streams.py --ref REF_DIR --test TEST_DIR --out VERDICT.json --mode exact|edited
           [--expect-differ d03 ...] [--after 2026-02-28T00:09:00]

    python scripts/compare_branch_streams.py digest STREAM_DIR OUT.json     # keep a stream as digests, then delete the raw files

    python scripts/compare_branch_streams.py inventory OUT.json --namelist namelist.input --hours H
           [--branch-generation GEN | --whole-stream] [--include-initial] [--domains N]

``--branch-generation GEN`` takes the seam (branch valid time) from the checkpoint generation itself
(identity.run_start + own_steps x dt_s of the root), never from wrfinput or a plan. ``inventory`` writes the
independently expected frame names from the namelist alone (start date, root time_step, parent_time_step_ratio,
history_interval): a domain writes history at its first own step at or after each history time (E275); ``--expect-frames
OUT.json`` then requires the TEST stream to hold exactly those frames (none missing, none extra) and REF to hold them.
In every mode every REF frame after the seam (``--whole-stream``: every REF frame) must also be in TEST.

``--ref``/``--test`` accept a stream directory or such a digest JSON (disk-light gates: per-file SHA-256, per
variable dtype/shape/byte SHA-256, non-finite flags, global attributes; ``max_abs`` only with raw files).
Every ``wrfout_d0N_*`` file of TEST must exist in REF with the same name (only frames strictly after
``--after`` are required; TEST frames at or before it fail). Per file: whole-file SHA-256, and per variable
the dtype/shape/byte digest plus every global attribute.

Every top-level file of a stream directory must be a well-formed ``wrfout_dNN_<time>`` frame: anything else
(``wrfout*.bak``, suffixed names, temporaries) is reported and FAILs; digests record such files with their SHA-256.

* ``exact``: every TEST frame is byte-identical to REF (file SHA), and so is every REF frame after
  ``--after`` (completeness). Verdict PASS / FAIL.
* ``edited``: domains not in ``--expect-differ`` must be byte-identical per variable; each expected domain must
  differ in at least one variable on the last frame; time-invariant geography (HGT, LU_INDEX, XLAT, XLONG,
  LANDMASK, MAPFAC_*, VAR_SSO, IVGTYP, ISLTYP, SHDMAX, SHDMIN, ALBBCK, SNOALB, PHB, MUB, PB, FNM, FNP, RDNW, RDN,
  DNW, DN, ZNU, ZNW, CF1-3) must be identical everywhere. Non-finite values anywhere FAIL. The JSON lists every
  differing variable per frame with max |diff|.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

STATIC = ("HGT", "LU_INDEX", "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V", "LANDMASK", "MAPFAC_M",
          "MAPFAC_U", "MAPFAC_V", "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY",
          "VAR_SSO", "IVGTYP", "ISLTYP", "SHDMAX", "SHDMIN", "ALBBCK", "SNOALB", "PHB", "MUB", "PB", "FNM", "FNP",
          "RDNW", "RDN", "DNW", "DN", "ZNU", "ZNW", "CF1", "CF2", "CF3", "COSALPHA", "SINALPHA", "F", "E")
NAME = re.compile(r"^wrfout_(d\d\d)_(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)$")


def seam_time(generation: Path) -> str:
    """Branch valid time of a checkpoint generation: run_start + root own_steps x root dt_s (frame-name format)."""
    manifest = json.loads((Path(generation) / "manifest.json").read_text())
    root = manifest["domains"][0]
    start = datetime.fromisoformat(manifest["identity"]["run_start"])
    seam = start + timedelta(seconds=int(manifest["own_steps"][root]) * float(manifest["dt_s"][root]))
    return seam.strftime("%Y-%m-%d_%H:%M:%S")


def _namelist_values(text: str) -> dict:
    values = {}
    for line in text.splitlines():
        line = line.split("!", 1)[0].strip()
        m = re.match(r"^([A-Za-z_0-9]+)\s*=\s*(.*?),?\s*$", line)
        if m:
            values[m.group(1).lower()] = [v.strip().strip("'\"") for v in m.group(2).split(",") if v.strip()]
    return values


def inventory(namelist: Path, hours: float, after: str | None, include_initial: bool, domains: int | None) -> list[str]:
    """Expected wrfout frame names from the namelist alone (independent of any produced stream)."""
    nl = _namelist_values(Path(namelist).read_text())
    n = int(domains or nl["max_dom"][0])
    start = datetime(*(int(nl.get(k, ["0"])[0]) for k in ("start_year", "start_month", "start_day", "start_hour",
                                                          "start_minute", "start_second")))
    if int(nl.get("time_step_fract_num", ["0"])[0]) != 0:
        raise ValueError("fractional root time_step is not supported by the inventory")
    dt = {1: float(nl["time_step"][0])}
    parent = [int(v) for v in nl.get("parent_id", ["1"] * n)]
    ratio = [int(v) for v in nl.get("parent_time_step_ratio", ["1"] * n)]
    for d in range(2, n + 1):
        dt[d] = dt[parent[d - 1]] / ratio[d - 1]
    interval = [float(v) * 60.0 for v in nl["history_interval"]]
    end_s = hours * 3600.0
    cutoff = None if after is None else (datetime.strptime(after, "%Y-%m-%d_%H:%M:%S") - start).total_seconds()
    names = []
    for d in range(1, n + 1):
        step, k = interval[d - 1 if len(interval) >= d else -1], 0 if include_initial else 1
        while k * step <= end_s + 1e-9:
            t = math.ceil(round(k * step / dt[d], 9)) * dt[d]
            if t <= end_s + 1e-9 and (cutoff is None or t > cutoff + 1e-9):
                names.append(f"wrfout_d{d:02d}_{(start + timedelta(seconds=t)):%Y-%m-%d_%H:%M:%S}")
            k += 1
    return sorted(set(names))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frames(root: Path) -> dict[str, Path]:
    out = {}
    for path in sorted(Path(root).glob("wrfout_d*")):
        if NAME.match(path.name):
            out[path.name] = path
    return out


def stray(root: Path) -> list[str]:
    """Top-level files of a stream directory that are not well-formed frames (wrfout*.bak, suffixed, .tmp, ...)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p.is_file() or p.is_symlink()) and not NAME.match(p.name))


def variables(path: Path) -> tuple[dict, dict]:
    import netCDF4
    data, attrs = {}, {}
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        for key in ds.ncattrs():
            value = ds.getncattr(key)
            attrs[key] = value.tolist() if hasattr(value, "tolist") else value
        for name, var in ds.variables.items():
            data[name] = np.asarray(var[...])
    return data, attrs


def digest_file(path: Path) -> dict:
    data, attrs = variables(path)
    return {"file_sha": sha(path), "attrs": attrs,
            "vars": {n: {"dtype": str(a.dtype), "shape": list(a.shape), "sha": hashlib.sha256(a.tobytes()).hexdigest()}
                     for n, a in data.items()},
            "nonfinite": sorted(n for n, a in data.items() if a.dtype.kind == "f" and not np.all(np.isfinite(a)))}


def load_stream(root: Path) -> dict:
    """{frame name: Path (raw) | dict (digest)}"""
    root = Path(root)
    if root.suffix == ".json":
        return json.loads(root.read_text())["frames"]
    return frames(root)


def load_stray(root: Path) -> list[str]:
    root = Path(root)
    if root.suffix == ".json":
        value = json.loads(root.read_text())
        if "stray" not in value:
            return ["<digest without a stray-file record>"]
        return list(value["stray"])
    return stray(root)


def compare_digest(a: dict, b: dict) -> dict:
    row = {"file_sha_equal": a["file_sha"] == b["file_sha"], "missing": sorted(set(a["vars"]) ^ set(b["vars"])),
           "attrs_differing": sorted(k for k in set(a["attrs"]) | set(b["attrs"]) if a["attrs"].get(k) != b["attrs"].get(k)),
           "nonfinite": b["nonfinite"], "differing": {}}
    for name in sorted(set(a["vars"]) & set(b["vars"])):
        if a["vars"][name] != b["vars"][name]:
            row["differing"][name] = {"dtype": [a["vars"][name]["dtype"], b["vars"][name]["dtype"]],
                                      "shape": [a["vars"][name]["shape"], b["vars"][name]["shape"]]}
    return row


def compare_any(ref, test) -> dict:
    if isinstance(ref, Path) and isinstance(test, Path):
        return compare_file(ref, test)
    return compare_digest(ref if isinstance(ref, dict) else digest_file(ref),
                          test if isinstance(test, dict) else digest_file(test))


def compare_file(ref: Path, test: Path) -> dict:
    a, attrs_a = variables(ref)
    b, attrs_b = variables(test)
    row = {"file_sha_equal": sha(ref) == sha(test), "differing": {}, "missing": sorted(set(a) ^ set(b)),
           "attrs_differing": sorted(k for k in set(attrs_a) | set(attrs_b) if attrs_a.get(k) != attrs_b.get(k)),
           "nonfinite": []}
    for name in sorted(set(a) & set(b)):
        x, y = a[name], b[name]
        if x.dtype.kind == "f" and not np.all(np.isfinite(y)):
            row["nonfinite"].append(name)
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            entry = {"dtype": [str(x.dtype), str(y.dtype)], "shape": [list(x.shape), list(y.shape)]}
            if x.shape == y.shape and x.dtype.kind in "fiu" and y.dtype.kind in "fiu":
                diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
                entry.update(max_abs=float(np.nanmax(diff)) if diff.size else 0.0,
                             cells=int(np.count_nonzero(diff)))
            row["differing"][name] = entry
    return row


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["inventory"]:
        ip = argparse.ArgumentParser(prog="compare_branch_streams.py inventory", allow_abbrev=False)
        ip.add_argument("out", type=Path)
        ip.add_argument("--namelist", type=Path, required=True)
        ip.add_argument("--hours", type=float, required=True)
        group = ip.add_mutually_exclusive_group(required=True)
        group.add_argument("--branch-generation", type=Path)
        group.add_argument("--whole-stream", action="store_true")
        ip.add_argument("--include-initial", action="store_true")
        ip.add_argument("--domains", type=int)
        a = ip.parse_args(argv[1:])
        after = seam_time(a.branch_generation) if a.branch_generation else None
        names = inventory(a.namelist, a.hours, after, a.include_initial, a.domains)
        a.out.write_text(json.dumps({"schema": "v50branch.frame_inventory.v1", "namelist": str(a.namelist), "after": after,
                                     "hours": a.hours, "frames": names}, indent=1) + "\n")
        print(f"inventory: {len(names)} frames -> {a.out}", file=sys.stderr)
        return 0
    if argv[:1] == ["digest"]:
        root, out = Path(argv[1]), Path(argv[2])
        result = {"schema": "v50branch.stream_digest.v1", "source": str(root.resolve()),
                  "frames": {name: digest_file(path) for name, path in frames(root).items()},
                  "stray": stray(root), "stray_sha256": {n: sha(root / n) for n in stray(root) if (root / n).is_file()}}
        if not result["frames"]:
            print(f"digest: no wrfout frames in {root}", file=sys.stderr)
            return 1
        out.write_text(json.dumps(result, indent=1, default=str) + "\n")
        print(f"digest: {len(result['frames'])} frames -> {out}", file=sys.stderr)
        return 0
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("exact", "edited"), required=True)
    seam = ap.add_mutually_exclusive_group(required=True)
    seam.add_argument("--after", help="branch valid time (ISO, e.g. 2026-02-28T00:09:00): frames must be later")
    seam.add_argument("--branch-generation", type=Path, help="take the seam from this checkpoint generation's manifest")
    ap.add_argument("--expect-frames", type=Path, help="inventory JSON: TEST must hold exactly these frames")
    ap.add_argument("--expect-differ", nargs="*", default=[])
    ap.add_argument("--whole-stream", action="store_true",
                    help="exact mode for a resumed stream: frames at/before --after are allowed and compared too")
    a = ap.parse_args(argv)
    after = seam_time(a.branch_generation) if a.branch_generation else a.after.replace("T", "_")[:19]
    ref, test = load_stream(a.ref), load_stream(a.test)
    later = lambda name: NAME.match(name).group(2) > after  # noqa: E731  (lexicographic == chronological here)
    verdict = {"schema": "v50branch.compare_streams.v1", "mode": a.mode, "ref": str(a.ref), "test": str(a.test),
               "after": after, "expect_differ": a.expect_differ, "frames": {}, "failures": []}
    fail = verdict["failures"].append
    verdict["stray"] = {"ref": load_stray(a.ref), "test": load_stray(a.test)}
    for side in ("ref", "test"):
        if verdict["stray"][side]:
            fail(f"{side} stream has files that are not well-formed frames: {verdict['stray'][side]}")
    if not test:
        fail("test stream has no wrfout frames")
    early = [n for n in test if not later(n)]
    if a.whole_stream and a.mode != "exact":
        fail("--whole-stream applies to exact mode only")
    if early and not a.whole_stream:
        fail(f"test frames at/before the branch time: {early}")
    missing = sorted(n for n in ref if (a.whole_stream or later(n)) and n not in test)  # completeness, every mode
    if missing:
        fail(f"ref frames {'of the whole stream' if a.whole_stream else 'after the branch time'} missing in test: {missing}")
    if a.expect_frames:
        expected = set(json.loads(a.expect_frames.read_text())["frames"])
        verdict["expected_frames"] = len(expected)
        if sorted(expected - set(test)):
            fail(f"expected frames missing in test: {sorted(expected - set(test))}")
        if sorted(set(test) - expected):
            fail(f"test frames not in the expected inventory: {sorted(set(test) - expected)}")
        if sorted(expected - set(ref)):
            fail(f"expected frames missing in ref: {sorted(expected - set(ref))}")
    last = {}
    for name in sorted(test):
        if name not in ref:
            fail(f"{name}: not in ref")
            continue
        row = compare_any(ref[name], test[name])
        verdict["frames"][name] = row
        domain = NAME.match(name).group(1)
        last[domain] = name
        if row["nonfinite"]:
            fail(f"{name}: non-finite values in {row['nonfinite']}")
        if row["missing"]:
            fail(f"{name}: variable sets differ {row['missing']}")
        if a.mode == "exact" and not row["file_sha_equal"]:
            fail(f"{name}: file bytes differ (variables {sorted(row['differing'])[:12]}, attrs {row['attrs_differing']})")
        if a.mode == "edited":
            static = sorted(set(row["differing"]) & set(STATIC))
            if static:
                fail(f"{name}: time-invariant geography differs {static}")
            if domain not in a.expect_differ and row["differing"]:
                fail(f"{name}: domain {domain} must be identical; differs in {sorted(row['differing'])[:12]}")
    if a.mode == "edited":
        for domain in a.expect_differ:
            if domain not in last:
                fail(f"expected-different domain {domain} has no frame")
            elif not verdict["frames"][last[domain]]["differing"]:
                fail(f"{last[domain]}: expected a response in {domain}, but it is byte-identical")
    verdict["verdict"] = "PASS" if not verdict["failures"] else "FAIL"
    a.out.write_text(json.dumps(verdict, indent=1, default=str) + "\n")
    print(f"compare_streams {a.mode}: {verdict['verdict']} ({len(verdict['frames'])} frames)"
          + "".join("\n  " + f for f in verdict["failures"][:20]), file=sys.stderr)
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
