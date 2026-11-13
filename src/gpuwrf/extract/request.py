"""Extract request (CLI ``--extract-spec v1 --extract-config JSON --extract-dir DIR``).

The request is bound at run start: its policy enters the restart output policy
only when enabled (missing == OFF, so OFF checkpoints stay byte-stable) and a
resume with a different request is refused before any model work.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REQUEST_SCHEMA = "gpuwrf.extract.request.v1"
SPECS = ("v1",)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_KEYS = {"schema", "run_id", "member", "issue_utc", "arm", "seed", "domains", "points_csv", "max_distance_m",
         "raw_history"}
_REQUIRED = {"schema", "run_id", "member", "issue_utc"}


def _parse_utc(text: str) -> datetime:
    value = str(text).strip()
    for fmt in ("%Y%m%d_%Hz", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%d_%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"extract issue_utc {text!r}: use YYYYmmdd_HHz or YYYY-mm-ddTHH:MM:SSZ (UTC)")


@dataclass(frozen=True)
class ExtractRequest:
    spec: str
    config_path: Path
    config_sha256: str
    extract_dir: Path
    run_id: str
    member: str
    issue_utc: datetime
    arm: str | None
    seed: int | None
    domains: tuple[str, ...] | None  # None = every run domain
    points_csv: tuple[Path, ...] = ()
    points_sha256: tuple[str, ...] = ()
    max_distance_m: tuple[tuple[str, float], ...] = ()  # per-nest override of 1.5*dx
    raw_history: bool = True  # False: extract only, no wrfout written (RAW off)

    def policy(self) -> dict:
        """Restart-bound policy (identity of the extract stream). Keys beyond the M1 three
        appear only when used, so an M1-style request keeps its policy bytes."""
        out = {"spec": self.spec, "config_sha256": self.config_sha256, "extract_dir": str(self.extract_dir)}
        if self.points_csv:
            out["points_sha256"] = list(self.points_sha256)
        if not self.raw_history:
            out["raw_history"] = "off"
        return out

    def profiles_file_name(self, domain: str) -> str:
        return f"profiles_{self.run_id}_{self.member}_{domain}.nc"

    def file_name(self, domain: str) -> str:
        return f"extract_{self.run_id}_{self.member}_{domain}.nc"


def load_request(spec: str, config_path: Path, extract_dir: Path) -> ExtractRequest:
    if spec not in SPECS:
        raise ValueError(f"--extract-spec {spec!r} unsupported; supported: {', '.join(SPECS)}")
    config_path = Path(config_path).resolve()
    data = config_path.read_bytes()
    doc = json.loads(data)
    if not isinstance(doc, dict):
        raise ValueError("extract config must be a JSON object")
    unknown = set(doc) - _KEYS
    if unknown:
        raise ValueError(f"extract config has unknown keys {sorted(unknown)}")
    absent = _REQUIRED - set(doc)
    if absent:
        raise ValueError(f"extract config misses required keys {sorted(absent)}")
    if doc["schema"] != REQUEST_SCHEMA:
        raise ValueError(f"extract config schema must be {REQUEST_SCHEMA}")
    for key in ("run_id", "member"):
        if not isinstance(doc[key], str) or not _ID.match(doc[key]):
            raise ValueError(f"extract config {key} must match {_ID.pattern}")
    arm = doc.get("arm")
    if arm is not None and (not isinstance(arm, str) or not _ID.match(arm)):
        raise ValueError("extract config arm must be an identifier string")
    seed = doc.get("seed")
    if seed is not None and type(seed) is not int:
        raise ValueError("extract config seed must be an integer or null")
    domains = doc.get("domains")
    if domains is not None:
        if (not isinstance(domains, list) or not domains
                or any(not isinstance(d, str) or not re.fullmatch(r"d[0-9]{2}", d) for d in domains)
                or len(set(domains)) != len(domains)):
            raise ValueError("extract config domains must be a non-empty unique list like ['d01','d02']")
        domains = tuple(sorted(domains))
    points = doc.get("points_csv", [])
    if isinstance(points, str):
        points = [points]
    if not isinstance(points, list) or any(not isinstance(p, str) or not p for p in points):
        raise ValueError("extract config points_csv must be a path or a list of paths")
    paths = []
    for p in points:
        path = Path(p)
        if not path.is_absolute():
            raise ValueError(f"extract points_csv {p!r} must be an absolute path (bound by content SHA)")
        paths.append(path.resolve())
    if len(set(paths)) != len(paths):
        raise ValueError("extract points_csv lists a file twice")
    shas = tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths)
    if paths:
        from gpuwrf.extract.points import load_point_csvs
        load_point_csvs(paths)  # format/duplicate validation before any model work
    limits = doc.get("max_distance_m", {})
    if (not isinstance(limits, dict) or any(not re.fullmatch(r"d[0-9]{2}", k) for k in limits)
            or any(type(v) not in (int, float) or not v > 0 for v in limits.values())):
        raise ValueError("extract config max_distance_m must map 'dNN' to a positive number of metres")
    if limits and not paths:
        raise ValueError("extract config max_distance_m needs points_csv")
    raw = doc.get("raw_history", "on")
    if raw not in ("on", "off"):
        raise ValueError("extract config raw_history must be 'on' or 'off'")
    return ExtractRequest(
        spec=spec, config_path=config_path, config_sha256=hashlib.sha256(data).hexdigest(),
        extract_dir=Path(extract_dir).resolve(), run_id=doc["run_id"], member=doc["member"],
        issue_utc=_parse_utc(doc["issue_utc"]), arm=arm, seed=seed, domains=domains,
        points_csv=tuple(paths), points_sha256=shas,
        max_distance_m=tuple(sorted((k, float(v)) for k, v in limits.items())), raw_history=raw == "on")


def recorded_policy(identity: dict):
    return identity.get("output_policy", {}).get("extract")


def require_resume_policy(generation: Path, request: ExtractRequest | None) -> None:
    """Manifest-only admission before model initialization (like --history-znt).

    Called after the existing boundary-coverage and journal refusals. With the extract OFF an absent
    manifest is left to the ordinary resume path (unchanged OFF behaviour)."""
    path = Path(generation) / "manifest.json"
    if request is None and not path.is_file():
        return
    identity = json.loads(path.read_text())["identity"]
    recorded = recorded_policy(identity)
    requested = None if request is None else request.policy()
    if recorded != requested:
        raise ValueError("--extract-* policy differs from checkpoint; resume must keep the same "
                         f"extract request (recorded={recorded}, requested={requested})")
    if request is not None:
        # A completed run already published this extract (manifest written last). An extension writes a chained
        # segment (<extract_dir>/segment_NN, see published_chain); every published part must be READY and bound to
        # this exact policy -- checked here, before model initialization and before any write.
        published_chain(request)


SEGMENT_PREFIX = "segment_"


def published_chain(request: ExtractRequest) -> list[tuple[Path, dict, str]]:
    """Published parts of ``request.extract_dir`` in order: the base extract, then segment_01, segment_02, ...
    (each written by one resumed extension of a completed run). Returns [(dir, manifest, manifest_sha256)].
    Refuses a published part that is not READY or whose recorded policy is not this request's policy.
    A segment directory without a manifest is an extension still in progress (ordinary resume continues it)."""
    base = Path(request.extract_dir)
    chain: list[tuple[Path, dict, str]] = []
    k = 0
    while True:
        part = base if k == 0 else base / f"{SEGMENT_PREFIX}{k:02d}"
        path = part / "manifest.json"
        if not path.is_file():
            break
        data = path.read_bytes()
        manifest = json.loads(data)
        if manifest.get("status") != "READY":
            raise ValueError(f"extract {part} was published with status {manifest.get('status')!r}; extending a "
                             "completed run needs every published part READY (start a fresh run/extract dir instead)")
        if manifest.get("policy") != request.policy():
            raise ValueError(f"extract {part} was published under policy {manifest.get('policy')!r}, not this "
                             f"request's {request.policy()!r}; an extension must keep the same extract request")
        # Every published artifact of this part must still exist with its recorded bytes and SHA256.
        for name, info in sorted((manifest.get("artifacts") or {}).items()):
            path_a = part / name
            if not path_a.is_file() or path_a.stat().st_size != info.get("bytes") \
                    or _sha256_file(path_a) != info.get("sha256"):
                raise ValueError(f"extract {part}: published artifact {name} is missing or changed "
                                 "(bytes/SHA256 differ from its manifest); refusing the extension")
        if not manifest.get("artifacts"):
            raise ValueError(f"extract {part}: manifest lists no artifacts; refusing the extension")
        # A segment must still sit on exactly the parents it was published on (same manifest SHA256s).
        if k > 0:
            seg = manifest.get("segment") or {}
            recorded = [(p.get("dir"), p.get("manifest_sha256")) for p in seg.get("parents") or []]
            current = [("." if q == base else q.name, sha) for q, _, sha in chain]
            if seg.get("index") != k or recorded != current:
                raise ValueError(f"extract {part}: its recorded parents {recorded} differ from the published "
                                 f"chain {current} (an ancestor manifest changed); refusing the extension")
        chain.append((part, manifest, hashlib.sha256(data).hexdigest()))
        k += 1
    return chain


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()
