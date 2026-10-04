#!/usr/bin/env python3
"""Backend-dark, fail-closed preflight for the v0234 nine-nest arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-gpt-ninenest-replay")
SPRINT = REPO / ".agent/sprints/2026-07-22-v0234-gpt-ninenest-replay"
INPUT = Path("<DATA_ROOT>/alisios/registry/wrf_cases/payload/canary_all7/run")
TRUTH = Path("<DATA_ROOT>/alisios/registry/wrf_cases/payload/canary_all7/run_cadvariant")
REGISTRY = Path("<DATA_ROOT>/alisios/registry/wrf_cases/cases/canary_all7")
WRF_ROOT = Path("<USER_HOME>/src/wrf_pristine/WRF")

INPUT_HASHES = {
    "namelist.input": "d78f5ffedd7f1b3cf774c5a9b340b7d8f97e60d6646aef10e23468a5094e5e03",
    "wrfbdy_d01": "f2edd0b3eec6151554f2263e961b162db2be6402a7ea86c9521eaabdfc083feb",
    "wrfinput_d01": "5f23828853181dda3f7c558ab618299a7925e26cf5fcf2495e70052cbe88a11a",
    "wrfinput_d02": "4cbaaeff4ba4643da6710842a89586d749b04e0b1ceeb536f1ccabfac7e627e0",
    "wrfinput_d03": "67568190eec6c883e811da5beb1228023b287d5b837d32e67267a2929958dc3a",
    "wrfinput_d04": "aa330c7ebe2369d832842c9c05d1c4a9066fdb2b1aaf8bf6e1876948c17c55f0",
    "wrfinput_d05": "b8c22b3824d4d77c6646d1924a948ad0f3d0fdb355562e7d0b36ae79c1e9389b",
    "wrfinput_d06": "fcb456a895e84ce179b00188953cd0e45ba71dbcd4d8eac09b5e42158ab81f18",
    "wrfinput_d07": "823e87b7f4faef349dc2d10613dc80de42fb6fefbfde4640d8c9b7ad7b8c4d8f",
    "wrfinput_d08": "5e35985590819f919ea3a1bbf5413730c42344612875a818fd6bc30bac7c64a3",
    "wrfinput_d09": "9474cee5d1b607f494c03cb34a66cd15c74b95de3b5f77a0a22b2a1e03b94e74",
}

TRUTH_HASHES = {
    "wrfout_d01_2026-04-28_18:20:06": "f035afc0298266ca1bc6254a692543a9b7e01baf53c68b6a92a50aa2c75f00d4",
    "wrfout_d01_2026-04-28_18:40:12": "a21e2c4c1aeb6474e54361e4e2a1df29bf18862f274d7e4ee7d651ea4059896c",
    "wrfout_d01_2026-04-28_19:00:00": "b16c2ffce6845d391cbd980689818348dd114119b74aadc840e8e9d398f17fd6",
    "wrfout_d02_2026-04-28_18:20:00": "fd383084a5ccaddbd556516ec3c580e4bfaa71c4c7bbca6219a6b0d4c53ec3fe",
    "wrfout_d02_2026-04-28_18:40:00": "50fcb726b77e87a51c98e58f8e0a5b3cc6fbb864e993e3735b40551244e45f40",
    "wrfout_d02_2026-04-28_19:00:00": "c37e1cabede827c2589a7ff672b1c879fef867c6d6e7c87674e68bafaa1c4753",
    "wrfout_d03_2026-04-28_18:20:00": "27b241d9993238b1674439ed0c2df7a95dab5111f5e788e868c5f100b18cc7a3",
    "wrfout_d03_2026-04-28_18:40:00": "94337cfde0a84d622316692065dca47f651f784865b749bcf3b304f445b43f53",
    "wrfout_d03_2026-04-28_19:00:00": "92f0e4eca1691fd9922a6f665b05312f84fd4146ffee22d7975b8f45c4f1beaf",
    "wrfout_d04_2026-04-28_18:20:00": "ca9f545f271285b562fd22de36e1553dc95d04238aba4b7241fcd3dfd51993f3",
    "wrfout_d04_2026-04-28_18:40:00": "b2ced32b55f9b3a09e3c40a3502051a31a07a7927208e4fce789f7247719bb6b",
    "wrfout_d04_2026-04-28_19:00:00": "5c8e8c4a207a7c0e4b0dd8a479a96bb53365bd55c5c9258dff85dd26165a844d",
    "wrfout_d05_2026-04-28_18:20:00": "61e7f4234f2b89e73c19ca74754faee21a7acd0c3e0b1f6a4ce19ae34e86e78d",
    "wrfout_d05_2026-04-28_18:40:00": "1ffe45ec97b7de7c70f3e2559691a4be7e2ebc351cc3eb03515b1cb5ede2502b",
    "wrfout_d05_2026-04-28_19:00:00": "17ba1a745c5f4aea8699730018ae4e6d94ecf721d9aa61af6b7a5f762b82f9d1",
    "wrfout_d06_2026-04-28_18:20:00": "6d0afce6267650e7212fbfaa40508bb4ce76e0732018a29af19640d171d30a5b",
    "wrfout_d06_2026-04-28_18:40:00": "826dacc7bb9fb772f32d13b13d9ad69e2253d104f8c3fcefcfa1e28440a8f89b",
    "wrfout_d06_2026-04-28_19:00:00": "9bc4ddd94e0c322261127f62f5b8bd7472aa6464b64249d3fef78c79b96e74b0",
    "wrfout_d07_2026-04-28_18:20:00": "e030558e69c515bfd4aa8ae5e995f856e0632d8cef5858aef6461765bb1bf9c7",
    "wrfout_d07_2026-04-28_18:40:00": "3c44647a5ca10f77408758129b47ae62009718265b012b8c59c3bdd03e9992ca",
    "wrfout_d07_2026-04-28_19:00:00": "c4bf8fd671961f999e329e1c1c3e6fbf1fce4240eaf82a6146b5a0e7fa7c549c",
    "wrfout_d08_2026-04-28_18:20:00": "895bda77743895a410d8542feb18aa58298657b18d7bd35e71328cac06c1983d",
    "wrfout_d08_2026-04-28_18:40:00": "201afb9f50dc2d9c522c179f31c192f058aa28543eb2ef30f0be1bbcbe3774b6",
    "wrfout_d08_2026-04-28_19:00:00": "139e42dcdfd772d17ed751b21054bf24a3d30b00953b257d22a1c0e06bf791ec",
    "wrfout_d09_2026-04-28_18:20:00": "5f064b48ec299dedd9ca22aca6703598dc6b783e9846466f09205ffc4f5d25cd",
    "wrfout_d09_2026-04-28_18:40:00": "cbc0d4adfb2c869cb1cce7f22a4809dc82702a8771920a38f1baba18f41156de",
    "wrfout_d09_2026-04-28_19:00:00": "7ebc22beb0e8c241f35bbea78534a58f7a3a235a940c5ac64400229400c0ee43",
}

RUNTIME_HASHES = {
    "run/MPTABLE.TBL": "7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d",
    "run/SOILPARM.TBL": "1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f",
    "run/GENPARM.TBL": "9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2",
    "run/RRTMG_LW_DATA": "bcfdee24b63a4c909522a329b8e16c539f0173c7e5aea2caf933ab4fe28c5c97",
    "run/RRTMG_SW_DATA": "a7d25f5b4d33be8629cbef7ecacc1ff413bf398a021297793e843ba1cc627baf",
    "run/CAMtr_volume_mixing_ratio": "9a427fd106f8e36b30e0b29266bff1398b025b82af5b878e5a7a8e9dfe268ca7",
}

REGISTRY_HASHES = {
    "forcing_provenance.json": "0779f080c43809d853376819ea052d30c1dc4fca036b71154e67b0336a554dd4",
    "origin.json": "d5626202d19421925aca7d745038d6ec94278177e45a35b3a84a70f30dc5b576",
    "physical_location.json": "b3a235a743313dd3ac46f7835531b896ae6c1147f53b1fb9e271afa0e091bdc4",
}

REPO_HASHES = {
    "src/gpuwrf/integration/nested_pipeline.py": "77b6a690647ec4b0b615f425338ca41f3e594c29566eeafc0f9e884e1f0e98c4",
    "scripts/compare_wrfout_grid.py": "5099e49d5ae5d6326599027e8ff18a39bc3918519d66d5a31f816bd89b67d43c",
    "proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json": "9df4fab8514523286f07b127eb8bf57a7545459c4448fef721ac44c0ecc351dc",
    "proofs/v019/release_prep/gate_summary.json": "0b781cc733b6071d642338cdf0acfabcc95a7fd0c4b7d618dd501ac6b174e288",
    "proofs/v019/release_prep/grid_compare_summary.json": "e37093640022fc98459c6ddd28e6d9cf20bbdbe6ebd7a2f84da58bbba84b35bd",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(payload: dict) -> str:
    unsigned = {k: v for k, v in payload.items() if k != "canonical_payload_sha256"}
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def verify_hashes(root: Path, expected: dict[str, str]) -> list[dict[str, object]]:
    rows = []
    for rel, wanted in expected.items():
        path = root / rel
        observed = sha256(path) if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": observed,
                "expected_sha256": wanted,
                "pass": observed == wanted,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--authorized-head", required=True)
    parser.add_argument("--cache-mode", choices=("cold_fresh", "aot_seeded_warm"), default="cold_fresh")
    parser.add_argument("--aot-seed-manifest", type=Path)
    parser.add_argument(
        "--input-mode",
        choices=("fresh_namespace_stage", "sealed_aot_source_stage"),
        default="fresh_namespace_stage",
    )
    parser.add_argument("--replay-input-stage", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if re.fullmatch(r"[0-9a-f]{64}", args.nonce) is None:
        raise SystemExit("nonce must be exactly 64 lowercase hex characters")
    if re.fullmatch(r"[0-9a-f]{40}", args.authorized_head) is None:
        raise SystemExit("authorized head must be exactly 40 lowercase hex characters")

    prefix = args.nonce[:16]
    namespace = Path(f"<DATA_ROOT>/wrf_gpu2/v0234_gpt_ninenest_replay_{prefix}")
    audit = json.loads((SPRINT / "CONSUMED_NONCE_AUDIT.json").read_text())
    prior_hits = []
    for sprint_dir in sorted((REPO / ".agent/sprints").glob("2026-07-2*")):
        if sprint_dir == SPRINT:
            continue
        for path in sprint_dir.rglob("*"):
            if path.is_file():
                try:
                    if args.nonce in path.read_text(errors="ignore"):
                        prior_hits.append(str(path.relative_to(REPO)))
                except OSError:
                    continue

    input_rows = verify_hashes(
        INPUT, {k: v for k, v in INPUT_HASHES.items() if k != "namelist.input"}
    )
    input_rows += verify_hashes(TRUTH, {"namelist.input": INPUT_HASHES["namelist.input"]})
    truth_rows = verify_hashes(TRUTH, TRUTH_HASHES)
    runtime_rows = verify_hashes(WRF_ROOT, RUNTIME_HASHES)
    registry_rows = verify_hashes(REGISTRY, REGISTRY_HASHES)
    repo_rows = verify_hashes(REPO, REPO_HASHES)

    source = (REPO / "src/gpuwrf/integration/nested_pipeline.py").read_text()
    launcher_expression_present = (
        'os.environ.get("GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE", "1") == "1"'
        in source
    )
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    model_dirty = git("status", "--porcelain", "--", "src/gpuwrf")
    all_rows = input_rows + truth_rows + runtime_rows + registry_rows + repo_rows
    disk = shutil.disk_usage(namespace.parent)
    seed_manifest = None
    seed_manifest_sha256 = None
    seed_manifest_canonical = None
    seed_manifest_pass = args.cache_mode == "cold_fresh"
    replay_input_stage = None
    replay_input_stage_manifest = None
    replay_input_stage_manifest_sha256 = None
    replay_input_stage_manifest_canonical = None
    replay_input_stage_rows: list[dict[str, object]] = []
    input_provenance_reuse_pass = args.input_mode == "fresh_namespace_stage"
    if args.cache_mode == "aot_seeded_warm" and args.aot_seed_manifest is not None:
        try:
            seed_manifest = json.loads(args.aot_seed_manifest.read_text())
            seed_manifest_sha256 = sha256(args.aot_seed_manifest)
            seed_manifest_canonical = seed_manifest.get("canonical_payload_sha256")
            seed_manifest_pass = (
                seed_manifest.get("verdict") == "AOT_SEED_MANIFEST_PASS"
                and seed_manifest.get("checks", {}).get("exactly_six_seed_files") is True
                and seed_manifest.get("checks", {}).get("all_blob_hashes_match_authenticated_meta") is True
                and seed_manifest.get("checks", {}).get("source_gpuwrf_tree_matches_current") is True
            )
        except (OSError, json.JSONDecodeError):
            seed_manifest_pass = False

    if (
        args.input_mode == "sealed_aot_source_stage"
        and seed_manifest is not None
        and args.replay_input_stage is not None
    ):
        try:
            replay_input_stage = args.replay_input_stage.resolve(strict=True)
            expected_stage = (Path(seed_manifest["source_namespace"]) / "input").resolve(strict=True)
            replay_input_stage_manifest = Path(seed_manifest["source_namespace"]) / "stage_manifest.json"
            replay_input_stage_manifest_sha256 = sha256(replay_input_stage_manifest)
            prior_stage_payload = json.loads(replay_input_stage_manifest.read_text())
            replay_input_stage_manifest_canonical = prior_stage_payload.get("canonical_payload_sha256")
            replay_input_stage_rows = verify_hashes(replay_input_stage, INPUT_HASHES)
            prior_rows = {
                row["name"]: (int(row["bytes"]), str(row["sha256"]), str(row["resolved"]))
                for row in prior_stage_payload["files"]
            }
            current_rows = {
                Path(row["path"]).name: (
                    int(row["bytes"]),
                    str(row["sha256"]),
                    str((replay_input_stage / Path(row["path"]).name).resolve(strict=True)),
                )
                for row in replay_input_stage_rows
            }
            input_provenance_reuse_pass = (
                args.cache_mode == "aot_seeded_warm"
                and replay_input_stage == expected_stage
                and prior_stage_payload.get("schema") == "wrfgpu2.v0234.ninenest-stage-manifest.v1"
                and prior_stage_payload.get("read_only_symlink_stage") is True
                and prior_rows == current_rows
                and all(bool(row["pass"]) for row in replay_input_stage_rows)
            )
        except (KeyError, OSError, json.JSONDecodeError, ValueError):
            input_provenance_reuse_pass = False

    checks = {
        "head_matches_authorization": head == args.authorized_head,
        "branch_matches": branch == "worker/gpt/v0234-ninenest-replay",
        "model_tree_clean": model_dirty == "",
        "all_pinned_hashes_match": all(bool(row["pass"]) for row in all_rows),
        "launcher_default_expression_present": launcher_expression_present,
        "nonce_not_consumed": args.nonce not in audit["consumed_nonces"],
        "nonce_absent_from_prior_sprints": not prior_hits,
        "namespace_absent": not namespace.exists(),
        "free_disk_at_least_8_gib": disk.free >= 8 * 1024**3,
        "cache_authority_valid_for_mode": seed_manifest_pass,
        "input_provenance_reuse_valid_for_mode": input_provenance_reuse_pass,
    }
    verdict = "STATIC_PREFLIGHT_PASS" if all(checks.values()) else "STATIC_PREFLIGHT_FAIL"
    payload = {
        "schema": "wrfgpu2.v0234.ninenest-static-preflight.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "gpu_commands": 0,
        "gpu_queries": 0,
        "jax_or_gpuwrf_imported": False,
        "nonce": args.nonce,
        "nonce_prefix": prefix,
        "namespace": str(namespace),
        "cache_destination": str(namespace / "jax_cache"),
        "authorized_head": args.authorized_head,
        "observed_head": head,
        "branch": branch,
        "checks": checks,
        "prior_nonce_hits": prior_hits,
        "free_disk_bytes": disk.free,
        "configuration": {
            "init": "2026-04-28T18:00:00Z",
            "hours": 1,
            "max_dom": 9,
            "history_interval_minutes": 20,
            "expected_wrfout_count": 27,
            "boundary_default_environment": None,
            "cache_mode": args.cache_mode,
            "input_mode": args.input_mode,
            "fresh_cache_destination": True,
            "aot_seed_manifest": str(args.aot_seed_manifest) if args.aot_seed_manifest else None,
            "aot_seed_manifest_sha256": seed_manifest_sha256,
            "aot_seed_manifest_canonical_payload_sha256": seed_manifest_canonical,
            "replay_input_stage": str(replay_input_stage) if replay_input_stage else None,
            "replay_input_stage_manifest": (
                str(replay_input_stage_manifest) if replay_input_stage_manifest else None
            ),
            "replay_input_stage_manifest_sha256": replay_input_stage_manifest_sha256,
            "replay_input_stage_manifest_canonical_payload_sha256": replay_input_stage_manifest_canonical,
            "namespace_input_is_symlink": args.input_mode == "sealed_aot_source_stage",
        },
        "files": {
            "inputs": input_rows,
            "truth": truth_rows,
            "runtime": runtime_rows,
            "registry": registry_rows,
            "repo_authorities": repo_rows,
            "reused_input_stage": replay_input_stage_rows,
        },
    }
    payload["canonical_payload_sha256"] = canonical(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "out": str(args.out), "canonical_payload_sha256": payload["canonical_payload_sha256"]}, indent=2))
    return 0 if verdict == "STATIC_PREFLIGHT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
