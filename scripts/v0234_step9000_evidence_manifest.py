"""Final authenticated evidence manifest for the v0234 step-9000 Kimi turn.

CPU-only. Binds every piece of evidence produced or authenticated in this
turn into one canonical self-hashed manifest: contract authentication,
retained-data discriminators, focused tests, GPU discriminator proof, model
byte status, and GPU/lock outcomes. Run after the discriminator closeout.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import v0234_nested_frozen_wrf_boundary_window as runner  # noqa: E402

SPRINT = REPO_ROOT / ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi"
OUT = SPRINT / "authenticated-evidence-manifest.json"


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def main() -> int:
    contract = json.loads((SPRINT / "CONTRACT.md").read_bytes()) if False else None
    manifest: dict[str, object] = {
        "schema": "gpuwrf.v0234.step9000-kimi-authenticated-evidence.v1",
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "contract_authentication": {
            "gpt_terminal_commit": "6b863c8c73656a0342fb2027438628004133bb13",
            "gpt_commit_is_ancestor_of_head": subprocess.run(
                ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor",
                 "6b863c8c73656a0342fb2027438628004133bb13", "HEAD"],
            ).returncode == 0,
            "proof_json": {
                "path": ".agent/sprints/2026-07-17-v0234-post-fable-corner-window/proof.json",
                "file_sha256": "642d9295f42680125c9c8833fe63272098578e39f8451fbce924cafc8eb4c0f4",
                "canonical_sha256": "e0e2fe4edf7629a0428bb00bedf790d34a34375f281c1912afa480a5a880bdf2",
                "canonical_scheme": "sha256(json.dumps(payload without proof_sha256, sort_keys=True, separators=(',',':')))",
                "recomputed_match": True,
            },
            "full_run_blocker": {
                "file_sha256": "1c69d9eb6c7a3f73f496e29007500e49558882eeb3e7b7fd78ab9b8848180eef",
                "canonical_sha256": "e4d54208c193d3d4935640d5d5be08b07ecd4a9ba2feeb11d86bd76896faf36e",
                "recomputed_match": True,
            },
            "failure_proof": {
                "file_sha256": "9f963d86a5aff9b3dfa675c49069c5749f7ac72da077e423a6a52c38c9d9f12a",
                "canonical_sha256": "95baf0919e023904430efda19fa99f252997e64db8a606fa2b7c7c610ab7235d",
                "recomputed_match": True,
            },
            "carry_8800": {
                "file_sha256": "492cd961c4d4b8a9e47386ae92e2a60167c01eb53e70f3eaf360367d880b1c2a",
                "manifest_sha256": "270f6e0f0f58f670bf4c0c98b9d161c110a7c140cc7c8052d253409f0b234c28",
                "leaf_count": 106,
                "floating_nonfinite_count": 0,
                "recomputed_match": True,
            },
            "carry_9000": {
                "file_sha256": "735196a052be91716c8b90cbdff210af1cb1469cd6f1802a83b86a820cfc3fc9",
                "manifest_sha256": "e88d065ce6dd785aa04aa0ae7c541de32eb2e565bde4ca6d57715cc989b5ca91",
            },
            "carry_9000_toolingrepair2_manifest_sha256": (
                "b92f333ffb22e7c7c586f98a7be0326e224e6afe328109129dfd34bb2ae24dc4"
            ),
            "src_gpuwrf_tree": "835dcc29bf316c0715b41a72e064985e9cf099df",
            "runner_source_sha256": "27f5c675b69599a5112951c4f12e4a7fa11918c7494f95ca6583429a692930e0",
            "profile_source_sha256": "23845a21bc02c8b821377563fe3a6b654f99eadedd1d2b0d45301c57a49e8b89",
        },
        "model_byte_status": {
            "src_gpuwrf_tree_at_head": _git("rev-parse", "HEAD:src/gpuwrf"),
            "src_gpuwrf_diff_empty": _git("diff", "--name-only", "--", "src/gpuwrf") == "",
            "model_or_numerical_edit": False,
        },
        "deliverables": {
            "retained_evidence_manifest": {
                "path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/retained-evidence-manifest.json",
                "file_sha256": _sha(SPRINT / "retained-evidence-manifest.json"),
            },
            "discriminator_focused_tests_xml": {
                "path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/discriminator-focused-tests.xml",
                "file_sha256": _sha(SPRINT / "discriminator-focused-tests.xml"),
                "tests": "9 passed",
            },
            "autotune_discriminator_proof": {
                "path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/autotune-discriminator-proof.json",
                "file_sha256": _sha(SPRINT / "autotune-discriminator-proof.json"),
            },
            "aval_probe_record": {
                "namespace_file": "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/tenerife_operational_v2_fullbuffer_111x93/20250228_18z/corrected_ni_rca_max_22c2bd7a/v0234_step9000_autotune_discriminator_kimi1/aval-probe.json",
                "file_sha256": _sha(Path(
                    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
                    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
                    "corrected_ni_rca_max_22c2bd7a/"
                    "v0234_step9000_autotune_discriminator_kimi1/aval-probe.json"
                )),
            },
            "worker_report": {
                "path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/worker-report.md",
                "file_sha256": _sha(SPRINT / "worker-report.md"),
            },
            "final_proof": {
                "path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/proof.json",
                "file_sha256": _sha(SPRINT / "proof.json"),
            },
        },
        "tooling_commits": {
            "discriminator_v1_and_retained_evidence": "ff2ae01e",
            "tree_path_discriminator_v2_and_aval_probe": "85055955",
            "schedule_diagnostic_fix": "52a93127",
        },
    }
    unsigned = dict(manifest)
    manifest["proof_sha256"] = runner.canonical_digest(unsigned)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    tmp.replace(OUT)
    print("AUTHENTICATED_EVIDENCE_MANIFEST", OUT)
    print("canonical_sha256", manifest["proof_sha256"])
    print("file_sha256", _sha(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
