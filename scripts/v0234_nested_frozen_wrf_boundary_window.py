#!/usr/bin/env python3
"""Reserved runner for the accepted nested frozen-WRF boundary bundle.

The module top level is intentionally Python-stdlib-only.  Admission authority,
environment, source, lock, cache, input, and PREEMPT checks all complete before
``_import_runtime`` can import JAX or gpuwrf.  ``--cpu-dry-run`` never calls that
import hook, never verifies/acquires a live GPU lease, and never executes model
code.

The admitted runtime reuses the corrected ordinary-bisection orchestration.  It
builds one three-domain prefix, compiles one ordinary d03 one-step executable,
and applies bounded health reductions only between completed dispatches.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import re
import stat
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "gpuwrf.v0234.nested-boundary-final-window.v1"
CPU_PROOF_SCHEMA = "gpuwrf.v0234.nested-boundary-final-runner-cpu-audit.v1"
AUDIT_ADMISSION = "READY_FOR_BOUNDARY_FINAL_GPU_REPLAY"

REPO_ROOT = Path(__file__).resolve().parents[1]
SPRINT_DIR = REPO_ROOT / ".agent/sprints/2026-07-14-v0234-nested-boundary-final-runner"
RUNNER_SOURCE = Path(__file__).resolve()

BASE_CANDIDATE_COMMIT = "4484be85c5b720dbed8b8a3bb792711255af9c1b"
PARTIAL_WIND_CANDIDATE_COMMIT = "2c13b73112d9877d603324d66b127ecad60bf7e3"
PARTIAL_WIND_CANDIDATE_TREE = "f3929bfa2125129181068ffee08245166f4981a8"
PARTIAL_WIND_OPERATIONAL_MODE_SHA256 = "66cd3511c464736f559494e86329618fcf2544de54704c8878d1678b10a31982"
THETA_CANDIDATE_COMMIT = "e06583f070c6bc1b14083dc5241ccdc618ab8f32"
THETA_OPERATIONAL_MODE_SHA256 = "276d90e3b68ccbd6db72b4c8739f6e4210a5551dd4f84f6c2d6678ee2cd943fa"
DIFFOPT_PARENT_COMMIT = "11c2a08528b72c9d3ca2e7ac9dffd8b37dd6fb8d"
DIFFOPT_PARENT_TREE = "dc74b84b74489ebfcdf79e16b696dd59299fd6b4"
DIFFOPT_PARENT_OPERATIONAL_MODE_SHA256 = "6f1a9e697c4cf2167f23ce6d25cf6995b83b301c0c896139e2f435c2479b9bf6"
DIFFOPT_PARENT_NESTED_PIPELINE_SHA256 = "9fa7396468f09b9e913f516c12425fc8f6847eaf2a47dc43af4fcf8aff82230a"
SCALAR_CANDIDATE_COMMIT = "aca6b55bdf0c5feac749f5f94358c73fb284507e"
SCALAR_CANDIDATE_TREE = "b26f0712d308db52d3fd46fddbf8c3926b47802d"
CANDIDATE_COMMIT = "cb46ef1b3179871382c1277adc09056ade5c5cec"
CANDIDATE_TREE = "5fc7310f95d71cfe87ac9fc9d64205419a77a843"
FINAL_NI_DIR = REPO_ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
NESTED_ADV_AMENDMENT = FINAL_NI_DIR / "contract-amendment-05.json"
NESTED_ADV_AMENDMENT_SHA256 = "4cfbfc886ecfb24afe4bc9faeabe395b1064363c1bd8d8670ce0992deff99e59"
NESTED_ADV_AMENDMENT_PAYLOAD_SHA256 = "e786f8cb3a92dc169fab128d4953626d14156f4e6163c4fb6de88973db280622"
NESTED_ADV_PROOF = FINAL_NI_DIR / "nested-advection-degrade-proof.json"
NESTED_ADV_PROOF_SHA256 = "3625019759b710d7b4e2ef7bc7b7eda516692eed9bfc79f4dfad5e52739db360"
NESTED_ADV_PROOF_PAYLOAD_SHA256 = "f92630fc1aa77ae0fbc0be10c3e6d1052d3a56cab2f5a0c3f5c3c9f648911fb0"
THETA_AMENDMENT = FINAL_NI_DIR / "contract-amendment-06.json"
THETA_AMENDMENT_SHA256 = "c773843a48594f948611bd2a551d1af0f4a218640e61bbe9c10482729a8542aa"
THETA_AMENDMENT_PAYLOAD_SHA256 = "a686a00538ae76abb84e823cba8910b002f353963044b5bcd30a1ed9a507f8f6"
THETA_SOURCE_PROOF = FINAL_NI_DIR / "theta-unlimited-source-proof.json"
THETA_SOURCE_PROOF_SHA256 = "18e7f7654fcf226f7b439c514ef0bc041f6bcb441ce19e8b895e0e645703f552"
THETA_SOURCE_PROOF_PAYLOAD_SHA256 = "e0735ca4ce22cf913379bd2c2bb80a9e7947f49c7914ae6d6aa1b96189bbd292"
DIFFOPT_AMENDMENT = FINAL_NI_DIR / "contract-amendment-11.json"
DIFFOPT_AMENDMENT_SHA256 = "b2fe695a68144a12a07b29b74f4bdbc1b90386834a062053e4b8c81089a18765"
DIFFOPT_CPU_AB_PROOF = FINAL_NI_DIR / "nested-diffopt1-rk1-forward-bundle-cpu-ab-proof.json"
DIFFOPT_CPU_AB_PROOF_SHA256 = "f54724d0a27dec1cc025bf64065b34dd50c9d92adbbdc54b2b6ab41dd7039e61"
DIFFOPT_CPU_AB_PAYLOAD_SHA256 = "96f3ceda7b41ec86bc007f5ef5ae9c705303d73687f0ecd1dcd6f73f53426322"
DIFFOPT_CANDIDATE_PROOF = FINAL_NI_DIR / "nested-diffopt1-rk1-forward-bundle-candidate-proof.json"
DIFFOPT_CANDIDATE_PROOF_SHA256 = "9c9c9f7fe44fe309575db127e3dab04c6da59ba8ff3a992be2c11019b9b7b9a0"
DIFFOPT_CANDIDATE_PAYLOAD_SHA256 = "2cb85732fdf24ca6fea399b80611db6b05d6483e8b9ae6c927abfef0887de87e"
SCALAR_AMENDMENT = FINAL_NI_DIR / "contract-amendment-12.json"
SCALAR_AMENDMENT_SHA256 = "a2d01ebdb3bdcb4deecb260a20b341668eb1b500e2af4e041b951cceb2938da8"
SCALAR_AMENDMENT_PAYLOAD_SHA256 = "2a8a6ae3b4967a44ff2f189499d1c0da9de6e6ea7287de5c3f33f477049a14f3"
SCALAR_ORACLE_PROOF = FINAL_NI_DIR / "nested-scalar-diffusion-source-oracle.json"
SCALAR_ORACLE_PROOF_SHA256 = "108fbaa27f25ad758a99296792221622f2d57257a09b24646bb172efe4b2131e"
SCALAR_ORACLE_PAYLOAD_SHA256 = "2c0f76140517b5cad5af95d262a75e5ac858eb70a413f189cd3f5baea9e0a8e9"
SCALAR_CPU_AB_PROOF = FINAL_NI_DIR / "nested-scalar-diffusion-complete-cpu-ab-proof.json"
SCALAR_CPU_AB_PROOF_SHA256 = "67c792a60bcac15468657d174b24091afaf2fd0055f3f11ad5234927c0449c2d"
SCALAR_CPU_AB_PAYLOAD_SHA256 = "6a81499a3c72ad33fdbfd9967422dd846ce03a671a144274927a59715f8dc18f"
SCALAR_CANDIDATE_PROOF = FINAL_NI_DIR / "nested-scalar-diffusion-candidate-proof.json"
SCALAR_CANDIDATE_PROOF_SHA256 = "f1d86e6d1d1452cf9633109f02091f1db66e4b0a9a0046d0346e8997ef8c8426"
SCALAR_CANDIDATE_PAYLOAD_SHA256 = "0890954fee27312ca9b6a5c680e5a34238c5b9081e81818687af08c6be1e1fb7"
T_SOURCE_AMENDMENT = FINAL_NI_DIR / "contract-amendment-13.json"
T_SOURCE_AMENDMENT_SHA256 = "63e5284de46e2e269f42b1bc7a3cc2708f77d2346f186491fdd22deef1131d5f"
T_SOURCE_AMENDMENT_PAYLOAD_SHA256 = "2b274ae10e02963de5ee58ec73e65a3a39f38b84bf9204fbe8405ec9cc51beb9"
T_SOURCE_CPU_AB_PROOF = FINAL_NI_DIR / "nested-t-source-cadence-cpu-ab-proof.json"
T_SOURCE_CPU_AB_PROOF_SHA256 = "036f84943145cd629b0688cec4093754b7f57bc6167a7722ad2ba4e9763331ce"
T_SOURCE_CPU_AB_PAYLOAD_SHA256 = "d3b6f72269d845370ef90f35434290a87c07f8836ee31ef6ea437f4b45cb47fb"
T_SOURCE_CANDIDATE_PROOF = FINAL_NI_DIR / "nested-t-source-cadence-candidate-proof.json"
T_SOURCE_CANDIDATE_PROOF_SHA256 = "292dbc9c6cb8a2f85d84454900ffc1e0592a7813101eaf55a11bb94843186bd4"
T_SOURCE_CANDIDATE_PAYLOAD_SHA256 = "ec2ca0c5a343f0d73ec4fb856d86f6319bada46d4d003ecd7a9c31e00018610f"
OWNER_OVERRIDE = SPRINT_DIR / "final-ni-full18h-owner-override.json"
OWNER_OVERRIDE_SHA256 = "fc9505236ffc3a2e08c935cb0a5e0f2b2fb03e4961f8a614ff90da268e96de0e"
OWNER_OVERRIDE_PAYLOAD_SHA256 = "33bf3e03dafb09cd91685486580e4ffa78ce962decf7d9940f00c1c661e15793"
TERMINAL_CANDIDATE_PROOF = FINAL_NI_DIR / "final-candidate-proof.json"
TERMINAL_CANDIDATE_PROOF_SHA256 = "99b9e7c5a4cbb0062d9c8e2cc5bceb3d81a063e12623a2f2793b05ecee944f58"
TERMINAL_CANDIDATE_PAYLOAD_SHA256 = "30d6fc157186bf48643e9886371be4d9f6b6afaa5c476b9ba203038319fff866"
CPU_CANDIDATE_PROOF = FINAL_NI_DIR / "candidate-cpu-proof.json"
CPU_CANDIDATE_PROOF_SHA256 = "419fb10b1b69d2e41daec7f350c361b481a4e1d2d45d1840dec335ef94937e37"
CPU_CANDIDATE_PAYLOAD_SHA256 = "622c591eeb43f405b8fa3aa65b7a503d82a0f96b33ac6c9a1405cca5fc0dade7"
GPU_POLICY_PROOF = FINAL_NI_DIR / "bounded-gpu-policy-proof.json"
GPU_POLICY_PROOF_SHA256 = "0ade432cae45e21c63740e3f9e297f2101e30ce6054135797cbff55f4f866595"
GPU_POLICY_PAYLOAD_SHA256 = "965cf67021332a01b85a8b7cc1d92cb8d45d72fa5c775c9c9c58771fc40179be"
INTERFACE_PROOF = FINAL_NI_DIR / "interface-transfer-proof.json"
INTERFACE_PROOF_SHA256 = "9c92ab5c3be4f2ce83110d75fffe42f51d38895eb5bae8509f644fcc02a7e926"
INTERFACE_PAYLOAD_SHA256 = "a44127d285d21721a1cb75651c8b311935f7d34682d9f59273b57b58a99ffdd8"
FINAL_NI_AUTHORITY_PROOF = FINAL_NI_DIR / "authority-proof.json"
FINAL_NI_AUTHORITY_PROOF_SHA256 = "0c17ab23cb35abf58f54e1ea89488c2795f9445acf829129f70cf437dc5d62e9"
FINAL_NI_AUTHORITY_PAYLOAD_SHA256 = "e837097198e3b6058cd9ee98e0b2a614d22df671c80c22ee0357cd66f3e36edc"
LOCALIZATION_PROOF = FINAL_NI_DIR / "carry-localization-proof.json"
LOCALIZATION_PROOF_SHA256 = "c5727c8a01ae3aa55b7f68bfd18e7339a5e27cd4be11c0c318e14bd1018aecc7"
LOCALIZATION_PAYLOAD_SHA256 = "7a8a7a2cec47e6c3aedfeeadaedd05721eaf36d0618900fb19f41d574337ad64"
SOURCE_INVARIANT_PROOF = FINAL_NI_DIR / "source-invariant-proof.json"
SOURCE_INVARIANT_PROOF_SHA256 = "1a39cf49b6e66d4b6632a0dc424d6a73530c66440ad37dcdd7e614e581c9ac49"
SOURCE_INVARIANT_PAYLOAD_SHA256 = "3836b62679c5944e7c3c62a20966e63f5d328168bb5f4d5fd95aa77a8f4dd242"
CORRECTED_CPU_ARM_PROOF = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_final_ni_fable5_cpuab/"
    "corrected/cpu-ab-corrected-proof.json"
)
CORRECTED_CPU_ARM_PROOF_SHA256 = "81d586861a8628a94a9307c52df0850318d5f641b28d6b79d334c876b98a61f8"
CORRECTED_CPU_ARM_PAYLOAD_SHA256 = "bccc84c65c92cb07c5575c0c5b9e87ee534f699e46ba3a7cd79ae2335ad448b1"
FALSIFIED_CPU_ARM_PROOF = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_final_ni_fable5_cpuab/"
    "falsified-stop200/cpu-ab-falsified-proof.json"
)
FALSIFIED_CPU_ARM_PROOF_SHA256 = "d096cd6fc37851261b60237440a580eb8db4d0813f4b8a970a19726b03c78f28"
FALSIFIED_CPU_ARM_PAYLOAD_SHA256 = "9fe217eda484b2c71cf3da27f662b55cf5d2e3473110b349af7e70396873c5d2"
RAW_GPU_PROOF = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_final_ni_fable5_terminal3/"
    "bounded-gpu-proof.json"
)
RAW_GPU_PROOF_SHA256 = "7cfaff7b267a4acd7254c7df92a940abfc327268e1260f709ec61a10e8d78e97"
RAW_GPU_PAYLOAD_SHA256 = "5749e87848f5bf0fc543364ac8f29a57a6c0f2d982c837485f3e801958b5dd5c"
LAUNCH_COMMAND = SPRINT_DIR / "nested-t-source-cadence-full18h-exact-launch-command.txt"
RUNNER_CPU_AUDIT = SPRINT_DIR / "nested-t-source-cadence-full18h-runner-cpu-proof.json"
BUNDLE_FLAG = "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE"
BUNDLE_FLAG_VALUE = "1"

CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
TERMINAL_CONTRACT = CASE_ROOT / "terminal_cpu_authority_contract_v1.json"
TERMINAL_CONTRACT_SHA256 = "26b16e21782400308ec4d58175907796bf8aaddb01062bb06be8b87a2a88848a"
INPUT_DIR = CASE_ROOT / "run/wrf"
CPU_MANIFEST = CASE_ROOT / "run/cpu_oracle_manifest.json"
CPU_MANIFEST_SHA256 = "c8f087d7d6ca2e0b8d2e959429eb81c4584d3d1f94ef944723774b347ba831ad"
INPUT_HASHES = {
    "namelist.input": "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838",
    "wrfbdy_d01": "1b5b20408b3384b2e81dc09f3ec029c3110117745e623dc0b7669f0976121fec",
    "wrfinput_d01": "afd069d201c14f0001308696f3b5b2f30f4e1b77f368d15a0d9d7c5b77dc0756",
    "wrfinput_d02": "ec281f234940d7353b9fdf544833db635b4388e83bab4a7a71e025f23f36b964",
    "wrfinput_d03": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
}
RETRY20_ROOT = CASE_ROOT / "gpu_validation_retry20_relative_rmse_3ee02c19"
RETRY20_WRF_ROOT = RETRY20_ROOT / "authority/wrf_root"
RETRY20_PAIRS = RETRY20_ROOT / "incremental-pairs.json"
RETRY20_PAIRS_SHA256 = "9436da45de29518abc25c7fbe7f458f0323057f5768dc2b38b6e96e4713b4ffd"
RETRY20_ATTESTATION = RETRY20_ROOT / "gpu-proof/retry20-runtime-source-attestation.json"
RETRY20_ATTESTATION_SHA256 = "fec1ea5e1be9ca6f272dbbf955a7cb96ade1710becedf87c6cc5b194a45e8a26"
RETRY20_FINAL_ACCEPT = RETRY20_ROOT / "retry20-final-accept.json"
RETRY20_FINAL_ACCEPT_SHA256 = "72e9e6df6e23dc3889586afdc27b782de19aba54a8c7f62060baf3d2087648c8"
RETRY20_CACHE_AUTHORITY = RETRY20_ROOT / "retry20-cache-source-authority.json"
RETRY20_CACHE_AUTHORITY_SHA256 = "35b03ad00f6b652c60fbc2c72a9e1228c8823c95f87bb4c13299ba5817ecdb5c"
LINEAGE_WORK_DIR = CASE_ROOT / "corrected_ni_rca_max_22c2bd7a"
LINEAGE_CACHE = LINEAGE_WORK_DIR / "cache/bb2ffe33dbff-8ec38f8e1a90/jit"

LOCK_ROOT = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-gpu-lock-v2")
LOCK_COMMIT = "8152309aff1e85e1052d44d549a5a5409e710bdd"
LOCK_WRAPPER = LOCK_ROOT / "scripts/with_gpu_lock.sh"
LOCK_VERIFIER = LOCK_ROOT / "scripts/gpu_lock_v2.py"
LOCK_WRAPPER_SHA256 = "c75b3a4eda17e94df921986e1c15fcc71e182e01d51b6fe877e4c52533077e1a"
LOCK_VERIFIER_SHA256 = "ec911f565ee9d2c58dee81d8d1e17411ed677be500a57e73ded5f793e2c4187d"
LOCK_LABEL = "v0234-nested-t-source-full18h"
LOCK_INTENT = "production-preemptible"
EXPECTED_AFFINITY = [12, 13, 14, 15]
CPU_FOCUSED_TEST_ARGS = (
    "tests/test_v0234_nested_frozen_wrf_boundary_window.py",
    "-k",
    "not real_schema_parent_cpu_rehearsal",
)

FAILED_PRODUCTION_NAMESPACE = "nested_boundary_final_985f5714"
CUSTOM_CALL_DIAGNOSTIC_NAMESPACE = "nested_boundary_final_985f5714_custom_call_diag1"
REPAIRED_PRODUCTION_NAMESPACE = "nested_boundary_final_985f5714_gatefix1"
OWNER_OVERRIDE_NAMESPACE = "nested_boundary_final_4484be85_full18h_owner_override1"
FULL_REPLAY_NAMESPACE = "nested_t_source_cb46ef1b_full18h_discriminator1"
REQUIRE_KNOWN_1500_V10_RECORD = False
REQUIRE_TOOLING_CRITIC_ACCEPT = False
TOOLING_CRITIC_SCHEMA = "gpuwrf.v0234.post-fable-fulltree-tooling-critic.v1"
TOOLING_CRITIC_VERDICT = "KIMI_TOOLING_CRITIC_ACCEPT"
TOOLING_CRITIC_PROFILE_SOURCE: Path | None = None
TOOLING_CRITIC_REVIEWED_PATHS: tuple[Path, ...] = ()
TOOLING_CRITIC_REQUIRED_PAYLOAD: dict[str, Any] = {}
TOOLING_CRITIC_AUTHORITY_HOOK: Callable[[], Mapping[str, Any]] | None = None
KNOWN_1500_V10_ARTIFACT: Path | None = None
KNOWN_1500_V10_ARTIFACT_SHA256: str | None = None
KNOWN_1500_V10_PAYLOAD_SHA256: str | None = None
ALLOW_DIAGNOSTIC_TERMINAL_WAKE_RED = False
DIAGNOSTIC_TERMINAL_ALLOWED_FIELDS: frozenset[str] = frozenset()
PARTIAL_WIND_REPLAY_NAMESPACE = "nested_advection_degrade_2c13b731_full18h_discriminator1"
CONTINUATION_NAMESPACE = "nested_boundary_final_985f5714_resume8800_ba7beb32"
PARENT_JOIN_NAMESPACE = "nested_boundary_final_985f5714_parent_join_resume8800_ba7beb32_repair1"
RC3_RUN_DIR = LINEAGE_WORK_DIR / REPAIRED_PRODUCTION_NAMESPACE
RC3_FAILURE_PROOF = RC3_RUN_DIR / "failure/failure-proof.json"
RC3_FAILURE_PROOF_SHA256 = "4a24288eeb6df6b06cea09ab8116f537485089c5a68aad00bd6dbf2b893d312c"
RC3_FAILURE_PROOF_PAYLOAD_SHA256 = "ba7beb322e142c2a42a0357ae67b0a7acd1e334b855009cb40691aa228cce597"
RC3_STEP8800_CARRY = RC3_RUN_DIR / "failure/first-failed-d03-step-8800.pkl"
RC3_STEP8800_CARRY_SHA256 = "4bfe21f007c30e44102715495916651fd97aea3c8b5f81329886c1258872e786"
RC3_STEP8800_WRFOUT = RC3_RUN_DIR / "output/wrfout_d03_2025-03-01_14:40:00"
RC3_STEP8800_WRFOUT_SHA256 = "3a6a5ac5edfc2bd9aff113f3c7553257fef6312f42ee2735aa47336d8e5ac19f"
RC3_LAST_PAIR = RC3_RUN_DIR / "frame-pairs/d03-step-08600.json"
RC3_LAST_PAIR_SHA256 = "2542195d4e3ddb56be29adfebb29af2c70f52db560cffeacc8e624f7b3ff6449"
RC3_RAW_OUTPUT_COUNTS = {"d01": 15, "d02": 15, "d03": 45}
RC3_FRAME_PAIR_COUNTS = {"d01": 15, "d02": 15, "d03": 44}
RC3_EXPECTED_PARENT_STEPS_AT_D03_8800 = {"d01": 978, "d02": 2934}
BOUNDARY_TARGET_TRANSITION_FIELDS = (
    "u_bdy", "v_bdy", "theta_bdy", "qv_bdy", "ph_bdy", "mu_bdy",
    "w_bdy", "p_bdy", "pb_bdy", "phb_bdy", "mub_bdy",
)
BOUNDARY_TARGET_TRANSITION_LEAF_INDICES = tuple(range(38, 49))

# The retained lower-only artifact contains only this JAX CUDA typed-FFI target.
# jax/_src/lax/linalg.py constructs it for tridiagonal_solve and
# jaxlib/gpu_sparse.py registers the CUDA/ROCm gtsv2 device-library targets.
# The spelling is intentionally exact: every other custom target fails closed.
ALLOWED_DEVICE_CUSTOM_CALL_TARGETS: frozenset[str] = frozenset({
    "cusparse_gtsv2_ffi",
})
FORBIDDEN_HLO_TEXT_TOKENS = (
    "xla_python_cpu_callback",
    "host_callback",
    "io_callback",
    "pure_callback",
    "debug_callback",
    "outside_compilation",
    "host_transfer",
    "device_to_host",
    "host_send",
    "host_recv",
    "stablehlo.send",
    "stablehlo.recv",
    "mhlo.send",
    "mhlo.recv",
    "corrected_ni_rca",
    "rca_acoustic",
    "phase_tap",
    "recorder",
)
FORBIDDEN_CUSTOM_TARGET_FRAGMENTS = (
    "callback", "outside_compilation", "host", "python", "debug", "send", "recv",
)

PREEMPT_PATHS = (
    Path("/tmp/PREEMPT_GPU"),
    Path("/tmp/PREEMPT_PRODUCTION"),
    Path("<DATA_ROOT>/alisios/state/PREEMPT_PRODUCTION"),
    Path("<DATA_ROOT>/alisios/state/PREEMPT_GPU"),
)
HOLD_PATHS = (
    Path("/tmp/HOLD_GPU"),
    Path("/tmp/HOLD_WRFGPU"),
    Path("<DATA_ROOT>/alisios/state/HOLD_GPU"),
    Path("<DATA_ROOT>/alisios/state/HOLD_WRFGPU"),
)
NIGHTLY_ACTIVE = Path("<DATA_ROOT>/alisios/state/nightly18z/active.json")
GPU_LOCK_HOLDER = Path("/tmp/wrf_gpu2_gpu.lock.holder")
PRODUCTION_GPU_PROCESS_MARKERS = ("gpuwrf", "nsys", "ncu")
PRODUCTION_GPU_PROCESS_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:gpuwrf|nsys|ncu)(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)
CPU_ONLY_JAX_PLATFORM_PATTERN = re.compile(
    r"(?<!\S)JAX_PLATFORM(?:S|_NAME)=(?:cpu|'cpu'|\"cpu\")(?=\s|$)",
    flags=re.IGNORECASE,
)
CUDA_HIDDEN_PATTERN = re.compile(
    r"(?<!\S)CUDA_VISIBLE_DEVICES=(?:''|\"\")?(?=\s|$)",
)

PREFIX_OWN_STEPS = {"d01": 1022, "d02": 3066, "d03": 9198}
WINDOW_OWN_STEPS = {"d01": 1045, "d02": 3135, "d03": 9405}
TERMINAL_OWN_STEPS = {"d01": 1200, "d02": 3600, "d03": 10800}
PARENT_JOIN_OWN_STEPS = {"d01": 978, "d02": 2934, "d03": 8800}
PARENT_ALIGNED_OWN_STEPS = {"d01": 978, "d02": 2934, "d03": 8802}
PARENT_RECON_SEGMENTS = (67,) * 14 + (40,)
PARENT_DOMAINS = ("d01", "d02")
PARENT_CATCHUP_D03_STEPS = (8801, 8802)
PARENT_STAGE_9000_ROOT_STEPS = 22
PARENT_STAGE_9405_ROOT_STEPS = 45
PARENT_STAGE_TERMINAL_ROOT_STEPS = 155
PREFIX_SEGMENTS = (67,) * 15 + (17,)
WINDOW_D03_STEPS = tuple(range(9199, 9406))
RUN_START = datetime(2025, 3, 1, tzinfo=timezone.utc)
HISTORY_INTERVAL_SECONDS = {"d01": 3600, "d02": 3600, "d03": 1200}
DT_SECONDS = {"d01": 54, "d02": 18, "d03": 6}
DECISIVE_D03_STEP = 9000
CHECKPOINT_D03_STEPS = (8800, 9000)
LATE_WINDOW_RETAIN_D03_STEPS = (9313, 9314, 9405)
FIRST_PROGRESS_D03_STEPS = (0, 200)

EARLY_CAUSAL_PRIOR = LINEAGE_WORK_DIR / (
    "nested_boundary_final_4484be85_full18h_owner_override1/output/"
    "wrfout_d03_2025-03-01_00:20:00"
)
EARLY_CAUSAL_PRIOR_SHA256 = "79492ab0b0f8482809a2486f72969973f13dbd35f374801cd787156238a72f7c"
EARLY_CAUSAL_PARTIAL_WIND = LINEAGE_WORK_DIR / (
    "nested_advection_degrade_2c13b731_full18h_discriminator1/output/"
    "wrfout_d03_2025-03-01_00:20:00"
)
EARLY_CAUSAL_PARTIAL_WIND_SHA256 = "3cf8dcda34ae469df86028209f5f8c2ed59ef4fb1cfd38e2148ee33bcc5ab599"
EARLY_CAUSAL_SCALAR_PARENT = LINEAGE_WORK_DIR / (
    "nested_scalar_diffusion_aca6b55b_full18h_discriminator1/output/"
    "wrfout_d03_2025-03-01_00:20:00"
)
EARLY_CAUSAL_SCALAR_PARENT_SHA256 = "bea988877a7ea23e5b99adfb5628993abac6b619c026c846547ad02bdbb33dc2"
EARLY_CAUSAL_RETRY20 = RETRY20_ROOT / "pair-snapshots/20250301T002000/gpu.nc"
EARLY_CAUSAL_RETRY20_SHA256 = "70e09cf3ca22711c66ef529e716ead53fb998d2f548e9939c35d7767ac3f9e62"
EARLY_CAUSAL_CPU = RETRY20_ROOT / "pair-snapshots/20250301T002000/cpu.nc"
EARLY_CAUSAL_CPU_SHA256 = "0a1157771f8b00f2c2c4fb66ec1cb63e4534cf3c305981ca81e1cfaad0d8d7f1"
EARLY_CAUSAL_RING1_BASELINE = {
    "T": {
        "prior_vs_cpu": 0.07439330567219023,
        "prior_vs_retry20": 0.05252442272211308,
        "partial_vs_cpu": 0.07753516846062372,
        "partial_vs_retry20": 0.0588941045970705,
        "scalar_vs_cpu": 0.07679712121970532,
        "scalar_vs_retry20": 0.058984730554649464,
    },
    "U": {
        "prior_vs_cpu": 0.2857256114890511,
        "prior_vs_retry20": 0.27772695352886917,
        "partial_vs_cpu": 0.20247207650654608,
        "partial_vs_retry20": 0.19592398176235976,
        "scalar_vs_cpu": 0.20146066974270455,
        "scalar_vs_retry20": 0.19546461478916494,
    },
}
EARLY_CAUSAL_RMSE_NOISE = 1.0e-12

FROZEN_1500_RMSE = {
    "T": 0.5421680888949643,
    "U": 1.1323567330094144,
    "V": 1.1318203205639872,
    "W": 0.18633111790197587,
    "T2": 1.352612988238872,
    "U10": 1.9737860008971808,
    "V10": 2.1128268857679338,
    "PSFC": 20.22747532736003,
}

ABSOLUTE_SCALE_CEILING = 1.0e6
AMPLIFICATION_CEILING = 1000.0
THETA_LIMITER_CEILING_K = 1000.0
STATE_HEALTH_FIELDS = (
    "u", "v", "w", "theta", "p_total", "p_perturbation",
    "ph_total", "ph_perturbation", "mu_total", "mu_perturbation",
)
SAVE_HEALTH_FIELDS = (
    "u_save", "v_save", "w_save", "t_save", "ph_save", "mu_save", "ww_save",
)
SCRATCH_HEALTH_FIELDS = ("t_2ave", "ww", "mudf", "muave", "muts")
ALL_SCALE_FIELDS = (
    *(f"state.{name}" for name in STATE_HEALTH_FIELDS),
    *SAVE_HEALTH_FIELDS,
    *SCRATCH_HEALTH_FIELDS,
)
STRICT_FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")
STATIC_FIELDS = ("XLAT", "XLONG", "HGT", "LANDMASK")

REQUIRED_PREIMPORT_ENV = {
    "JAX_PLATFORMS": "cuda",
    "JAX_ENABLE_X64": "true",
    "CUDA_VISIBLE_DEVICES": "0",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "GPUWRF_ALLOCATOR": "cuda_async",
    "GPUWRF_FINITE_CHECK": "1",
    "GPUWRF_NESTED_FUSE": "0",
    "GPUWRF_NESTED_DEFUSE_COMPILE": "0",
    "GPUWRF_NESTED_PARALLEL_COMPILE": "0",
    "GPUWRF_NESTED_AOT": "0",
    "GPUWRF_AOT_VERIFY": "0",
    "GPUWRF_NESTED_ASYNC_OUTPUT": "0",
    "GPUWRF_NEST_OUTPUT_PIPELINE": "0",
    "GPUWRF_BATCH_ENSEMBLE": "1",
    "GPUWRF_NESTED_SYNC_MODE": "root",
    "GPUWRF_BITWISE": "1",
    "GPUWRF_ADVANCE_CHUNK_LOOP": "fori",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "JAX_ENABLE_COMPILATION_CACHE": "false",
    BUNDLE_FLAG: BUNDLE_FLAG_VALUE,
}
INFRASTRUCTURE_GPUWRF_ENV = {
    "GPUWRF_WRF_ROOT",
    "GPUWRF_NESTED_BUNDLE_APPROVED_SHA",
    "GPUWRF_FINAL_NI_OWNER_OVERRIDE",
    "GPUWRF_NESTED_BUNDLE_RUNNER_SHA",
    "GPUWRF_NESTED_BUNDLE_RUNNER_AUDIT",
    "GPUWRF_NESTED_BUNDLE_RUNNER_AUDIT_SHA256",
    "GPUWRF_POST_FABLE_TOOLING_CRITIC",
    "GPUWRF_POST_FABLE_TOOLING_CRITIC_SHA256",
    "GPUWRF_GPU_LOCK_HELD",
    "GPUWRF_GPU_LOCK_FD",
    "GPUWRF_GPU_LOCK_FILE",
    "GPUWRF_GPU_LOCK_HOLDER_FILE",
    "GPUWRF_GPU_LOCK_LABEL",
    "GPUWRF_GPU_LOCK_TOKEN",
}
FORBIDDEN_PERSISTENT_CACHE_ENV = (
    "GPUWRF_CACHE",
    "GPUWRF_JAX_CACHE_DIR",
    "JAX_COMPILATION_CACHE_DIR",
)
FORBIDDEN_NON_GPUWRF_ENV = {
    "JAX_DEBUG_NANS",
    "JAX_DEBUG_INFS",
    "JAX_DISABLE_JIT",
    "XLA_FLAGS",
    "NVIDIA_TF32_OVERRIDE",
}
FORBIDDEN_OVERRIDE_FRAGMENTS = (
    "RECORDER", "RCA", "PHASE_TAP", "CALLBACK", "SUMMARY", "SANIT",
    "TOLERANCE", "RELAX_STRENGTH", "ACOUSTIC_PRECISION", "DISABLE_GUARDS",
)

ACCEPTED_SOURCE_HASHES = {
    "src/gpuwrf/coupling/boundary_apply.py": "af37186341d5a4b4a7911c111bb025e98356bc4edb6bb2358934a017b895023e",
    "src/gpuwrf/dynamics/core/acoustic.py": "a09dd90b7cac264894654adc03d6d9e93e5ea1fcf2349eac3f45f3b9c7511506",
    "src/gpuwrf/dynamics/explicit_diffusion.py": "9caabc9e38df44c670bc4a10b9a4c2c25497e0060744219a19aec284154514f1",
    "src/gpuwrf/integration/nested_pipeline.py": "292ad0763c913830a7e1d9528f32337b2a769d27b1f89eee086189ac47852abc",
    "src/gpuwrf/nesting/__init__.py": "cc3b106012f35f27c8a63ccbecae944ebcfc435f6be6789d2b3d974595b777de",
    "src/gpuwrf/nesting/boundary_construction.py": "78cf40d8d72b29752921d9f5a1a65e6b4abfb8042ed81407d2b2a87e46fafd27",
    "src/gpuwrf/nesting/interp.py": "b0569005ca7f87d9b85ed0cc0fd778b1f2fd8fbc3c8bd72a849b24868e78b9f3",
    "src/gpuwrf/nesting/moving_driver.py": "f04aae53c031a6f42bad7cb14e4839b2a25d374caef75b7130e0942fdb6e728a",
    "src/gpuwrf/runtime/domain_tree.py": "d01fbfc9c7fe423a9f932fb6e77fa06e4a7f71858d3493bee942d08c7a714ed4",
    "src/gpuwrf/runtime/operational_mode.py": "d73973967bfb8253ac47f900c713c7e21fb4e476839eaf04a840fa669d474ae2",
}
CANDIDATE_CLEAN_AUTHORITY = {
    "candidate_clean_at_offline_admission": True,
    "base_owner_override_proof_sha256": OWNER_OVERRIDE_PAYLOAD_SHA256,
    "nested_advection_amendment_proof_sha256": NESTED_ADV_AMENDMENT_PAYLOAD_SHA256,
    "theta_amendment_proof_sha256": THETA_AMENDMENT_PAYLOAD_SHA256,
    "theta_source_proof_sha256": THETA_SOURCE_PROOF_PAYLOAD_SHA256,
    "diffopt_amendment_file_sha256": DIFFOPT_AMENDMENT_SHA256,
    "diffopt_cpu_ab_proof_sha256": DIFFOPT_CPU_AB_PAYLOAD_SHA256,
    "diffopt_candidate_proof_sha256": DIFFOPT_CANDIDATE_PAYLOAD_SHA256,
    "scalar_amendment_payload_sha256": SCALAR_AMENDMENT_PAYLOAD_SHA256,
    "scalar_oracle_proof_sha256": SCALAR_ORACLE_PAYLOAD_SHA256,
    "scalar_complete_cpu_ab_proof_sha256": SCALAR_CPU_AB_PAYLOAD_SHA256,
    "scalar_candidate_proof_sha256": SCALAR_CANDIDATE_PAYLOAD_SHA256,
    "t_source_amendment_payload_sha256": T_SOURCE_AMENDMENT_PAYLOAD_SHA256,
    "t_source_cpu_ab_proof_sha256": T_SOURCE_CPU_AB_PAYLOAD_SHA256,
    "t_source_candidate_proof_sha256": T_SOURCE_CANDIDATE_PAYLOAD_SHA256,
    "critic_not_requested_for_this_evidence_ranked_continuation": True,
    "runner_changes_restricted_to_contract_runner_tests_and_reports": True,
}


class RunnerGateError(RuntimeError):
    """Fail-closed runner error with a stable gate code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class WindowFalsified(RunnerGateError):
    pass


class MetricGateFailure(RunnerGateError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize_pytest_output(value: str) -> str:
    """Remove only elapsed-time noise from an otherwise exact focused result."""

    return re.sub(r"\bin \d+(?:\.\d+)?s\b", "in <elapsed>s", value)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        pickle.dump(payload, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _require_sha(path: Path, expected: str, code: str) -> dict[str, Any]:
    digest = sha256_file(path)
    if digest != expected:
        raise RunnerGateError(code, f"{path}: expected={expected} actual={digest}")
    stat_result = path.stat()
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": digest,
        "bytes": stat_result.st_size,
        "mode": oct(stat_result.st_mode & 0o777),
    }


def read_authenticated_json(
    path: Path, expected_sha256: str, code: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink():
        raise RunnerGateError(code, f"symlink rejected: {path}")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RunnerGateError(code, f"not a regular file: {path}")
        raw = handle.read()
        after = os.fstat(handle.fileno())
    stable = (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size == len(raw)
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
    )
    digest = hashlib.sha256(raw).hexdigest()
    if not stable or digest != expected_sha256:
        raise RunnerGateError(
            code,
            f"{path}: stable={stable} expected={expected_sha256} actual={digest}",
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerGateError(code, f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RunnerGateError(code, f"top-level JSON object required: {path}")
    return payload, {
        "path": str(path.resolve(strict=True)),
        "sha256": digest,
        "bytes": len(raw),
        "mode": oct(before.st_mode & 0o777),
        "single_read_stable": True,
    }


def assert_rc3_continuation_source_authority() -> dict[str, Any]:
    """Authenticate the immutable rc3 source and prove continuation inventory.

    This is stdlib-only and safe before JAX/model import.  It deliberately treats
    a wrfout as output evidence, never as a substitute for an operational carry.
    """

    proof, proof_row = read_authenticated_json(
        RC3_FAILURE_PROOF,
        RC3_FAILURE_PROOF_SHA256,
        "RC3_FAILURE_PROOF_HASH",
    )
    unsigned = dict(proof)
    embedded = unsigned.pop("proof_sha256", None)
    canonical = canonical_digest(unsigned)
    first = proof.get("first_failed") or {}
    last = proof.get("last_healthy") or {}
    manifest = first.get("manifest") or {}
    leaves = manifest.get("leaves") or []
    transition_rows = []
    for index, name in zip(
        BOUNDARY_TARGET_TRANSITION_LEAF_INDICES,
        BOUNDARY_TARGET_TRANSITION_FIELDS,
    ):
        row = leaves[index] if index < len(leaves) else {}
        shape = list(row.get("shape") or [])
        transition_rows.append({
            "leaf_index": index,
            "field": name,
            "shape": shape,
            "leading_time_records": shape[0] if shape else None,
            "dtype": row.get("dtype"),
            "nonfinite_count": row.get("nonfinite_count"),
            "sha256": row.get("sha256"),
        })
    expected_time = RUN_START + timedelta(seconds=8800 * DT_SECONDS["d03"])
    if (
        proof.get("schema") != "gpuwrf.v0234.nested-boundary-final-window-failure.v1"
        or proof.get("status") != "ATOMIC_FAILURE_CAPTURED"
        or embedded != RC3_FAILURE_PROOF_PAYLOAD_SHA256
        or canonical != RC3_FAILURE_PROOF_PAYLOAD_SHA256
        or proof.get("failure", {}).get("code") != "OUTPUT_PAIR_EXCEPTION"
        or first.get("domain") != "d03"
        or first.get("step") != 8800
        or first.get("file_sha256") != RC3_STEP8800_CARRY_SHA256
        or Path(str(first.get("path", ""))).resolve() != RC3_STEP8800_CARRY.resolve()
        or manifest.get("leaf_count") != 106
        or manifest.get("floating_nonfinite_count") != 0
        or last.get("domain") != "d03"
        or last.get("step") != 8600
        or len(transition_rows) != 11
        or any(row["leading_time_records"] != 2 for row in transition_rows)
        or expected_time != datetime(2025, 3, 1, 14, 40, tzinfo=timezone.utc)
    ):
        raise RunnerGateError("RC3_FAILURE_SEMANTICS", "retained step-8800 authority changed")

    carry_row = _require_sha(
        RC3_STEP8800_CARRY, RC3_STEP8800_CARRY_SHA256, "RC3_STEP8800_CARRY_HASH",
    )
    wrfout_row = _require_sha(
        RC3_STEP8800_WRFOUT, RC3_STEP8800_WRFOUT_SHA256, "RC3_STEP8800_WRFOUT_HASH",
    )
    last_pair, last_pair_row = read_authenticated_json(
        RC3_LAST_PAIR, RC3_LAST_PAIR_SHA256, "RC3_LAST_PAIR_HASH",
    )
    if (
        last_pair.get("domain") != "d03"
        or last_pair.get("valid_time") != "2025-03-01T14:20:00+00:00"
        or (last_pair.get("finite_identity") or {}).get("own_step") != 8600
        or (last_pair.get("finite_identity") or {}).get("passed") is not True
        or RC3_STEP8800_WRFOUT.name != "wrfout_d03_2025-03-01_14:40:00"
    ):
        raise RunnerGateError("RC3_TIME_AUTHORITY", "last pair or 14:40 output changed")

    raw_names: dict[str, set[str]] = {name: set() for name in ("d01", "d02", "d03")}
    for path in (RC3_RUN_DIR / "output").glob("wrfout_d0*"):
        parts = path.name.split("_", 2)
        if len(parts) >= 2 and parts[1] in raw_names:
            raw_names[parts[1]].add(path.name)
    raw_counts = {name: len(values) for name, values in raw_names.items()}

    pair_names: dict[str, set[str]] = {name: set() for name in ("d01", "d02", "d03")}
    for path in (RC3_RUN_DIR / "frame-pairs").glob("d0*-step-*.json"):
        name = path.name.split("-", 1)[0]
        if name in pair_names:
            pair_names[name].add(path.name)
    pair_counts = {name: len(values) for name, values in pair_names.items()}
    if raw_counts != RC3_RAW_OUTPUT_COUNTS or pair_counts != RC3_FRAME_PAIR_COUNTS:
        raise RunnerGateError(
            "RC3_INVENTORY_CHANGED",
            repr({"raw": raw_counts, "pairs": pair_counts}),
        )

    carry_candidates = {"d01": [], "d02": [], "d03": []}
    for path in RC3_RUN_DIR.rglob("*.pkl"):
        for name in carry_candidates:
            if f"-{name}-" in path.name:
                carry_candidates[name].append(str(path.resolve()))
    carry_candidates = {
        name: sorted(paths) for name, paths in carry_candidates.items()
    }
    missing_parent_carries = [
        name for name in ("d01", "d02") if not carry_candidates[name]
    ]
    terminal_names = {
        name: {
            f"wrfout_{name}_{stamp}"
            for stamp in standard_frame_schedule(TERMINAL_OWN_STEPS)[name]
        }
        for name in ("d01", "d02", "d03")
    }
    missing_terminal_outputs = {
        name: sorted(terminal_names[name] - raw_names[name])
        for name in terminal_names
    }
    blockers = []
    if missing_parent_carries:
        blockers.append({
            "code": "PARENT_OPERATIONAL_CARRIES_NOT_RETAINED",
            "domains": missing_parent_carries,
            "expected_steps_at_d03_8800": RC3_EXPECTED_PARENT_STEPS_AT_D03_8800,
            "reason": "wrfout lacks the promoted operational scratch needed for bit-exact restart",
        })
    for name in ("d01", "d02"):
        if missing_terminal_outputs[name]:
            blockers.append({
                "code": "TERMINAL_PARENT_OUTPUTS_UNPRODUCIBLE_WITH_D03_ONLY_CARRY",
                "domain": name,
                "missing": missing_terminal_outputs[name],
            })
    return {
        "failure_proof": {
            **proof_row,
            "canonical_payload_sha256": canonical,
        },
        "step8800_carry": carry_row,
        "step8800_wrfout": wrfout_row,
        "last_healthy_pair": last_pair_row,
        "step": 8800,
        "valid_time": expected_time.isoformat(),
        "leaf_count": 106,
        "floating_nonfinite_count": 0,
        "boundary_target_transition": {
            "source": "build_child_boundary_package.two_time",
            "semantics": "[old child ring, new parent target]",
            "initial_time_records": 1,
            "operational_time_records": 2,
            "rows": transition_rows,
        },
        "raw_output_counts": raw_counts,
        "frame_pair_counts": pair_counts,
        "carry_candidates": carry_candidates,
        "missing_parent_carries": missing_parent_carries,
        "missing_terminal_outputs": missing_terminal_outputs,
        "blockers": blockers,
        "continuation_ready": not blockers,
        "source_namespace_preserved": str(RC3_RUN_DIR.resolve()),
        "fresh_namespace": str((LINEAGE_WORK_DIR / PARENT_JOIN_NAMESPACE).resolve()),
        "superseded_d03_only_continuation_namespace": str(
            (LINEAGE_WORK_DIR / CONTINUATION_NAMESPACE).resolve()
        ),
    }


def validate_tooling_critic_acceptance(
    environment: Mapping[str, str],
    *,
    required: bool,
) -> dict[str, Any]:
    """Fail before JAX import unless Kimi accepted the exact tooling candidate."""

    if not required:
        return {"required": False, "checked": False}
    path_value = environment.get("GPUWRF_POST_FABLE_TOOLING_CRITIC", "")
    expected_sha256 = environment.get(
        "GPUWRF_POST_FABLE_TOOLING_CRITIC_SHA256", ""
    )
    if not path_value or len(expected_sha256) != 64:
        raise RunnerGateError(
            "TOOLING_CRITIC_ENV",
            "missing critic proof path or exact file SHA-256",
        )
    path = Path(path_value).resolve()
    payload, row = read_authenticated_json(
        path, expected_sha256, "TOOLING_CRITIC_FILE"
    )
    unsigned = dict(payload)
    embedded = unsigned.pop("proof_sha256", None)
    observed = canonical_digest(unsigned)
    candidate = str(payload.get("candidate_commit", ""))
    head = _git(REPO_ROOT, "rev-parse", "HEAD")
    reviewed_paths = tuple(dict.fromkeys(
        (
            RUNNER_SOURCE.resolve(),
            LAUNCH_COMMAND.resolve(),
            *(path.resolve() for path in TOOLING_CRITIC_REVIEWED_PATHS),
        )
    ))
    candidate_is_ancestor = bool(
        candidate
        and subprocess.run(
            (
                "git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor",
                candidate, head,
            ),
            check=False,
        ).returncode == 0
    )
    reviewed_path_diff = (
        _git(
            REPO_ROOT,
            "diff",
            "--name-only",
            candidate,
            head,
            "--",
            *(str(path.relative_to(REPO_ROOT)) for path in reviewed_paths),
        ).splitlines()
        if candidate_is_ancestor
        else ["candidate-not-ancestor"]
    )
    profile_source_sha256 = (
        None
        if TOOLING_CRITIC_PROFILE_SOURCE is None
        else sha256_file(TOOLING_CRITIC_PROFILE_SOURCE)
    )
    required_payload_exact = all(
        payload.get(name) == expected
        for name, expected in TOOLING_CRITIC_REQUIRED_PAYLOAD.items()
    )
    extra_authority = (
        {"required": False}
        if TOOLING_CRITIC_AUTHORITY_HOOK is None
        else dict(TOOLING_CRITIC_AUTHORITY_HOOK())
    )
    if (
        payload.get("schema") != TOOLING_CRITIC_SCHEMA
        or payload.get("verdict") != TOOLING_CRITIC_VERDICT
        or embedded != observed
        or payload.get("runner_source_sha256") != sha256_file(RUNNER_SOURCE)
        or payload.get("exact_launcher_sha256") != sha256_file(LAUNCH_COMMAND)
        or payload.get("model_code_changed") is not False
        or payload.get("full_tree_gpu_replay_admitted") is not True
        or (
            TOOLING_CRITIC_PROFILE_SOURCE is not None
            and payload.get("profile_source_sha256") != profile_source_sha256
        )
        or not required_payload_exact
        or not candidate_is_ancestor
        or reviewed_path_diff
    ):
        raise RunnerGateError("TOOLING_CRITIC_SEMANTICS", str(path))
    return {
        **row,
        "required": True,
        "checked": True,
        "canonical_payload_sha256": observed,
        "verdict": payload["verdict"],
        "candidate_commit": candidate,
        "candidate_is_ancestor": candidate_is_ancestor,
        "profile_source_sha256": profile_source_sha256,
        "required_payload_exact": required_payload_exact,
        "reviewed_paths": [str(path) for path in reviewed_paths],
        "reviewed_paths_unchanged_since_candidate": not reviewed_path_diff,
        "extra_authority": extra_authority,
    }


def validate_preimport_environment(
    environment: Mapping[str, str],
    *,
    require_runner_audit: bool,
    require_tooling_critic: bool = False,
) -> dict[str, Any]:
    actual = {name: environment.get(name) for name in REQUIRED_PREIMPORT_ENV}
    if actual != REQUIRED_PREIMPORT_ENV:
        raise RunnerGateError(
            "ENV_FROZEN_LANE",
            f"expected={REQUIRED_PREIMPORT_ENV!r} actual={actual!r}",
        )
    if environment.get("GPUWRF_NESTED_BUNDLE_APPROVED_SHA") != CANDIDATE_COMMIT:
        raise RunnerGateError("ENV_CANDIDATE", CANDIDATE_COMMIT)
    if Path(environment.get("GPUWRF_FINAL_NI_OWNER_OVERRIDE", "")).resolve() != OWNER_OVERRIDE.resolve():
        raise RunnerGateError("ENV_OWNER_OVERRIDE", str(OWNER_OVERRIDE))
    persistent_cache_env = sorted(
        name for name in FORBIDDEN_PERSISTENT_CACHE_ENV if name in environment
    )
    if persistent_cache_env:
        raise RunnerGateError(
            "ENV_PERSISTENT_CACHE_FORBIDDEN", repr(persistent_cache_env),
        )
    wrf_root = environment.get("GPUWRF_WRF_ROOT", "")
    if Path(wrf_root).resolve() != RETRY20_WRF_ROOT.resolve():
        raise RunnerGateError("ENV_WRF_ROOT", wrf_root)
    if require_runner_audit:
        for name in (
            "GPUWRF_NESTED_BUNDLE_RUNNER_SHA",
            "GPUWRF_NESTED_BUNDLE_RUNNER_AUDIT",
            "GPUWRF_NESTED_BUNDLE_RUNNER_AUDIT_SHA256",
        ):
            if not environment.get(name):
                raise RunnerGateError("ENV_RUNNER_AUDIT", f"missing {name}")
    tooling_critic = validate_tooling_critic_acceptance(
        environment, required=require_tooling_critic,
    )
    unexpected = sorted(
        name for name in environment
        if name.startswith("GPUWRF_")
        and name not in set(REQUIRED_PREIMPORT_ENV) | INFRASTRUCTURE_GPUWRF_ENV
    )
    if unexpected:
        raise RunnerGateError("ENV_UNAPPROVED_GPUWRF", repr(unexpected))
    forbidden_named = sorted(
        name for name in environment
        if name in FORBIDDEN_NON_GPUWRF_ENV
        or (
            name.startswith("GPUWRF_")
            and name not in set(REQUIRED_PREIMPORT_ENV) | INFRASTRUCTURE_GPUWRF_ENV
            and any(fragment in name for fragment in FORBIDDEN_OVERRIDE_FRAGMENTS)
        )
    )
    if forbidden_named:
        raise RunnerGateError("ENV_FORBIDDEN_OVERRIDE", repr(forbidden_named))
    return {
        "validated_before_jax_import": True,
        "required_environment": actual,
        "candidate": CANDIDATE_COMMIT,
        "persistent_cache": {
            "enabled": False,
            "forbidden_environment": list(FORBIDDEN_PERSISTENT_CACHE_ENV),
        },
        "wrf_root": str(Path(wrf_root).resolve()),
        "runner_audit_required": require_runner_audit,
        "tooling_critic": tooling_critic,
        "unexpected_gpuwrf": unexpected,
        "forbidden_named": forbidden_named,
    }


def _has_production_gpu_process_marker(argv: Sequence[str]) -> bool:
    """Match production process names, not substrings inside Python identifiers."""

    return PRODUCTION_GPU_PROCESS_PATTERN.search(" ".join(argv)) is not None


def _declares_explicit_cpu_only_no_cuda(argv: Sequence[str]) -> bool:
    """Recognize wrappers that bind JAX to CPU *and* hide every CUDA device.

    Both declarations are required.  A lone CPU or CUDA token remains
    ambiguous and therefore cannot suppress the production-GPU fail-closed
    gate.  This handles shell ``-c`` payloads where harmless source text can
    contain the identifier ``gpuwrf``.
    """

    command = " ".join(argv)
    return bool(
        CPU_ONLY_JAX_PLATFORM_PATTERN.search(command)
        and CUDA_HIDDEN_PATTERN.search(command)
    )


def _active_production_gpu_processes() -> list[dict[str, Any]]:
    """Mirror the manager wake's GPU-job evidence without querying the GPU."""

    excluded = {os.getpid()}
    cursor = os.getppid()
    while cursor > 1 and cursor not in excluded:
        excluded.add(cursor)
        try:
            status = (Path("/proc") / str(cursor) / "status").read_text()
            parent_line = next(
                line for line in status.splitlines() if line.startswith("PPid:")
            )
            cursor = int(parent_line.split()[1])
        except (OSError, StopIteration, ValueError, IndexError):
            break
    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        if not argv:
            continue
        if not _has_production_gpu_process_marker(argv):
            continue
        if _declares_explicit_cpu_only_no_cuda(argv):
            continue
        # Manager/agent text can mention gpuwrf while doing no model work.
        executable = Path(argv[0]).name.lower()
        if executable in {"codex", "claude", "rg", "grep", "pgrep"}:
            continue
        rows.append({"pid": int(entry.name), "argv": argv})
    return sorted(rows, key=lambda row: row["pid"])


def assert_preemption_clear(label: str) -> dict[str, Any]:
    present = [str(path) for path in PREEMPT_PATHS if path.exists() or path.is_symlink()]
    if present:
        raise RunnerGateError("PREEMPT", f"{label}: {present}")
    holds = [str(path) for path in HOLD_PATHS if path.exists() or path.is_symlink()]
    if holds:
        raise RunnerGateError("HOLD", f"{label}: {holds}")
    active_gpu_processes = _active_production_gpu_processes()
    if active_gpu_processes:
        raise RunnerGateError(
            "PREEMPT_PRODUCTION_GPU_ACTIVE",
            f"{label}: {active_gpu_processes!r}",
        )
    own_lock = bool(
        os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
        and os.environ.get("GPUWRF_GPU_LOCK_HOLDER_FILE")
        == str(GPU_LOCK_HOLDER)
        and os.environ.get("GPUWRF_GPU_LOCK_TOKEN")
    )
    foreign_holder = bool(
        (GPU_LOCK_HOLDER.exists() or GPU_LOCK_HOLDER.is_symlink()) and not own_lock
    )
    if foreign_holder:
        raise RunnerGateError(
            "PREEMPT_PRODUCTION_GPU_LOCKED", f"{label}: {GPU_LOCK_HOLDER}"
        )
    nightly: dict[str, Any] | None = None
    if NIGHTLY_ACTIVE.exists():
        nightly = json.loads(NIGHTLY_ACTIVE.read_text())
    nightly_active = bool(nightly is not None and nightly.get("active") is True)
    nightly_status = None if nightly is None else nightly.get("status")
    return {
        "label": label,
        "sentinels_absent": [str(path) for path in PREEMPT_PATHS],
        "holds_absent": [str(path) for path in HOLD_PATHS],
        "nightly_active": None if nightly is None else nightly.get("active"),
        "nightly_status": nightly_status,
        "nightly_run_id": None if nightly is None else nightly.get("run_id"),
        "nightly_non_gpu_admitted": nightly_active,
        "production_gpu_processes": active_gpu_processes,
        "foreign_gpu_lock_holder": foreign_holder,
        "owner_policy": (
            "active Nightly state is informational; only explicit PREEMPT/HOLD, "
            "a foreign lock holder, or a live production GPU process revokes WRFGPU"
        ),
    }


def _authenticated_canonical_json(
    path: Path,
    file_sha256: str,
    payload_sha256: str,
    code: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, row = read_authenticated_json(path, file_sha256, f"{code}_FILE")
    unsigned = dict(payload)
    embedded = unsigned.pop("proof_sha256", None)
    observed = canonical_digest(unsigned)
    if embedded != payload_sha256 or observed != payload_sha256:
        raise RunnerGateError(
            f"{code}_CANONICAL",
            f"embedded={embedded} expected={payload_sha256} observed={observed}",
        )
    return payload, {**row, "canonical_payload_sha256": observed}


def assert_final_candidate_proof_authority() -> dict[str, Any]:
    """Authenticate the owner-authorized terminal candidate without a critic."""

    nested_amendment, nested_amendment_row = _authenticated_canonical_json(
        NESTED_ADV_AMENDMENT,
        NESTED_ADV_AMENDMENT_SHA256,
        NESTED_ADV_AMENDMENT_PAYLOAD_SHA256,
        "NESTED_ADV_AMENDMENT",
    )
    nested_proof, nested_proof_row = _authenticated_canonical_json(
        NESTED_ADV_PROOF,
        NESTED_ADV_PROOF_SHA256,
        NESTED_ADV_PROOF_PAYLOAD_SHA256,
        "NESTED_ADV_PROOF",
    )
    theta_amendment, theta_amendment_row = _authenticated_canonical_json(
        THETA_AMENDMENT,
        THETA_AMENDMENT_SHA256,
        THETA_AMENDMENT_PAYLOAD_SHA256,
        "THETA_AMENDMENT",
    )
    theta_source, theta_source_row = _authenticated_canonical_json(
        THETA_SOURCE_PROOF,
        THETA_SOURCE_PROOF_SHA256,
        THETA_SOURCE_PROOF_PAYLOAD_SHA256,
        "THETA_SOURCE_PROOF",
    )
    diffopt_amendment, diffopt_amendment_row = read_authenticated_json(
        DIFFOPT_AMENDMENT,
        DIFFOPT_AMENDMENT_SHA256,
        "DIFFOPT_AMENDMENT_FILE",
    )
    diffopt_cpu, diffopt_cpu_row = _authenticated_canonical_json(
        DIFFOPT_CPU_AB_PROOF,
        DIFFOPT_CPU_AB_PROOF_SHA256,
        DIFFOPT_CPU_AB_PAYLOAD_SHA256,
        "DIFFOPT_CPU_AB_PROOF",
    )
    diffopt_candidate, diffopt_candidate_row = _authenticated_canonical_json(
        DIFFOPT_CANDIDATE_PROOF,
        DIFFOPT_CANDIDATE_PROOF_SHA256,
        DIFFOPT_CANDIDATE_PAYLOAD_SHA256,
        "DIFFOPT_CANDIDATE_PROOF",
    )
    scalar_amendment, scalar_amendment_row = read_authenticated_json(
        SCALAR_AMENDMENT,
        SCALAR_AMENDMENT_SHA256,
        "SCALAR_AMENDMENT_FILE",
    )
    scalar_amendment_canonical = canonical_digest(scalar_amendment)
    if scalar_amendment_canonical != SCALAR_AMENDMENT_PAYLOAD_SHA256:
        raise RunnerGateError(
            "SCALAR_AMENDMENT_CANONICAL",
            f"expected={SCALAR_AMENDMENT_PAYLOAD_SHA256} observed={scalar_amendment_canonical}",
        )
    scalar_amendment_row = {
        **scalar_amendment_row,
        "canonical_payload_sha256": scalar_amendment_canonical,
    }
    scalar_oracle, scalar_oracle_row = _authenticated_canonical_json(
        SCALAR_ORACLE_PROOF,
        SCALAR_ORACLE_PROOF_SHA256,
        SCALAR_ORACLE_PAYLOAD_SHA256,
        "SCALAR_ORACLE_PROOF",
    )
    scalar_cpu, scalar_cpu_row = _authenticated_canonical_json(
        SCALAR_CPU_AB_PROOF,
        SCALAR_CPU_AB_PROOF_SHA256,
        SCALAR_CPU_AB_PAYLOAD_SHA256,
        "SCALAR_CPU_AB_PROOF",
    )
    scalar_candidate, scalar_candidate_row = _authenticated_canonical_json(
        SCALAR_CANDIDATE_PROOF,
        SCALAR_CANDIDATE_PROOF_SHA256,
        SCALAR_CANDIDATE_PAYLOAD_SHA256,
        "SCALAR_CANDIDATE_PROOF",
    )
    t_source_amendment, t_source_amendment_row = read_authenticated_json(
        T_SOURCE_AMENDMENT,
        T_SOURCE_AMENDMENT_SHA256,
        "T_SOURCE_AMENDMENT_FILE",
    )
    t_source_amendment_canonical = canonical_digest(t_source_amendment)
    if t_source_amendment_canonical != T_SOURCE_AMENDMENT_PAYLOAD_SHA256:
        raise RunnerGateError(
            "T_SOURCE_AMENDMENT_CANONICAL",
            f"expected={T_SOURCE_AMENDMENT_PAYLOAD_SHA256} "
            f"observed={t_source_amendment_canonical}",
        )
    t_source_amendment_row = {
        **t_source_amendment_row,
        "canonical_payload_sha256": t_source_amendment_canonical,
    }
    t_source_cpu, t_source_cpu_row = _authenticated_canonical_json(
        T_SOURCE_CPU_AB_PROOF,
        T_SOURCE_CPU_AB_PROOF_SHA256,
        T_SOURCE_CPU_AB_PAYLOAD_SHA256,
        "T_SOURCE_CPU_AB_PROOF",
    )
    t_source_candidate, t_source_candidate_row = _authenticated_canonical_json(
        T_SOURCE_CANDIDATE_PROOF,
        T_SOURCE_CANDIDATE_PROOF_SHA256,
        T_SOURCE_CANDIDATE_PAYLOAD_SHA256,
        "T_SOURCE_CANDIDATE_PROOF",
    )
    owner, owner_row = _authenticated_canonical_json(
        OWNER_OVERRIDE,
        OWNER_OVERRIDE_SHA256,
        OWNER_OVERRIDE_PAYLOAD_SHA256,
        "OWNER_OVERRIDE",
    )
    terminal, terminal_row = _authenticated_canonical_json(
        TERMINAL_CANDIDATE_PROOF,
        TERMINAL_CANDIDATE_PROOF_SHA256,
        TERMINAL_CANDIDATE_PAYLOAD_SHA256,
        "TERMINAL_CANDIDATE_PROOF",
    )
    cpu, cpu_row = _authenticated_canonical_json(
        CPU_CANDIDATE_PROOF,
        CPU_CANDIDATE_PROOF_SHA256,
        CPU_CANDIDATE_PAYLOAD_SHA256,
        "CPU_CANDIDATE_PROOF",
    )
    gpu, gpu_row = _authenticated_canonical_json(
        GPU_POLICY_PROOF,
        GPU_POLICY_PROOF_SHA256,
        GPU_POLICY_PAYLOAD_SHA256,
        "GPU_POLICY_PROOF",
    )
    interface, interface_row = _authenticated_canonical_json(
        INTERFACE_PROOF,
        INTERFACE_PROOF_SHA256,
        INTERFACE_PAYLOAD_SHA256,
        "INTERFACE_PROOF",
    )
    frozen, frozen_row = _authenticated_canonical_json(
        FINAL_NI_AUTHORITY_PROOF,
        FINAL_NI_AUTHORITY_PROOF_SHA256,
        FINAL_NI_AUTHORITY_PAYLOAD_SHA256,
        "FINAL_NI_AUTHORITY_PROOF",
    )
    localization, localization_row = _authenticated_canonical_json(
        LOCALIZATION_PROOF,
        LOCALIZATION_PROOF_SHA256,
        LOCALIZATION_PAYLOAD_SHA256,
        "LOCALIZATION_PROOF",
    )
    source, source_row = _authenticated_canonical_json(
        SOURCE_INVARIANT_PROOF,
        SOURCE_INVARIANT_PROOF_SHA256,
        SOURCE_INVARIANT_PAYLOAD_SHA256,
        "SOURCE_INVARIANT_PROOF",
    )
    corrected, corrected_row = _authenticated_canonical_json(
        CORRECTED_CPU_ARM_PROOF,
        CORRECTED_CPU_ARM_PROOF_SHA256,
        CORRECTED_CPU_ARM_PAYLOAD_SHA256,
        "CORRECTED_CPU_ARM_PROOF",
    )
    falsified, falsified_row = _authenticated_canonical_json(
        FALSIFIED_CPU_ARM_PROOF,
        FALSIFIED_CPU_ARM_PROOF_SHA256,
        FALSIFIED_CPU_ARM_PAYLOAD_SHA256,
        "FALSIFIED_CPU_ARM_PROOF",
    )
    raw_gpu, raw_gpu_row = _authenticated_canonical_json(
        RAW_GPU_PROOF,
        RAW_GPU_PROOF_SHA256,
        RAW_GPU_PAYLOAD_SHA256,
        "RAW_GPU_PROOF",
    )

    owner_review = owner.get("review_policy") or {}
    owner_execution = owner.get("execution") or {}
    owner_cache = owner.get("cache_policy") or {}
    owner_monitoring = owner.get("monitoring") or {}
    terminal_candidate = terminal.get("candidate") or {}
    terminal_gates = terminal.get("terminal_gates") or {}
    terminal_components = terminal.get("components") or {}
    cpu_ab = cpu.get("cpu_ab_discriminator") or {}
    cpu_gates = cpu_ab.get("gates") or {}
    operator = cpu.get("operator_gates") or {}
    gpu_gates = gpu.get("gates") or {}
    interface_hlo = interface.get("candidate_on", {}).get("forbidden_hlo_tokens") or {}
    model_lineage = terminal_candidate.get("model_source_lineage") or {}
    expected_lineage = {
        "src/gpuwrf/coupling/boundary_apply.py": ACCEPTED_SOURCE_HASHES[
            "src/gpuwrf/coupling/boundary_apply.py"
        ],
        "src/gpuwrf/dynamics/core/acoustic.py": ACCEPTED_SOURCE_HASHES[
            "src/gpuwrf/dynamics/core/acoustic.py"
        ],
    }
    lineage_green = all(
        path in model_lineage
        and model_lineage[path].get("identical") is True
        and set(
            value for key, value in model_lineage[path].items()
            if key.endswith("_sha256")
        ) == {expected}
        for path, expected in expected_lineage.items()
    )
    component_expectations = {
        "authority": FINAL_NI_AUTHORITY_PAYLOAD_SHA256,
        "carry_localization": LOCALIZATION_PAYLOAD_SHA256,
        "source_invariant": SOURCE_INVARIANT_PAYLOAD_SHA256,
        "interface_transfer": INTERFACE_PAYLOAD_SHA256,
        "cpu_candidate": CPU_CANDIDATE_PAYLOAD_SHA256,
        "bounded_gpu_policy": GPU_POLICY_PAYLOAD_SHA256,
        "raw_bounded_gpu": RAW_GPU_PAYLOAD_SHA256,
    }
    components_green = all(
        (terminal_components.get(name) or {}).get("proof_sha256") == expected
        for name, expected in component_expectations.items()
    )
    corrected_ref = cpu_ab.get("corrected_arm") or {}
    falsified_ref = cpu_ab.get("falsified_arm") or {}
    corrected_obs = corrected.get("observation") or {}
    falsified_obs = falsified.get("observation") or {}
    raw_ref = gpu.get("raw_proof") or {}
    if (
        owner.get("schema") != "gpuwrf.v0234.final-ni-full18h-owner-override.v1"
        or owner.get("verdict") != "OWNER_OVERRIDE_FULL18H_AUTHORIZED"
        or owner.get("candidate_commit") != BASE_CANDIDATE_COMMIT
        or owner.get("terminal_candidate_proof_sha256")
        != TERMINAL_CANDIDATE_PAYLOAD_SHA256
        or owner.get("terminal_cpu_authority_sha256") != TERMINAL_CONTRACT_SHA256
        or owner_review.get("independent_critic_cancelled") is not True
        or owner_review.get("additional_review_required") is not False
        or owner_review.get("additional_manager_authorization_required") is not False
        or owner_execution.get("fresh_namespace") != OWNER_OVERRIDE_NAMESPACE
        or owner_execution.get("direct_same_process_to_18h") is not True
        or owner_execution.get("terminal_counts") != {"d01": 19, "d02": 19, "d03": 55}
        or owner_cache.get("persistent_unauthenticated_cache_allowed") is not False
        or owner_cache.get("authenticated_retry20_lineage_cache_allowed") is not False
        or owner_cache.get("runtime_persistent_cache_used") is not False
        or owner_monitoring.get("progress_aware") is not True
        or int(owner_monitoring.get("minimum_inactivity_stall_seconds", 0)) < 3600
        or terminal.get("schema") != "gpuwrf.v0234.final-ni-fable5-terminal-candidate.v1"
        or terminal.get("verdict") != "READY_FOR_GPT_CRITIC"
        or not terminal_gates
        or not all(value is True for value in terminal_gates.values())
        or not components_green
        or not lineage_green
        or cpu.get("schema") != "gpuwrf.v0234.final-ni-fable5-candidate-cpu.v1"
        or cpu.get("verdict") != "CPU_SOURCE_CANDIDATE_GREEN_BOUNDED_GPU_PENDING"
        or not cpu_gates
        or not all(value is True for value in cpu_gates.values())
        or operator.get("production_carry_leaf_count") != 106
        or operator.get("candidate_off_exact") is not True
        or operator.get("ordinary_candidate_on_finite") is not True
        or operator.get("new_loop_host_device_transfer") is not False
        or operator.get("observer_or_callback_added") is not False
        or any(int(value) != 0 for value in (operator.get("forbidden_hlo_tokens") or {}).values())
        or Path(str(corrected_ref.get("path", ""))).resolve() != CORRECTED_CPU_ARM_PROOF.resolve()
        or corrected_ref.get("proof_sha256") != CORRECTED_CPU_ARM_PAYLOAD_SHA256
        or Path(str(falsified_ref.get("path", ""))).resolve() != FALSIFIED_CPU_ARM_PROOF.resolve()
        or falsified_ref.get("proof_sha256") != FALSIFIED_CPU_ARM_PAYLOAD_SHA256
        or corrected.get("schema") != "gpuwrf.v0234.final-ni-fable5-cpu-ab-arm.v1"
        or corrected.get("arm") != "corrected"
        or corrected.get("backend") != "cpu"
        or corrected.get("gpu_commands") != 0
        or corrected_obs.get("finite_guard", {}).get("raised") is not False
        or corrected_obs.get("nonfinite_fields") != {}
        or corrected_obs.get("ph_beyond_ring0_bit_frozen") is not False
        or not float(corrected_obs.get("ph_interior_max_abs_change", 0.0)) > 0.0
        or falsified.get("schema") != "gpuwrf.v0234.final-ni-fable5-cpu-ab-arm.v1"
        or falsified.get("arm") != "falsified"
        or falsified.get("backend") != "cpu"
        or falsified.get("gpu_commands") != 0
        or falsified_obs.get("finite_guard", {}).get("raised") is not True
        or falsified_obs.get("ni_first_nonfinite_scan_index") != [0, 1, 1]
        or falsified_obs.get("ph_beyond_ring0_bit_frozen") is not True
        or gpu.get("schema") != "gpuwrf.v0234.final-ni-fable5-bounded-gpu-policy.v1"
        or gpu.get("verdict") != "GPU_DISCRIMINATOR_POLICY_GREEN_RAW_GATE_MISAPPLIED"
        or not gpu_gates
        or not all(value is True for value in gpu_gates.values())
        or Path(str(raw_ref.get("path", ""))).resolve() != RAW_GPU_PROOF.resolve()
        or raw_ref.get("proof_sha256") != RAW_GPU_PAYLOAD_SHA256
        or raw_gpu.get("schema") != "gpuwrf.v0234.final-ni-fable5-gpu-discriminator.v1"
        or raw_gpu.get("verdict") != "GPU_DISCRIMINATOR_RED"
        or raw_gpu.get("proof_sha256") != RAW_GPU_PAYLOAD_SHA256
        or interface.get("verdict")
        != "CANDIDATE_OFF_E0D0_EXACT_CANDIDATE_ON_INTERFACE_CLEAN"
        or interface.get("production_interface", {}).get("leaf_count") != 106
        or interface.get("candidate_off", {}).get("exact") is not True
        or interface.get("candidate_on", {}).get("output_finite") is not True
        or any(int(value) != 0 for value in interface_hlo.values())
        or frozen.get("verdict") != "AUTHORITY_GREEN"
        or localization.get("verdict") != "INTERIOR_PH_FREEZE_CONFIRMED"
        or source.get("verdict") != "SOURCE_DISCREPANCY_PROVEN"
    ):
        raise RunnerGateError(
            "FINAL_CANDIDATE_AUTHORITY_SEMANTICS",
            repr({
                "components_green": components_green,
                "lineage_green": lineage_green,
                "terminal_verdict": terminal.get("verdict"),
                "cpu_verdict": cpu.get("verdict"),
                "gpu_verdict": gpu.get("verdict"),
            }),
        )
    accepted = nested_amendment.get("accepted_candidate") or {}
    offline = accepted.get("offline_proof") or {}
    causal_gate = nested_amendment.get("causal_gpu_gate") or {}
    if (
        nested_amendment.get("verdict")
        != "NESTED_ADVECTION_DISCRIMINATOR_RUNNER_AMENDED_FAIL_CLOSED"
        or nested_amendment.get("owner_authority", {}).get("gpu_hold_released") is not True
        or accepted.get("commit") != PARTIAL_WIND_CANDIDATE_COMMIT
        or accepted.get("tree") != PARTIAL_WIND_CANDIDATE_TREE
        or accepted.get("operational_mode_sha256")
        != PARTIAL_WIND_OPERATIONAL_MODE_SHA256
        or offline.get("proof_sha256") != NESTED_ADV_PROOF_PAYLOAD_SHA256
        or causal_gate.get("namespace") != PARTIAL_WIND_REPLAY_NAMESPACE
        or causal_gate.get("first_gate") != "d03 step 200 / 00:20"
        or nested_proof.get("verdict")
        != "NESTED_ADVECTION_DEGRADE_OFFLINE_CAUSAL_GATE_GREEN"
        or nested_proof.get("proof_sha256") != NESTED_ADV_PROOF_PAYLOAD_SHA256
        or nested_proof.get("gpu_commands") != 0
    ):
        raise RunnerGateError(
            "NESTED_ADV_CANDIDATE_AUTHORITY_SEMANTICS",
            repr({
                "amendment_verdict": nested_amendment.get("verdict"),
                "accepted": accepted,
                "causal_gate": causal_gate,
                "proof_verdict": nested_proof.get("verdict"),
            }),
        )
    theta_parent = theta_amendment.get("falsified_parent_candidate") or {}
    theta_gate = theta_amendment.get("gpu_gate") or {}
    theta_git = theta_source.get("git") or {}
    theta_gates = theta_source.get("gates") or {}
    theta_runtime = theta_source.get("source", {}).get("runtime") or {}
    if (
        theta_amendment.get("schema")
        != "gpuwrf.v0234.final-ni-fable5-contract-amendment.v6"
        or theta_amendment.get("verdict")
        != "THETA_UNLIMITED_SOURCE_REPAIR_AMENDED_FAIL_CLOSED"
        or theta_parent.get("commit") != PARTIAL_WIND_CANDIDATE_COMMIT
        or theta_parent.get("verdict") != "PARTIAL_WIND_DISCRIMINATOR_ONLY"
        or theta_gate.get("first_gate")
        != "fresh canonical full-history process through d03 step 200 / 00:20"
        or "continue the same process" not in str(theta_gate.get("green_action", ""))
        or theta_source.get("schema") != "gpuwrf.v0234.theta-unlimited-source-proof.v1"
        or theta_source.get("verdict") != "THETA_UNLIMITED_OFFLINE_CAUSAL_GATE_GREEN"
        or theta_source.get("proof_sha256") != THETA_SOURCE_PROOF_PAYLOAD_SHA256
        or theta_source.get("gpu_commands_run") != 0
        or theta_source.get("gpu_queries_run") != 0
        or not theta_gates
        or not all(value is True for value in theta_gates.values())
        or theta_git.get("head") != THETA_CANDIDATE_COMMIT
        or theta_git.get("partial_wind_candidate") != PARTIAL_WIND_CANDIDATE_COMMIT
        or theta_git.get("changed_model_files") != ["src/gpuwrf/runtime/operational_mode.py"]
        or theta_runtime.get("operational_mode_sha256")
        != THETA_OPERATIONAL_MODE_SHA256
    ):
        raise RunnerGateError(
            "THETA_CANDIDATE_AUTHORITY_SEMANTICS",
            repr({
                "amendment_verdict": theta_amendment.get("verdict"),
                "partial_parent": theta_parent,
                "gpu_gate": theta_gate,
                "proof_verdict": theta_source.get("verdict"),
                "proof_git": theta_git,
            }),
        )
    diffopt_model = diffopt_candidate.get("candidate") or {}
    diffopt_contract = diffopt_candidate.get("contract") or {}
    diffopt_cpu_ref = diffopt_candidate.get("authenticated_cpu_ab") or {}
    diffopt_scope = diffopt_candidate.get("scope") or {}
    diffopt_checks = diffopt_cpu.get("checks") or {}
    if (
        diffopt_amendment.get("schema")
        != "gpuwrf.v0234.final-ni-fable5-contract-amendment.v11"
        or diffopt_amendment.get("verdict")
        != "NESTED_DIFFOPT1_106_LEAF_T_INIT_INVERSION_AMENDED_FAIL_CLOSED"
        or diffopt_amendment.get("identity_constraints", {}).get(
            "operational_carry_leaves"
        ) != 106
        or diffopt_candidate.get("schema")
        != "gpuwrf.v0234.nested-diffopt1-rk1-forward-bundle-candidate.v1"
        or diffopt_candidate.get("verdict")
        != "NESTED_DIFFOPT1_RK1_FORWARD_GPU_STEP200_ADMITTED"
        or diffopt_candidate.get("proof_sha256")
        != DIFFOPT_CANDIDATE_PAYLOAD_SHA256
        or diffopt_model.get("commit") != DIFFOPT_PARENT_COMMIT
        or diffopt_model.get("tree") != DIFFOPT_PARENT_TREE
        or diffopt_model.get("parent_partial_wind_commit")
        != PARTIAL_WIND_CANDIDATE_COMMIT
        or diffopt_model.get("model_files") != {
            "src/gpuwrf/integration/nested_pipeline.py": DIFFOPT_PARENT_NESTED_PIPELINE_SHA256,
            "src/gpuwrf/runtime/operational_mode.py": DIFFOPT_PARENT_OPERATIONAL_MODE_SHA256,
        }
        or diffopt_contract.get("file_sha256") != DIFFOPT_AMENDMENT_SHA256
        or diffopt_cpu_ref.get("file_sha256") != DIFFOPT_CPU_AB_PROOF_SHA256
        or diffopt_cpu_ref.get("proof_sha256") != DIFFOPT_CPU_AB_PAYLOAD_SHA256
        or diffopt_cpu_ref.get("interface")
        != "106 leaves in and 106 leaves out for both arms"
        or diffopt_cpu_ref.get("all_leaves_finite") is not True
        or diffopt_cpu_ref.get("forbidden_callback_targets") != []
        or diffopt_cpu.get("schema")
        != "gpuwrf.v0234.nested-diffopt1-rk1-forward-bundle-cpu-ab.v1"
        or diffopt_cpu.get("verdict") != "NESTED_DIFFOPT1_RK1_FORWARD_CPU_AB_GREEN"
        or not diffopt_checks
        or not all(value is True for value in diffopt_checks.values())
        or diffopt_cpu.get("parent", {}).get("stablehlo_sha256")
        != "f5042ec46e3d596a08ad1b0af5dc1a4953c7bd86a5199bfacfa6cc26f50e8d67"
        or diffopt_cpu.get("candidate", {}).get("stablehlo_sha256")
        != "3b5a2e36ac9a99a7dd7311fd51569cf119aad4d70afcb2c9b715c3b1f8003ab7"
        or diffopt_scope.get("gpu_commands_run_for_candidate") != 0
        or diffopt_scope.get("gpu_queries_run_for_candidate") != 0
        or diffopt_scope.get("full_forecasts_run_for_candidate") != 0
    ):
        raise RunnerGateError(
            "DIFFOPT_CANDIDATE_AUTHORITY_SEMANTICS",
            repr({
                "amendment_verdict": diffopt_amendment.get("verdict"),
                "candidate": diffopt_model,
                "candidate_verdict": diffopt_candidate.get("verdict"),
                "cpu_verdict": diffopt_cpu.get("verdict"),
            }),
        )
    scalar_model = scalar_candidate.get("candidate") or {}
    scalar_contract = scalar_candidate.get("contract") or {}
    scalar_oracle_ref = scalar_candidate.get("source_oracle") or {}
    scalar_cpu_ref = scalar_candidate.get("complete_cpu_ab") or {}
    scalar_causal = scalar_candidate.get("causal_scope") or {}
    scalar_gpu = scalar_candidate.get("gpu_discriminator") or {}
    scalar_cpu_checks = scalar_cpu.get("checks") or {}
    scalar_oracle_checks = scalar_oracle.get("checks") or {}
    expected_scalar_model_files = {
        "src/gpuwrf/dynamics/explicit_diffusion.py": ACCEPTED_SOURCE_HASHES[
            "src/gpuwrf/dynamics/explicit_diffusion.py"
        ],
        "src/gpuwrf/runtime/operational_mode.py": ACCEPTED_SOURCE_HASHES[
            "src/gpuwrf/runtime/operational_mode.py"
        ],
    }
    if (
        scalar_amendment.get("schema")
        != "gpuwrf.v0234.final-ni-fable5-contract-amendment.v12"
        or scalar_amendment.get("verdict")
        != "STEP200_T_SCALAR_RCA_AMENDED_FAIL_CLOSED"
        or scalar_amendment.get("identity_constraints", {}).get(
            "operational_carry_leaves"
        ) != 106
        or scalar_candidate.get("schema")
        != "gpuwrf.v0234.nested-scalar-diffusion-candidate.v1"
        or scalar_candidate.get("verdict")
        != "NESTED_SCALAR_DIFFUSION_GPU_STEP200_ADMITTED"
        or scalar_candidate.get("proof_sha256")
        != SCALAR_CANDIDATE_PAYLOAD_SHA256
        or scalar_model.get("commit") != SCALAR_CANDIDATE_COMMIT
        or scalar_model.get("tree") != SCALAR_CANDIDATE_TREE
        or scalar_model.get("parent_falsified_commit") != DIFFOPT_PARENT_COMMIT
        or scalar_model.get("parent_partial_wind_commit")
        != PARTIAL_WIND_CANDIDATE_COMMIT
        or scalar_model.get("model_files") != expected_scalar_model_files
        or scalar_contract.get("file_sha256") != SCALAR_AMENDMENT_SHA256
        or scalar_contract.get("canonical_payload_sha256")
        != SCALAR_AMENDMENT_PAYLOAD_SHA256
        or scalar_oracle_ref.get("file_sha256") != SCALAR_ORACLE_PROOF_SHA256
        or scalar_oracle_ref.get("proof_sha256") != SCALAR_ORACLE_PAYLOAD_SHA256
        or scalar_oracle.get("schema")
        != "gpuwrf.v0234.nested-scalar-diffusion-source-oracle.v1"
        or scalar_oracle.get("verdict")
        != "NESTED_SCALAR_DIFFUSION_SOURCE_ORACLE_GREEN"
        or not scalar_oracle_checks
        or not all(value is True for value in scalar_oracle_checks.values())
        or scalar_cpu_ref.get("file_sha256") != SCALAR_CPU_AB_PROOF_SHA256
        or scalar_cpu_ref.get("proof_sha256") != SCALAR_CPU_AB_PAYLOAD_SHA256
        or scalar_cpu_ref.get("input_leaf_count") != 106
        or scalar_cpu_ref.get("output_leaf_count") != 106
        or scalar_cpu_ref.get("all_leaves_finite") is not True
        or scalar_cpu_ref.get("forbidden_callback_targets") != []
        or scalar_cpu.get("schema")
        != "gpuwrf.v0234.nested-scalar-diffusion-complete-cpu-ab.v1"
        or scalar_cpu.get("verdict")
        != "NESTED_SCALAR_DIFFUSION_COMPLETE_CPU_AB_GREEN"
        or not scalar_cpu_checks
        or not all(value is True for value in scalar_cpu_checks.values())
        or scalar_cpu.get("candidate_commit") != SCALAR_CANDIDATE_COMMIT
        or scalar_cpu.get("candidate_B", {}).get("stablehlo_sha256")
        != "4c72b498a31f4cefe162d41827bd5cae6ba348de15e47b76fa4ac8146cabd044"
        or scalar_causal.get("direct_momentum_change") is not False
        or scalar_causal.get("carry_or_namelist_leaf_change") is not False
        or scalar_causal.get("new_loop_host_device_transfer") is not False
        or scalar_causal.get("observer_or_callback_added") is not False
        or scalar_causal.get("coefficient_sweep_or_speculative_toggle") is not False
        or scalar_gpu.get("gpu_commands_run_for_candidate") != 0
        or scalar_gpu.get("gpu_queries_run_for_candidate") != 0
        or scalar_gpu.get("full_forecasts_run_for_candidate") != 0
        or scalar_gpu.get("first_gate")
        != "fresh canonical full-history process through d03 step 200 / 00:20"
        or "continue the identical compiled same-process run directly to 18h"
        not in str(scalar_gpu.get("green_action", ""))
    ):
        raise RunnerGateError(
            "SCALAR_CANDIDATE_AUTHORITY_SEMANTICS",
            repr({
                "amendment_verdict": scalar_amendment.get("verdict"),
                "candidate": scalar_model,
                "candidate_verdict": scalar_candidate.get("verdict"),
                "oracle_verdict": scalar_oracle.get("verdict"),
                "cpu_verdict": scalar_cpu.get("verdict"),
            }),
        )
    t_source_model = t_source_candidate.get("candidate") or {}
    t_source_contract = t_source_candidate.get("contract") or {}
    t_source_cpu_ref = t_source_candidate.get("complete_cpu_ab") or {}
    t_source_scope = t_source_candidate.get("causal_scope") or {}
    t_source_gpu = t_source_candidate.get("gpu_discriminator") or {}
    t_source_checks = t_source_cpu.get("checks") or {}
    expected_t_source_model_files = {
        "src/gpuwrf/integration/nested_pipeline.py": ACCEPTED_SOURCE_HASHES[
            "src/gpuwrf/integration/nested_pipeline.py"
        ],
    }
    if (
        t_source_amendment.get("schema")
        != "gpuwrf.v0234.final-ni-fable5-contract-amendment.v13"
        or t_source_amendment.get("verdict")
        != "NESTED_T_SOURCE_CADENCE_REPAIR_AMENDED_FAIL_CLOSED"
        or t_source_amendment.get("parent_amendment", {}).get("file_sha256")
        != SCALAR_AMENDMENT_SHA256
        or t_source_candidate.get("schema")
        != "gpuwrf.v0234.nested-t-source-cadence-candidate.v1"
        or t_source_candidate.get("verdict")
        != "NESTED_T_SOURCE_CADENCE_GPU_STEP200_ADMITTED"
        or t_source_candidate.get("proof_sha256")
        != T_SOURCE_CANDIDATE_PAYLOAD_SHA256
        or t_source_model.get("commit") != CANDIDATE_COMMIT
        or t_source_model.get("tree") != CANDIDATE_TREE
        or t_source_model.get("parent_falsified_commit")
        != SCALAR_CANDIDATE_COMMIT
        or t_source_model.get("partial_wind_commit")
        != PARTIAL_WIND_CANDIDATE_COMMIT
        or t_source_model.get("model_files") != expected_t_source_model_files
        or t_source_contract.get("file_sha256") != T_SOURCE_AMENDMENT_SHA256
        or t_source_contract.get("canonical_payload_sha256")
        != T_SOURCE_AMENDMENT_PAYLOAD_SHA256
        or t_source_cpu_ref.get("file_sha256") != T_SOURCE_CPU_AB_PROOF_SHA256
        or t_source_cpu_ref.get("proof_sha256") != T_SOURCE_CPU_AB_PAYLOAD_SHA256
        or t_source_cpu_ref.get("input_leaf_count") != 106
        or t_source_cpu_ref.get("output_leaf_count") != 106
        or t_source_cpu_ref.get("all_leaves_finite") is not True
        or t_source_cpu_ref.get("forbidden_callback_targets") != []
        or t_source_cpu_ref.get("candidate_stablehlo_sha256")
        != "f2973b9806db3a660936c6e1593995c62d8d66f13936f316de28c067914f59f2"
        or t_source_cpu.get("schema")
        != "gpuwrf.v0234.nested-t-source-cadence-cpu-ab.v1"
        or t_source_cpu.get("verdict")
        != "NESTED_T_SOURCE_CADENCE_CPU_AB_GREEN"
        or t_source_cpu.get("candidate_commit") != CANDIDATE_COMMIT
        or not t_source_checks
        or not all(value is True for value in t_source_checks.values())
        or t_source_cpu.get("candidate_B", {}).get("stablehlo_sha256")
        != "f2973b9806db3a660936c6e1593995c62d8d66f13936f316de28c067914f59f2"
        or t_source_scope.get("only_model_delta")
        != ["src/gpuwrf/integration/nested_pipeline.py"]
        or t_source_scope.get("new_carry_or_result_leaves") != 0
        or t_source_scope.get("new_loop_host_device_transfers") != 0
        or t_source_scope.get("observer_or_callback_added") is not False
        or t_source_scope.get("coefficient_sweep_or_speculative_toggle") is not False
        or t_source_scope.get("partial_wind_edge_stencils_retained") is not True
        or t_source_scope.get("v10_link_claimed") is not False
        or t_source_gpu.get("gpu_commands_run_for_candidate") != 0
        or t_source_gpu.get("gpu_queries_run_for_candidate") != 0
        or t_source_gpu.get("full_forecasts_run_for_candidate") != 0
        or t_source_gpu.get("first_gate")
        != "fresh canonical full-history process through d03 step 200 / 00:20"
        or "continue the identical compiled same-process run directly"
        not in str(t_source_gpu.get("green_action", ""))
    ):
        raise RunnerGateError(
            "T_SOURCE_CANDIDATE_AUTHORITY_SEMANTICS",
            repr({
                "amendment_verdict": t_source_amendment.get("verdict"),
                "candidate": t_source_model,
                "candidate_verdict": t_source_candidate.get("verdict"),
                "cpu_verdict": t_source_cpu.get("verdict"),
            }),
        )
    return {
        "owner_override": owner_row,
        "owner_override_verdict": owner["verdict"],
        "critic_cancelled_by_owner": True,
        "terminal_candidate": terminal_row,
        "cpu_candidate": cpu_row,
        "gpu_policy": gpu_row,
        "interface": interface_row,
        "frozen_authority": frozen_row,
        "carry_localization": localization_row,
        "source_invariant": source_row,
        "corrected_cpu_arm": corrected_row,
        "falsified_cpu_arm": falsified_row,
        "raw_bounded_gpu": raw_gpu_row,
        "nested_advection_amendment": nested_amendment_row,
        "nested_advection_offline_proof": nested_proof_row,
        "nested_advection_partial_wind_commit": PARTIAL_WIND_CANDIDATE_COMMIT,
        "theta_amendment": theta_amendment_row,
        "theta_source_proof": theta_source_row,
        "theta_candidate_commit": THETA_CANDIDATE_COMMIT,
        "diffopt_amendment": diffopt_amendment_row,
        "diffopt_cpu_ab": diffopt_cpu_row,
        "diffopt_candidate": diffopt_candidate_row,
        "diffopt_parent_candidate_commit": DIFFOPT_PARENT_COMMIT,
        "scalar_amendment": scalar_amendment_row,
        "scalar_oracle": scalar_oracle_row,
        "scalar_cpu_ab": scalar_cpu_row,
        "scalar_candidate": scalar_candidate_row,
        "scalar_candidate_commit": SCALAR_CANDIDATE_COMMIT,
        "t_source_amendment": t_source_amendment_row,
        "t_source_cpu_ab": t_source_cpu_row,
        "t_source_candidate": t_source_candidate_row,
        "t_source_candidate_commit": CANDIDATE_COMMIT,
        "earliest_causal_gate": "d03 step 200 / 00:20",
        "model_lineage_green": lineage_green,
        "component_bindings_green": components_green,
        "full_18h_authorized": True,
        "additional_review_or_manager_authority_required": False,
    }


def assert_candidate_source_authority(
    environment: Mapping[str, str], *, require_clean: bool,
) -> dict[str, Any]:
    head = _git(REPO_ROOT, "rev-parse", "HEAD")
    approved_runner = environment.get("GPUWRF_NESTED_BUNDLE_RUNNER_SHA")
    if approved_runner and approved_runner != head:
        raise RunnerGateError("RUNNER_HEAD", f"approved={approved_runner} head={head}")
    if require_clean and _git(REPO_ROOT, "status", "--porcelain"):
        raise RunnerGateError("WORKTREE_DIRTY", "GPU launch/audited dry-run requires clean worktree")
    candidate_tree = _git(REPO_ROOT, "rev-parse", f"{CANDIDATE_COMMIT}^{{tree}}")
    if candidate_tree != CANDIDATE_TREE:
        raise RunnerGateError("CANDIDATE_TREE", candidate_tree)
    ancestor = subprocess.run(
        ("git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", CANDIDATE_COMMIT, head),
        check=False,
    ).returncode == 0
    if not ancestor:
        raise RunnerGateError("CANDIDATE_ANCESTRY", f"{CANDIDATE_COMMIT} !<= {head}")
    model_diff = _git(REPO_ROOT, "diff", "--name-only", CANDIDATE_COMMIT, "--", "src/gpuwrf")
    if model_diff:
        raise RunnerGateError("ACCEPTED_MODEL_CHANGED", model_diff)
    runner_delta = _git(REPO_ROOT, "diff", "--name-only", CANDIDATE_COMMIT, head).splitlines()
    allowed_exact = {
        str(RUNNER_SOURCE.relative_to(REPO_ROOT)),
        "scripts/v0234_nested_scalar_diffusion_complete_cpu_ab.py",
        "tests/test_v0234_nested_frozen_wrf_boundary_window.py",
    }
    allowed_prefixes = (
        str(SPRINT_DIR.relative_to(REPO_ROOT)) + "/",
        str(FINAL_NI_DIR.relative_to(REPO_ROOT)) + "/",
    )
    unexpected_runner_delta = sorted(
        path for path in runner_delta
        if path not in allowed_exact
        and not any(path.startswith(prefix) for prefix in allowed_prefixes)
    )
    if unexpected_runner_delta:
        raise RunnerGateError("RUNNER_ONLY_DELTA", repr(unexpected_runner_delta))
    source_rows = {}
    for relative, expected in ACCEPTED_SOURCE_HASHES.items():
        path = REPO_ROOT / relative
        source_rows[relative] = _require_sha(path, expected, "ACCEPTED_SOURCE_HASH")
    return {
        "runner_head": head,
        "approved_runner_head": approved_runner,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_tree": candidate_tree,
        "candidate_is_ancestor": ancestor,
        "accepted_model_diff_empty": True,
        "runner_only_delta": runner_delta,
        "unexpected_runner_delta": unexpected_runner_delta,
        "accepted_model_tree": _git(REPO_ROOT, "rev-parse", f"{CANDIDATE_COMMIT}:src/gpuwrf"),
        "accepted_sources": source_rows,
        "runner_source_sha256": sha256_file(RUNNER_SOURCE),
        "worktree_clean": not bool(_git(REPO_ROOT, "status", "--porcelain")),
    }


def assert_input_retry20_cache_authority(environment: Mapping[str, str]) -> dict[str, Any]:
    terminal_payload, terminal = read_authenticated_json(
        TERMINAL_CONTRACT, TERMINAL_CONTRACT_SHA256, "TERMINAL_CONTRACT_HASH",
    )
    cpu_manifest_payload, cpu_manifest = read_authenticated_json(
        CPU_MANIFEST, CPU_MANIFEST_SHA256, "CPU_MANIFEST_HASH",
    )
    contract_manifest = (terminal_payload.get("corrected_cpu_qa") or {}).get("manifest") or {}
    terminal_process = terminal_payload.get("terminal_cpu_process") or {}
    if (
        terminal_payload.get("schema") != "tenerife_terminal_cpu_authority_contract_v1"
        or terminal_process.get("wrf_status") != "success"
        or terminal_process.get("wrf_exit_code") != 0
        or terminal_process.get("terminal_all_processes_absent") is not True
        or contract_manifest.get("path") != str(CPU_MANIFEST)
        or contract_manifest.get("sha256") != CPU_MANIFEST_SHA256
        or cpu_manifest_payload.get("status") != "complete"
        or cpu_manifest_payload.get("qa_status") != "pass"
        or cpu_manifest_payload.get("all_required_raw_and_thin_fields_finite") is not True
        or cpu_manifest_payload.get("raw_frame_counts") != {"d01": 19, "d02": 19, "d03": 55}
        or cpu_manifest_payload.get("raw_frame_schedules")
        != standard_frame_schedule(TERMINAL_OWN_STEPS)
        or terminal_payload.get("parent_schedule_contract", {}).get("history_interval_minutes")
        != [60, 60, 20]
    ):
        raise RunnerGateError("CPU_TERMINAL_SEMANTICS", "terminal CPU manifest changed")
    inputs = {}
    for name, expected in INPUT_HASHES.items():
        path = INPUT_DIR / name
        row = _require_sha(path, expected, "INPUT_HASH")
        if path.stat().st_mode & 0o222:
            raise RunnerGateError("INPUT_WRITABLE", str(path))
        inputs[name] = row
    retained = {
        "runtime_attestation": _require_sha(
            RETRY20_ATTESTATION, RETRY20_ATTESTATION_SHA256, "RETRY20_ATTESTATION_HASH"
        ),
        "final_accept": _require_sha(
            RETRY20_FINAL_ACCEPT, RETRY20_FINAL_ACCEPT_SHA256, "RETRY20_ACCEPT_HASH"
        ),
        "incremental_pairs": _require_sha(
            RETRY20_PAIRS, RETRY20_PAIRS_SHA256, "RETRY20_PAIRS_HASH"
        ),
    }
    if not RETRY20_WRF_ROOT.is_dir() or RETRY20_WRF_ROOT.stat().st_mode & 0o222:
        raise RunnerGateError("WRF_ROOT_AUTHORITY", str(RETRY20_WRF_ROOT))
    if Path(environment["GPUWRF_WRF_ROOT"]).resolve() != RETRY20_WRF_ROOT.resolve():
        raise RunnerGateError("WRF_ROOT_ENV", environment["GPUWRF_WRF_ROOT"])
    if not LINEAGE_WORK_DIR.is_dir():
        raise RunnerGateError("LINEAGE_WORK_DIR_MISSING", str(LINEAGE_WORK_DIR))
    expected_cache_environment = {
        "GPUWRF_JAX_CACHE": "0",
        "GPUWRF_JAX_CACHE_LOCK": "0",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
    }
    actual_cache_environment = {
        name: environment.get(name) for name in expected_cache_environment
    }
    forbidden_cache_environment = sorted(
        name for name in FORBIDDEN_PERSISTENT_CACHE_ENV if name in environment
    )
    if (
        actual_cache_environment != expected_cache_environment
        or forbidden_cache_environment
    ):
        raise RunnerGateError(
            "PERSISTENT_CACHE_DISABLED",
            f"actual={actual_cache_environment!r} forbidden={forbidden_cache_environment!r}",
        )
    return {
        "terminal_contract": terminal,
        "cpu_manifest": cpu_manifest,
        "terminal_frame_counts": cpu_manifest_payload["raw_frame_counts"],
        "terminal_frame_schedules": cpu_manifest_payload["raw_frame_schedules"],
        "corrected_inputs": inputs,
        "input_authority_sha256": canonical_digest(inputs),
        "retry20": retained,
        "wrf_root": {
            "path": str(RETRY20_WRF_ROOT.resolve()),
            "mode": oct(RETRY20_WRF_ROOT.stat().st_mode & 0o777),
        },
        "lineage_work_dir": str(LINEAGE_WORK_DIR.resolve()),
        "runtime_cache": {
            "persistent": False,
            "trusted_paths": [],
            "required_environment": expected_cache_environment,
            "forbidden_environment": list(FORBIDDEN_PERSISTENT_CACHE_ENV),
        },
    }


def assert_lock_source_authority() -> dict[str, Any]:
    commit = _git(LOCK_ROOT, "rev-parse", "HEAD")
    if commit != LOCK_COMMIT:
        raise RunnerGateError("LOCK_COMMIT", commit)
    return {
        "root": str(LOCK_ROOT.resolve()),
        "commit": commit,
        "wrapper": _require_sha(LOCK_WRAPPER, LOCK_WRAPPER_SHA256, "LOCK_WRAPPER_HASH"),
        "verifier": _require_sha(LOCK_VERIFIER, LOCK_VERIFIER_SHA256, "LOCK_VERIFIER_HASH"),
        "required_label": LOCK_LABEL,
        "required_intent": LOCK_INTENT,
    }


def assert_live_lock_authority(environment: Mapping[str, str]) -> dict[str, Any]:
    required = {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_FD": "9",
        "GPUWRF_GPU_LOCK_FILE": "/tmp/wrf_gpu2_gpu.lock",
        "GPUWRF_GPU_LOCK_HOLDER_FILE": "/tmp/wrf_gpu2_gpu.lock.holder",
        "GPUWRF_GPU_LOCK_LABEL": LOCK_LABEL,
    }
    for name, expected in required.items():
        if environment.get(name) != expected:
            raise RunnerGateError("LOCK_ENV", f"{name}={environment.get(name)!r}")
    token = environment.get("GPUWRF_GPU_LOCK_TOKEN", "")
    if not token:
        raise RunnerGateError("LOCK_TOKEN", "missing")
    spec = importlib.util.spec_from_file_location("v0234_nested_bundle_lock_v2", LOCK_VERIFIER)
    if spec is None or spec.loader is None:
        raise RunnerGateError("LOCK_IMPORT", str(LOCK_VERIFIER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lock_path = Path(required["GPUWRF_GPU_LOCK_FILE"])
    lease = module._read_sidecar(lock_path)
    module._validate_schema(dict(lease))
    if (
        lease.get("intent") != LOCK_INTENT
        or lease.get("label") != LOCK_LABEL
        or lease.get("lease_id") != token
    ):
        raise RunnerGateError("LOCK_LEASE", "label/intent/token mismatch")
    module._validate_lock_lease_fd(lock_path, 9, lease)
    probe_fd = module._open_gpu_lock(lock_path)
    try:
        module._validate_live_lease(lock_path, probe_fd, expected=lease)
    finally:
        os.close(probe_fd)
    return {
        "live_verified_before_jax_import": True,
        "intent": lease["intent"],
        "label": lease["label"],
        "lease_id_sha256": sha256_text(token),
    }


def validate_runner_audit(environment: Mapping[str, str]) -> dict[str, Any]:
    path = Path(environment["GPUWRF_NESTED_BUNDLE_RUNNER_AUDIT"]).resolve()
    expected = environment["GPUWRF_NESTED_BUNDLE_RUNNER_AUDIT_SHA256"]
    payload, row = read_authenticated_json(path, expected, "RUNNER_AUDIT_HASH")
    if (
        payload.get("schema") != CPU_PROOF_SCHEMA
        or payload.get("verdict") != AUDIT_ADMISSION
        or payload.get("candidate_commit") != CANDIDATE_COMMIT
        or payload.get("candidate_tree") != CANDIDATE_TREE
        or payload.get("runner_source_sha256") != sha256_file(RUNNER_SOURCE)
        or payload.get("candidate_clean_authority") != CANDIDATE_CLEAN_AUTHORITY
        or payload.get("deterministic_payload") is not True
        or payload.get("gpu_commands_run") != 0
        or payload.get("jax_imported") is not False
        or payload.get("focused_tests", {}).get("returncode") != 0
        or not isinstance(payload.get("focused_tests", {}).get("stdout_sha256"), str)
        or len(payload.get("focused_tests", {}).get("stdout_sha256", "")) != 64
        or payload.get("static_audit", {}).get("passed") is not True
        or payload.get("exact_launch_audit", {}).get("passed") is not True
    ):
        raise RunnerGateError("RUNNER_AUDIT_SEMANTICS", str(path))
    audit_head = str(payload.get("runner_head_at_audit", ""))
    current_head = _git(REPO_ROOT, "rev-parse", "HEAD")
    if not audit_head or subprocess.run(
        ("git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", audit_head, current_head),
        check=False,
    ).returncode != 0:
        raise RunnerGateError("RUNNER_AUDIT_ANCESTRY", f"audit={audit_head} head={current_head}")
    source_diff = _git(
        REPO_ROOT,
        "diff", "--name-only", audit_head, current_head, "--",
        str(RUNNER_SOURCE.relative_to(REPO_ROOT)),
    )
    if source_diff:
        raise RunnerGateError("RUNNER_AUDIT_SOURCE_CHANGED", source_diff)
    return {**row, "verdict": payload["verdict"], "runner_source_sha256": payload["runner_source_sha256"]}


def assert_admission_authority(
    environment: Mapping[str, str],
    *,
    run_dir: Path,
    proof_output: Path,
    require_live_lock: bool,
    require_runner_audit: bool,
    require_clean: bool,
) -> dict[str, Any]:
    preimport = validate_preimport_environment(
        environment,
        require_runner_audit=require_runner_audit,
        require_tooling_critic=bool(
            require_live_lock and REQUIRE_TOOLING_CRITIC_ACCEPT
        ),
    )
    if run_dir == LINEAGE_WORK_DIR.resolve() or LINEAGE_WORK_DIR.resolve() not in run_dir.parents:
        raise RunnerGateError("RUN_DIR_LINEAGE", str(run_dir))
    if run_dir.exists() or run_dir.is_symlink():
        raise RunnerGateError("RUN_DIR_EXISTS", str(run_dir))
    if proof_output.exists() or proof_output.is_symlink():
        raise RunnerGateError("PROOF_EXISTS", str(proof_output))
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != EXPECTED_AFFINITY:
        raise RunnerGateError("CPU_AFFINITY", repr(affinity))
    authority = {
        "preimport": preimport,
        "candidate": assert_candidate_source_authority(environment, require_clean=require_clean),
        "final_candidate_and_owner_override": assert_final_candidate_proof_authority(),
        "input_retry20_cache": assert_input_retry20_cache_authority(environment),
        "lock_source": assert_lock_source_authority(),
        "preemption": assert_preemption_clear("preimport-admission"),
        "cpu_affinity": affinity,
        "run_dir_nonexistent": str(run_dir),
        "proof_output_nonexistent": str(proof_output),
    }
    if require_runner_audit:
        authority["runner_audit"] = validate_runner_audit(environment)
    if require_live_lock:
        authority["live_lock"] = assert_live_lock_authority(environment)
    else:
        authority["live_lock"] = {
            "checked": False,
            "reason": "cpu-dry-run-does-not-touch-or-acquire-gpu-lock",
        }
    return authority


def standard_output_steps(end_steps: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    """Exact corrected scheduler steps, including explicit lead-zero history."""

    result: dict[str, tuple[int, ...]] = {}
    for name in ("d01", "d02", "d03"):
        dt = DT_SECONDS[name]
        interval = HISTORY_INTERVAL_SECONDS[name]
        terminal = int(end_steps[name])
        values = [0]
        index = 1
        while True:
            own_step = int(math.ceil(index * interval / dt))
            if own_step > terminal:
                break
            values.append(own_step)
            index += 1
        result[name] = tuple(values)
    return result


def standard_frame_schedule(end_steps: Mapping[str, int]) -> dict[str, list[str]]:
    return {
        name: [
            (RUN_START + timedelta(seconds=step * DT_SECONDS[name])).strftime("%Y-%m-%d_%H:%M:%S")
            for step in steps
        ]
        for name, steps in standard_output_steps(end_steps).items()
    }


def schedule_clock_oracle() -> dict[str, Any]:
    prefix_root = sum(PREFIX_SEGMENTS)
    prefix = {"d01": prefix_root, "d02": prefix_root * 3, "d03": prefix_root * 9}
    window_root = (WINDOW_OWN_STEPS["d03"] - PREFIX_OWN_STEPS["d03"]) // 9
    window = {
        "d01": prefix["d01"] + window_root,
        "d02": prefix["d02"] + window_root * 3,
        "d03": prefix["d03"] + window_root * 9,
    }
    sampled = list(range(prefix["d03"] + 1, window["d03"] + 1))
    prefix_outputs = standard_output_steps(PREFIX_OWN_STEPS)
    window_outputs = standard_output_steps(WINDOW_OWN_STEPS)
    terminal_outputs = standard_output_steps(TERMINAL_OWN_STEPS)
    parent_root = sum(PARENT_RECON_SEGMENTS)
    parent_join = {"d01": parent_root, "d02": parent_root * 3, "d03": 8800}
    parent_aligned = dict(PARENT_ALIGNED_OWN_STEPS)
    parent_9000 = {
        "d01": parent_aligned["d01"] + PARENT_STAGE_9000_ROOT_STEPS,
        "d02": parent_aligned["d02"] + 3 * PARENT_STAGE_9000_ROOT_STEPS,
        "d03": parent_aligned["d03"] + 9 * PARENT_STAGE_9000_ROOT_STEPS,
    }
    parent_9405 = {
        "d01": parent_9000["d01"] + PARENT_STAGE_9405_ROOT_STEPS,
        "d02": parent_9000["d02"] + 3 * PARENT_STAGE_9405_ROOT_STEPS,
        "d03": parent_9000["d03"] + 9 * PARENT_STAGE_9405_ROOT_STEPS,
    }
    parent_terminal = {
        "d01": parent_9405["d01"] + PARENT_STAGE_TERMINAL_ROOT_STEPS,
        "d02": parent_9405["d02"] + 3 * PARENT_STAGE_TERMINAL_ROOT_STEPS,
        "d03": parent_9405["d03"] + 9 * PARENT_STAGE_TERMINAL_ROOT_STEPS,
    }
    parent_prefix_outputs = standard_output_steps(PARENT_JOIN_OWN_STEPS)
    passed = bool(
        prefix == PREFIX_OWN_STEPS
        and window == WINDOW_OWN_STEPS
        and sampled == list(WINDOW_D03_STEPS)
        and prefix_outputs["d03"][-1] == DECISIVE_D03_STEP
        and {name: len(values) for name, values in prefix_outputs.items()}
        == {"d01": 16, "d02": 16, "d03": 46}
        and {name: len(values) for name, values in window_outputs.items()}
        == {"d01": 16, "d02": 16, "d03": 48}
        and {name: len(values) for name, values in terminal_outputs.items()}
        == {"d01": 19, "d02": 19, "d03": 55}
        and window_outputs["d03"][-3:] == (9000, 9200, 9400)
        and parent_join == PARENT_JOIN_OWN_STEPS
        and parent_aligned == PARENT_ALIGNED_OWN_STEPS
        and parent_9000 == {"d01": 1000, "d02": 3000, "d03": 9000}
        and parent_9405 == WINDOW_OWN_STEPS
        and parent_terminal == TERMINAL_OWN_STEPS
        and {name: len(parent_prefix_outputs[name]) for name in PARENT_DOMAINS}
        == {"d01": RC3_RAW_OUTPUT_COUNTS["d01"], "d02": RC3_RAW_OUTPUT_COUNTS["d02"]}
        and 8800 - (2934 - 1) * 3 == 1
    )
    return {
        "passed": passed,
        "prefix_segments": list(PREFIX_SEGMENTS),
        "prefix_build_count": 1,
        "prefix_own_steps": prefix,
        "window_root_steps": window_root,
        "window_own_steps": window,
        "d03_dispatch_steps": sampled,
        "prefix_output_steps": {name: list(values) for name, values in prefix_outputs.items()},
        "window_output_steps": {name: list(values) for name, values in window_outputs.items()},
        "terminal_output_steps": {name: list(values) for name, values in terminal_outputs.items()},
        "window_output_counts": {name: len(values) for name, values in window_outputs.items()},
        "terminal_output_counts": {name: len(values) for name, values in terminal_outputs.items()},
        "first_progress_d03_steps": list(FIRST_PROGRESS_D03_STEPS),
        "decisive_d03_step": DECISIVE_D03_STEP,
        "second_prefix_or_rebuild": False,
        "parent_join": {
            "segments": list(PARENT_RECON_SEGMENTS),
            "parent_only_own_steps": {name: parent_join[name] for name in PARENT_DOMAINS},
            "retained_d03_step": parent_join["d03"],
            "zero_d03_dispatches_before_join": True,
            "d03_subcycle_position_at_join": 8800 - (2934 - 1) * 3,
            "catchup_d03_steps": list(PARENT_CATCHUP_D03_STEPS),
            "aligned_own_steps": parent_aligned,
            "stage_9000_root_steps": PARENT_STAGE_9000_ROOT_STEPS,
            "stage_9000_own_steps": parent_9000,
            "stage_9405_root_steps": PARENT_STAGE_9405_ROOT_STEPS,
            "stage_9405_own_steps": parent_9405,
            "stage_terminal_root_steps": PARENT_STAGE_TERMINAL_ROOT_STEPS,
            "terminal_own_steps": parent_terminal,
            "retained_parent_output_counts": {
                name: len(parent_prefix_outputs[name]) for name in PARENT_DOMAINS
            },
        },
    }


def audit_exact_launch_command(path: Path) -> dict[str, Any]:
    text = path.read_text()
    required = (
        "/usr/bin/env -i",
        "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE=1",
        f"GPUWRF_NESTED_BUNDLE_APPROVED_SHA={CANDIDATE_COMMIT}",
        f"GPUWRF_FINAL_NI_OWNER_OVERRIDE={OWNER_OVERRIDE}",
        "GPUWRF_JAX_CACHE=0",
        "GPUWRF_JAX_CACHE_LOCK=0",
        "JAX_ENABLE_COMPILATION_CACHE=false",
        "GPUWRF_NESTED_BUNDLE_RUNNER_SHA=\"$RUNNER_SHA\"",
        "GPUWRF_NESTED_BUNDLE_RUNNER_AUDIT_SHA256=\"$AUDIT_SHA\"",
        str(LOCK_WRAPPER),
        f"--label {LOCK_LABEL}",
        "--intent production-preemptible",
        "/usr/bin/taskset -c 12-15",
        "-m scripts.v0234_nested_frozen_wrf_boundary_window",
        f"RUN_DIR={LINEAGE_WORK_DIR / FULL_REPLAY_NAMESPACE}",
        'PROOF="$RUN_DIR/full-terminal-proof.json"',
        "--direct-terminal",
    )
    if REQUIRE_KNOWN_1500_V10_RECORD:
        required = (*required, "--record-known-1500-v10-red")
    if REQUIRE_TOOLING_CRITIC_ACCEPT:
        required = (
            *required,
            'GPUWRF_POST_FABLE_TOOLING_CRITIC="$CRITIC"',
            'GPUWRF_POST_FABLE_TOOLING_CRITIC_SHA256="$CRITIC_SHA"',
        )
    missing = [token for token in required if token not in text]
    forbidden = [
        token for token in (
            "GPUWRF_NORMAL_BDY_RELAX_STRENGTH",
            "GPUWRF_CORRECTED_NI_RCA",
            "GPUWRF_PHASE_TAP",
            "GPUWRF_SANITIZER",
            "GPUWRF_TOLERANCE",
            "GPUWRF_NESTED_BUNDLE_CRITIC_ACCEPTANCE",
            "GPUWRF_JAX_CACHE_DIR=",
            "JAX_COMPILATION_CACHE_DIR=",
            "GPUWRF_CACHE=",
            "nvidia-smi",
            "rocm-smi",
            "--cpu-dry-run",
            "--bootstrap-audit",
            "--lower-only-custom-call-diagnostic",
            "--hold-seconds",
            "--continuation-authority",
            "--parent-join-resume",
        ) if token in text
    ]
    passed = bool(
        not missing
        and not forbidden
        and text.count("-m scripts.v0234_nested_frozen_wrf_boundary_window") == 1
        and text.count("/scripts/with_gpu_lock.sh") == 1
    )
    return {
        "passed": passed,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "missing_required_tokens": missing,
        "forbidden_tokens": forbidden,
        "lock_commit_bound_by_preimport_authority": LOCK_COMMIT,
        "one_model_process": text.count("-m scripts.v0234_nested_frozen_wrf_boundary_window") == 1,
        "known_1500_v10_record_required": REQUIRE_KNOWN_1500_V10_RECORD,
        "tooling_critic_accept_required": REQUIRE_TOOLING_CRITIC_ACCEPT,
        "shell_syntax_checked_separately": True,
    }


def assert_same_carry_continuity(
    prefix_carries: Mapping[str, Any], window_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    names = tuple(prefix_carries)
    same = names == tuple(window_inputs) and all(
        prefix_carries[name] is window_inputs[name] for name in names
    )
    if not same:
        raise RunnerGateError("CARRY_CONTINUITY", "prefix/window carry object changed")
    return {
        "same_object_per_domain": True,
        "domains": list(names),
        "prefix_rebuild_count": 0,
        "recompile_count_at_boundary": 0,
    }


def evaluate_health_summary(
    step: int,
    summary: Mapping[str, Any],
    *,
    step9313_scale_baseline: Sequence[float] | None,
) -> dict[str, Any]:
    nonfinite_by_leaf = [int(value) for value in summary["nonfinite_by_leaf"]]
    scale = [float(value) for value in summary["scale_maxabs"]]
    violations: list[dict[str, Any]] = []
    if len(nonfinite_by_leaf) != 106:
        violations.append({"gate": "complete_106_leaf_carry", "actual": len(nonfinite_by_leaf)})
    total_nonfinite = sum(nonfinite_by_leaf)
    if total_nonfinite:
        violations.append({"gate": "complete_carry_finite", "actual": total_nonfinite})
    if 9214 <= step <= 9313 and float(summary["theta_max"]) >= THETA_LIMITER_CEILING_K:
        violations.append({
            "gate": "theta_below_existing_limiter_ceiling",
            "actual": float(summary["theta_max"]),
            "ceiling": THETA_LIMITER_CEILING_K,
        })
    if step == 9313:
        if float(summary["corner_mu_pert_maxabs"]) >= 2000.0:
            violations.append({"gate": "corner_mu_pert_abs_lt_2000", "actual": float(summary["corner_mu_pert_maxabs"])})
        for name, actual in zip(("u", "v"), summary["wind_maxabs"][:2]):
            if float(actual) >= 80.0:
                violations.append({"gate": f"{name}_abs_lt_80", "actual": float(actual)})
        if float(summary["wind_maxabs"][2]) >= 30.0:
            violations.append({"gate": "w_abs_lt_30", "actual": float(summary["wind_maxabs"][2])})
    if step in (9314, 9405):
        if step9313_scale_baseline is None or len(step9313_scale_baseline) != len(ALL_SCALE_FIELDS):
            violations.append({"gate": "step9313_scale_baseline_present", "actual": None})
        else:
            for name, actual, baseline in zip(ALL_SCALE_FIELDS, scale, step9313_scale_baseline):
                amplification = actual / max(float(baseline), 1.0)
                if actual > ABSOLUTE_SCALE_CEILING:
                    violations.append({"gate": "absolute_scale_le_1e6", "field": name, "actual": actual})
                if amplification > AMPLIFICATION_CEILING:
                    violations.append({"gate": "amplification_le_1000", "field": name, "actual": amplification})
        for name, actual in zip(("p_total", "mu_total"), summary["positive_min"]):
            if not float(actual) > 0.0:
                violations.append({"gate": "strictly_positive", "field": name, "actual": float(actual)})
    if step == 9315 and int(summary["ni_nonfinite_count"]):
        violations.append({"gate": "state_Ni_finite", "actual": int(summary["ni_nonfinite_count"])})
    return {
        "step": int(step),
        "passed": not violations,
        "complete_carry_nonfinite_count": total_nonfinite,
        "theta_max": float(summary["theta_max"]),
        "corner_mu_pert_maxabs": float(summary["corner_mu_pert_maxabs"]),
        "wind_maxabs": [float(value) for value in summary["wind_maxabs"]],
        "positive_min": [float(value) for value in summary["positive_min"]],
        "scale_maxabs": scale,
        "ni_nonfinite_count": int(summary["ni_nonfinite_count"]),
        "violations": violations,
    }


def compare_no_worse_metrics(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any],
) -> dict[str, Any]:
    rows = {}
    violations = []
    candidate_metrics = candidate.get("metrics") or {}
    baseline_metrics = baseline.get("metrics") or {}
    for field in STRICT_FIELDS:
        if field not in candidate_metrics or field not in baseline_metrics:
            violations.append({"field": field, "gate": "metric_present"})
            continue
        actual = float(candidate_metrics[field]["rmse"])
        retained = float(baseline_metrics[field]["rmse"])
        frozen = FROZEN_1500_RMSE[field]
        authority_exact = retained == frozen
        passed = authority_exact and actual <= frozen
        rows[field] = {
            "candidate_rmse": actual,
            "retry20_rmse": retained,
            "frozen_maximum_rmse": frozen,
            "retry20_authority_exact": authority_exact,
            "no_worse": passed,
        }
        if not passed:
            violations.append({
                "field": field,
                "gate": "rmse_no_worse_than_frozen_retry20",
                "actual": actual,
                "retry20": retained,
                "frozen_maximum": frozen,
                "retry20_authority_exact": authority_exact,
            })
    static_pass = bool(candidate.get("per_frame_static_pass"))
    if not static_pass:
        violations.append({"field": list(STATIC_FIELDS), "gate": "frozen_geometry_exact"})
    return {
        "passed": not violations,
        "strict_fields": rows,
        "static_exact": static_pass,
        "violations": violations,
        "v10_causal_use": False,
        "v10_note": "V10 is reported as one strict metric; no common Ni mechanism is inferred.",
    }


def authenticate_known_1500_v10_observation() -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the one retained V10-only red that this discriminator may cross."""

    if (
        KNOWN_1500_V10_ARTIFACT is None
        or KNOWN_1500_V10_ARTIFACT_SHA256 is None
        or KNOWN_1500_V10_PAYLOAD_SHA256 is None
    ):
        raise RunnerGateError(
            "KNOWN_1500_V10_AUTHORITY_UNBOUND",
            "record-and-continue requested without an exact retained artifact",
        )
    payload, row = _authenticated_canonical_json(
        KNOWN_1500_V10_ARTIFACT,
        KNOWN_1500_V10_ARTIFACT_SHA256,
        KNOWN_1500_V10_PAYLOAD_SHA256,
        "KNOWN_1500_V10_ARTIFACT",
    )
    decisive = payload.get("decisive_1500") or {}
    candidate = payload.get("candidate") or {}
    expected_time = (
        RUN_START + timedelta(seconds=DECISIVE_D03_STEP * DT_SECONDS["d03"])
    ).isoformat()
    fields = decisive.get("strict_fields") or {}
    violations = decisive.get("violations") or []
    if (
        payload.get("domain") != "d03"
        or payload.get("own_step") != DECISIVE_D03_STEP
        or payload.get("valid_time") != expected_time
        or payload.get("passed") is not False
        or decisive.get("passed") is not False
        or decisive.get("static_exact") is not True
        or decisive.get("v10_causal_use") is not False
        or set(fields) != set(STRICT_FIELDS)
        or any(
            row_value.get("retry20_authority_exact") is not True
            for row_value in fields.values()
        )
        or fields.get("V10", {}).get("no_worse") is not False
        or any(
            fields[field].get("no_worse") is not True
            for field in STRICT_FIELDS
            if field != "V10"
        )
        or len(violations) != 1
        or violations[0].get("field") != "V10"
        or not isinstance(candidate.get("sha256"), str)
        or len(candidate["sha256"]) != 64
    ):
        raise RunnerGateError(
            "KNOWN_1500_V10_AUTHORITY_SEMANTICS",
            str(KNOWN_1500_V10_ARTIFACT),
        )
    return payload, {**row, "candidate_sha256": candidate["sha256"]}


def classify_exact_known_1500_v10_red(
    decisive: Mapping[str, Any],
    *,
    candidate_sha256: str,
    authority: Mapping[str, Any],
    candidate_path: Path | None = None,
    cpu_path: Path | None = None,
    runtime: SimpleNamespace | None = None,
) -> dict[str, Any]:
    """Admit only an exact replay of the authenticated, V10-only observation."""

    expected_decisive = authority.get("decisive_1500") or {}
    expected_candidate = (authority.get("candidate") or {}).get("sha256")
    exact_decisive = dict(decisive) == expected_decisive
    exact_candidate = candidate_sha256 == expected_candidate
    return {
        "schema": "gpuwrf.v0234.post-fable-known-1500-v10-record.v1",
        "passed": bool(exact_decisive and exact_candidate),
        "classification": "KNOWN_V10_ONLY_RED_EXACT" if exact_decisive and exact_candidate else "OBSERVATION_DRIFT",
        "record_and_continue_for_late_ni_only": bool(exact_decisive and exact_candidate),
        "waiver_or_reclassification": False,
        "release_blocker_remains": True,
        "decisive_payload_exact": exact_decisive,
        "candidate_frame_sha256_exact": exact_candidate,
        "expected_candidate_sha256": expected_candidate,
        "observed_candidate_sha256": candidate_sha256,
        "expected_decisive_payload_sha256": canonical_digest(expected_decisive),
        "observed_decisive_payload_sha256": canonical_digest(dict(decisive)),
    }


def validate_continuation_authority(
    payload: Mapping[str, Any],
    *,
    nonce: str,
    runner_commit: str,
    window_proof_sha256: str,
    now: datetime,
    not_before: datetime | None = None,
) -> dict[str, Any]:
    try:
        issued = datetime.fromisoformat(str(payload["issued_utc"]))
        expires = datetime.fromisoformat(str(payload["expires_utc"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise RunnerGateError("CONTINUATION_TIME", repr(payload)) from exc
    expected = {
        "schema": "gpuwrf.v0234.nested-frozen-wrf-boundary-continuation-authorization.v1",
        "decision": "CONTINUE_SAME_PROCESS_TO_18H",
        "nonce": nonce,
        "runner_commit": runner_commit,
        "window_proof_sha256": window_proof_sha256,
        "reuse_live_carries": True,
        "reuse_compiled_d03_one_step": True,
        "no_second_prefix": True,
        "terminal_d03_step": 10800,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RunnerGateError("CONTINUATION_BINDING", f"{key}={payload.get(key)!r}")
    if issued.tzinfo is None or expires.tzinfo is None or not (issued <= now < expires):
        raise RunnerGateError("CONTINUATION_FRESHNESS", f"issued={issued} now={now} expires={expires}")
    if not_before is not None and issued < not_before:
        raise RunnerGateError(
            "CONTINUATION_NOT_FRESH_FOR_HOLD",
            f"issued={issued} hold_started={not_before}",
        )
    if (expires - issued).total_seconds() > 900.0:
        raise RunnerGateError("CONTINUATION_LIFETIME", str(expires - issued))
    return {"authorized": True, "issued_utc": issued.isoformat(), "expires_utc": expires.isoformat()}


def hold_for_fresh_continuation(
    carries: Any,
    *,
    authority_path: Path,
    hold_seconds: float,
    runner_commit: str,
    window_proof_sha256: str,
    continuation: Callable[[Any], Any],
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    nonce = hashlib.sha256(
        f"{runner_commit}:{window_proof_sha256}:{os.getpid()}:{time.time_ns()}".encode()
    ).hexdigest()
    request_path = authority_path.with_name("continuation-request.json")
    request_time = now_fn()
    request = {
        "schema": "gpuwrf.v0234.nested-frozen-wrf-boundary-continuation-request.v1",
        "nonce": nonce,
        "runner_commit": runner_commit,
        "window_proof_sha256": window_proof_sha256,
        "authority_path": str(authority_path),
        "hold_seconds": float(hold_seconds),
        "request_utc": request_time.isoformat(),
        "full_continuation_currently_authorized": False,
    }
    atomic_write_json(request_path, request)
    deadline = time.monotonic() + max(0.0, float(hold_seconds))
    while time.monotonic() <= deadline:
        if authority_path.is_file() and not authority_path.is_symlink():
            payload = json.loads(authority_path.read_text())
            admission = validate_continuation_authority(
                payload,
                nonce=nonce,
                runner_commit=runner_commit,
                window_proof_sha256=window_proof_sha256,
                now=now_fn(),
                not_before=request_time,
            )
            result = continuation(carries)
            return {
                "decision": "AUTHORIZED_CONTINUATION_EXECUTED",
                "same_live_carries_passed": result is carries or result is not None,
                "admission": admission,
                "request": request,
            }
        if time.monotonic() >= deadline:
            break
        sleep_fn(min(1.0, max(0.0, deadline - time.monotonic())))
    return {
        "decision": "HOLD_EXPIRED_NO_CONTINUATION",
        "same_live_carries_preserved_until_exit": True,
        "request": request,
    }


def static_source_audit() -> dict[str, Any]:
    source = RUNNER_SOURCE.read_text()
    tree = ast.parse(source)
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module or "")
    forbidden_top = sorted(
        name for name in top_imports
        if name.split(".")[0] in {"jax", "gpuwrf", "numpy", "netCDF4"}
    )
    runtime_source = inspect.getsource(_run_window)
    dispatch_offset = runtime_source.find("value = executable(")
    health_offset = runtime_source.find("summary = materialize_health(health_executable, value, runtime)")
    preempt_offset = runtime_source.find("assert_preemption_clear(")
    ordered = 0 <= preempt_offset < dispatch_offset < health_offset
    output_source = inspect.getsource(StandardCadenceOutput.__call__)
    writer_offset = output_source.find("writer_result = self.writer(")
    retain_offset = output_source.find("retained = _retain_complete_carry(")
    checkpoint_health_offset = output_source.find("health_raw = materialize_health(")
    pair_offset = output_source.find("pair = self.pairer.pair(")
    output_ordered = 0 <= writer_offset < retain_offset < checkpoint_health_offset < pair_offset
    model_lower_source = inspect.getsource(_lower_domain_once)
    model_compile_source = inspect.getsource(_lower_compile_domain_once)
    health_compile_source = inspect.getsource(_build_health_executable)
    lazy_health_source = inspect.getsource(LazyLiveCarryHealthExecutable)
    runtime_main_source = inspect.getsource(_runtime_main)
    parent_main_source = inspect.getsource(_parent_join_runtime_main)
    parent_scheduler_source = inspect.getsource(parent_scheduler_contract)
    parent_recon_source = inspect.getsource(run_parent_only_reconstitution)
    parent_rehearsal_source = inspect.getsource(run_real_schema_parent_cpu_rehearsal)
    parent_blocker_source = inspect.getsource(_retain_parent_join_blocker)
    parent_advance_source = inspect.getsource(ParentJoinContinuationAdvance)
    parent_target_source = inspect.getsource(compare_join_boundary_target_records)
    terminal_union_source = inspect.getsource(build_terminal_union_and_verdict)
    full_terminal_source = inspect.getsource(build_full_terminal_verdict)
    diagnostic_source = inspect.getsource(_lower_only_custom_call_diagnostic)
    explicit_lower_count = lambda value: len(re.findall(r"lowered\s*=\s*[^\n]+\.lower\(", value))
    explicit_compile_counts = {
        "production_lower_calls_in_source": explicit_lower_count(model_lower_source),
        "production_compile_calls_in_source": model_compile_source.count(".compile()"),
        "health_lower_calls_in_source": explicit_lower_count(health_compile_source),
        "health_compile_calls_in_source": health_compile_source.count(".compile()"),
    }
    compile_counts_exact = explicit_compile_counts == {
        "production_lower_calls_in_source": 1,
        "production_compile_calls_in_source": 1,
        "health_lower_calls_in_source": 1,
        "health_compile_calls_in_source": 1,
    }
    pairer_source = inspect.getsource(IncrementalFramePairer)
    known_v10_source = inspect.getsource(classify_exact_known_1500_v10_red)
    critic_source = inspect.getsource(validate_tooling_critic_acceptance)
    late_retain_source = inspect.getsource(retain_late_window_carry_and_frame)
    continuation_source = inspect.getsource(_run_authorized_continuation)
    host_file_pairing_only = all(
        token not in pairer_source for token in ("device_get", "block_until_ready", "jax.", "jnp.")
    )
    early_causal_fail_closed = bool(
        "early_causal_ring1_metrics(" in pairer_source
        and 'step == 200' in pairer_source
        and 'artifact["early_causal_00_20"]["passed"] is True' in pairer_source
        and 'artifact["early_causal_00_20"]' in pairer_source
        and 'or artifact["decisive_1500"]' in pairer_source
    )
    known_v10_record_fail_closed = bool(
        "dict(decisive) == expected_decisive" in known_v10_source
        and "candidate_sha256 == expected_candidate" in known_v10_source
        and '"waiver_or_reclassification": False' in known_v10_source
        and '"release_blocker_remains": True' in known_v10_source
        and 'not artifact["passed"] and not artifact["recorded_known_v10_red_and_continued"]'
        in pairer_source
    )
    tooling_critic_preimport_gate = bool(
        "TOOLING_CRITIC_VERDICT" in critic_source
        and "runner_source_sha256" in critic_source
        and "exact_launcher_sha256" in critic_source
        and "full_tree_gpu_replay_admitted" in critic_source
        and "require_live_lock and REQUIRE_TOOLING_CRITIC_ACCEPT"
        in inspect.getsource(assert_admission_authority)
    )
    late_window_retention_fail_closed = bool(
        "LATE_WINDOW_RETAIN_D03_STEPS" in runtime_source
        and "retain_late_window_carry_and_frame(" in runtime_source
        and "_retain_complete_carry(" in late_retain_source
        and "all_numeric_nonfinite" in late_retain_source
        and "static_identity" in late_retain_source
        and "explicit CONTRACT.md retention" in late_retain_source
    )
    continuation_first_red_retained = bool(
        "_retain_failure_pair(" in continuation_source
        and "CONTINUATION_NONFINITE" in continuation_source
        and "except MetricGateFailure as exc:" in continuation_source
    )
    canonical_fresh_output_path = bool(
        'output_dir = args.run_dir / "gpu-output"' in runtime_main_source
    )
    terminal_v10_red_honest = bool(
        "FULL_18H_LATE_NI_COMPLETE_KNOWN_V10_RED_REMAINS" in runtime_main_source
        and '"all_incremental_pairs_passed": not args.record_known_1500_v10_red'
        in runtime_main_source
        and '"known_1500_v10_red_remains_release_blocker"' in runtime_main_source
    )
    retain_offset = model_lower_source.find("atomically_retain_lowered_hlo_audit(")
    policy_offset = model_lower_source.find("evaluate_stablehlo_policy(")
    hlo_artifact_precedes_gate = 0 <= retain_offset < policy_offset
    diagnostic_never_compiles = bool(
        diagnostic_source.count("_lower_d03_once(") == 1
        and ".compile(" not in diagnostic_source
        and "_build_health_executable" not in diagnostic_source
    )
    lazy_health_binds_live_shape_once = bool(
        "LazyLiveCarryHealthExecutable(runtime)" in runtime_main_source
        and "LazyLiveCarryHealthExecutable(runtime)" in parent_main_source
        and "_build_health_executable(initial_carries" not in runtime_main_source
        and "HEALTH_LIVE_SIGNATURE_DRIFT" in lazy_health_source
        and "signature_recompile_allowed" in lazy_health_source
        and "self.compile_calls = 1" in lazy_health_source
    )
    forbidden_dispatch_tokens = [
        token for token in ("recorder", "phase_tap", "capture_rca", "callback_summary")
        if token in runtime_source.lower()
    ]
    parent_prejoin_isolation = bool(
        "PARENT_DOMAINS" in parent_recon_source
        and "PARENT_PREJOIN_D03_DISPATCH" in parent_recon_source
        and "d03_dispatches != 0" in parent_recon_source
        and "d03_dispatches_before_join" in parent_main_source
        and parent_main_source.find("run_parent_only_reconstitution(")
        < parent_main_source.find("load_authenticated_retained_d03_carry(")
        < parent_main_source.find("runtime._operational_force(")
    )
    parent_program_scope = bool(
        "LazyOneStepDomainExecutable" in parent_recon_source
        and "programs[name].advance" in parent_recon_source
        and "compile_calls\"] != 1" in parent_recon_source
        and "LazyOneStepDomainExecutable" in parent_main_source
    )
    parent_mixed_integral_projection = bool(
        'cadence_all != {"d01": 67, "d02": 200, "d03": 200}'
        in parent_scheduler_source
        and 'set(nonintegral_all) != {"d01"}' in parent_scheduler_source
        and "if name in nonintegral_all" in parent_scheduler_source
        and 'set(nonintegral) != {"d01"}' in parent_scheduler_source
    )
    parent_real_schema_rehearsal = bool(
        "runtime.ordinary.load_corrected_tree(run_dir)" in parent_rehearsal_source
        and "runtime.DomainHierarchy.from_edges(" in parent_rehearsal_source
        and "runtime._PerDomainWrfoutWriter(" in parent_rehearsal_source
        and "runtime._emit_initial_history_frames(" in parent_rehearsal_source
        and "edge_lookup(parent_nests[0])" in parent_rehearsal_source
        and "runtime.run_domain_tree_callbacks(" in parent_rehearsal_source
        and "identity_advance" in parent_rehearsal_source
        and '"ordinary_model_compiles": 0' in parent_rehearsal_source
        and "LazyOneStepDomainExecutable" not in parent_rehearsal_source
        and ".lower(" not in parent_rehearsal_source
        and ".compile(" not in parent_rehearsal_source
    )
    parent_blocker_retains_stage_traceback = bool(
        '"stage": getattr(args, "_parent_join_stage", "UNSET")' in parent_blocker_source
        and '"stage_trace": list(getattr(args, "_parent_join_stages", []))'
        in parent_blocker_source
        and '"traceback": traceback_text' in parent_blocker_source
        and "traceback.format_exc()" in inspect.getsource(main)
    )
    parent_health_after_dispatch = bool(
        parent_advance_source.find("value = self.programs[name].one_step(")
        < parent_advance_source.find("summary = materialize_health(")
        and "if name != \"d03\"" in parent_advance_source
    )
    target_record_fail_closed = bool(
        "retained[1]" in parent_target_source
        and "rebuilt[1]" in parent_target_source
        and "rebuilt_carry_used_for_continuation\": False" in parent_target_source
        and "PARENT_JOIN_TARGET_IDENTITY" in parent_main_source
        and "del rebuilt_d03" in parent_main_source
    )
    terminal_union_is_snapshot_only = bool(
        "immutable_file_snapshot" in terminal_union_source
        and "terminal_mode=True" in terminal_union_source
        and "state.get(\"matched_count\") != 55" in terminal_union_source
        and "final_verdict(" in terminal_union_source
    )
    terminal_cpu_uses_canonical_contract = bool(
        "validate_terminal_contract_authority(" in terminal_union_source
        and "validate_cpu_terminal_manifest(" not in terminal_union_source
        and "TERMINAL_CONTRACT" in terminal_union_source
    )
    domain_specific_interface_audit = bool(
        'domain_leaf_contract = 106 if domain == "d03" else None' in model_lower_source
        and "input_tree == output_tree and input_avals == output_avals" in model_lower_source
        and "domain_leaf_contract_passed" in model_lower_source
    )
    direct_full_terminal_flow = bool(
        "if args.direct_terminal:" in runtime_main_source
        and "terminal_result = continuation(result.carries)" in runtime_main_source
        and "build_full_terminal_verdict(" in runtime_main_source
        and '"manager_pause_at_9405": False' in runtime_main_source
        and "validate_gpu_completion_frames(" in full_terminal_source
        and "final_verdict(" in full_terminal_source
        and 'completion["counts"] != {"d01": 19, "d02": 19, "d03": 55}'
        in full_terminal_source
    )
    subprocess_calls = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    gpu_query_tokens = [
        token for token in ("nvidia-smi", "rocm-smi")
        if any(token in call for call in subprocess_calls)
    ]
    model_diff = _git(REPO_ROOT, "diff", "--name-only", CANDIDATE_COMMIT, "--", "src/gpuwrf")
    runtime_hook_count = sum(
        1 for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_import_runtime"
    )
    cpu_dry_source = inspect.getsource(_cpu_dry_run)
    cpu_dry_import_free = "_import_runtime" not in cpu_dry_source
    passed = bool(
        not forbidden_top
        and ordered
        and not forbidden_dispatch_tokens
        and not gpu_query_tokens
        and not model_diff
        and runtime_hook_count == 1
        and cpu_dry_import_free
        and output_ordered
        and compile_counts_exact
        and host_file_pairing_only
        and early_causal_fail_closed
        and known_v10_record_fail_closed
        and tooling_critic_preimport_gate
        and late_window_retention_fail_closed
        and continuation_first_red_retained
        and canonical_fresh_output_path
        and terminal_v10_red_honest
        and hlo_artifact_precedes_gate
        and diagnostic_never_compiles
        and lazy_health_binds_live_shape_once
        and parent_prejoin_isolation
        and parent_program_scope
        and parent_mixed_integral_projection
        and parent_real_schema_rehearsal
        and parent_blocker_retains_stage_traceback
        and parent_health_after_dispatch
        and target_record_fail_closed
        and terminal_union_is_snapshot_only
        and terminal_cpu_uses_canonical_contract
        and domain_specific_interface_audit
        and direct_full_terminal_flow
        and "_emit_initial_history_frames" in inspect.getsource(_runtime_main)
        and "output=output" in inspect.getsource(_run_authorized_continuation)
    )
    return {
        "passed": passed,
        "runner_source_sha256": sha256_text(source),
        "top_level_imports": top_imports,
        "forbidden_top_level_imports": forbidden_top,
        "preempt_dispatch_health_ordered": ordered,
        "normal_writer_checkpoint_health_pair_ordered": output_ordered,
        "explicit_compile_counts": explicit_compile_counts,
        "explicit_compile_counts_exact": compile_counts_exact,
        "incremental_pairing_uses_host_files_only": host_file_pairing_only,
        "early_00_20_causal_gate_fail_closed": early_causal_fail_closed,
        "known_1500_v10_record_fail_closed": known_v10_record_fail_closed,
        "tooling_critic_acceptance_preimport_gate": tooling_critic_preimport_gate,
        "late_window_9313_9314_9405_retention_fail_closed": (
            late_window_retention_fail_closed
        ),
        "continuation_first_new_red_retained": continuation_first_red_retained,
        "canonical_fresh_gpu_output_path": canonical_fresh_output_path,
        "terminal_v10_red_semantics_honest": terminal_v10_red_honest,
        "lowered_hlo_artifact_precedes_gate": hlo_artifact_precedes_gate,
        "lower_only_diagnostic_never_compiles": diagnostic_never_compiles,
        "lazy_health_binds_first_live_shape_exactly_once": lazy_health_binds_live_shape_once,
        "parent_prejoin_zero_d03_dispatch": parent_prejoin_isolation,
        "parent_one_compile_per_domain": parent_program_scope,
        "parent_mixed_integral_scheduler_projection": parent_mixed_integral_projection,
        "parent_real_schema_identity_rehearsal": parent_real_schema_rehearsal,
        "parent_blocker_retains_stage_and_traceback": parent_blocker_retains_stage_traceback,
        "parent_continuation_health_after_dispatch": parent_health_after_dispatch,
        "parent_join_target_record_fail_closed": target_record_fail_closed,
        "terminal_union_uses_immutable_snapshots": terminal_union_is_snapshot_only,
        "terminal_cpu_uses_marker_free_canonical_contract": terminal_cpu_uses_canonical_contract,
        "domain_specific_exact_interface_audit": domain_specific_interface_audit,
        "direct_full_terminal_19_19_55_flow": direct_full_terminal_flow,
        "explicit_lead_zero_normal_history": True,
        "continuation_retains_normal_output_callback": True,
        "dispatch_forbidden_tokens": forbidden_dispatch_tokens,
        "accepted_model_diff": model_diff.splitlines() if model_diff else [],
        "runtime_import_is_single_explicit_hook": runtime_hook_count == 1,
        "cpu_dry_run_calls_runtime_import": not cpu_dry_import_free,
        "subprocess_calls": subprocess_calls,
        "gpu_query_command_tokens": gpu_query_tokens,
    }


def _import_runtime() -> SimpleNamespace:
    """The only JAX/gpuwrf import hook; called after full admission."""

    import jax
    import jax.numpy as jnp
    import numpy as np
    from netCDF4 import Dataset

    from gpuwrf.integration.nested_pipeline import (
        _PerDomainWrfoutWriter,
        _emit_initial_history_frames,
        _nested_sync_mode_from_env,
    )
    from gpuwrf.profiling.transfer_audit import block_until_ready
    from gpuwrf.contracts import state as state_contract
    from gpuwrf.contracts.grid import DomainHierarchy
    from gpuwrf.runtime.domain_tree import (
        _operational_advance_factory,
        _operational_force,
        run_domain_tree_callbacks,
        run_operational_domain_tree,
    )
    from gpuwrf.runtime.finite_state_guard import assert_state_finite_at_boundary
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori, build_clock_base
    from gpuwrf.runtime import compile_cache as compile_cache_runtime
    from scripts import v0234_corrected_fullbuffer_gate as comparator
    from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

    return SimpleNamespace(**locals())


def _assert_cuda_runtime(runtime: SimpleNamespace) -> dict[str, Any]:
    devices = runtime.jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RunnerGateError("CUDA_RUNTIME", repr(devices))
    return {
        "jax_version": runtime.jax.__version__,
        "jaxlib_version": getattr(runtime.jax.lib, "__version__", "unknown"),
        "backend": runtime.jax.default_backend(),
        "devices": [str(device) for device in devices],
    }


def assert_runtime_cache_disabled(runtime: SimpleNamespace) -> dict[str, Any]:
    status = dict(runtime.compile_cache_runtime.CACHE_STATUS)
    jax_enabled = bool(runtime.jax.config.jax_enable_compilation_cache)
    if (
        status.get("source") != "disabled-by-GPUWRF_JAX_CACHE"
        or status.get("enabled") is not False
        or status.get("dir") is not None
        or jax_enabled
    ):
        raise RunnerGateError(
            "RUNTIME_PERSISTENT_CACHE",
            f"compile_cache={status!r} jax_enable_compilation_cache={jax_enabled!r}",
        )
    return {
        "persistent": False,
        "gpuwrf_compile_cache": status,
        "jax_enable_compilation_cache": jax_enabled,
    }


CUSTOM_CALL_PATTERN = re.compile(
    r"\bstablehlo\.custom_call\s+@(?:\"([^\"\n]+)\"|([A-Za-z_.$][A-Za-z0-9_.$-]*))"
)


def extract_stablehlo_custom_calls(stablehlo: str) -> dict[str, Any]:
    syntax_offsets = [
        match.start() for match in re.finditer(r"\bstablehlo\.custom_call\b", stablehlo)
    ]
    occurrences = []
    for index, match in enumerate(CUSTOM_CALL_PATTERN.finditer(stablehlo)):
        target = match.group(1) or match.group(2)
        snippet_start = max(0, match.start() - 256)
        snippet_end = min(len(stablehlo), match.end() + 1024)
        snippet = stablehlo[snippet_start:snippet_end]
        occurrences.append({
            "index": index,
            "target": target,
            "byte_offset": match.start(),
            "line": stablehlo.count("\n", 0, match.start()) + 1,
            "snippet_start": snippet_start,
            "snippet_end": snippet_end,
            "snippet": snippet,
            "snippet_sha256": sha256_text(snippet),
        })
    parsed_offsets = [row["byte_offset"] for row in occurrences]
    extraction_complete = len(syntax_offsets) == len(parsed_offsets)
    return {
        "stablehlo_sha256": sha256_text(stablehlo),
        "stablehlo_bytes": len(stablehlo.encode()),
        "custom_call_syntax_occurrence_count": len(syntax_offsets),
        "parsed_custom_call_count": len(occurrences),
        "extraction_complete": extraction_complete,
        "syntax_offsets": syntax_offsets,
        "parsed_offsets": parsed_offsets,
        "targets": sorted({row["target"] for row in occurrences}),
        "occurrences": occurrences,
    }


def evaluate_stablehlo_policy(
    stablehlo: str,
    *,
    allowed_custom_targets: frozenset[str] = ALLOWED_DEVICE_CUSTOM_CALL_TARGETS,
) -> dict[str, Any]:
    extraction = extract_stablehlo_custom_calls(stablehlo)
    lowered = stablehlo.lower()
    forbidden_text = sorted({
        token for token in FORBIDDEN_HLO_TEXT_TOKENS if token in lowered
    })
    forbidden_targets = sorted({
        target for target in extraction["targets"]
        if any(fragment in target.lower() for fragment in FORBIDDEN_CUSTOM_TARGET_FRAGMENTS)
    })
    unknown_targets = sorted(
        set(extraction["targets"]) - set(allowed_custom_targets)
    )
    unparsed_custom_call_count = (
        extraction["custom_call_syntax_occurrence_count"]
        - extraction["parsed_custom_call_count"]
    )
    passed = bool(
        extraction["extraction_complete"]
        and unparsed_custom_call_count == 0
        and not forbidden_text
        and not forbidden_targets
        and not unknown_targets
    )
    return {
        "passed": passed,
        "allowed_device_custom_call_targets": sorted(allowed_custom_targets),
        "custom_call_targets": extraction["targets"],
        "forbidden_text_tokens": forbidden_text,
        "forbidden_custom_call_targets": forbidden_targets,
        "unknown_custom_call_targets": unknown_targets,
        "unparsed_custom_call_count": unparsed_custom_call_count,
        "extraction_complete": extraction["extraction_complete"],
    }


def atomically_retain_lowered_hlo_audit(path: Path, stablehlo: str) -> dict[str, Any]:
    artifact = {
        "schema": "gpuwrf.v0234.nested-boundary-final-lowered-hlo-custom-calls.v1",
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_tree": CANDIDATE_TREE,
        "compile_calls_before_artifact": 0,
        "dispatch_calls_before_artifact": 0,
        "extraction": extract_stablehlo_custom_calls(stablehlo),
    }
    artifact["proof_sha256"] = canonical_digest(artifact)
    atomic_write_json(path, artifact)
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "proof_sha256": artifact["proof_sha256"],
        "extraction": artifact["extraction"],
    }


def health_carry_signature(carry: Any, runtime: SimpleNamespace) -> dict[str, Any]:
    """Shape/dtype-only signature; reading it performs no device transfer."""

    leaves = runtime.jax.tree_util.tree_leaves(carry)
    return {
        "treedef": str(runtime.jax.tree_util.tree_structure(carry)),
        "leaf_count": len(leaves),
        "leaves": [
            {"index": index, "shape": list(leaf.shape), "dtype": str(leaf.dtype)}
            for index, leaf in enumerate(leaves)
        ],
    }


def _build_health_executable(live_carry: Any, runtime: SimpleNamespace) -> tuple[Any, dict[str, Any]]:
    jax, jnp = runtime.jax, runtime.jnp

    def reduction(carry: Any) -> dict[str, Any]:
        def finite_maxabs(value: Any) -> Any:
            array = jnp.asarray(value)
            return jnp.max(jnp.where(jnp.isfinite(array), jnp.abs(array), 0.0))

        def finite_max(value: Any) -> Any:
            array = jnp.asarray(value)
            return jnp.max(jnp.where(jnp.isfinite(array), array, 0.0))

        def finite_min_or_zero(value: Any) -> Any:
            array = jnp.asarray(value)
            return jnp.min(jnp.where(jnp.isfinite(array), array, 0.0))

        leaves = jax.tree_util.tree_leaves(carry)
        nonfinite = []
        for leaf in leaves:
            if jnp.issubdtype(leaf.dtype, jnp.inexact):
                nonfinite.append(jnp.count_nonzero(~jnp.isfinite(leaf)))
            else:
                nonfinite.append(jnp.asarray(0, dtype=jnp.int32))
        named = [getattr(carry.state, name) for name in STATE_HEALTH_FIELDS]
        named += [getattr(carry, name) for name in SAVE_HEALTH_FIELDS]
        named += [getattr(carry, name) for name in SCRATCH_HEALTH_FIELDS]
        scale = jnp.stack([finite_maxabs(value) for value in named]).astype(jnp.float64)
        mu_pert = carry.state.mu_perturbation
        corners = jnp.stack((mu_pert[0, 0], mu_pert[0, -1], mu_pert[-1, 0], mu_pert[-1, -1]))
        return {
            "nonfinite_by_leaf": jnp.stack(nonfinite).astype(jnp.int64),
            "scale_maxabs": scale,
            "positive_min": jnp.stack((
                finite_min_or_zero(carry.state.p_total),
                finite_min_or_zero(carry.state.mu_total),
            )).astype(jnp.float64),
            "wind_maxabs": jnp.stack((
                finite_maxabs(carry.state.u),
                finite_maxabs(carry.state.v),
                finite_maxabs(carry.state.w),
            )).astype(jnp.float64),
            "corner_mu_pert_maxabs": finite_maxabs(corners).astype(jnp.float64),
            "theta_max": finite_max(carry.state.theta).astype(jnp.float64),
            "ni_nonfinite_count": jnp.count_nonzero(~jnp.isfinite(carry.state.Ni)).astype(jnp.int64),
        }

    lower_started = time.perf_counter()
    compiled_signature = health_carry_signature(live_carry, runtime)
    if compiled_signature["leaf_count"] != 106:
        raise RunnerGateError(
            "HEALTH_LIVE_LEAF_COUNT", str(compiled_signature["leaf_count"]),
        )
    lowered = jax.jit(reduction).lower(live_carry)
    lower_seconds = time.perf_counter() - lower_started
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    hlo_policy = evaluate_stablehlo_policy(stablehlo)
    if not hlo_policy["passed"]:
        raise RunnerGateError("HEALTH_HLO", json.dumps(hlo_policy, sort_keys=True))
    compile_started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started
    return executable, {
        "separate_from_model_hlo": True,
        "lower_calls": 1,
        "compile_calls": 1,
        "lower_wall_seconds": lower_seconds,
        "compile_wall_seconds": compile_seconds,
        "stablehlo_sha256": sha256_text(stablehlo),
        "stablehlo_bytes": len(stablehlo.encode()),
        "hlo_policy": hlo_policy,
        "forbidden_tokens_found": [],
        "lazy_live_carry_compile": True,
        "compiled_input_signature": compiled_signature,
        "boundary_target_transition_fields": list(BOUNDARY_TARGET_TRANSITION_FIELDS),
        "boundary_target_transition_leaf_indices": list(BOUNDARY_TARGET_TRANSITION_LEAF_INDICES),
        "bounded_outputs": {
            "nonfinite_by_leaf": 106,
            "scale_maxabs": len(ALL_SCALE_FIELDS),
            "other_scalars": 9,
        },
    }


class LazyLiveCarryHealthExecutable:
    """Compile health once on the first operational carry, then fail on drift."""

    def __init__(self, runtime: SimpleNamespace, builder: Callable[..., Any] | None = None) -> None:
        self.runtime = runtime
        self.builder = builder
        self.executable: Any | None = None
        self.compiled_signature: dict[str, Any] | None = None
        self.compile_audit: dict[str, Any] | None = None
        self.compile_calls = 0

    def executable_for(self, carry: Any) -> Any:
        signature = health_carry_signature(carry, self.runtime)
        if signature["leaf_count"] != 106:
            raise RunnerGateError("HEALTH_LIVE_LEAF_COUNT", str(signature["leaf_count"]))
        if self.executable is None:
            builder = self.builder or _build_health_executable
            executable, audit = builder(carry, self.runtime)
            if audit.get("lower_calls") != 1 or audit.get("compile_calls") != 1:
                raise RunnerGateError("HEALTH_COMPILE_AUDIT", repr(audit))
            self.executable = executable
            self.compiled_signature = signature
            self.compile_audit = {
                **audit,
                "lazy_compile_trigger": "first-authenticated-live-or-post-dispatch-carry",
                "initial_placeholder_carry_was_not_compiled": True,
                "signature_recompile_allowed": False,
            }
            self.compile_calls = 1
            print(
                "NESTED_BOUNDARY_HEALTH_COMPILE_COMPLETE "
                f"seconds={float(audit.get('compile_wall_seconds', 0.0)):.6f} "
                f"leaf_count={signature['leaf_count']}",
                flush=True,
            )
        elif signature != self.compiled_signature:
            raise RunnerGateError(
                "HEALTH_LIVE_SIGNATURE_DRIFT",
                json.dumps({
                    "compiled": self.compiled_signature,
                    "called": signature,
                    "compile_calls": self.compile_calls,
                }, sort_keys=True),
            )
        if self.compile_calls != 1 or self.executable is None:
            raise RunnerGateError("HEALTH_COMPILE_COUNT", str(self.compile_calls))
        return self.executable

    def audit(self) -> dict[str, Any]:
        if self.compile_audit is None or self.compile_calls != 1:
            raise RunnerGateError("HEALTH_NOT_COMPILED_ON_LIVE_CARRY", str(self.compile_calls))
        return dict(self.compile_audit)


def materialize_health(executable: Any, carry: Any, runtime: SimpleNamespace) -> dict[str, Any]:
    compiled = (
        executable.executable_for(carry)
        if isinstance(executable, LazyLiveCarryHealthExecutable)
        else executable
    )
    host = runtime.jax.device_get(compiled(carry))
    np = runtime.np
    return {
        "nonfinite_by_leaf": np.asarray(host["nonfinite_by_leaf"], dtype=np.int64).tolist(),
        "scale_maxabs": np.asarray(host["scale_maxabs"], dtype=np.float64).tolist(),
        "positive_min": np.asarray(host["positive_min"], dtype=np.float64).tolist(),
        "wind_maxabs": np.asarray(host["wind_maxabs"], dtype=np.float64).tolist(),
        "corner_mu_pert_maxabs": float(np.asarray(host["corner_mu_pert_maxabs"])),
        "theta_max": float(np.asarray(host["theta_max"])),
        "ni_nonfinite_count": int(np.asarray(host["ni_nonfinite_count"])),
    }


def carry_interface_signature(carry: Any, runtime: SimpleNamespace) -> dict[str, Any]:
    leaves = runtime.jax.tree_util.tree_leaves(carry)
    return {
        "treedef": str(runtime.jax.tree_util.tree_structure(carry)),
        "leaf_count": len(leaves),
        "leaves": [
            {"shape": list(leaf.shape), "dtype": str(leaf.dtype)}
            for leaf in leaves
        ],
    }


def _lower_domain_once(
    tree: Any,
    domain: str,
    carry: Any,
    runtime: SimpleNamespace,
    *,
    artifact_path: Path,
    start_step: int,
) -> tuple[Any, Any, Any, int, dict[str, Any]]:
    jax, jnp = runtime.jax, runtime.jnp
    namelist = tree.domains[domain].namelist
    clock = runtime.build_clock_base(namelist)
    cadence = int(namelist.radiation_cadence_steps)
    lower_started = time.perf_counter()
    lowered = runtime._advance_chunk_fori.lower(
        carry,
        namelist,
        jnp.asarray(int(start_step), dtype=jnp.int32),
        clock,
        n_steps=1,
        cadence=cadence,
    )
    lower_seconds = time.perf_counter() - lower_started
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    retained_hlo = atomically_retain_lowered_hlo_audit(artifact_path, stablehlo)
    hlo_policy = evaluate_stablehlo_policy(stablehlo)
    input_tree = jax.tree_util.tree_structure(carry)
    output_tree = jax.tree_util.tree_structure(lowered.out_info)
    avals = lambda value: [
        {"shape": list(leaf.shape), "dtype": str(leaf.dtype)}
        for leaf in jax.tree_util.tree_leaves(value)
    ]
    input_avals = avals(carry)
    output_avals = avals(lowered.out_info)
    interface_equal = bool(input_tree == output_tree and input_avals == output_avals)
    domain_leaf_contract = 106 if domain == "d03" else None
    domain_leaf_contract_passed = bool(
        domain_leaf_contract is None
        or len(input_avals) == len(output_avals) == domain_leaf_contract
    )
    return lowered, namelist, clock, cadence, {
        "callable": "gpuwrf.runtime.operational_mode._advance_chunk_fori",
        "domain": domain,
        "first_start_step": int(start_step),
        "lower_calls": 1,
        "compile_calls": 0,
        "lower_wall_seconds": lower_seconds,
        "n_steps": 1,
        "input_leaf_count": len(input_avals),
        "output_leaf_count": len(output_avals),
        "input_output_treedef_and_avals_identical": interface_equal,
        "domain_leaf_contract": domain_leaf_contract,
        "domain_leaf_contract_passed": domain_leaf_contract_passed,
        "stablehlo_sha256": retained_hlo["extraction"]["stablehlo_sha256"],
        "stablehlo_bytes": retained_hlo["extraction"]["stablehlo_bytes"],
        "lowered_hlo_artifact": retained_hlo,
        "hlo_policy": hlo_policy,
        "model_hlo_observer_free": hlo_policy["passed"],
        "host_health_or_output_values_feed_model": False,
    }


def _lower_compile_domain_once(
    tree: Any,
    domain: str,
    carry: Any,
    runtime: SimpleNamespace,
    *,
    artifact_path: Path,
    start_step: int,
) -> tuple[Any, Any, Any, int, dict[str, Any]]:
    lowered, namelist, clock, cadence, audit = _lower_domain_once(
        tree, domain, carry, runtime,
        artifact_path=artifact_path, start_step=start_step,
    )
    if (
        not audit["hlo_policy"]["passed"]
        or not audit["input_output_treedef_and_avals_identical"]
        or not audit["domain_leaf_contract_passed"]
    ):
        raise RunnerGateError(
            "ORDINARY_ONE_STEP_AUDIT",
            json.dumps({
                "hlo_policy": audit["hlo_policy"],
                "interface_equal": audit["input_output_treedef_and_avals_identical"],
                "domain_leaf_contract": audit["domain_leaf_contract"],
                "domain_leaf_contract_passed": audit["domain_leaf_contract_passed"],
                "lowered_hlo_artifact": audit["lowered_hlo_artifact"],
            }, sort_keys=True),
        )
    compile_started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started
    output_tree = runtime.jax.tree_util.tree_structure(lowered.out_info)
    if executable.out_tree != output_tree:
        raise RunnerGateError("ORDINARY_COMPILED_TREE", "output tree changed after compile")
    return executable, namelist, clock, cadence, {
        **audit,
        "compile_calls": 1,
        "compile_wall_seconds": compile_seconds,
    }


def _lower_d03_once(
    tree: Any,
    prefix_carry: Any,
    runtime: SimpleNamespace,
    *,
    artifact_path: Path,
) -> tuple[Any, Any, Any, int, dict[str, Any]]:
    return _lower_domain_once(
        tree, "d03", prefix_carry, runtime,
        artifact_path=artifact_path, start_step=9199,
    )


def _lower_compile_d03_once(
    tree: Any,
    prefix_carry: Any,
    runtime: SimpleNamespace,
    *,
    artifact_path: Path,
) -> tuple[Any, Any, Any, int, dict[str, Any]]:
    return _lower_compile_domain_once(
        tree, "d03", prefix_carry, runtime,
        artifact_path=artifact_path, start_step=9199,
    )


class LazyOneStepDomainExecutable:
    """Compile one ordinary one-step program on its first operational input."""

    def __init__(
        self,
        tree: Any,
        domain: str,
        runtime: SimpleNamespace,
        artifact_path: Path,
        *,
        builder: Callable[..., tuple[Any, Any, Any, int, dict[str, Any]]] = _lower_compile_domain_once,
    ) -> None:
        self.tree = tree
        self.domain = domain
        self.runtime = runtime
        self.artifact_path = artifact_path
        self.builder = builder
        self.executable: Any | None = None
        self.namelist: Any | None = None
        self.clock: Any | None = None
        self.cadence: int | None = None
        self.signature: dict[str, Any] | None = None
        self.compile_audit: dict[str, Any] | None = None
        self.compile_calls = 0
        self.dispatch_calls = 0

    def _bind(self, carry: Any, start_step: int) -> None:
        signature = carry_interface_signature(carry, self.runtime)
        if signature["leaf_count"] != 106:
            raise RunnerGateError(
                "ORDINARY_LIVE_LEAF_COUNT",
                f"domain={self.domain} leaves={signature['leaf_count']}",
            )
        if self.executable is None:
            executable, namelist, clock, cadence, audit = self.builder(
                self.tree,
                self.domain,
                carry,
                self.runtime,
                artifact_path=self.artifact_path,
                start_step=int(start_step),
            )
            if audit.get("lower_calls") != 1 or audit.get("compile_calls") != 1:
                raise RunnerGateError("ORDINARY_COMPILE_AUDIT", repr(audit))
            self.executable = executable
            self.namelist = namelist
            self.clock = clock
            self.cadence = int(cadence)
            self.signature = signature
            self.compile_audit = {
                **audit,
                "lazy_compile_trigger": "first-operational-domain-input",
                "signature_recompile_allowed": False,
            }
            self.compile_calls = 1
            print(
                "NESTED_BOUNDARY_DOMAIN_COMPILE_COMPLETE "
                f"domain={self.domain} first_step={int(start_step)} "
                f"seconds={float(audit.get('compile_wall_seconds', 0.0)):.6f}",
                flush=True,
            )
        elif signature != self.signature:
            raise RunnerGateError(
                "ORDINARY_LIVE_SIGNATURE_DRIFT",
                json.dumps({
                    "domain": self.domain,
                    "compiled": self.signature,
                    "called": signature,
                    "compile_calls": self.compile_calls,
                }, sort_keys=True),
            )
        if self.compile_calls != 1 or self.executable is None:
            raise RunnerGateError(
                "ORDINARY_COMPILE_COUNT", f"domain={self.domain} count={self.compile_calls}",
            )

    def one_step(self, carry: Any, start_step: int) -> Any:
        self._bind(carry, int(start_step))
        self.dispatch_calls += 1
        return self.executable(
            carry,
            self.namelist,
            self.runtime.jnp.asarray(int(start_step), dtype=self.runtime.jnp.int32),
            self.clock,
            n_steps=1,
            cadence=self.cadence,
        )

    def advance(self, carry: Any, start_step: int, n_steps: int) -> Any:
        value = carry
        for offset in range(int(n_steps)):
            value = self.one_step(value, int(start_step) + offset)
        return value

    def audit(self) -> dict[str, Any]:
        if self.compile_audit is None or self.compile_calls != 1:
            raise RunnerGateError(
                "ORDINARY_NOT_COMPILED", f"domain={self.domain} count={self.compile_calls}",
            )
        return {
            **self.compile_audit,
            "compile_calls": self.compile_calls,
            "dispatch_calls": self.dispatch_calls,
        }


def load_cpu_frame_index() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    manifest, authority = read_authenticated_json(
        CPU_MANIFEST, CPU_MANIFEST_SHA256, "CPU_MANIFEST_HASH",
    )
    if (
        manifest.get("raw_frame_counts") != {"d01": 19, "d02": 19, "d03": 55}
        or manifest.get("raw_frame_schedules") != standard_frame_schedule(TERMINAL_OWN_STEPS)
    ):
        raise RunnerGateError("CPU_FRAME_MANIFEST", "counts or schedules changed")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    raw = manifest.get("raw_artifacts") or {}
    for name in ("d01", "d02", "d03"):
        rows = raw.get(name)
        if not isinstance(rows, list) or len(rows) != manifest["raw_frame_counts"][name]:
            raise RunnerGateError("CPU_FRAME_INVENTORY", name)
        for row in rows:
            valid = str(row.get("valid_time"))
            path = Path(str(row.get("path", "")))
            expected_name = f"wrfout_{name}_{valid}"
            if path.parent != INPUT_DIR or path.name != expected_name or len(str(row.get("sha256", ""))) != 64:
                raise RunnerGateError("CPU_FRAME_ROW", repr(row))
            key = (name, valid)
            if key in index:
                raise RunnerGateError("CPU_FRAME_DUPLICATE", repr(key))
            index[key] = dict(row)
    return index, authority


def stable_file_authority(path: Path, code: str) -> dict[str, Any]:
    """Hash one regular non-symlink source through a stable single descriptor."""

    if path.is_symlink():
        raise RunnerGateError(code, f"symlink rejected: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RunnerGateError(code, f"not a regular file: {path}")
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(handle.fileno())
    current = path.stat()
    stable = bool(
        before.st_dev == after.st_dev == current.st_dev
        and before.st_ino == after.st_ino == current.st_ino
        and before.st_size == after.st_size == current.st_size
        and before.st_mtime_ns == after.st_mtime_ns == current.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns == current.st_ctime_ns
    )
    if not stable:
        raise RunnerGateError(code, f"source changed while hashing: {path}")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": digest.hexdigest(),
        "bytes": before.st_size,
        "device": before.st_dev,
        "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "mode": oct(before.st_mode & 0o777),
    }


def verify_stable_file_authority(authority: Mapping[str, Any], code: str) -> None:
    current = stable_file_authority(Path(str(authority["path"])), code)
    if current != dict(authority):
        raise RunnerGateError(code, json.dumps({
            "expected": dict(authority), "actual": current,
        }, sort_keys=True))


def _exact_array_descriptor(value: Any, runtime: SimpleNamespace) -> dict[str, Any]:
    np = runtime.np
    array = np.asanyarray(value)
    if np.ma.isMaskedArray(array):
        mask = np.ascontiguousarray(np.ma.getmaskarray(array))
        data = np.ascontiguousarray(np.ma.getdata(array))
        return {
            "kind": "masked-array",
            "dtype": str(data.dtype),
            "shape": list(data.shape),
            "data_sha256": hashlib.sha256(data.tobytes(order="C")).hexdigest(),
            "mask_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
        }
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.kind == "O":
        values = contiguous.tolist()
        return {
            "kind": "object-array",
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "values": values,
            "values_sha256": canonical_digest(values),
        }
    raw = contiguous.tobytes(order="C")
    return {
        "kind": "array",
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _exact_attribute_descriptor(value: Any, runtime: SimpleNamespace) -> dict[str, Any]:
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}
    return _exact_array_descriptor(value, runtime)


def netcdf_semantic_bit_manifest(path: Path, runtime: SimpleNamespace) -> dict[str, Any]:
    """Hash every NetCDF value and ordered metadata field, independent of HDF5 layout."""

    source = stable_file_authority(path, "NETCDF_SOURCE_UNSTABLE")
    with runtime.Dataset(path, "r") as dataset:
        dataset.set_auto_maskandscale(False)
        dimensions = [
            {
                "name": name,
                "size": len(dimension),
                "unlimited": bool(dimension.isunlimited()),
            }
            for name, dimension in dataset.dimensions.items()
        ]
        global_attributes = [
            {
                "name": name,
                "value": _exact_attribute_descriptor(dataset.getncattr(name), runtime),
            }
            for name in dataset.ncattrs()
        ]
        variables = []
        for name, variable in dataset.variables.items():
            try:
                filters = variable.filters()
            except (AttributeError, RuntimeError):
                filters = None
            try:
                chunking = variable.chunking()
            except (AttributeError, RuntimeError):
                chunking = None
            try:
                endian = variable.endian()
            except (AttributeError, RuntimeError):
                endian = None
            variables.append({
                "name": name,
                "dtype": str(variable.dtype),
                "dimensions": list(variable.dimensions),
                "shape": list(variable.shape),
                "attributes": [
                    {
                        "name": attr,
                        "value": _exact_attribute_descriptor(variable.getncattr(attr), runtime),
                    }
                    for attr in variable.ncattrs()
                ],
                "chunking": chunking,
                "filters": filters,
                "endian": endian,
                "data": _exact_array_descriptor(variable[...], runtime),
            })
        semantic = {
            "data_model": dataset.data_model,
            "file_format": dataset.file_format,
            "disk_format": getattr(dataset, "disk_format", None),
            "dimensions": dimensions,
            "global_attributes": global_attributes,
            "variables": variables,
        }
    verify_stable_file_authority(source, "NETCDF_SOURCE_CHANGED_AFTER_READ")
    return {
        "source": source,
        "semantic": semantic,
        "semantic_sha256": canonical_digest(semantic),
    }


def compare_netcdf_semantic_bits(
    reference: Path,
    candidate: Path,
    runtime: SimpleNamespace,
) -> dict[str, Any]:
    left = netcdf_semantic_bit_manifest(reference, runtime)
    right = netcdf_semantic_bit_manifest(candidate, runtime)
    differences: list[dict[str, Any]] = []
    left_semantic = left["semantic"]
    right_semantic = right["semantic"]
    for key in ("data_model", "file_format", "disk_format", "dimensions", "global_attributes"):
        if left_semantic[key] != right_semantic[key]:
            differences.append({
                "category": "metadata", "path": key,
                "reference": left_semantic[key], "candidate": right_semantic[key],
            })
    left_variables = {row["name"]: row for row in left_semantic["variables"]}
    right_variables = {row["name"]: row for row in right_semantic["variables"]}
    for name in sorted(set(left_variables) | set(right_variables)):
        before = left_variables.get(name)
        after = right_variables.get(name)
        if before is None or after is None:
            differences.append({
                "category": "metadata", "path": f"variables.{name}",
                "reference_present": before is not None,
                "candidate_present": after is not None,
            })
            continue
        before_metadata = {key: value for key, value in before.items() if key != "data"}
        after_metadata = {key: value for key, value in after.items() if key != "data"}
        if before_metadata != after_metadata:
            differences.append({
                "category": "metadata", "path": f"variables.{name}.metadata",
                "reference": before_metadata, "candidate": after_metadata,
            })
        if before["data"] != after["data"]:
            differences.append({
                "category": "data", "path": f"variables.{name}.data",
                "reference": before["data"], "candidate": after["data"],
            })
    passed = bool(
        left["semantic_sha256"] == right["semantic_sha256"] and not differences
    )
    return {
        "passed": passed,
        "reference": left["source"],
        "candidate": right["source"],
        "reference_semantic_sha256": left["semantic_sha256"],
        "candidate_semantic_sha256": right["semantic_sha256"],
        "difference_count": len(differences),
        "first_differences": differences[:20],
        "all_variable_values_bit_exact": passed,
        "all_dimensions_attributes_and_variable_metadata_exact": passed,
    }


class ParentPrefixBitwiseOutput:
    """Write reconstructed parent frames and fail closed against retained GPU frames."""

    wants_carry = True

    def __init__(self, writer: Any, output_dir: Path, runtime: SimpleNamespace) -> None:
        self.writer = writer
        self.output_dir = output_dir
        self.runtime = runtime
        self.rows: list[dict[str, Any]] = []
        self.counts = {name: 0 for name in PARENT_DOMAINS}
        self.calls: set[tuple[str, int]] = set()

    def __call__(self, name: str, step: int, carry: Any) -> dict[str, Any]:
        if name not in PARENT_DOMAINS:
            raise MetricGateFailure("PARENT_OUTPUT_DOMAIN", name)
        key = (name, int(step))
        if key in self.calls:
            raise MetricGateFailure("PARENT_OUTPUT_DUPLICATE", repr(key))
        self.calls.add(key)
        valid = RUN_START + timedelta(seconds=int(step) * DT_SECONDS[name])
        stamp = valid.strftime("%Y-%m-%d_%H:%M:%S")
        writer_result = self.writer(name, int(step), carry)
        candidate = self.output_dir / f"wrfout_{name}_{stamp}"
        reference = RC3_RUN_DIR / "output" / f"wrfout_{name}_{stamp}"
        if not candidate.is_file() or not reference.is_file():
            raise MetricGateFailure(
                "PARENT_PREFIX_FRAME_MISSING", f"candidate={candidate} reference={reference}",
            )
        comparison = compare_netcdf_semantic_bits(reference, candidate, self.runtime)
        row = {
            "domain": name,
            "own_step": int(step),
            "valid_time": valid.isoformat(),
            "writer_result": writer_result,
            "comparison": comparison,
        }
        self.rows.append(row)
        self.counts[name] += 1
        if not comparison["passed"]:
            raise MetricGateFailure(
                "PARENT_PREFIX_BIT_IDENTITY", json.dumps(row, sort_keys=True),
            )
        return row


def authenticate_early_causal_cpu_path(path: Path) -> dict[str, Any]:
    """Bind the selected manifest frame by bytes, not by duplicate pathname."""

    return _require_sha(
        path, EARLY_CAUSAL_CPU_SHA256, "EARLY_CAUSAL_SELECTED_CPU",
    )


def early_causal_ring1_metrics(
    np: Any,
    *,
    current: Mapping[str, Any],
    prior: Mapping[str, Any],
    partial: Mapping[str, Any],
    scalar: Mapping[str, Any],
    retry20: Mapping[str, Any],
    cpu: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail-closed 00:20 discriminator for the source-predicted ring-1 repair."""

    rows: dict[str, Any] = {}
    for field in ("T", "U"):
        arrays = {
            role: np.asarray(values[field], dtype=np.float64)
            for role, values in (
                ("current", current),
                ("prior", prior),
                ("partial", partial),
                ("scalar", scalar),
                ("retry20", retry20),
                ("cpu", cpu),
            )
        }
        shape = arrays["current"].shape
        if any(value.shape != shape for value in arrays.values()):
            raise MetricGateFailure(
                "EARLY_CAUSAL_SHAPE", f"{field}:{ {k: v.shape for k, v in arrays.items()} }",
            )
        ny, nx = shape[-2:]
        y, x = np.ogrid[:ny, :nx]
        distance = np.minimum.reduce((
            np.broadcast_to(y, (ny, nx)),
            np.broadcast_to(x, (ny, nx)),
            np.broadcast_to(ny - 1 - y, (ny, nx)),
            np.broadcast_to(nx - 1 - x, (ny, nx)),
        ))
        ring1 = distance == 1
        if len(shape) == 3:
            ring1 = np.broadcast_to(ring1, shape)

        def _rmse(left: Any, right: Any) -> float:
            delta = (left - right)[ring1]
            return float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))))

        observed = {
            "current_vs_cpu": _rmse(arrays["current"], arrays["cpu"]),
            "current_vs_retry20": _rmse(arrays["current"], arrays["retry20"]),
            "prior_vs_cpu": _rmse(arrays["prior"], arrays["cpu"]),
            "prior_vs_retry20": _rmse(arrays["prior"], arrays["retry20"]),
            "partial_vs_cpu": _rmse(arrays["partial"], arrays["cpu"]),
            "partial_vs_retry20": _rmse(arrays["partial"], arrays["retry20"]),
            "scalar_vs_cpu": _rmse(arrays["scalar"], arrays["cpu"]),
            "scalar_vs_retry20": _rmse(arrays["scalar"], arrays["retry20"]),
        }
        baseline = EARLY_CAUSAL_RING1_BASELINE[field]
        baseline_authenticated = all(
            observed[name] == float(expected) for name, expected in baseline.items()
        )
        gates = {
            "baseline_authenticated": baseline_authenticated,
            "closer_to_cpu_than_prior": (
                observed["current_vs_cpu"] < observed["prior_vs_cpu"]
            ),
            "closer_to_retry20_than_prior": (
                observed["current_vs_retry20"] < observed["prior_vs_retry20"]
            ),
        }
        if field == "T":
            gates.update({
                "closer_to_cpu_than_partial_wind_candidate": (
                    observed["current_vs_cpu"] < observed["partial_vs_cpu"]
                ),
                "closer_to_retry20_than_partial_wind_candidate": (
                    observed["current_vs_retry20"] < observed["partial_vs_retry20"]
                ),
                "closer_to_cpu_than_scalar_parent": (
                    observed["current_vs_cpu"] < observed["scalar_vs_cpu"]
                ),
                "closer_to_retry20_than_scalar_parent": (
                    observed["current_vs_retry20"] < observed["scalar_vs_retry20"]
                ),
            })
        else:
            gates.update({
                "partial_wind_candidate_improves_cpu_baseline": (
                    observed["partial_vs_cpu"] < observed["prior_vs_cpu"]
                ),
                "partial_wind_candidate_improves_retry20_baseline": (
                    observed["partial_vs_retry20"] < observed["prior_vs_retry20"]
                ),
                "retains_partial_wind_improvement_vs_cpu": (
                    observed["current_vs_cpu"] < observed["prior_vs_cpu"]
                ),
                "retains_partial_wind_improvement_vs_retry20": (
                    observed["current_vs_retry20"] < observed["prior_vs_retry20"]
                ),
            })
        rows[field] = {
            **observed,
            "frozen_baseline": dict(baseline),
            "gates": gates,
            "passed": all(gates.values()),
        }
    return {
        "schema": "gpuwrf.v0234.nested-t-source-cadence-early-causal-gate.v1",
        "domain": "d03",
        "own_step": 200,
        "valid_time": "2025-03-01T00:20:00+00:00",
        "ring": 1,
        "u_policy": "retain the source-backed 2c mechanism and remain better than immutable 4484 against both anchors",
        "fields": rows,
        "passed": all(row["passed"] for row in rows.values()),
        "v10_policy": "not part of this causal gate; remains a separate terminal metric",
    }


class IncrementalFramePairer:
    """Host-only finite/static-identity pairing after each normal wrfout closes."""

    def __init__(
        self,
        runtime: SimpleNamespace,
        pair_dir: Path,
        *,
        record_known_1500_v10_red: bool = False,
    ) -> None:
        self.runtime = runtime
        self.pair_dir = pair_dir
        self.record_known_1500_v10_red = bool(record_known_1500_v10_red)
        self.known_1500_v10_authority: dict[str, Any] | None = None
        self.known_1500_v10_authority_row: dict[str, Any] | None = None
        if self.record_known_1500_v10_red:
            (
                self.known_1500_v10_authority,
                self.known_1500_v10_authority_row,
            ) = authenticate_known_1500_v10_observation()
        self.cpu_index, self.cpu_manifest_authority = load_cpu_frame_index()
        self.retry20, self.retry20_authority = read_authenticated_json(
            RETRY20_PAIRS, RETRY20_PAIRS_SHA256, "RETRY20_PAIRS_HASH",
        )
        decisive_stamp = (RUN_START + timedelta(seconds=DECISIVE_D03_STEP * DT_SECONDS["d03"])).isoformat()
        decisive_rows = [row for row in self.retry20.get("pairs", []) if row.get("valid_time") == decisive_stamp]
        if (
            self.retry20.get("frozen_geometry_policy") != "terminal_nested_boundary_v1"
            or len(decisive_rows) != 1
        ):
            raise RunnerGateError("RETRY20_DECISIVE_AUTHORITY", decisive_stamp)
        self.decisive_retry20 = decisive_rows[0]
        baseline_metrics = self.decisive_retry20.get("metrics") or {}
        if any(float(baseline_metrics.get(field, {}).get("rmse", math.nan)) != limit for field, limit in FROZEN_1500_RMSE.items()):
            raise RunnerGateError("RETRY20_FROZEN_THRESHOLDS", decisive_stamp)
        self.early_authority = {
            "prior": _require_sha(
                EARLY_CAUSAL_PRIOR, EARLY_CAUSAL_PRIOR_SHA256,
                "EARLY_CAUSAL_PRIOR",
            ),
            "retry20": _require_sha(
                EARLY_CAUSAL_RETRY20, EARLY_CAUSAL_RETRY20_SHA256,
                "EARLY_CAUSAL_RETRY20",
            ),
            "partial_wind": _require_sha(
                EARLY_CAUSAL_PARTIAL_WIND, EARLY_CAUSAL_PARTIAL_WIND_SHA256,
                "EARLY_CAUSAL_PARTIAL_WIND",
            ),
            "scalar_parent": _require_sha(
                EARLY_CAUSAL_SCALAR_PARENT, EARLY_CAUSAL_SCALAR_PARENT_SHA256,
                "EARLY_CAUSAL_SCALAR_PARENT",
            ),
            "cpu": _require_sha(
                EARLY_CAUSAL_CPU, EARLY_CAUSAL_CPU_SHA256,
                "EARLY_CAUSAL_CPU",
            ),
        }
        self.rows: list[dict[str, Any]] = []
        self.counts = {"d01": 0, "d02": 0, "d03": 0}

    def _generic_identity_pair(
        self,
        name: str,
        step: int,
        valid: datetime,
        cpu_path: Path,
        candidate_path: Path,
    ) -> dict[str, Any]:
        np = self.runtime.np
        expected_stamp = valid.strftime("%Y-%m-%d_%H:%M:%S")
        with self.runtime.Dataset(cpu_path) as cpu, self.runtime.Dataset(candidate_path) as candidate:
            cpu_stamp = self.runtime.comparator._decoded_times(cpu)
            candidate_stamp = self.runtime.comparator._decoded_times(candidate)
            cpu_fields = set(cpu.variables)
            candidate_fields = set(candidate.variables)
            missing_required_cpu = sorted(set(STRICT_FIELDS + STATIC_FIELDS) - cpu_fields)
            missing_required_candidate = sorted(set(STRICT_FIELDS + STATIC_FIELDS) - candidate_fields)
            common_numeric: list[str] = []
            incompatible: list[str] = []
            nonfinite: list[dict[str, Any]] = []
            for field in sorted(cpu_fields & candidate_fields):
                if field == "Times":
                    continue
                cpu_var = cpu.variables[field]
                candidate_var = candidate.variables[field]
                if not (
                    np.issubdtype(cpu_var.dtype, np.number)
                    and np.issubdtype(candidate_var.dtype, np.number)
                ):
                    continue
                if cpu_var.shape != candidate_var.shape or cpu_var.dimensions != candidate_var.dimensions:
                    incompatible.append(field)
                    continue
                common_numeric.append(field)
                for side, dataset in (("cpu", cpu), ("candidate", candidate)):
                    values = self.runtime.comparator._variable_array(dataset, field)
                    count = int(np.count_nonzero(~np.isfinite(values)))
                    if count:
                        nonfinite.append({"side": side, "field": field, "count": count})
            static_identity = {}
            for field in STATIC_FIELDS:
                if field in cpu_fields and field in candidate_fields and field not in incompatible:
                    left = self.runtime.comparator._variable_array(cpu, field)
                    right = self.runtime.comparator._variable_array(candidate, field)
                    static_identity[field] = bool(np.array_equal(left, right))
                else:
                    static_identity[field] = False
        passed = bool(
            cpu_stamp == candidate_stamp == expected_stamp
            and not missing_required_cpu
            and not missing_required_candidate
            and not (set(incompatible) & set(STRICT_FIELDS + STATIC_FIELDS))
            and not nonfinite
            and all(static_identity.values())
        )
        return {
            "passed": passed,
            "domain": name,
            "own_step": int(step),
            "valid_time": valid.isoformat(),
            "cpu_times": cpu_stamp,
            "candidate_times": candidate_stamp,
            "common_numeric_field_count": len(common_numeric),
            "incompatible_common_numeric": incompatible,
            "missing_required_cpu": missing_required_cpu,
            "missing_required_candidate": missing_required_candidate,
            "nonfinite": nonfinite,
            "static_identity": static_identity,
        }

    def pair(self, name: str, step: int, candidate_path: Path) -> dict[str, Any]:
        valid = RUN_START + timedelta(seconds=int(step) * DT_SECONDS[name])
        stamp = valid.strftime("%Y-%m-%d_%H:%M:%S")
        cpu_authority = self.cpu_index.get((name, stamp))
        if cpu_authority is None:
            raise MetricGateFailure("CPU_FRAME_UNPAIRED", f"{name}:{stamp}")
        cpu_path = Path(cpu_authority["path"])
        cpu_sha256 = sha256_file(cpu_path)
        if cpu_sha256 != cpu_authority["sha256"] or cpu_path.stat().st_size != cpu_authority["bytes"]:
            raise MetricGateFailure("CPU_FRAME_AUTHORITY", f"{name}:{stamp}")
        candidate_sha256 = sha256_file(candidate_path)
        generic = self._generic_identity_pair(name, step, valid, cpu_path, candidate_path)
        artifact: dict[str, Any] = {
            "schema": "gpuwrf.v0234.nested-boundary-final-frame-pair.v1",
            "domain": name,
            "own_step": int(step),
            "valid_time": valid.isoformat(),
            "cpu": {**cpu_authority, "observed_sha256": cpu_sha256},
            "candidate": {
                "path": str(candidate_path.resolve()),
                "bytes": candidate_path.stat().st_size,
                "sha256": candidate_sha256,
            },
            "finite_identity": generic,
            "d03_full_pair": None,
            "decisive_1500": None,
            "known_1500_v10_record": None,
            "early_causal_00_20": None,
        }
        strict_rmse: dict[str, float] = {}
        if name == "d03":
            try:
                full = self.runtime.comparator.compare_pair(
                    cpu_path,
                    candidate_path,
                    valid,
                    frozen_geometry_policy="terminal_nested_boundary_v1",
                )
                strict_rmse = {
                    field: float(full["metrics"][field]["rmse"])
                    for field in STRICT_FIELDS
                }
                artifact["d03_full_pair"] = {
                    "valid_time": full["valid_time"],
                    "cpu_sha256": full["cpu_sha256"],
                    "candidate_sha256": full["gpu_sha256"],
                    "frozen_geo_sha256": full["frozen_geo_sha256"],
                    "frozen_geometry_policy": full["frozen_geometry_policy"],
                    "per_frame_static_pass": full["per_frame_static_pass"],
                    "strict_rmse": strict_rmse,
                    "common_compatible_numeric_field_count": len(
                        full["inventory"]["common_compatible_numeric_fields"]
                    ),
                    "cpu_only_fields": full["inventory"]["cpu_only_fields"],
                    "candidate_only_fields": full["inventory"]["gpu_only_fields"],
                    "incompatible_common_fields": full["inventory"]["common_incompatible_fields"],
                }
            except Exception as exc:
                artifact["d03_full_pair"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "passed": False,
                }
                generic["passed"] = False
                full = None
            if full is not None and full.get("per_frame_static_pass") is not True:
                generic["passed"] = False
            if full is not None and step == DECISIVE_D03_STEP:
                artifact["decisive_1500"] = compare_no_worse_metrics(full, self.decisive_retry20)
                if self.record_known_1500_v10_red:
                    if self.known_1500_v10_authority is None:
                        raise RunnerGateError(
                            "KNOWN_1500_V10_AUTHORITY_MISSING", "pairer was not fail-closed"
                        )
                    artifact["known_1500_v10_record"] = classify_exact_known_1500_v10_red(
                        artifact["decisive_1500"],
                        candidate_sha256=candidate_sha256,
                        authority=self.known_1500_v10_authority,
                        candidate_path=candidate_path,
                        cpu_path=cpu_path,
                        runtime=self.runtime,
                    )
            if full is not None and step == 200:
                profile_early_handler = globals().get("PROFILE_EARLY_CAUSAL_HANDLER")
                if callable(profile_early_handler):
                    artifact["early_causal_00_20"] = profile_early_handler(
                        runtime=self.runtime,
                        candidate_path=candidate_path,
                        candidate_sha256=candidate_sha256,
                        cpu_path=cpu_path,
                        cpu_sha256=cpu_sha256,
                        strict_rmse=strict_rmse,
                        finite_identity=generic,
                    )
                else:
                    selected_cpu = authenticate_early_causal_cpu_path(cpu_path)
                    with (
                        self.runtime.Dataset(candidate_path) as current_ds,
                        self.runtime.Dataset(EARLY_CAUSAL_PRIOR) as prior_ds,
                        self.runtime.Dataset(EARLY_CAUSAL_PARTIAL_WIND) as partial_ds,
                        self.runtime.Dataset(EARLY_CAUSAL_SCALAR_PARENT) as scalar_ds,
                        self.runtime.Dataset(EARLY_CAUSAL_RETRY20) as retry_ds,
                        self.runtime.Dataset(cpu_path) as cpu_ds,
                    ):
                        role_arrays = {
                            "current": {
                                field: current_ds.variables[field][0] for field in ("T", "U")
                            },
                            "prior": {
                                field: prior_ds.variables[field][0] for field in ("T", "U")
                            },
                            "partial": {
                                field: partial_ds.variables[field][0] for field in ("T", "U")
                            },
                            "scalar": {
                                field: scalar_ds.variables[field][0] for field in ("T", "U")
                            },
                            "retry20": {
                                field: retry_ds.variables[field][0] for field in ("T", "U")
                            },
                            "cpu": {
                                field: cpu_ds.variables[field][0] for field in ("T", "U")
                            },
                        }
                        artifact["early_causal_00_20"] = early_causal_ring1_metrics(
                            self.runtime.np, **role_arrays,
                        )
                        artifact["early_causal_00_20"]["selected_cpu_authority"] = selected_cpu
        artifact["passed"] = bool(
            generic["passed"]
            and (
                artifact["decisive_1500"] is None
                or artifact["decisive_1500"]["passed"] is True
            )
            and (
                artifact["early_causal_00_20"] is None
                or artifact["early_causal_00_20"]["passed"] is True
            )
        )
        artifact["recorded_known_v10_red_and_continued"] = bool(
            not artifact["passed"]
            and generic["passed"]
            and artifact["early_causal_00_20"] is None
            and artifact["decisive_1500"] is not None
            and artifact["decisive_1500"].get("passed") is False
            and artifact["known_1500_v10_record"] is not None
            and artifact["known_1500_v10_record"].get("passed") is True
        )
        artifact["proof_sha256"] = canonical_digest(artifact)
        artifact_path = self.pair_dir / f"{name}-step-{int(step):05d}.json"
        atomic_write_json(artifact_path, artifact)
        row = {
            "domain": name,
            "own_step": int(step),
            "valid_time": valid.isoformat(),
            "cpu_sha256": cpu_sha256,
            "candidate_sha256": candidate_sha256,
            "finite_identity_pass": generic["passed"],
            "static_identity": generic["static_identity"],
            "strict_rmse": strict_rmse,
            "scientific_pair_pass": artifact["passed"],
            "decisive_1500": artifact["decisive_1500"],
            "known_1500_v10_record": artifact["known_1500_v10_record"],
            "recorded_known_v10_red_and_continued": artifact[
                "recorded_known_v10_red_and_continued"
            ],
            "early_causal_00_20": artifact["early_causal_00_20"],
            "artifact": str(artifact_path.resolve()),
            "artifact_sha256": sha256_file(artifact_path),
            "artifact_payload_sha256": artifact["proof_sha256"],
        }
        self.rows.append(row)
        self.counts[name] += 1
        if name == "d03" and step in FIRST_PROGRESS_D03_STEPS:
            early = artifact["early_causal_00_20"]
            print(
                "NESTED_BOUNDARY_FIRST_RESULT "
                f"valid={valid.isoformat()} finite_identity={int(generic['passed'])} "
                f"V10_RMSE={strict_rmse.get('V10')} "
                f"causal_gate={None if early is None else int(early['passed'])}",
                flush=True,
            )
        if not artifact["passed"] and not artifact["recorded_known_v10_red_and_continued"]:
            detail = (
                artifact["early_causal_00_20"]
                or artifact["known_1500_v10_record"]
                or artifact["decisive_1500"]
                or generic
            )
            raise MetricGateFailure("INCREMENTAL_FRAME_PAIR", json.dumps(detail, sort_keys=True))
        return row


def seed_retained_prefix_pairs(
    pairer: IncrementalFramePairer,
) -> dict[str, Any]:
    """Re-pair every immutable gatefix1 frame through and including d03 14:40."""

    rows = []
    expected_steps = standard_output_steps(PARENT_JOIN_OWN_STEPS)
    for name in ("d01", "d02", "d03"):
        for step in expected_steps[name]:
            valid = RUN_START + timedelta(seconds=int(step) * DT_SECONDS[name])
            path = RC3_RUN_DIR / "output" / (
                f"wrfout_{name}_{valid.strftime('%Y-%m-%d_%H:%M:%S')}"
            )
            rows.append(pairer.pair(name, int(step), path))
    expected_counts = RC3_RAW_OUTPUT_COUNTS
    if pairer.counts != expected_counts:
        raise RunnerGateError(
            "RETAINED_PREFIX_PAIR_COUNTS",
            f"expected={expected_counts} actual={pairer.counts}",
        )
    last_d03 = [
        row for row in rows
        if row["domain"] == "d03" and row["own_step"] == 8800
    ]
    if (
        len(last_d03) != 1
        or last_d03[0]["valid_time"] != "2025-03-01T14:40:00+00:00"
        or last_d03[0]["finite_identity_pass"] is not True
    ):
        raise RunnerGateError("RETAINED_1440_PAIR", repr(last_d03))
    return {
        "counts": dict(pairer.counts),
        "row_count": len(rows),
        "retained_1440": last_d03[0],
        "sources": [
            {
                "domain": row["domain"],
                "own_step": row["own_step"],
                "candidate_sha256": row["candidate_sha256"],
            }
            for row in rows
        ],
    }


class StandardCadenceOutput:
    """Synchronous normal writer plus host-only incremental pairing and checkpoints."""

    wants_carry = True

    def __init__(
        self,
        writer: Any,
        pairer: IncrementalFramePairer,
        output_dir: Path,
        health_executable: Any,
        runtime: SimpleNamespace,
        run_dir: Path,
    ) -> None:
        self.writer = writer
        self.pairer = pairer
        self.output_dir = output_dir
        self.health_executable = health_executable
        self.runtime = runtime
        self.run_dir = run_dir
        self.scheduler_calls: list[tuple[str, int]] = []
        self.emitted: list[dict[str, Any]] = []
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.failed_carry: Any | None = None
        self.failed_step: int | None = None
        self.failed_domain: str | None = None
        self.previous_alarm_carry: dict[str, Any] = {}
        self.previous_alarm_step: dict[str, int] = {}
        self.last_alarm_carry: dict[str, Any] = {}
        self.last_alarm_step: dict[str, int] = {}

    def __call__(self, name: str, step: int, carry: Any) -> dict[str, Any]:
        step = int(step)
        key = (name, step)
        if key in self.scheduler_calls:
            raise MetricGateFailure("DUPLICATE_OUTPUT_ALARM", repr(key))
        self.scheduler_calls.append(key)
        if name in self.last_alarm_carry:
            self.previous_alarm_carry[name] = self.last_alarm_carry[name]
            self.previous_alarm_step[name] = self.last_alarm_step[name]
        self.last_alarm_carry[name] = carry
        self.last_alarm_step[name] = step
        valid = RUN_START + timedelta(seconds=step * DT_SECONDS[name])
        try:
            assert_preemption_clear(f"normal-output-{name}-{step}-pre")
            writer_result = self.writer(name, step, carry)
            path = self.output_dir / f"wrfout_{name}_{valid.strftime('%Y-%m-%d_%H:%M:%S')}"
            if not path.is_file():
                raise MetricGateFailure("WRFOUT_MISSING", str(path))
            checkpoint = None
            if name == "d03" and step in CHECKPOINT_D03_STEPS:
                retained = _retain_complete_carry(
                    self.runtime,
                    self.run_dir / "checkpoints",
                    role="authenticated",
                    domain=name,
                    step=step,
                    carry=carry,
                )
                health_raw = materialize_health(self.health_executable, carry, self.runtime)
                health = evaluate_health_summary(step, health_raw, step9313_scale_baseline=None)
                checkpoint = {"carry": retained, "health": health, "wrfout_retained": str(path.resolve())}
                self.checkpoints[str(step)] = checkpoint
                if not health["passed"]:
                    raise MetricGateFailure("CHECKPOINT_HEALTH", json.dumps(health["violations"], sort_keys=True))
            pair = self.pairer.pair(name, step, path)
        except BaseException as exc:
            self.failed_carry = carry
            self.failed_step = step
            self.failed_domain = name
            if isinstance(exc, MetricGateFailure):
                raise
            if isinstance(exc, RunnerGateError):
                raise MetricGateFailure(exc.code, exc.detail) from exc
            raise MetricGateFailure(
                "OUTPUT_PAIR_EXCEPTION", f"{type(exc).__name__}: {exc}",
            ) from exc
        row = {
            "domain": name,
            "own_step": step,
            "valid_time": valid.isoformat(),
            "wrfout": str(path.resolve()),
            "writer_result": writer_result,
            "pair": pair,
            "checkpoint": checkpoint,
        }
        self.emitted.append(row)
        return row


def _run_prefix(
    tree: Any,
    names: tuple[str, ...],
    initial_carries: dict[str, Any],
    dt_by_domain: dict[str, float],
    *,
    output: StandardCadenceOutput,
    cadence: dict[str, int],
    nonintegral: dict[str, tuple[int, ...]],
    runtime: SimpleNamespace,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    block_between, root_sync_cadence = runtime._nested_sync_mode_from_env()
    carries = initial_carries
    own_steps = {name: 0 for name in names}
    first_carry_ids = {name: id(value) for name, value in initial_carries.items()}
    aggregate_events: Counter[str] = Counter()
    rows = []
    for index, root_steps in enumerate(PREFIX_SEGMENTS):
        try:
            assert_preemption_clear(f"prefix-segment-{index}-pre")
        except RunnerGateError as exc:
            failure = _retain_failure_pair(
                runtime,
                run_dir,
                last_step=own_steps["d03"],
                last_carry=carries["d03"],
                failed_step=own_steps["d03"],
                failed_carry=carries["d03"],
                failure={
                    "code": exc.code,
                    "detail": exc.detail,
                    "dispatch_withheld": True,
                    "no_first_bad_output_exists": True,
                },
            )
            raise WindowFalsified(exc.code, failure["proof_sha256"]) from exc
        segment_input_carries = carries
        segment_input_steps = dict(own_steps)
        segment_started = time.perf_counter()
        try:
            result = runtime.run_operational_domain_tree(
                tree,
                root_steps=root_steps,
                feedback_enabled=False,
                output=output,
                output_cadence_steps=cadence,
                output_alarm_steps=nonintegral,
                block_between=block_between,
                root_sync_cadence=root_sync_cadence,
                carries=carries,
                initial_own_steps=own_steps,
            )
        except MetricGateFailure as exc:
            domain = output.failed_domain or "d03"
            last_carry = output.previous_alarm_carry.get(domain, segment_input_carries[domain])
            last_step = output.previous_alarm_step.get(domain, segment_input_steps[domain])
            failed_carry = output.failed_carry or output.last_alarm_carry.get(domain)
            failed_step = output.failed_step or output.last_alarm_step.get(domain)
            if failed_carry is None or failed_step is None:
                raise
            failure = _retain_failure_pair(
                runtime,
                run_dir,
                domain=domain,
                last_step=int(last_step),
                last_carry=last_carry,
                failed_step=int(failed_step),
                failed_carry=failed_carry,
                failure={"code": exc.code, "detail": exc.detail, "metric_gate": True},
            )
            raise WindowFalsified(exc.code, failure["proof_sha256"]) from exc
        runtime.block_until_ready(tuple(result.states[name].theta for name in names))
        try:
            for name in names:
                step = int(result.own_steps[name])
                runtime.assert_state_finite_at_boundary(
                    result.states[name], domain=name, step=step,
                    sim_time_s=step * float(dt_by_domain[name]),
                )
        except BaseException as exc:
            failure = _retain_failure_pair(
                runtime,
                run_dir,
                last_step=segment_input_steps["d03"],
                last_carry=segment_input_carries["d03"],
                failed_step=int(result.own_steps["d03"]),
                failed_carry=result.carries["d03"],
                failure={"code": "PREFIX_FINITE_BOUNDARY", "detail": f"{type(exc).__name__}: {exc}"},
            )
            raise WindowFalsified("PREFIX_FINITE_BOUNDARY", failure["proof_sha256"]) from exc
        events = Counter(event[0] for event in result.events)
        aggregate_events.update(events)
        carries = result.carries
        own_steps = dict(result.own_steps)
        rows.append({
            "index": index,
            "root_steps": root_steps,
            "own_steps": own_steps,
            "event_counts": dict(events),
            "wall_seconds": time.perf_counter() - segment_started,
        })
        print(f"NESTED_BUNDLE_PREFIX segment={index + 1}/{len(PREFIX_SEGMENTS)} own_steps={own_steps}", flush=True)
    if own_steps != PREFIX_OWN_STEPS:
        raise RunnerGateError("PREFIX_CLOCKS", repr(own_steps))
    return carries, own_steps, {
        "build_count": 1,
        "segments": rows,
        "initial_carry_object_ids": first_carry_ids,
        "prefix_output_carry_object_ids": {name: id(value) for name, value in carries.items()},
        "event_counts": dict(sorted(aggregate_events.items())),
        "own_steps": own_steps,
        "normal_output_counts": dict(output.pairer.counts),
        "normal_output_emitted": [
            {"domain": row["domain"], "valid_time": row["valid_time"]}
            for row in output.emitted
        ],
        "checkpoints": output.checkpoints,
    }


def _retain_complete_carry(
    runtime: SimpleNamespace,
    directory: Path,
    *,
    role: str,
    step: int,
    carry: Any,
    domain: str = "d03",
) -> dict[str, Any]:
    host = runtime.jax.device_get(carry)
    path = directory / f"{role}-{domain}-step-{step}.pkl"
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    manifest_before = runtime.ordinary.host_tree_manifest(host)
    atomic_write_pickle(path, host)
    with path.open("rb") as handle:
        reread = pickle.load(handle)
    manifest_after = runtime.ordinary.host_tree_manifest(reread)
    identity = runtime.ordinary.compare_manifests(manifest_before, manifest_after)
    if not identity["all_leaf_bytes_equal"]:
        raise RunnerGateError("RETAIN_REREAD_IDENTITY", role)
    return {
        "role": role,
        "domain": domain,
        "step": int(step),
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "manifest": manifest_before,
        "reread_identity": identity,
    }


def retain_late_window_carry_and_frame(
    runtime: SimpleNamespace,
    pairer: IncrementalFramePairer,
    frame_writer: Any,
    run_dir: Path,
    *,
    step: int,
    carry: Any,
    health: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain the contract-required exact-step carry and a finite d03 frame."""

    if step not in LATE_WINDOW_RETAIN_D03_STEPS:
        raise RunnerGateError("LATE_WINDOW_RETAIN_STEP", str(step))
    retained = _retain_complete_carry(
        runtime,
        run_dir / "late-window-carries",
        role="late-window",
        domain="d03",
        step=step,
        carry=carry,
    )
    writer_result = frame_writer("d03", step, carry)
    frame_path = Path(writer_result["wrfout"]).resolve()
    if not frame_path.is_file():
        raise MetricGateFailure("LATE_WINDOW_FRAME_MISSING", str(frame_path))
    frame_sha256 = sha256_file(frame_path)
    reference_stamp = (
        RUN_START + timedelta(seconds=9400 * DT_SECONDS["d03"])
    ).strftime("%Y-%m-%d_%H:%M:%S")
    reference_authority = pairer.cpu_index.get(("d03", reference_stamp))
    if reference_authority is None:
        raise MetricGateFailure("LATE_WINDOW_STATIC_REFERENCE", reference_stamp)
    reference_path = Path(reference_authority["path"])
    reference_sha256 = sha256_file(reference_path)
    if (
        reference_sha256 != reference_authority["sha256"]
        or reference_path.stat().st_size != reference_authority["bytes"]
    ):
        raise MetricGateFailure("LATE_WINDOW_STATIC_REFERENCE_AUTHORITY", reference_stamp)

    np = runtime.np
    strict: dict[str, Any] = {}
    nonfinite: list[dict[str, Any]] = []
    with runtime.Dataset(frame_path) as candidate, runtime.Dataset(reference_path) as reference:
        decoded_time = runtime.comparator._decoded_times(candidate)
        expected_time = (
            RUN_START + timedelta(seconds=step * DT_SECONDS["d03"])
        ).strftime("%Y-%m-%d_%H:%M:%S")
        for field in STRICT_FIELDS:
            values = runtime.comparator._variable_array(candidate, field)
            finite_count = int(np.count_nonzero(~np.isfinite(values)))
            if finite_count:
                nonfinite.append({"field": field, "count": finite_count})
            contiguous = np.ascontiguousarray(values)
            strict[field] = {
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
                "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
                "nonfinite_count": finite_count,
                "minimum": None if finite_count else float(np.min(contiguous)),
                "maximum": None if finite_count else float(np.max(contiguous)),
            }
        static_identity = {
            field: bool(np.array_equal(
                runtime.comparator._variable_array(candidate, field),
                runtime.comparator._variable_array(reference, field),
            ))
            for field in STATIC_FIELDS
        }
        for field, variable in candidate.variables.items():
            if field == "Times" or not np.issubdtype(variable.dtype, np.number):
                continue
            values = runtime.comparator._variable_array(candidate, field)
            count = int(np.count_nonzero(~np.isfinite(values)))
            if count and field not in STRICT_FIELDS:
                nonfinite.append({"field": field, "count": count})
    frame_sha256_after = sha256_file(frame_path)
    passed = bool(
        decoded_time == expected_time
        and frame_sha256_after == frame_sha256
        and not nonfinite
        and all(static_identity.values())
        and health.get("passed") is True
    )
    row = {
        "schema": "gpuwrf.v0234.post-fable-late-window-retained-step.v1",
        "passed": passed,
        "domain": "d03",
        "own_step": step,
        "valid_time": expected_time,
        "carry": retained,
        "frame": {
            "path": str(frame_path),
            "bytes": frame_path.stat().st_size,
            "sha256": frame_sha256,
            "stable_after_inspection": frame_sha256_after == frame_sha256,
            "decoded_time": decoded_time,
        },
        "strict_field_manifests": strict,
        "all_numeric_nonfinite": nonfinite,
        "static_identity": static_identity,
        "static_reference": {
            **reference_authority,
            "observed_sha256": reference_sha256,
        },
        "complete_carry_health": dict(health),
        "host_transfer_authority": (
            "explicit CONTRACT.md retention at d03 9313/9314/9405; "
            "outside the production one-step HLO"
        ),
    }
    row["proof_sha256"] = canonical_digest(row)
    if not passed:
        raise MetricGateFailure("LATE_WINDOW_RETAINED_FRAME_GATE", json.dumps(row, sort_keys=True))
    return row


def prepare_rc3_continuation_cpu_namespace(
    runtime: SimpleNamespace,
    run_dir: Path,
) -> dict[str, Any]:
    """Build the fresh host-only continuation input proof without GPU work."""

    run_dir = run_dir.resolve()
    expected = (LINEAGE_WORK_DIR / CONTINUATION_NAMESPACE).resolve()
    if run_dir != expected:
        raise RunnerGateError(
            "CONTINUATION_NAMESPACE", f"expected={expected} actual={run_dir}",
        )
    devices = runtime.jax.devices()
    if not devices or any(device.platform != "cpu" for device in devices):
        raise RunnerGateError("CONTINUATION_CPU_PREP_DEVICE", repr(devices))
    source_before = assert_rc3_continuation_source_authority()
    if run_dir.exists() or run_dir.is_symlink():
        raise RunnerGateError("CONTINUATION_NAMESPACE_EXISTS", str(run_dir))
    run_dir.mkdir(parents=True, exist_ok=False)

    with RC3_STEP8800_CARRY.open("rb") as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read()
        after = os.fstat(handle.fileno())
    stable = bool(
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size == len(raw)
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
        and hashlib.sha256(raw).hexdigest() == RC3_STEP8800_CARRY_SHA256
    )
    if not stable:
        raise RunnerGateError("CONTINUATION_CARRY_SINGLE_READ", str(RC3_STEP8800_CARRY))
    host_carry = pickle.loads(raw)
    loaded_manifest = runtime.ordinary.host_tree_manifest(host_carry)
    failure_payload, _failure_row = read_authenticated_json(
        RC3_FAILURE_PROOF,
        RC3_FAILURE_PROOF_SHA256,
        "CONTINUATION_FAILURE_PROOF_REREAD",
    )
    expected_manifest = failure_payload["first_failed"]["manifest"]
    source_identity = runtime.ordinary.compare_manifests(expected_manifest, loaded_manifest)
    if (
        loaded_manifest.get("leaf_count") != 106
        or loaded_manifest.get("floating_nonfinite_count") != 0
        or source_identity.get("all_leaf_bytes_equal") is not True
    ):
        raise RunnerGateError("CONTINUATION_DESERIALIZE_IDENTITY", repr(source_identity))

    roundtripped = pickle.loads(pickle.dumps(host_carry, protocol=5))
    roundtrip_manifest = runtime.ordinary.host_tree_manifest(roundtripped)
    roundtrip_identity = runtime.ordinary.compare_manifests(
        loaded_manifest, roundtrip_manifest,
    )
    if roundtrip_identity.get("all_leaf_bytes_equal") is not True:
        raise RunnerGateError("CONTINUATION_MEMORY_ROUNDTRIP", repr(roundtrip_identity))
    retained = _retain_complete_carry(
        runtime,
        run_dir / "input",
        role="roundtrip-authenticated",
        domain="d03",
        step=8800,
        carry=roundtripped,
    )
    if retained["reread_identity"].get("all_leaf_bytes_equal") is not True:
        raise RunnerGateError("CONTINUATION_FILE_ROUNDTRIP", repr(retained))
    retained_identity = runtime.ordinary.compare_manifests(
        loaded_manifest, retained["manifest"],
    )
    if retained_identity.get("all_leaf_bytes_equal") is not True:
        raise RunnerGateError("CONTINUATION_RETAINED_IDENTITY", repr(retained_identity))

    pairer = IncrementalFramePairer(runtime, run_dir / "frame-pairs")
    pair_row = pairer.pair("d03", 8800, RC3_STEP8800_WRFOUT)
    if (
        pair_row.get("finite_identity_pass") is not True
        or pair_row.get("own_step") != 8800
        or pair_row.get("valid_time") != "2025-03-01T14:40:00+00:00"
        or pairer.counts != {"d01": 0, "d02": 0, "d03": 1}
    ):
        raise RunnerGateError("CONTINUATION_1440_PAIR", repr(pair_row))

    source_after = assert_rc3_continuation_source_authority()
    immutable_keys = ("failure_proof", "step8800_carry", "step8800_wrfout", "last_healthy_pair")
    if any(source_before[key] != source_after[key] for key in immutable_keys):
        raise RunnerGateError("RC3_SOURCE_CHANGED_DURING_PREP", repr(immutable_keys))

    proof = {
        "schema": "gpuwrf.v0234.nested-boundary-final-resume8800-input.v1",
        "verdict": (
            "READY_FOR_RESUME8800_GPU"
            if source_after["continuation_ready"]
            else "RESUME8800_BLOCKED_PARENT_OPERATIONAL_CARRIES_NOT_RETAINED"
        ),
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_tree": CANDIDATE_TREE,
        "source_authority_before": source_before,
        "source_authority_after": source_after,
        "source_namespace_immutable": True,
        "source_deserialize_identity": source_identity,
        "memory_roundtrip_identity": roundtrip_identity,
        "retained_roundtrip_identity": retained_identity,
        "retained_roundtrip_carry": retained,
        "retained_1440_pair": pair_row,
        "exact_resume": {
            "domain": "d03",
            "step": 8800,
            "valid_time": "2025-03-01T14:40:00+00:00",
            "leaf_count": 106,
            "floating_nonfinite_count": 0,
        },
        "gpu_commands_run": 0,
        "gpu_queries_run": 0,
        "gpu_locks_acquired_or_verified": 0,
        "model_lowers": 0,
        "model_compiles": 0,
        "model_dispatches": 0,
        "health_lowers": 0,
        "health_compiles": 0,
        "parent_output_substitution": False,
        "wrfout_restart_reconstruction": False,
        "full_prefix_rerun": False,
    }
    proof["proof_sha256"] = canonical_digest(proof)
    proof_path = run_dir / "continuation-input-proof.json"
    atomic_write_json(proof_path, proof)
    return {
        "path": str(proof_path.resolve()),
        "file_sha256": sha256_file(proof_path),
        "proof_sha256": proof["proof_sha256"],
        "verdict": proof["verdict"],
        "blockers": source_after["blockers"],
    }


def load_authenticated_retained_d03_carry(
    runtime: SimpleNamespace,
    run_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Load retained post-8800 d03 once and prove host/device/file identity."""

    source_before = stable_file_authority(
        RC3_STEP8800_CARRY, "PARENT_JOIN_RETAINED_D03_SOURCE",
    )
    if source_before["sha256"] != RC3_STEP8800_CARRY_SHA256:
        raise RunnerGateError(
            "PARENT_JOIN_RETAINED_D03_HASH",
            f"expected={RC3_STEP8800_CARRY_SHA256} actual={source_before['sha256']}",
        )
    raw = RC3_STEP8800_CARRY.read_bytes()
    verify_stable_file_authority(source_before, "PARENT_JOIN_RETAINED_D03_CHANGED")
    host = pickle.loads(raw)
    host_manifest = runtime.ordinary.host_tree_manifest(host)
    failure, failure_authority = read_authenticated_json(
        RC3_FAILURE_PROOF,
        RC3_FAILURE_PROOF_SHA256,
        "PARENT_JOIN_FAILURE_PROOF",
    )
    expected_manifest = failure["first_failed"]["manifest"]
    deserialize_identity = runtime.ordinary.compare_manifests(
        expected_manifest, host_manifest,
    )
    if (
        host_manifest.get("leaf_count") != 106
        or host_manifest.get("floating_nonfinite_count") != 0
        or deserialize_identity.get("all_leaf_bytes_equal") is not True
    ):
        raise RunnerGateError(
            "PARENT_JOIN_D03_DESERIALIZE_IDENTITY", repr(deserialize_identity),
        )
    device = runtime.jax.device_put(host)
    runtime.block_until_ready(device.state.theta)
    device_roundtrip = runtime.ordinary.host_tree_manifest(
        runtime.jax.device_get(device),
    )
    device_identity = runtime.ordinary.compare_manifests(
        host_manifest, device_roundtrip,
    )
    if device_identity.get("all_leaf_bytes_equal") is not True:
        raise RunnerGateError("PARENT_JOIN_D03_DEVICE_IDENTITY", repr(device_identity))
    retained = _retain_complete_carry(
        runtime,
        run_dir / "authenticated-input",
        role="retained-post-dispatch",
        domain="d03",
        step=8800,
        carry=device,
    )
    retained_identity = runtime.ordinary.compare_manifests(
        host_manifest, retained["manifest"],
    )
    if retained_identity.get("all_leaf_bytes_equal") is not True:
        raise RunnerGateError("PARENT_JOIN_D03_RETAINED_IDENTITY", repr(retained_identity))
    return device, {
        "source": source_before,
        "failure_proof": failure_authority,
        "deserialize_identity": deserialize_identity,
        "device_roundtrip_identity": device_identity,
        "fresh_retained_identity": retained_identity,
        "fresh_retained": retained,
        "own_step": 8800,
        "valid_time": "2025-03-01T14:40:00+00:00",
    }


def compare_join_boundary_target_records(
    retained_d03: Any,
    rebuilt_d03: Any,
    runtime: SimpleNamespace,
    *,
    edge: Any,
) -> dict[str, Any]:
    """Compare only WRF's new-parent target record; never feed rebuilt data onward."""

    np = runtime.np
    rows = []
    for leaf_index, field in zip(
        BOUNDARY_TARGET_TRANSITION_LEAF_INDICES,
        BOUNDARY_TARGET_TRANSITION_FIELDS,
    ):
        retained = np.ascontiguousarray(
            runtime.jax.device_get(getattr(retained_d03.state, field))
        )
        rebuilt = np.ascontiguousarray(
            runtime.jax.device_get(getattr(rebuilt_d03.state, field))
        )
        retained_target = np.ascontiguousarray(retained[1]) if retained.ndim else retained
        rebuilt_target = np.ascontiguousarray(rebuilt[1]) if rebuilt.ndim else rebuilt
        left_raw = retained_target.tobytes(order="C")
        right_raw = rebuilt_target.tobytes(order="C")
        exact = bool(
            retained.shape == rebuilt.shape
            and retained.dtype == rebuilt.dtype
            and retained.shape[0] == rebuilt.shape[0] == 2
            and left_raw == right_raw
        )
        rows.append({
            "leaf_index": leaf_index,
            "field": field,
            "retained_shape": list(retained.shape),
            "rebuilt_shape": list(rebuilt.shape),
            "retained_dtype": str(retained.dtype),
            "rebuilt_dtype": str(rebuilt.dtype),
            "target_record_index": 1,
            "target_bytes": len(left_raw),
            "retained_target_sha256": hashlib.sha256(left_raw).hexdigest(),
            "rebuilt_target_sha256": hashlib.sha256(right_raw).hexdigest(),
            "bit_exact": exact,
        })
    ratio = int(edge.parent_grid_ratio)
    cadence = {
        "parent": "d02",
        "child": "d03",
        "parent_grid_ratio": ratio,
        "d02_step": PARENT_JOIN_OWN_STEPS["d02"],
        "d03_step": PARENT_JOIN_OWN_STEPS["d03"],
        "d03_subcycle_position": (
            PARENT_JOIN_OWN_STEPS["d03"]
            - (PARENT_JOIN_OWN_STEPS["d02"] - 1) * ratio
        ),
        "d02_clock_seconds": PARENT_JOIN_OWN_STEPS["d02"] * DT_SECONDS["d02"],
        "d03_clock_seconds": PARENT_JOIN_OWN_STEPS["d03"] * DT_SECONDS["d03"],
    }
    passed = bool(
        len(rows) == 11
        and all(row["bit_exact"] for row in rows)
        and ratio == 3
        and cadence["d03_subcycle_position"] == 1
        and cadence["d02_clock_seconds"] == 52812
        and cadence["d03_clock_seconds"] == 52800
    )
    return {
        "passed": passed,
        "semantics": "compare record[1] new-parent targets; rebuilt carry is discarded",
        "record_zero_excluded_reason": (
            "retained child is post-step8800 while original force used pre-step8800 child ring"
        ),
        "cadence": cadence,
        "rows": rows,
        "rebuilt_carry_used_for_continuation": False,
    }


def parent_scheduler_contract(
    runtime: SimpleNamespace,
    dt_by_domain: dict[str, float],
) -> tuple[dict[str, int], dict[str, tuple[int, ...]], dict[str, Any]]:
    """Project the real mixed integral/non-integral schedule onto d01/d02."""

    cadence_all, nonintegral_all, schedule = runtime.ordinary.scheduler_contract(
        ("d01", "d02", "d03"), dt_by_domain, total_steps=TERMINAL_OWN_STEPS,
    )
    if (
        cadence_all != {"d01": 67, "d02": 200, "d03": 200}
        or set(nonintegral_all) != {"d01"}
        or set((schedule.get("nonintegral_output_alarms") or {})) != {"d01"}
    ):
        raise RunnerGateError(
            "PARENT_SCHEDULER_SCHEMA",
            repr({
                "cadence": cadence_all,
                "nonintegral": sorted(nonintegral_all),
                "schedule_nonintegral": sorted(
                    (schedule.get("nonintegral_output_alarms") or {})
                ),
            }),
        )
    cadence = {name: cadence_all[name] for name in PARENT_DOMAINS}
    nonintegral = {
        name: nonintegral_all[name]
        for name in PARENT_DOMAINS
        if name in nonintegral_all
    }
    if cadence != {"d01": 67, "d02": 200} or set(nonintegral) != {"d01"}:
        raise RunnerGateError(
            "PARENT_SCHEDULER_PROJECTION", repr((cadence, nonintegral)),
        )
    return cadence, nonintegral, schedule


def run_parent_only_reconstitution(
    tree: Any,
    initial_carries: dict[str, Any],
    dt_by_domain: dict[str, float],
    runtime: SimpleNamespace,
    run_dir: Path,
    *,
    stage_fn: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, int], dict[str, LazyOneStepDomainExecutable], dict[str, Any]]:
    """Recreate d01/d02 only, with exactly one ordinary one-step program each."""

    parent_output_dir = run_dir / "parent-reconstruction-output"
    parent_output_dir.mkdir(parents=True, exist_ok=False)
    if stage_fn is not None:
        stage_fn("PARENT_RECON_SCHEDULER_CONTRACT")
    cadence, nonintegral, schedule = parent_scheduler_contract(runtime, dt_by_domain)
    if stage_fn is not None:
        stage_fn("PARENT_RECON_SCHEDULER_PROJECTION")
    if stage_fn is not None:
        stage_fn("PARENT_RECON_HIERARCHY")
    parent_nests = tuple(
        spec for spec in tree.hierarchy.nests
        if spec.parent == "d01" and spec.child == "d02"
    )
    if len(parent_nests) != 1:
        raise RunnerGateError("PARENT_HIERARCHY_EDGE", repr(parent_nests))
    hierarchy = runtime.DomainHierarchy.from_edges(
        PARENT_DOMAINS, parent_nests, max_dom=tree.hierarchy.max_dom,
    )
    parent_carries = {name: initial_carries[name] for name in PARENT_DOMAINS}
    own_steps = {name: 0 for name in PARENT_DOMAINS}
    parent_bundles = {name: tree.domains[name] for name in PARENT_DOMAINS}
    edge_lookup = runtime.ordinary._edge_lookup(tree)
    resolved_parent_edge = edge_lookup(parent_nests[0])
    if (
        hierarchy.order != PARENT_DOMAINS
        or set(parent_carries) != set(PARENT_DOMAINS)
        or set(parent_bundles) != set(PARENT_DOMAINS)
        or resolved_parent_edge.parent != "d01"
        or resolved_parent_edge.child != "d02"
    ):
        raise RunnerGateError(
            "PARENT_REAL_SCHEMA",
            repr({
                "hierarchy": hierarchy.order,
                "carries": sorted(parent_carries),
                "bundles": sorted(parent_bundles),
                "edge": (resolved_parent_edge.parent, resolved_parent_edge.child),
            }),
        )
    if stage_fn is not None:
        stage_fn("PARENT_RECON_WRITER")
    writer = runtime._PerDomainWrfoutWriter(
        output_dir=parent_output_dir,
        input_dir=INPUT_DIR,
        run_start=RUN_START,
        bundles=parent_bundles,
        output_cadence_steps=cadence,
        dt_by_domain={name: dt_by_domain[name] for name in PARENT_DOMAINS},
        async_writer=None,
        output_pipeline=None,
    )
    if set(writer.bundles) != set(PARENT_DOMAINS) or set(writer.written) != set(PARENT_DOMAINS):
        raise RunnerGateError(
            "PARENT_WRITER_SCHEMA",
            repr((sorted(writer.bundles), sorted(writer.written))),
        )
    output = ParentPrefixBitwiseOutput(writer, parent_output_dir, runtime)
    programs = {
        name: LazyOneStepDomainExecutable(
            tree,
            name,
            runtime,
            run_dir / f"ordinary-one-step-{name}-lowered-hlo.json",
        )
        for name in PARENT_DOMAINS
    }
    d03_dispatches = 0

    def advance(name: str, carry: Any, start_step: int, n_steps: int) -> Any:
        nonlocal d03_dispatches
        if name not in programs:
            d03_dispatches += int(n_steps)
            raise RunnerGateError("PARENT_PREJOIN_D03_DISPATCH", repr((name, start_step, n_steps)))
        assert_preemption_clear(f"parent-reconstitution-{name}-step-{int(start_step)}-pre")
        return programs[name].advance(carry, int(start_step), int(n_steps))

    if stage_fn is not None:
        stage_fn("PARENT_RECON_INITIAL_HISTORY")
    initial_history = runtime._emit_initial_history_frames(
        output, PARENT_DOMAINS, parent_carries, enabled=True,
    )
    block_between, root_sync_cadence = runtime._nested_sync_mode_from_env()
    segment_rows = []
    for index, root_steps in enumerate(PARENT_RECON_SEGMENTS):
        if stage_fn is not None:
            stage_fn(f"PARENT_RECON_SEGMENT_{index + 1:02d}")
        assert_preemption_clear(f"parent-reconstitution-segment-{index}-pre")
        result = runtime.run_domain_tree_callbacks(
            hierarchy,
            parent_carries,
            root_steps=int(root_steps),
            advance=advance,
            force=runtime._operational_force,
            feedback=None,
            feedback_enabled=False,
            output=output,
            output_cadence_steps=cadence,
            output_alarm_steps=nonintegral,
            block_between=block_between,
            root_sync_cadence=root_sync_cadence,
            edge_lookup=edge_lookup,
            fused_cascade=None,
            initial_own_steps=own_steps,
        )
        runtime.block_until_ready(tuple(result.states[name].theta for name in PARENT_DOMAINS))
        for name in PARENT_DOMAINS:
            step = int(result.own_steps[name])
            runtime.assert_state_finite_at_boundary(
                result.states[name], domain=name, step=step,
                sim_time_s=step * float(dt_by_domain[name]),
            )
        parent_carries = result.carries
        own_steps = dict(result.own_steps)
        segment_rows.append({
            "index": index,
            "root_steps": int(root_steps),
            "own_steps": own_steps,
            "event_counts": dict(Counter(event[0] for event in result.events)),
        })
        print(
            "NESTED_BOUNDARY_PARENT_RECON "
            f"segment={index + 1}/{len(PARENT_RECON_SEGMENTS)} own_steps={own_steps}",
            flush=True,
        )
    expected_parent_steps = {name: PARENT_JOIN_OWN_STEPS[name] for name in PARENT_DOMAINS}
    expected_counts = {"d01": 15, "d02": 15}
    if own_steps != expected_parent_steps:
        raise RunnerGateError(
            "PARENT_RECON_CLOCKS", f"expected={expected_parent_steps} actual={own_steps}",
        )
    if output.counts != expected_counts or len(output.rows) != 30:
        raise RunnerGateError(
            "PARENT_RECON_OUTPUT_COUNTS",
            f"expected={expected_counts} actual={output.counts} rows={len(output.rows)}",
        )
    if d03_dispatches != 0:
        raise RunnerGateError("PARENT_PREJOIN_D03_DISPATCH_COUNT", str(d03_dispatches))
    program_audits = {name: programs[name].audit() for name in PARENT_DOMAINS}
    if any(row["compile_calls"] != 1 for row in program_audits.values()):
        raise RunnerGateError("PARENT_COMPILE_COUNTS", repr(program_audits))
    if not all(row["comparison"]["passed"] for row in output.rows):
        raise RunnerGateError("PARENT_RECON_BIT_IDENTITY", "one or more frame comparisons failed")
    return parent_carries, own_steps, programs, {
        "schedule": schedule,
        "real_schema": {
            "hierarchy_order": list(hierarchy.order),
            "carry_domains": sorted(parent_carries),
            "bundle_domains": sorted(parent_bundles),
            "writer_domains": sorted(writer.written),
            "edge": [resolved_parent_edge.parent, resolved_parent_edge.child],
            "cadence": cadence,
            "nonintegral_override_domains": sorted(nonintegral),
        },
        "initial_history": initial_history,
        "segments": segment_rows,
        "own_steps": own_steps,
        "output_counts": dict(output.counts),
        "frame_comparisons": output.rows,
        "programs": program_audits,
        "d03_dispatches_before_join": d03_dispatches,
        "all_retained_parent_gpu_values_and_metadata_bit_exact": True,
    }


def run_real_schema_parent_cpu_rehearsal(run_dir: Path) -> dict[str, Any]:
    """Exercise the actual parent schema on CPU with identity model stepping."""

    runtime = _import_runtime()
    devices = runtime.jax.devices()
    if not devices or any(device.platform != "cpu" for device in devices):
        raise RunnerGateError("PARENT_REHEARSAL_CPU_DEVICE", repr(devices))
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    original_gpu_device = runtime.state_contract._gpu_device
    runtime.state_contract._gpu_device = lambda: devices[0]
    try:
        tree, names, initial_carries, dt_by_domain, load_authority = (
            runtime.ordinary.load_corrected_tree(run_dir)
        )
    finally:
        runtime.state_contract._gpu_device = original_gpu_device
    if runtime.state_contract._gpu_device is not original_gpu_device:
        raise RunnerGateError("PARENT_REHEARSAL_DEVICE_HOOK_RESTORE", "identity mismatch")
    cadence, nonintegral, schedule = parent_scheduler_contract(runtime, dt_by_domain)
    parent_nests = tuple(
        spec for spec in tree.hierarchy.nests
        if spec.parent == "d01" and spec.child == "d02"
    )
    if len(parent_nests) != 1:
        raise RunnerGateError("PARENT_REHEARSAL_EDGE_COUNT", repr(parent_nests))
    hierarchy = runtime.DomainHierarchy.from_edges(
        PARENT_DOMAINS, parent_nests, max_dom=tree.hierarchy.max_dom,
    )
    carries = {name: initial_carries[name] for name in PARENT_DOMAINS}
    bundles = {name: tree.domains[name] for name in PARENT_DOMAINS}
    edge_lookup = runtime.ordinary._edge_lookup(tree)
    resolved = edge_lookup(parent_nests[0])
    leaf_counts = {
        name: len(runtime.jax.tree_util.tree_leaves(carries[name]))
        for name in PARENT_DOMAINS
    }
    if (
        names != ("d01", "d02", "d03")
        or hierarchy.order != PARENT_DOMAINS
        or set(carries) != set(PARENT_DOMAINS)
        or set(bundles) != set(PARENT_DOMAINS)
        or leaf_counts != {"d01": 115, "d02": 106}
        or (resolved.parent, resolved.child) != ("d01", "d02")
    ):
        raise RunnerGateError(
            "PARENT_REHEARSAL_REAL_SCHEMA",
            repr({
                "names": names,
                "hierarchy": hierarchy.order,
                "carries": sorted(carries),
                "bundles": sorted(bundles),
                "leaf_counts": leaf_counts,
                "edge": (resolved.parent, resolved.child),
            }),
        )

    output_dir = run_dir / "initial-history"
    output_dir.mkdir(parents=True, exist_ok=False)
    writer = runtime._PerDomainWrfoutWriter(
        output_dir=output_dir,
        input_dir=INPUT_DIR,
        run_start=RUN_START,
        bundles=bundles,
        output_cadence_steps=cadence,
        dt_by_domain={name: dt_by_domain[name] for name in PARENT_DOMAINS},
        async_writer=None,
        output_pipeline=None,
    )
    if set(writer.bundles) != set(PARENT_DOMAINS) or set(writer.written) != set(PARENT_DOMAINS):
        raise RunnerGateError(
            "PARENT_REHEARSAL_WRITER_SCHEMA",
            repr((sorted(writer.bundles), sorted(writer.written))),
        )
    class RehearsalSchedulerOutput:
        wants_carry = True

        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []
            self.counts = {name: 0 for name in PARENT_DOMAINS}

        def __call__(self, name: str, step: int, carry: Any) -> Any:
            if name not in self.counts:
                raise RunnerGateError("PARENT_REHEARSAL_OUTPUT_DOMAIN", name)
            self.calls.append((name, int(step)))
            self.counts[name] += 1
            return writer(name, int(step), carry)

    scheduler_output = RehearsalSchedulerOutput()
    initial_history = runtime._emit_initial_history_frames(
        scheduler_output, PARENT_DOMAINS, carries, enabled=True,
    )
    initial_rows = []
    for name, result in zip(PARENT_DOMAINS, initial_history):
        path = Path(str(result["wrfout"]))
        with runtime.Dataset(path, "r") as dataset:
            decoded = runtime.comparator._decoded_times(dataset)
            variables = len(dataset.variables)
        initial_rows.append({
            "domain": name,
            "own_step": result["own_step"],
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "decoded_time": decoded,
            "variable_count": variables,
        })
    if (
        [row["domain"] for row in initial_rows] != list(PARENT_DOMAINS)
        or any(row["own_step"] != 0 for row in initial_rows)
        or any(row["decoded_time"] != "2025-03-01_00:00:00" for row in initial_rows)
        or any(row["variable_count"] <= 0 for row in initial_rows)
        or scheduler_output.calls != [("d01", 0), ("d02", 0)]
        or scheduler_output.counts != {"d01": 1, "d02": 1}
        or {name: len(paths) for name, paths in writer.written.items()}
        != {"d01": 1, "d02": 1}
    ):
        raise RunnerGateError(
            "PARENT_REHEARSAL_INITIAL_HISTORY",
            repr({
                "rows": initial_rows,
                "calls": scheduler_output.calls,
                "counts": scheduler_output.counts,
                "writer_counts": {
                    name: len(paths) for name, paths in writer.written.items()
                },
            }),
        )

    advance_calls: list[dict[str, Any]] = []
    force_calls: list[dict[str, Any]] = []

    def identity_advance(name: str, carry: Any, start_step: int, n_steps: int) -> Any:
        advance_calls.append({
            "domain": name,
            "start_step": int(start_step),
            "n_steps": int(n_steps),
            "same_object": True,
        })
        return carry

    def identity_force(edge: Any, _parent: Any, child: Any) -> Any:
        force_calls.append({
            "parent": edge.parent,
            "child": edge.child,
            "ratio": int(edge.parent_grid_ratio),
            "weights_present": edge.weights is not None,
        })
        return child

    initial_output_calls = list(scheduler_output.calls)
    initial_output_counts = dict(scheduler_output.counts)
    result = runtime.run_domain_tree_callbacks(
        hierarchy,
        carries,
        root_steps=1,
        advance=identity_advance,
        force=identity_force,
        feedback=None,
        feedback_enabled=False,
        output=scheduler_output,
        output_cadence_steps=cadence,
        output_alarm_steps=nonintegral,
        block_between=False,
        root_sync_cadence=None,
        edge_lookup=edge_lookup,
        fused_cascade=None,
        initial_own_steps={"d01": 0, "d02": 0},
    )
    event_counts = dict(Counter(event[0] for event in result.events))
    expected_advance = [
        {"domain": "d01", "start_step": 1, "n_steps": 1, "same_object": True},
        {"domain": "d02", "start_step": 1, "n_steps": 3, "same_object": True},
    ]
    if (
        dict(result.own_steps) != {"d01": 1, "d02": 3}
        or advance_calls != expected_advance
        or force_calls != [{
            "parent": "d01", "child": "d02", "ratio": 3, "weights_present": True,
        }]
        or scheduler_output.calls != initial_output_calls
        or scheduler_output.counts != initial_output_counts
        or event_counts != {"advance": 2, "force": 1}
    ):
        raise RunnerGateError(
            "PARENT_REHEARSAL_FIRST_RECURSION",
            repr({
                "own_steps": dict(result.own_steps),
                "advance": advance_calls,
                "force": force_calls,
                "output": scheduler_output.calls,
                "output_counts": scheduler_output.counts,
                "events": event_counts,
            }),
        )
    proof = {
        "schema": "gpuwrf.v0234.parent-join-real-schema-cpu-rehearsal.v1",
        "verdict": "PARENT_REAL_SCHEMA_REHEARSAL_GREEN",
        "candidate_commit": CANDIDATE_COMMIT,
        "devices": [str(device) for device in devices],
        "load_authority": load_authority,
        "domains": {
            "loaded": list(names),
            "hierarchy": list(hierarchy.order),
            "carries": sorted(carries),
            "bundles": sorted(bundles),
            "writer": sorted(writer.written),
            "scheduler_output": sorted(scheduler_output.counts),
            "leaf_counts": leaf_counts,
        },
        "edge_lookup": force_calls[0],
        "scheduler": {
            "cadence": cadence,
            "nonintegral_override_domains": sorted(nonintegral),
            "contract": schedule,
            "initial_history": initial_rows,
            "initial_history_calls": [list(row) for row in initial_output_calls],
            "initial_history_counts": initial_output_counts,
            "writer_counts": {
                name: len(paths) for name, paths in writer.written.items()
            },
            "first_recursive_own_steps": dict(result.own_steps),
            "advance_calls": advance_calls,
            "force_calls": force_calls,
            "output_calls_after_first_recursion": [
                list(row) for row in scheduler_output.calls
            ],
            "output_counts_after_first_recursion": dict(scheduler_output.counts),
            "event_counts": event_counts,
        },
        "ordinary_model_programs_constructed": 0,
        "ordinary_model_lowers": 0,
        "ordinary_model_compiles": 0,
        "ordinary_model_dispatches": 0,
        "identity_advance_only": True,
        "cpu_state_constructor_override_restored": True,
        "gpu_commands_or_queries": 0,
    }
    proof["proof_sha256"] = canonical_digest(proof)
    atomic_write_json(run_dir / "real-schema-rehearsal.json", proof)
    return proof


class ParentJoinContinuationAdvance:
    """One-step ordinary dispatch plus post-dispatch complete-carry health."""

    def __init__(
        self,
        programs: Mapping[str, LazyOneStepDomainExecutable],
        health_executable: LazyLiveCarryHealthExecutable,
        runtime: SimpleNamespace,
        run_dir: Path,
        retained_d03: Any,
    ) -> None:
        self.programs = dict(programs)
        self.health_executable = health_executable
        self.runtime = runtime
        self.run_dir = run_dir
        self.rows: list[dict[str, Any]] = []
        self.step9313_scale_baseline: list[float] | None = None
        self.previous_healthy_carry = retained_d03
        self.previous_healthy_step = 8800
        self.last_healthy_carry = retained_d03
        self.last_healthy_step = 8800

    def __call__(self, name: str, carry: Any, start_step: int, n_steps: int) -> Any:
        if name not in self.programs:
            raise RunnerGateError("CONTINUATION_PROGRAM_DOMAIN", name)
        value = carry
        for offset in range(int(n_steps)):
            step = int(start_step) + offset
            assert_preemption_clear(f"parent-join-{name}-step-{step}-pre")
            before = value
            value = self.programs[name].one_step(value, step)
            if name != "d03":
                continue
            summary = materialize_health(self.health_executable, value, self.runtime)
            gate = evaluate_health_summary(
                step,
                summary,
                step9313_scale_baseline=self.step9313_scale_baseline,
            )
            if step == 9313 and gate["passed"]:
                self.step9313_scale_baseline = list(gate["scale_maxabs"])
            self.rows.append(gate)
            if not gate["passed"]:
                failure = _retain_failure_pair(
                    self.runtime,
                    self.run_dir,
                    last_step=self.last_healthy_step,
                    last_carry=before,
                    failed_step=step,
                    failed_carry=value,
                    failure={"code": "HEALTH_GATE", "gate": gate},
                )
                raise WindowFalsified("HEALTH_GATE", failure["proof_sha256"])
            self.previous_healthy_carry = self.last_healthy_carry
            self.previous_healthy_step = self.last_healthy_step
            self.last_healthy_carry = value
            self.last_healthy_step = step
            if step % 10 == 0 or step in (9313, 9314, 9315, 9405):
                print(
                    f"NESTED_BUNDLE_HEALTH step={step} finite=1 violations=0",
                    flush=True,
                )
        return value


def run_parent_join_stage(
    tree: Any,
    carries: dict[str, Any],
    own_steps: dict[str, int],
    root_steps: int,
    expected_steps: Mapping[str, int],
    *,
    advance: ParentJoinContinuationAdvance,
    output: StandardCadenceOutput,
    cadence: dict[str, int],
    nonintegral: dict[str, tuple[int, ...]],
    runtime: SimpleNamespace,
    label: str,
) -> Any:
    assert_preemption_clear(f"{label}-pre")
    block_between, root_sync_cadence = runtime._nested_sync_mode_from_env()
    result = runtime.run_domain_tree_callbacks(
        tree.hierarchy,
        carries,
        root_steps=int(root_steps),
        advance=advance,
        force=runtime._operational_force,
        feedback=None,
        feedback_enabled=False,
        output=output,
        output_cadence_steps=cadence,
        output_alarm_steps=nonintegral,
        block_between=block_between,
        root_sync_cadence=root_sync_cadence,
        edge_lookup=runtime.ordinary._edge_lookup(tree),
        fused_cascade=None,
        initial_own_steps=own_steps,
    )
    runtime.block_until_ready(tuple(result.states[name].theta for name in tree.hierarchy.order))
    if dict(result.own_steps) != dict(expected_steps):
        raise RunnerGateError(
            f"{label.upper()}_CLOCKS",
            f"expected={dict(expected_steps)} actual={dict(result.own_steps)}",
        )
    for name in tree.hierarchy.order:
        step = int(result.own_steps[name])
        runtime.assert_state_finite_at_boundary(
            result.states[name], domain=name, step=step,
            sim_time_s=step * float(DT_SECONDS[name]),
        )
    return result


def build_terminal_union_and_verdict(
    run_dir: Path,
    continuation_output: Path,
    runtime: SimpleNamespace,
) -> dict[str, Any]:
    """Snapshot immutable prefix + continuation, then run standard 19/19/55 gate."""

    union_dir = run_dir / "gpu-output"
    union_dir.mkdir(parents=True, exist_ok=False)
    source_before = assert_rc3_continuation_source_authority()
    schedules = standard_frame_schedule(TERMINAL_OWN_STEPS)
    prefix_schedules = standard_frame_schedule(PARENT_JOIN_OWN_STEPS)
    snapshots = []
    for name in ("d01", "d02", "d03"):
        prefix = set(prefix_schedules[name])
        for stamp in schedules[name]:
            source_root = RC3_RUN_DIR / "output" if stamp in prefix else continuation_output
            source = source_root / f"wrfout_{name}_{stamp}"
            target = union_dir / source.name
            if not source.is_file():
                raise RunnerGateError("TERMINAL_UNION_SOURCE_MISSING", str(source))
            snapshots.append(runtime.comparator.immutable_file_snapshot(source, target))
    source_after = assert_rc3_continuation_source_authority()
    immutable_keys = (
        "failure_proof", "step8800_carry", "step8800_wrfout", "last_healthy_pair",
        "raw_output_counts", "frame_pair_counts",
    )
    if any(source_before[key] != source_after[key] for key in immutable_keys):
        raise RunnerGateError("TERMINAL_PREFIX_SOURCE_CHANGED", repr(immutable_keys))
    completion = runtime.comparator.validate_gpu_completion_frames(
        union_dir, terminal_mode=True,
    )
    if completion["counts"] != {"d01": 19, "d02": 19, "d03": 55}:
        raise RunnerGateError("TERMINAL_UNION_COUNTS", repr(completion["counts"]))
    terminal_cpu_authority = runtime.comparator.validate_terminal_contract_authority(
        TERMINAL_CONTRACT, interval_seconds=0.0,
    )
    terminal_cache = {
        "cpu": terminal_cpu_authority["terminal_cpu"],
        "gpu": completion,
    }
    pair_state_path = run_dir / "incremental-pairs.json"
    matched = -1
    state: dict[str, Any] | None = None
    for _attempt in range(120):
        state = runtime.comparator.incremental_pair(
            INPUT_DIR,
            union_dir,
            pair_state_path,
            terminal_cache=terminal_cache,
            terminal_mode=True,
        )
        current = int(state["matched_count"])
        if current == 55:
            break
        if current < matched:
            raise RunnerGateError("TERMINAL_PAIR_REGRESSION", repr((matched, current)))
        matched = current
    if state is None or state.get("matched_count") != 55:
        raise RunnerGateError(
            "TERMINAL_PAIR_INCOMPLETE", repr(None if state is None else state.get("matched_count")),
        )
    plot = run_dir / "identity-numbers-first.jpg"
    final = runtime.comparator.final_verdict(
        CPU_MANIFEST,
        pair_state_path,
        plot,
        union_dir,
        terminal_cache=terminal_cache,
    )
    if final.get("verdict") != "PASS" or final.get("matched_frames") != 55:
        raise RunnerGateError("TERMINAL_STANDARD_VERDICT", json.dumps(final, sort_keys=True))
    return {
        "union_dir": str(union_dir.resolve()),
        "snapshot_count": len(snapshots),
        "prefix_source_immutable": True,
        "completion": completion,
        "terminal_cpu_authority": terminal_cpu_authority,
        "incremental_pair_state": {
            "path": str(pair_state_path.resolve()),
            "sha256": sha256_file(pair_state_path),
            "matched_count": state["matched_count"],
            "authority_sha256": state["authority_sha256"],
        },
        "final_verdict": final,
        "identity_plot": {
            "path": str(plot.resolve()),
            "sha256": sha256_file(plot),
            "bytes": plot.stat().st_size,
        },
    }


def build_full_terminal_verdict(
    run_dir: Path,
    output_dir: Path,
    runtime: SimpleNamespace,
) -> dict[str, Any]:
    """Run the canonical terminal comparator on one complete fresh output stream."""

    completion = runtime.comparator.validate_gpu_completion_frames(
        output_dir, terminal_mode=True,
    )
    if completion["counts"] != {"d01": 19, "d02": 19, "d03": 55}:
        raise RunnerGateError("FULL_TERMINAL_COUNTS", repr(completion["counts"]))
    terminal_cpu_authority = runtime.comparator.validate_terminal_contract_authority(
        TERMINAL_CONTRACT, interval_seconds=0.0,
    )
    terminal_cache = {
        "cpu": terminal_cpu_authority["terminal_cpu"],
        "gpu": completion,
    }
    pair_state_path = run_dir / "incremental-pairs.json"
    state: dict[str, Any] | None = None
    matched = -1
    for _attempt in range(120):
        state = runtime.comparator.incremental_pair(
            INPUT_DIR,
            output_dir,
            pair_state_path,
            terminal_cache=terminal_cache,
            terminal_mode=True,
        )
        current = int(state["matched_count"])
        if current == 55:
            break
        if current < matched:
            raise RunnerGateError("FULL_TERMINAL_PAIR_REGRESSION", repr((matched, current)))
        matched = current
    if state is None or state.get("matched_count") != 55:
        raise RunnerGateError(
            "FULL_TERMINAL_PAIR_INCOMPLETE",
            repr(None if state is None else state.get("matched_count")),
        )
    plot = run_dir / "identity-numbers-first.jpg"
    final = runtime.comparator.final_verdict(
        CPU_MANIFEST,
        pair_state_path,
        plot,
        output_dir,
        terminal_cache=terminal_cache,
    )
    diagnostic_wake_admission: dict[str, Any] | None = None
    if final.get("verdict") != "PASS" or final.get("matched_frames") != 55:
        failures = final.get("failures") or []
        failure_fields = {
            str(row.get("field"))
            for row in failures
            if isinstance(row, Mapping)
        }
        failure_gates = {
            str(row.get("gate"))
            for row in failures
            if isinstance(row, Mapping)
        }
        diagnostic_wake_only = bool(
            ALLOW_DIAGNOSTIC_TERMINAL_WAKE_RED
            and final.get("verdict") == "FAIL_IDENTITY"
            and failures
            and failure_fields <= DIAGNOSTIC_TERMINAL_ALLOWED_FIELDS
            and failure_gates == {"STRICT_POOLED_RMSE"}
        )
        if not diagnostic_wake_only:
            raise RunnerGateError(
                "FULL_TERMINAL_STANDARD_VERDICT", json.dumps(final, sort_keys=True),
            )
        state = runtime.comparator.load_exact_json(pair_state_path)
        runtime.comparator._identity_plot(state, plot)
        diagnostic_wake_admission = {
            "classification": "KNOWN_WAKE_RELEASE_BLOCKER_AT_TERMINAL",
            "release_gate_passed": False,
            "waiver_or_tolerance_change": False,
            "isolation_only": True,
            "allowed_failure_fields": sorted(DIAGNOSTIC_TERMINAL_ALLOWED_FIELDS),
            "observed_failure_fields": sorted(failure_fields),
            "observed_failure_gates": sorted(failure_gates),
        }
    return {
        "output_dir": str(output_dir.resolve()),
        "single_fresh_process_stream": True,
        "completion": completion,
        "terminal_cpu_authority": terminal_cpu_authority,
        "incremental_pair_state": {
            "path": str(pair_state_path.resolve()),
            "sha256": sha256_file(pair_state_path),
            "matched_count": state["matched_count"],
            "authority_sha256": state["authority_sha256"],
        },
        "final_verdict": final,
        "diagnostic_wake_admission": diagnostic_wake_admission,
        "identity_plot": {
            "path": str(plot.resolve()),
            "sha256": sha256_file(plot),
            "bytes": plot.stat().st_size,
        },
    }


def _retain_failure_pair(
    runtime: SimpleNamespace,
    run_dir: Path,
    *,
    domain: str = "d03",
    last_step: int,
    last_carry: Any,
    failed_step: int,
    failed_carry: Any,
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    directory = run_dir / "failure"
    last = _retain_complete_carry(
        runtime,
        directory,
        role="last-healthy",
        domain=domain,
        step=last_step,
        carry=last_carry,
    )
    failed = _retain_complete_carry(
        runtime,
        directory,
        role="first-failed",
        domain=domain,
        step=failed_step,
        carry=failed_carry,
    )
    payload = {
        "schema": "gpuwrf.v0234.nested-boundary-final-window-failure.v1",
        "status": "ATOMIC_FAILURE_CAPTURED",
        "domain": domain,
        "failure": dict(failure),
        "last_healthy": last,
        "first_failed": failed,
        "aggregate_sha256": canonical_digest({
            "last": [last["file_sha256"], last["manifest"]["manifest_sha256"]],
            "failed": [failed["file_sha256"], failed["manifest"]["manifest_sha256"]],
        }),
    }
    payload["proof_sha256"] = canonical_digest(payload)
    atomic_write_json(directory / "failure-proof.json", payload)
    return payload


def _run_window(
    tree: Any,
    prefix_carries: dict[str, Any],
    prefix_steps: dict[str, int],
    *,
    output: StandardCadenceOutput,
    cadence: dict[str, int],
    nonintegral: dict[str, tuple[int, ...]],
    executable: Any,
    namelist: Any,
    clock: Any,
    d03_cadence: int,
    health_executable: Any,
    late_frame_writer: Any,
    runtime: SimpleNamespace,
    run_dir: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    normal_advance = runtime._operational_advance_factory(tree)
    block_between, root_sync_cadence = runtime._nested_sync_mode_from_env()
    rows: list[dict[str, Any]] = []
    sampled: list[int] = []
    retained_late_window: dict[str, dict[str, Any]] = {}
    step9313_scale_baseline: list[float] | None = None
    last_healthy_carry = prefix_carries["d03"]
    last_healthy_step = 9198
    previous_healthy_carry = last_healthy_carry
    previous_healthy_step = last_healthy_step

    def advance(name: str, carry: Any, start_step: int, n_steps: int) -> Any:
        nonlocal step9313_scale_baseline
        nonlocal last_healthy_carry, last_healthy_step
        nonlocal previous_healthy_carry, previous_healthy_step
        if name != "d03":
            return normal_advance(name, carry, start_step, n_steps)
        value = carry
        for offset in range(int(n_steps)):
            native_step = int(start_step + offset)
            try:
                assert_preemption_clear(f"window-d03-step-{native_step}-pre")
            except RunnerGateError as exc:
                failure = _retain_failure_pair(
                    runtime,
                    run_dir,
                    last_step=native_step - 1,
                    last_carry=value,
                    failed_step=native_step - 1,
                    failed_carry=value,
                    failure={
                        "code": exc.code,
                        "detail": exc.detail,
                        "dispatch_withheld": True,
                        "no_first_bad_output_exists": True,
                    },
                )
                raise WindowFalsified(exc.code, failure["proof_sha256"]) from exc
            before = value
            value = executable(
                value,
                namelist,
                runtime.jnp.asarray(native_step, dtype=runtime.jnp.int32),
                clock,
                n_steps=1,
                cadence=d03_cadence,
            )
            summary = materialize_health(health_executable, value, runtime)
            gate = evaluate_health_summary(
                native_step,
                summary,
                step9313_scale_baseline=step9313_scale_baseline,
            )
            if native_step == 9313 and gate["passed"]:
                step9313_scale_baseline = list(gate["scale_maxabs"])
            rows.append(gate)
            sampled.append(native_step)
            if not gate["passed"]:
                failure = _retain_failure_pair(
                    runtime,
                    run_dir,
                    last_step=last_healthy_step,
                    last_carry=before,
                    failed_step=native_step,
                    failed_carry=value,
                    failure={"code": "HEALTH_GATE", "gate": gate},
                )
                raise WindowFalsified("HEALTH_GATE", failure["proof_sha256"])
            previous_healthy_carry, previous_healthy_step = last_healthy_carry, last_healthy_step
            last_healthy_carry, last_healthy_step = value, native_step
            if native_step in LATE_WINDOW_RETAIN_D03_STEPS:
                retained_late_window[str(native_step)] = retain_late_window_carry_and_frame(
                    runtime,
                    output.pairer,
                    late_frame_writer,
                    run_dir,
                    step=native_step,
                    carry=value,
                    health=gate,
                )
            if native_step % 10 == 0 or native_step in (9313, 9314, 9315, 9405):
                print(
                    f"NESTED_BUNDLE_HEALTH step={native_step} finite=1 violations=0",
                    flush=True,
                )
        return value

    try:
        result = runtime.run_domain_tree_callbacks(
            tree.hierarchy,
            prefix_carries,
            root_steps=23,
            advance=advance,
            force=runtime._operational_force,
            feedback=None,
            feedback_enabled=False,
            output=output,
            output_cadence_steps=cadence,
            output_alarm_steps=nonintegral,
            block_between=block_between,
            root_sync_cadence=root_sync_cadence,
            edge_lookup=runtime.ordinary._edge_lookup(tree),
            fused_cascade=None,
            initial_own_steps=prefix_steps,
        )
    except MetricGateFailure as exc:
        failed = output.failed_carry or last_healthy_carry
        failed_step = output.failed_step or last_healthy_step
        failed_domain = output.failed_domain or "d03"
        failure = _retain_failure_pair(
            runtime,
            run_dir,
            domain=failed_domain,
            last_step=previous_healthy_step,
            last_carry=previous_healthy_carry,
            failed_step=failed_step,
            failed_carry=failed,
            failure={"code": exc.code, "detail": exc.detail},
        )
        raise WindowFalsified(exc.code, failure["proof_sha256"]) from exc
    runtime.block_until_ready(tuple(result.states[name].theta for name in tree.hierarchy.order))
    if dict(result.own_steps) != WINDOW_OWN_STEPS:
        raise RunnerGateError("WINDOW_CLOCKS", repr(result.own_steps))
    if sampled != list(WINDOW_D03_STEPS):
        raise RunnerGateError("WINDOW_SAMPLING", f"first={sampled[:2]} last={sampled[-2:]} count={len(sampled)}")
    proof = {
        "own_steps": dict(result.own_steps),
        "sampled_steps": sampled,
        "exact_9199_through_9405": True,
        "health_rows": rows,
        "step9313_scale_baseline": step9313_scale_baseline,
        "model_dispatches": len(sampled),
        "health_materialization": "after each completed d03 dispatch, outside production HLO",
        "retained_late_window": retained_late_window,
        "normal_output_counts": dict(output.pairer.counts),
    }
    live = {
        "last_healthy_carry": last_healthy_carry,
        "last_healthy_step": last_healthy_step,
        "previous_healthy_carry": previous_healthy_carry,
        "previous_healthy_step": previous_healthy_step,
    }
    return result, proof, live


def _run_authorized_continuation(
    carries: dict[str, Any],
    *,
    tree: Any,
    own_steps: dict[str, int],
    cadence: dict[str, int],
    nonintegral: dict[str, tuple[int, ...]],
    executable: Any,
    namelist: Any,
    clock: Any,
    d03_cadence: int,
    health_executable: Any,
    output: StandardCadenceOutput,
    runtime: SimpleNamespace,
    run_dir: Path,
) -> Any:
    """Same-process terminal continuation, retaining the first new red carry."""

    normal_advance = runtime._operational_advance_factory(tree)
    last_healthy_carry = carries["d03"]
    last_healthy_step = int(own_steps["d03"])

    def advance(name: str, carry: Any, start_step: int, n_steps: int) -> Any:
        nonlocal last_healthy_carry, last_healthy_step
        if name != "d03":
            return normal_advance(name, carry, start_step, n_steps)
        value = carry
        for offset in range(int(n_steps)):
            step = int(start_step + offset)
            assert_preemption_clear(f"authorized-continuation-d03-step-{step}-pre")
            value = executable(
                value, namelist, runtime.jnp.asarray(step, dtype=runtime.jnp.int32),
                clock, n_steps=1, cadence=d03_cadence,
            )
            summary = materialize_health(health_executable, value, runtime)
            if sum(int(v) for v in summary["nonfinite_by_leaf"]):
                failure = _retain_failure_pair(
                    runtime,
                    run_dir,
                    last_step=last_healthy_step,
                    last_carry=last_healthy_carry,
                    failed_step=step,
                    failed_carry=value,
                    failure={
                        "code": "CONTINUATION_NONFINITE",
                        "nonfinite_by_leaf": [
                            int(v) for v in summary["nonfinite_by_leaf"]
                        ],
                    },
                )
                raise WindowFalsified(
                    "CONTINUATION_NONFINITE", failure["proof_sha256"]
                )
            last_healthy_carry, last_healthy_step = value, step
        return value

    block_between, root_sync_cadence = runtime._nested_sync_mode_from_env()
    try:
        result = runtime.run_domain_tree_callbacks(
            tree.hierarchy,
            carries,
            root_steps=TERMINAL_OWN_STEPS["d01"] - own_steps["d01"],
            advance=advance,
            force=runtime._operational_force,
            feedback=None,
            feedback_enabled=False,
            output=output,
            output_cadence_steps=cadence,
            output_alarm_steps=nonintegral,
            block_between=block_between,
            root_sync_cadence=root_sync_cadence,
            edge_lookup=runtime.ordinary._edge_lookup(tree),
            fused_cascade=None,
            initial_own_steps=own_steps,
        )
    except MetricGateFailure as exc:
        domain = output.failed_domain or "d03"
        failed_carry = output.failed_carry
        failed_step = output.failed_step
        if failed_carry is None or failed_step is None:
            raise
        prior_carry = output.previous_alarm_carry.get(
            domain, carries[domain]
        )
        prior_step = output.previous_alarm_step.get(
            domain, int(own_steps[domain])
        )
        failure = _retain_failure_pair(
            runtime,
            run_dir,
            domain=domain,
            last_step=int(prior_step),
            last_carry=prior_carry,
            failed_step=int(failed_step),
            failed_carry=failed_carry,
            failure={"code": exc.code, "detail": exc.detail, "metric_gate": True},
        )
        raise WindowFalsified(exc.code, failure["proof_sha256"]) from exc
    if dict(result.own_steps) != TERMINAL_OWN_STEPS:
        raise RunnerGateError("CONTINUATION_CLOCKS", repr(result.own_steps))
    expected_counts = {name: len(values) for name, values in standard_output_steps(TERMINAL_OWN_STEPS).items()}
    if output.pairer.counts != expected_counts:
        raise RunnerGateError(
            "CONTINUATION_OUTPUT_COUNTS",
            f"expected={expected_counts} actual={output.pairer.counts}",
        )
    return result


def execute_after_preflight(preflight: Callable[[], Any], importer: Callable[[], Any]) -> Any:
    """Small test seam proving failed admission cannot reach the runtime import hook."""

    authority = preflight()
    return authority, importer()


def _cpu_dry_run(
    args: argparse.Namespace,
    authority: Mapping[str, Any],
) -> int:
    if "jax" in sys.modules or any(name.startswith("gpuwrf") for name in sys.modules):
        raise RunnerGateError("CPU_DRY_IMPORT_PRE", "JAX/gpuwrf already imported")
    schedule = schedule_clock_oracle()
    static = static_source_audit()
    launch = audit_exact_launch_command(LAUNCH_COMMAND)
    if not schedule["passed"] or not static["passed"] or not launch["passed"]:
        raise RunnerGateError(
            "CPU_STATIC_AUDIT",
            repr({"schedule": schedule, "static": static, "launch": launch}),
        )
    focused = {
        "command": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *CPU_FOCUSED_TEST_ARGS,
        ],
        "returncode": 0,
        "stdout_sha256": None,
        "stdout_tail": None,
    }
    if args.run_focused_tests:
        completed = subprocess.run(
            focused["command"], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        normalized_stdout = normalize_pytest_output(completed.stdout)
        focused.update({
            "returncode": completed.returncode,
            "stdout_sha256": sha256_text(normalized_stdout),
            "stdout_tail": normalized_stdout[-4000:],
            "stdout_normalization": "pytest elapsed seconds replaced with <elapsed>s",
        })
        if completed.returncode:
            raise RunnerGateError("FOCUSED_TESTS", completed.stdout[-4000:])
    if "jax" in sys.modules or any(name.startswith("gpuwrf") for name in sys.modules):
        raise RunnerGateError("CPU_DRY_IMPORT_POST", "JAX/gpuwrf imported during CPU audit")
    proof = {
        "schema": CPU_PROOF_SCHEMA,
        "verdict": AUDIT_ADMISSION,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_tree": CANDIDATE_TREE,
        "runner_head_at_audit": authority["candidate"]["runner_head"],
        "runner_source_sha256": sha256_file(RUNNER_SOURCE),
        "candidate_clean_authority": CANDIDATE_CLEAN_AUTHORITY,
        "authority": authority,
        "schedule_clock_oracle": schedule,
        "static_audit": static,
        "exact_launch_audit": launch,
        "focused_tests": focused,
        "exact_launch_command_path": str(LAUNCH_COMMAND.resolve()),
        "exact_launch_command_sha256": sha256_file(LAUNCH_COMMAND),
        "gpu_commands_run": 0,
        "gpu_queries_run": 0,
        "gpu_lock_acquired_or_verified": False,
        "model_imported_or_executed": False,
        "jax_imported": False,
        "standard_output_contract": {
            "window_counts": schedule["window_output_counts"],
            "terminal_counts": schedule["terminal_output_counts"],
            "initial_history_explicit": True,
            "incremental_pairing_synchronous_after_normal_writer": True,
            "first_progress_d03_steps": list(FIRST_PROGRESS_D03_STEPS),
            "decisive_d03_step": DECISIVE_D03_STEP,
            "frozen_1500_rmse": FROZEN_1500_RMSE,
            "early_causal_ring1_fields": ["T", "U"],
            "early_causal_baseline": EARLY_CAUSAL_RING1_BASELINE,
            "early_causal_u_policy": (
                "retain the 2c source mechanism and remain better than 4484 "
                "against CPU WRF and Retry20"
            ),
        },
        "cpu_dry_run": True,
        "deterministic_payload": True,
    }
    proof["proof_sha256"] = canonical_digest(proof)
    atomic_write_json(args.proof_output, proof)
    print(json.dumps({
        "verdict": proof["verdict"],
        "proof": str(args.proof_output),
        "proof_sha256": proof["proof_sha256"],
        "gpu_commands_run": 0,
        "jax_imported": False,
    }, sort_keys=True), flush=True)
    return 0


def _lower_only_custom_call_diagnostic(
    args: argparse.Namespace,
    authority: dict[str, Any],
    runtime: SimpleNamespace,
) -> int:
    if args.run_dir.name != CUSTOM_CALL_DIAGNOSTIC_NAMESPACE:
        raise RunnerGateError(
            "DIAGNOSTIC_NAMESPACE",
            f"expected={CUSTOM_CALL_DIAGNOSTIC_NAMESPACE} actual={args.run_dir.name}",
        )
    authority["cuda_runtime"] = _assert_cuda_runtime(runtime)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    tree, names, initial_carries, dt_by_domain, load_authority = runtime.ordinary.load_corrected_tree(
        args.run_dir,
    )
    if names != ("d01", "d02", "d03") or dt_by_domain != {
        "d01": 54.0, "d02": 18.0, "d03": 6.0,
    }:
        raise RunnerGateError("DIAGNOSTIC_DOMAIN_AUTHORITY", repr((names, dt_by_domain)))
    assert_preemption_clear("diagnostic-pre-lower")
    _lowered, _namelist, _clock, _cadence, lower_audit = _lower_d03_once(
        tree,
        initial_carries["d03"],
        runtime,
        artifact_path=args.run_dir / "ordinary-one-step-lowered-hlo.json",
    )
    if lower_audit["lower_calls"] != 1 or lower_audit["compile_calls"] != 0:
        raise RunnerGateError("DIAGNOSTIC_COMPILE_COUNT", repr(lower_audit))
    authority["inputs_post"] = assert_input_retry20_cache_authority(os.environ)
    authority["lock_post"] = assert_live_lock_authority(os.environ)
    authority["preemption_post"] = assert_preemption_clear("diagnostic-complete")
    proof = {
        "schema": "gpuwrf.v0234.nested-boundary-final-custom-call-lower-only.v1",
        "verdict": "LOWER_ONLY_CUSTOM_CALL_DIAGNOSTIC_COMPLETE",
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_tree": CANDIDATE_TREE,
        "authority": authority,
        "load_authority": load_authority,
        "lower_audit": lower_audit,
        "lower_calls": 1,
        "compile_calls": 0,
        "model_dispatches": 0,
        "health_lowers_or_compiles": 0,
        "diagnostic_namespace": str(args.run_dir),
    }
    proof["proof_sha256"] = canonical_digest(proof)
    atomic_write_json(args.proof_output, proof)
    print(json.dumps({
        "verdict": proof["verdict"],
        "custom_call_targets": lower_audit["hlo_policy"]["custom_call_targets"],
        "policy_passed": lower_audit["hlo_policy"]["passed"],
        "compile_calls": 0,
        "model_dispatches": 0,
        "artifact": lower_audit["lowered_hlo_artifact"]["path"],
        "artifact_sha256": lower_audit["lowered_hlo_artifact"]["file_sha256"],
        "proof_sha256": proof["proof_sha256"],
    }, sort_keys=True), flush=True)
    return 0


def _parent_join_runtime_main(
    args: argparse.Namespace,
    authority: dict[str, Any],
    runtime: SimpleNamespace,
) -> int:
    def stage(value: str) -> None:
        args._parent_join_stage = value
        stages = getattr(args, "_parent_join_stages", None)
        if stages is None:
            stages = []
            args._parent_join_stages = stages
        if not stages or stages[-1] != value:
            stages.append(value)

    started = datetime.now(timezone.utc)
    stage("CUDA_RUNTIME")
    authority["cuda_runtime"] = _assert_cuda_runtime(runtime)
    stage("RUNTIME_CACHE_DISABLED")
    authority["runtime_cache"] = assert_runtime_cache_disabled(runtime)
    stage("NAMESPACE_CREATE")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    stage("RC3_SOURCE_AUTHORITY")
    source_before = assert_rc3_continuation_source_authority()
    stage("DOMAIN_LOAD")
    load_started = time.perf_counter()
    tree, names, initial_carries, dt_by_domain, load_authority = (
        runtime.ordinary.load_corrected_tree(args.run_dir)
    )
    load_wall_seconds = time.perf_counter() - load_started
    if names != ("d01", "d02", "d03") or dt_by_domain != {
        "d01": 54.0, "d02": 18.0, "d03": 6.0,
    }:
        raise RunnerGateError("PARENT_JOIN_DOMAIN_AUTHORITY", repr((names, dt_by_domain)))
    cadence, nonintegral, scheduler = runtime.ordinary.scheduler_contract(
        names, dt_by_domain, total_steps=TERMINAL_OWN_STEPS,
    )

    stage("PARENT_RECONSTRUCTION")
    parent_carries, parent_steps, parent_programs, parent_proof = (
        run_parent_only_reconstitution(
            tree, initial_carries, dt_by_domain, runtime, args.run_dir,
            stage_fn=stage,
        )
    )
    if parent_proof["d03_dispatches_before_join"] != 0:
        raise RunnerGateError(
            "PARENT_PREJOIN_D03_DISPATCH_COUNT",
            str(parent_proof["d03_dispatches_before_join"]),
        )
    stage("RETAINED_D03_AUTHENTICATION")
    retained_d03, retained_d03_proof = load_authenticated_retained_d03_carry(
        runtime, args.run_dir,
    )
    d03_specs = tuple(
        spec for spec in tree.hierarchy.nests
        if spec.parent == "d02" and spec.child == "d03"
    )
    if len(d03_specs) != 1:
        raise RunnerGateError("PARENT_JOIN_D03_EDGE", repr(d03_specs))
    edge = runtime.ordinary._edge_lookup(tree)(d03_specs[0])
    stage("BOUNDARY_TARGET_JOIN")
    assert_preemption_clear("parent-join-target-build-pre")
    rebuilt_d03 = runtime._operational_force(
        edge, parent_carries["d02"], retained_d03,
    )
    target_identity = compare_join_boundary_target_records(
        retained_d03, rebuilt_d03, runtime, edge=edge,
    )
    del rebuilt_d03
    if not target_identity["passed"]:
        raise RunnerGateError(
            "PARENT_JOIN_TARGET_IDENTITY", json.dumps(target_identity, sort_keys=True),
        )
    if parent_steps != {"d01": 978, "d02": 2934}:
        raise RunnerGateError("PARENT_JOIN_CLOCKS", repr(parent_steps))
    join_carries = {
        "d01": parent_carries["d01"],
        "d02": parent_carries["d02"],
        "d03": retained_d03,
    }
    join_steps = dict(PARENT_JOIN_OWN_STEPS)
    authenticated_join_carries = {
        name: _retain_complete_carry(
            runtime,
            args.run_dir / "authenticated-join-carries",
            role="parent-join",
            domain=name,
            step=join_steps[name],
            carry=join_carries[name],
        )
        for name in names
    }
    source_at_join = assert_rc3_continuation_source_authority()
    source_keys = (
        "failure_proof", "step8800_carry", "step8800_wrfout", "last_healthy_pair",
        "raw_output_counts", "frame_pair_counts",
    )
    if any(source_before[key] != source_at_join[key] for key in source_keys):
        raise RunnerGateError("PARENT_JOIN_SOURCE_CHANGED", repr(source_keys))
    print(
        "PARENT_JOIN_GREEN "
        "d01_step=978 d02_step=2934 d03_step=8800 "
        "parent_frames=30 target_leaves=11 d03_prejoin_dispatches=0",
        flush=True,
    )

    stage("RETAINED_PREFIX_PAIRING")
    continuation_output = args.run_dir / "continuation-output"
    continuation_output.mkdir(parents=True, exist_ok=False)
    writer = runtime._PerDomainWrfoutWriter(
        output_dir=continuation_output,
        input_dir=INPUT_DIR,
        run_start=RUN_START,
        bundles=tree.domains,
        output_cadence_steps=cadence,
        dt_by_domain=dt_by_domain,
        async_writer=None,
        output_pipeline=None,
    )
    pairer = IncrementalFramePairer(runtime, args.run_dir / "frame-pairs")
    retained_prefix_pairs = seed_retained_prefix_pairs(pairer)
    health_executable = LazyLiveCarryHealthExecutable(runtime)
    output = StandardCadenceOutput(
        writer,
        pairer,
        continuation_output,
        health_executable,
        runtime,
        args.run_dir,
    )
    d03_program = LazyOneStepDomainExecutable(
        tree,
        "d03",
        runtime,
        args.run_dir / "ordinary-one-step-d03-lowered-hlo.json",
    )
    programs = {**parent_programs, "d03": d03_program}
    advance = ParentJoinContinuationAdvance(
        programs, health_executable, runtime, args.run_dir, retained_d03,
    )

    stage("D03_CATCHUP_8801_8802")
    assert_preemption_clear("parent-join-d03-catchup-pre")
    join_carries["d03"] = advance(
        "d03", join_carries["d03"], PARENT_CATCHUP_D03_STEPS[0],
        len(PARENT_CATCHUP_D03_STEPS),
    )
    runtime.block_until_ready(join_carries["d03"].state.theta)
    aligned_steps = dict(PARENT_ALIGNED_OWN_STEPS)
    aligned_carry = _retain_complete_carry(
        runtime,
        args.run_dir / "authenticated-join-carries",
        role="aligned",
        domain="d03",
        step=aligned_steps["d03"],
        carry=join_carries["d03"],
    )

    stage9000_expected = {"d01": 1000, "d02": 3000, "d03": 9000}
    stage("COUPLED_STAGE_9000")
    stage9000 = run_parent_join_stage(
        tree,
        join_carries,
        aligned_steps,
        PARENT_STAGE_9000_ROOT_STEPS,
        stage9000_expected,
        advance=advance,
        output=output,
        cadence=cadence,
        nonintegral=nonintegral,
        runtime=runtime,
        label="parent_join_9000",
    )
    decisive_rows = [
        row for row in pairer.rows
        if row["domain"] == "d03" and row["own_step"] == DECISIVE_D03_STEP
    ]
    if (
        pairer.counts != {"d01": 16, "d02": 16, "d03": 46}
        or len(decisive_rows) != 1
        or not decisive_rows[0].get("decisive_1500")
        or decisive_rows[0]["decisive_1500"].get("passed") is not True
    ):
        raise RunnerGateError(
            "PARENT_JOIN_9000_GATE",
            repr({"counts": pairer.counts, "decisive": decisive_rows}),
        )
    print(
        "PARENT_JOIN_9000_GREEN "
        f"V10_RMSE={decisive_rows[0]['strict_rmse'].get('V10')}",
        flush=True,
    )

    stage("COUPLED_STAGE_9405")
    stage9405 = run_parent_join_stage(
        tree,
        stage9000.carries,
        dict(stage9000.own_steps),
        PARENT_STAGE_9405_ROOT_STEPS,
        WINDOW_OWN_STEPS,
        advance=advance,
        output=output,
        cadence=cadence,
        nonintegral=nonintegral,
        runtime=runtime,
        label="parent_join_9405",
    )
    step9405_rows = [row for row in advance.rows if row["step"] == 9405]
    if (
        pairer.counts != {"d01": 16, "d02": 16, "d03": 48}
        or len(step9405_rows) != 1
        or step9405_rows[0]["passed"] is not True
    ):
        raise RunnerGateError(
            "PARENT_JOIN_9405_GATE",
            repr({"counts": pairer.counts, "health": step9405_rows}),
        )
    print("PARENT_JOIN_9405_GREEN finite=1 violations=0", flush=True)

    stage("COUPLED_STAGE_TERMINAL")
    terminal = run_parent_join_stage(
        tree,
        stage9405.carries,
        dict(stage9405.own_steps),
        PARENT_STAGE_TERMINAL_ROOT_STEPS,
        TERMINAL_OWN_STEPS,
        advance=advance,
        output=output,
        cadence=cadence,
        nonintegral=nonintegral,
        runtime=runtime,
        label="parent_join_terminal",
    )
    if pairer.counts != {"d01": 19, "d02": 19, "d03": 55}:
        raise RunnerGateError("PARENT_JOIN_TERMINAL_COUNTS", repr(pairer.counts))
    terminal_carries = {
        name: _retain_complete_carry(
            runtime,
            args.run_dir / "terminal-carries",
            role="terminal",
            domain=name,
            step=TERMINAL_OWN_STEPS[name],
            carry=terminal.carries[name],
        )
        for name in names
    }
    program_audits = {name: programs[name].audit() for name in names}
    if any(row["compile_calls"] != 1 for row in program_audits.values()):
        raise RunnerGateError("PARENT_JOIN_COMPILE_COUNTS", repr(program_audits))
    health_audit = health_executable.audit()
    stage("TERMINAL_UNION_AND_IDENTITY")
    terminal_identity = build_terminal_union_and_verdict(
        args.run_dir, continuation_output, runtime,
    )
    source_after = assert_rc3_continuation_source_authority()
    if any(source_before[key] != source_after[key] for key in source_keys):
        raise RunnerGateError("PARENT_JOIN_SOURCE_CHANGED_FINAL", repr(source_keys))
    authority["inputs_post"] = assert_input_retry20_cache_authority(os.environ)
    authority["lock_post"] = assert_live_lock_authority(os.environ)
    authority["preemption_post"] = assert_preemption_clear("parent-join-complete")
    finished = datetime.now(timezone.utc)
    stage("FINAL_PROOF")
    proof = {
        "schema": "gpuwrf.v0234.nested-boundary-parent-join-terminal.v1",
        "verdict": "PARENT_JOIN_TERMINAL_GREEN",
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_tree": CANDIDATE_TREE,
        "authority": authority,
        "source_authority_before": source_before,
        "source_authority_at_join": source_at_join,
        "source_authority_after": source_after,
        "load_authority": load_authority,
        "scheduler": scheduler,
        "parent_reconstitution": parent_proof,
        "retained_d03": retained_d03_proof,
        "boundary_target_join": target_identity,
        "authenticated_join_carries": authenticated_join_carries,
        "aligned_d03_carry": aligned_carry,
        "retained_prefix_pairs": retained_prefix_pairs,
        "stage_9000": {
            "own_steps": dict(stage9000.own_steps),
            "decisive": decisive_rows[0],
        },
        "stage_9405": {
            "own_steps": dict(stage9405.own_steps),
            "health": step9405_rows[0],
        },
        "terminal": {
            "own_steps": dict(terminal.own_steps),
            "frame_pair_counts": dict(pairer.counts),
            "complete_carries": terminal_carries,
            "identity": terminal_identity,
        },
        "programs": program_audits,
        "health_program": health_audit,
        "d03_health": {
            "first_step": advance.rows[0]["step"],
            "last_step": advance.rows[-1]["step"],
            "row_count": len(advance.rows),
            "all_passed": all(row["passed"] for row in advance.rows),
            "materialized_only_after_completed_dispatch": True,
        },
        "scope": {
            "parent_reconstitution_runs": 1,
            "d03_prefix_dispatches": 0,
            "d01_one_step_compiles_or_loads": 1,
            "d02_one_step_compiles_or_loads": 1,
            "d03_one_step_compiles_or_loads": 1,
            "health_compiles": 1,
            "model_or_numerical_edit": False,
            "scientific_output_substitution": False,
            "retained_prefix_snapshotted_not_rewritten": True,
            "manager_waits": 0,
            "gpu_queries": 0,
        },
        "stage_trace": list(args._parent_join_stages),
        "mechanism_separation": {
            "ni": "nested-boundary candidate gate",
            "v10": "independent strict metric report",
            "common_root_proved": False,
        },
        "timing": {
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "wall_seconds": (finished - started).total_seconds(),
            "domain_load_wall_seconds": load_wall_seconds,
        },
    }
    proof["proof_sha256"] = canonical_digest(proof)
    atomic_write_json(args.proof_output, proof)
    print(json.dumps({
        "verdict": proof["verdict"],
        "proof": str(args.proof_output),
        "proof_sha256": proof["proof_sha256"],
        "counts": pairer.counts,
        "identity_plot": terminal_identity["identity_plot"],
    }, sort_keys=True), flush=True)
    return 0


def _retain_parent_join_blocker(
    args: argparse.Namespace,
    code: str,
    detail: str,
    *,
    traceback_text: str,
) -> dict[str, Any]:
    blocker = {
        "schema": "gpuwrf.v0234.nested-boundary-parent-join-blocker.v1",
        "verdict": "PARENT_JOIN_BLOCKED",
        "failure_code": code,
        "detail": detail,
        "stage": getattr(args, "_parent_join_stage", "UNSET"),
        "stage_trace": list(getattr(args, "_parent_join_stages", [])),
        "traceback": traceback_text,
        "candidate_commit": CANDIDATE_COMMIT,
        "model_or_numerical_edit": False,
        "namespace_preserved": str(args.run_dir),
    }
    blocker["proof_sha256"] = canonical_digest(blocker)
    blocker_path = args.run_dir / "parent-join-blocker.json"
    if not blocker_path.exists() and not blocker_path.is_symlink():
        atomic_write_json(blocker_path, blocker)
    print(json.dumps({
        "verdict": blocker["verdict"],
        "failure_code": code,
        "detail": detail,
        "stage": blocker["stage"],
        "proof": str(blocker_path),
        "proof_sha256": blocker["proof_sha256"],
    }, sort_keys=True), flush=True)
    return blocker


def _retain_full_run_blocker(
    args: argparse.Namespace,
    code: str,
    detail: str,
    *,
    traceback_text: str,
) -> dict[str, Any]:
    blocker = {
        "schema": "gpuwrf.v0234.nested-boundary-full18h-blocker.v1",
        "verdict": "FULL_18H_BLOCKED",
        "failure_code": code,
        "detail": detail,
        "stage": getattr(args, "_full_run_stage", "UNSET"),
        "stage_trace": list(getattr(args, "_full_run_stages", [])),
        "traceback": traceback_text,
        "candidate_commit": CANDIDATE_COMMIT,
        "model_or_numerical_edit": False,
        "namespace_preserved": str(args.run_dir),
    }
    blocker["proof_sha256"] = canonical_digest(blocker)
    blocker_path = args.run_dir / "full-run-blocker.json"
    if not blocker_path.exists() and not blocker_path.is_symlink():
        atomic_write_json(blocker_path, blocker)
    print(json.dumps({
        "verdict": blocker["verdict"],
        "failure_code": code,
        "detail": detail,
        "stage": blocker["stage"],
        "proof": str(blocker_path),
        "proof_sha256": blocker["proof_sha256"],
    }, sort_keys=True), flush=True)
    return blocker


def _runtime_main(args: argparse.Namespace, authority: dict[str, Any], runtime: SimpleNamespace) -> int:
    def stage(value: str) -> None:
        args._full_run_stage = value
        stages = getattr(args, "_full_run_stages", None)
        if stages is None:
            stages = []
            args._full_run_stages = stages
        if not stages or stages[-1] != value:
            stages.append(value)

    started = datetime.now(timezone.utc)
    stage("CUDA_RUNTIME")
    authority["cuda_runtime"] = _assert_cuda_runtime(runtime)
    stage("RUNTIME_CACHE_DISABLED")
    authority["runtime_cache"] = assert_runtime_cache_disabled(runtime)
    stage("NAMESPACE_CREATE")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    # The canonical terminal comparator is path-bound to ``gpu-output``.
    # Prior direct runs stopped before reaching this latent tooling gate.
    output_dir = args.run_dir / "gpu-output"
    output_dir.mkdir(parents=True, exist_ok=False)
    stage("DOMAIN_LOAD")
    load_started = time.perf_counter()
    tree, names, initial_carries, dt_by_domain, load_authority = runtime.ordinary.load_corrected_tree(args.run_dir)
    load_wall_seconds = time.perf_counter() - load_started
    if names != ("d01", "d02", "d03") or dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
        raise RunnerGateError("DOMAIN_AUTHORITY", f"names={names} dt={dt_by_domain}")
    cadence, nonintegral, schedule = runtime.ordinary.scheduler_contract(
        names, dt_by_domain, total_steps=TERMINAL_OWN_STEPS,
    )
    stage("D03_LOWER_COMPILE")
    assert_preemption_clear("pre-lower")
    executable, namelist, clock, d03_cadence, ordinary_audit = _lower_compile_d03_once(
        tree,
        initial_carries["d03"],
        runtime,
        artifact_path=args.run_dir / "ordinary-one-step-lowered-hlo.json",
    )
    health_executable = LazyLiveCarryHealthExecutable(runtime)
    print(
        "NESTED_BOUNDARY_COMPILE_COMPLETE "
        f"ordinary_seconds={ordinary_audit['compile_wall_seconds']:.6f} "
        "health_seconds=deferred_to_first_live_carry",
        flush=True,
    )
    assert_preemption_clear("post-compile-pre-initial-output")
    writer = runtime._PerDomainWrfoutWriter(
        output_dir=output_dir,
        input_dir=INPUT_DIR,
        run_start=RUN_START,
        bundles=tree.domains,
        output_cadence_steps=cadence,
        dt_by_domain=dt_by_domain,
        async_writer=None,
        output_pipeline=None,
    )
    pairer = IncrementalFramePairer(
        runtime,
        args.run_dir / "frame-pairs",
        record_known_1500_v10_red=args.record_known_1500_v10_red,
    )
    output = StandardCadenceOutput(
        writer,
        pairer,
        output_dir,
        health_executable,
        runtime,
        args.run_dir,
    )
    late_frame_dir = args.run_dir / "late-window-frames"
    late_frame_dir.mkdir(parents=True, exist_ok=False)
    late_frame_writer = runtime._PerDomainWrfoutWriter(
        output_dir=late_frame_dir,
        input_dir=INPUT_DIR,
        run_start=RUN_START,
        bundles=tree.domains,
        output_cadence_steps=cadence,
        dt_by_domain=dt_by_domain,
        async_writer=None,
        output_pipeline=None,
    )
    stage("INITIAL_HISTORY")
    try:
        initial_history = runtime._emit_initial_history_frames(
            output, names, initial_carries, enabled=True,
        )
    except MetricGateFailure as exc:
        domain = output.failed_domain or "d03"
        failed = output.failed_carry or initial_carries[domain]
        failed_step = output.failed_step or 0
        failure = _retain_failure_pair(
            runtime,
            args.run_dir,
            domain=domain,
            last_step=0,
            last_carry=initial_carries[domain],
            failed_step=failed_step,
            failed_carry=failed,
            failure={"code": exc.code, "detail": exc.detail, "initial_history": True},
        )
        raise WindowFalsified(exc.code, failure["proof_sha256"]) from exc
    stage(f"PREFIX_TO_{PREFIX_OWN_STEPS['d03']}")
    prefix_carries, prefix_steps, prefix_proof = _run_prefix(
        tree,
        names,
        initial_carries,
        dt_by_domain,
        output=output,
        cadence=cadence,
        nonintegral=nonintegral,
        runtime=runtime,
        run_dir=args.run_dir,
    )
    prefix_health_raw = materialize_health(health_executable, prefix_carries["d03"], runtime)
    health_audit = health_executable.audit()
    prefix_terminal_step = int(prefix_steps["d03"])
    prefix_health = evaluate_health_summary(
        prefix_terminal_step,
        prefix_health_raw,
        step9313_scale_baseline=None,
    )
    if not prefix_health["passed"] or prefix_steps != PREFIX_OWN_STEPS:
        last_carry = output.last_alarm_carry.get("d03", prefix_carries["d03"])
        last_step = output.last_alarm_step.get("d03", prefix_terminal_step)
        failure = _retain_failure_pair(
            runtime,
            args.run_dir,
            last_step=int(last_step),
            last_carry=last_carry,
            failed_step=prefix_terminal_step,
            failed_carry=prefix_carries["d03"],
            failure={
                "code": f"PREFIX_{prefix_terminal_step}_HEALTH",
                "violations": prefix_health["violations"],
                "clocks": prefix_steps,
            },
        )
        raise WindowFalsified(
            f"PREFIX_{prefix_terminal_step}_HEALTH",
            failure["proof_sha256"],
        )
    expected_prefix_counts = {
        name: len(values) for name, values in standard_output_steps(PREFIX_OWN_STEPS).items()
    }
    if pairer.counts != expected_prefix_counts:
        raise RunnerGateError(
            "PREFIX_OUTPUT_COUNTS",
            f"expected={expected_prefix_counts} actual={pairer.counts}",
        )
    if set(output.checkpoints) != {str(step) for step in CHECKPOINT_D03_STEPS}:
        raise RunnerGateError("CHECKPOINT_INVENTORY", repr(output.checkpoints))
    profile_prefix_terminal = globals().get("PROFILE_PREFIX_TERMINAL_HANDLER")
    if callable(profile_prefix_terminal):
        return int(profile_prefix_terminal(
            args=args,
            authority=authority,
            runtime=runtime,
            tree=tree,
            names=names,
            initial_carries=initial_carries,
            dt_by_domain=dt_by_domain,
            load_authority=load_authority,
            cadence=cadence,
            nonintegral=nonintegral,
            schedule=schedule,
            ordinary_audit=ordinary_audit,
            health_executable=health_executable,
            output=output,
            pairer=pairer,
            prefix_carries=prefix_carries,
            prefix_steps=prefix_steps,
            prefix_proof=prefix_proof,
            prefix_health=prefix_health,
            initial_history=initial_history,
            started=started,
            load_wall_seconds=load_wall_seconds,
        ))
    window_inputs = prefix_carries
    continuity = assert_same_carry_continuity(prefix_carries, window_inputs)
    assert_preemption_clear("pre-window")
    stage("WINDOW_9199_9405")
    result, window_proof, live = _run_window(
        tree,
        window_inputs,
        prefix_steps,
        output=output,
        cadence=cadence,
        nonintegral=nonintegral,
        executable=executable,
        namelist=namelist,
        clock=clock,
        d03_cadence=d03_cadence,
        health_executable=health_executable,
        late_frame_writer=late_frame_writer,
        runtime=runtime,
        run_dir=args.run_dir,
    )
    expected_window_counts = {
        name: len(values) for name, values in standard_output_steps(WINDOW_OWN_STEPS).items()
    }
    if pairer.counts != expected_window_counts:
        raise RunnerGateError(
            "WINDOW_OUTPUT_COUNTS",
            f"expected={expected_window_counts} actual={pairer.counts}",
        )
    decisive_rows = [
        row for row in pairer.rows
        if row["domain"] == "d03" and row["own_step"] == DECISIVE_D03_STEP
    ]
    decisive_contract_gate = bool(
        len(decisive_rows) == 1
        and (
            (
                args.record_known_1500_v10_red
                and decisive_rows[0]["decisive_1500"].get("passed") is False
                and decisive_rows[0].get("scientific_pair_pass") is False
                and decisive_rows[0].get("recorded_known_v10_red_and_continued") is True
                and (decisive_rows[0].get("known_1500_v10_record") or {}).get("passed")
                is True
            )
            or (
                not args.record_known_1500_v10_red
                and decisive_rows[0]["decisive_1500"].get("passed") is True
            )
        )
    )
    if not decisive_contract_gate:
        raise RunnerGateError("DECISIVE_1500_GATE", repr(decisive_rows))
    if set(window_proof["retained_late_window"]) != {
        str(step) for step in LATE_WINDOW_RETAIN_D03_STEPS
    }:
        raise RunnerGateError(
            "LATE_WINDOW_RETAIN_INVENTORY",
            repr(sorted(window_proof["retained_late_window"])),
        )
    green_carries = {
        name: (
            window_proof["retained_late_window"]["9405"]["carry"]
            if name == "d03"
            else _retain_complete_carry(
                runtime,
                args.run_dir / "green-carries",
                role="green",
                domain=name,
                step=int(result.own_steps[name]),
                carry=result.carries[name],
            )
        )
        for name in names
    }
    inputs_post = assert_input_retry20_cache_authority(os.environ)
    authority["inputs_post"] = inputs_post
    authority["lock_post"] = assert_live_lock_authority(os.environ)
    authority["preemption_post"] = assert_preemption_clear("window-complete")
    finished = datetime.now(timezone.utc)
    proof: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "WINDOW_COMPLETE",
        "verdict": (
            "LATE_NI_WINDOW_COMPLETE_KNOWN_V10_RED_REMAINS"
            if args.record_known_1500_v10_red
            else "COUPLED_FORCEDOWN_WINDOW_GREEN"
        ),
        "authority": authority,
        "load_authority": load_authority,
        "scheduler": schedule,
        "initial_history": {
            "emitted_by_normal_writer": True,
            "results": initial_history,
        },
        "prefix": prefix_proof,
        "prefix_9198_health": prefix_health,
        "carry_continuity": continuity,
        "ordinary_one_step_program": ordinary_audit,
        "health_program": health_audit,
        "window": window_proof,
        "incremental_frame_pairs": {
            "cpu_manifest_authority": pairer.cpu_manifest_authority,
            "retry20_authority": pairer.retry20_authority,
            "known_1500_v10_authority": pairer.known_1500_v10_authority_row,
            "early_causal_authority": pairer.early_authority,
            "counts": dict(pairer.counts),
            "expected_counts": expected_window_counts,
            "rows": pairer.rows,
            "all_outputs_retained": True,
            "terminal_continuation_counts": {"d01": 19, "d02": 19, "d03": 55},
            "jpg_can_be_built_from_retained_terminal_stream_without_rerun": True,
        },
        "checkpoint_carries": output.checkpoints,
        "green_complete_carries": green_carries,
        "decisive_1500": decisive_rows[0],
        "known_1500_v10_red_remains": bool(args.record_known_1500_v10_red),
        "outputs": [
            {
                "domain": row["domain"],
                "own_step": row["own_step"],
                "valid_time": row["valid_time"],
                "wrfout": row["wrfout"],
            }
            for row in output.emitted
        ],
        "mechanism_separation": {
            "ni_track": "candidate boundary-bundle falsification window",
            "v10_track": "reported independently; no common cause claimed",
            "common_root_proved": False,
        },
        "scope": {
            "prefix_builds": 1,
            "d03_one_step_lowers": 1,
            "d03_one_step_compiles": 1,
            "health_lowers": 1,
            "health_compiles": 1,
            "initial_history_writes": 3,
            "normal_output_callback_only": True,
            "incremental_pairing_uses_closed_host_files_only": True,
            "full_18h_continuation_before_fresh_authority": bool(args.direct_terminal),
            "model_or_numerical_edit": False,
            "observers_callbacks_or_summaries_in_model_hlo": False,
        },
        "timing": {
            "started_utc": started.isoformat(),
            "finished_window_utc": finished.isoformat(),
            "window_wall_seconds": (finished - started).total_seconds(),
            "domain_load_wall_seconds": load_wall_seconds,
            "ordinary_one_step_lower_wall_seconds": ordinary_audit["lower_wall_seconds"],
            "ordinary_one_step_compile_wall_seconds": ordinary_audit["compile_wall_seconds"],
            "health_lower_wall_seconds": health_audit["lower_wall_seconds"],
            "health_compile_wall_seconds": health_audit["compile_wall_seconds"],
        },
    }
    proof["proof_sha256"] = canonical_digest(proof)
    window_proof_path = args.run_dir / "window-proof.json" if args.direct_terminal else args.proof_output
    atomic_write_json(window_proof_path, proof)
    continuation_path = args.continuation_authority or (args.run_dir / "manager-continuation-authority.json")

    def continuation(carries: Any) -> Any:
        return _run_authorized_continuation(
            carries,
            tree=tree,
            own_steps=dict(result.own_steps),
            cadence=cadence,
            nonintegral=nonintegral,
            executable=executable,
            namelist=namelist,
            clock=clock,
            d03_cadence=d03_cadence,
            health_executable=health_executable,
            output=output,
            runtime=runtime,
            run_dir=args.run_dir,
        )

    if args.direct_terminal:
        stage("DIRECT_TERMINAL_CONTINUATION")
        terminal_result = continuation(result.carries)
        runtime.block_until_ready(tuple(
            terminal_result.states[name].theta for name in names
        ))
        for name in names:
            runtime.assert_state_finite_at_boundary(
                terminal_result.states[name],
                domain=name,
                step=TERMINAL_OWN_STEPS[name],
                sim_time_s=TERMINAL_OWN_STEPS[name] * float(dt_by_domain[name]),
            )
        terminal_carries = {
            name: _retain_complete_carry(
                runtime,
                args.run_dir / "terminal-carries",
                role="terminal",
                domain=name,
                step=TERMINAL_OWN_STEPS[name],
                carry=terminal_result.carries[name],
            )
            for name in names
        }
        stage("TERMINAL_19_19_55_IDENTITY_V10_JPG")
        terminal_identity = build_full_terminal_verdict(
            args.run_dir, output_dir, runtime,
        )
        if pairer.counts != {"d01": 19, "d02": 19, "d03": 55}:
            raise RunnerGateError("FULL_PAIRER_COUNTS", repr(pairer.counts))
        if not all(row["finite_identity_pass"] for row in pairer.rows):
            raise RunnerGateError("FULL_INCREMENTAL_PAIR", "one or more frame pairs failed")
        authority["terminal_inputs_post"] = assert_input_retry20_cache_authority(os.environ)
        authority["terminal_lock_post"] = assert_live_lock_authority(os.environ)
        authority["terminal_preemption_post"] = assert_preemption_clear("terminal-complete")
        finished_terminal = datetime.now(timezone.utc)
        terminal_proof = {
            "schema": "gpuwrf.v0234.nested-boundary-full18h-final.v1",
            "status": "TERMINAL_COMPLETE",
            "verdict": (
                "FULL_18H_LATE_NI_COMPLETE_KNOWN_V10_RED_REMAINS"
                if args.record_known_1500_v10_red
                else "FULL_18H_IDENTITY_GREEN"
            ),
            "candidate_commit": CANDIDATE_COMMIT,
            "authority": authority,
            "window_proof": {
                "path": str(window_proof_path.resolve()),
                "file_sha256": sha256_file(window_proof_path),
                "payload_sha256": proof["proof_sha256"],
            },
            "terminal_own_steps": dict(terminal_result.own_steps),
            "terminal_output_counts": dict(pairer.counts),
            "terminal_carries": terminal_carries,
            "terminal_identity_v10_jpg": terminal_identity,
            "all_incremental_pairs_passed": not args.record_known_1500_v10_red,
            "all_finite_identity_pairs_passed": True,
            "known_1500_v10_red": (
                decisive_rows[0] if args.record_known_1500_v10_red else None
            ),
            "known_1500_v10_red_remains_release_blocker": bool(
                args.record_known_1500_v10_red
            ),
            "record_and_continue_scope": (
                "late-Ni discriminator only; no waiver or reclassification"
                if args.record_known_1500_v10_red
                else None
            ),
            "direct_same_process_continuation": True,
            "manager_pause_at_9405": False,
            "model_or_numerical_edit": False,
            "timing": {
                "started_utc": started.isoformat(),
                "finished_utc": finished_terminal.isoformat(),
                "wall_seconds": (finished_terminal - started).total_seconds(),
            },
        }
        terminal_proof["proof_sha256"] = canonical_digest(terminal_proof)
        atomic_write_json(args.proof_output, terminal_proof)
        print(json.dumps({
            "verdict": terminal_proof["verdict"],
            "proof": str(args.proof_output),
            "proof_sha256": terminal_proof["proof_sha256"],
            "counts": pairer.counts,
            "identity_plot": terminal_identity["identity_plot"],
        }, sort_keys=True), flush=True)
        return 0

    hold = hold_for_fresh_continuation(
        result.carries,
        authority_path=continuation_path,
        hold_seconds=args.hold_seconds,
        runner_commit=authority["candidate"]["runner_head"],
        window_proof_sha256=proof["proof_sha256"],
        continuation=continuation,
    )
    hold_path = args.run_dir / "continuation-hold-result.json"
    atomic_write_json(hold_path, hold)
    print(json.dumps({
        "verdict": proof["verdict"],
        "proof": str(args.proof_output),
        "proof_sha256": proof["proof_sha256"],
        "hold": hold["decision"],
    }, sort_keys=True), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--proof-output", type=Path, required=True)
    parser.add_argument("--cpu-dry-run", action="store_true")
    parser.add_argument("--bootstrap-audit", action="store_true")
    parser.add_argument("--run-focused-tests", action="store_true")
    parser.add_argument("--lower-only-custom-call-diagnostic", action="store_true")
    parser.add_argument("--parent-join-resume", action="store_true")
    parser.add_argument("--direct-terminal", action="store_true")
    parser.add_argument("--record-known-1500-v10-red", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=300.0)
    parser.add_argument("--continuation-authority", type=Path)
    args = parser.parse_args(argv)
    args._parent_join_stage = "PRE_RUNTIME"
    args._parent_join_stages = ["PRE_RUNTIME"]
    args._full_run_stage = "PRE_RUNTIME"
    args._full_run_stages = ["PRE_RUNTIME"]
    args.run_dir = args.run_dir.resolve()
    args.proof_output = args.proof_output.resolve()
    if args.continuation_authority is not None:
        args.continuation_authority = args.continuation_authority.resolve()
    if args.bootstrap_audit and not args.cpu_dry_run:
        raise RunnerGateError("BOOTSTRAP_SCOPE", "--bootstrap-audit is CPU-dry-run only")
    if args.lower_only_custom_call_diagnostic and args.cpu_dry_run:
        raise RunnerGateError("DIAGNOSTIC_SCOPE", "lower-only diagnostic is a locked runtime mode")
    if args.parent_join_resume and (args.cpu_dry_run or args.lower_only_custom_call_diagnostic):
        raise RunnerGateError("PARENT_JOIN_SCOPE", "parent join is a production runtime mode")
    if args.direct_terminal and (
        args.cpu_dry_run or args.lower_only_custom_call_diagnostic or args.parent_join_resume
    ):
        raise RunnerGateError("DIRECT_TERMINAL_SCOPE", "direct terminal is exclusive production mode")
    if args.record_known_1500_v10_red and not (
        args.direct_terminal or args.cpu_dry_run
    ):
        raise RunnerGateError(
            "KNOWN_1500_V10_SCOPE",
            "record-and-continue is limited to direct-terminal replay or its CPU audit",
        )
    if (
        REQUIRE_KNOWN_1500_V10_RECORD
        and args.direct_terminal
        and not args.record_known_1500_v10_red
    ):
        raise RunnerGateError(
            "KNOWN_1500_V10_RECORD_REQUIRED",
            "post-Fable direct-terminal replay must preserve the known red",
        )
    if (
        not args.cpu_dry_run
        and not args.lower_only_custom_call_diagnostic
        and args.run_dir.name != (
            PARENT_JOIN_NAMESPACE
            if args.parent_join_resume
            else FULL_REPLAY_NAMESPACE
            if args.direct_terminal
            else REPAIRED_PRODUCTION_NAMESPACE
        )
    ):
        expected_namespace = (
            PARENT_JOIN_NAMESPACE
            if args.parent_join_resume
            else FULL_REPLAY_NAMESPACE
            if args.direct_terminal
            else REPAIRED_PRODUCTION_NAMESPACE
        )
        raise RunnerGateError(
            "PRODUCTION_NAMESPACE",
            f"expected={expected_namespace} actual={args.run_dir.name}",
        )
    require_runner_audit = not args.bootstrap_audit
    authority = assert_admission_authority(
        os.environ,
        run_dir=args.run_dir,
        proof_output=args.proof_output,
        require_live_lock=not args.cpu_dry_run,
        require_runner_audit=require_runner_audit,
        require_clean=True,
    )
    if args.cpu_dry_run:
        return _cpu_dry_run(args, authority)
    authority, runtime = execute_after_preflight(lambda: authority, _import_runtime)
    if args.lower_only_custom_call_diagnostic:
        return _lower_only_custom_call_diagnostic(args, authority, runtime)
    try:
        if args.parent_join_resume:
            return _parent_join_runtime_main(args, authority, runtime)
        return _runtime_main(args, authority, runtime)
    except WindowFalsified as exc:
        if args.parent_join_resume and args.run_dir.is_dir():
            _retain_parent_join_blocker(
                args, exc.code, exc.detail, traceback_text=traceback.format_exc(),
            )
            return 3
        if args.direct_terminal and args.run_dir.is_dir():
            _retain_full_run_blocker(
                args, exc.code, exc.detail, traceback_text=traceback.format_exc(),
            )
            return 3
        profile_window_handler = globals().get("PROFILE_WINDOW_FALSIFIED_HANDLER")
        if callable(profile_window_handler) and args.run_dir.is_dir():
            return int(profile_window_handler(args, exc))
        failure_output = args.run_dir / "window-falsified.json"
        if not failure_output.exists():
            atomic_write_json(failure_output, {
                "schema": SCHEMA,
                "verdict": "COUPLED_FORCEDOWN_WINDOW_FALSIFIED",
                "failure_code": exc.code,
                "detail": exc.detail,
            })
        print(json.dumps({
            "verdict": "COUPLED_FORCEDOWN_WINDOW_FALSIFIED",
            "failure_code": exc.code,
            "detail": exc.detail,
        }, sort_keys=True), flush=True)
        return 3
    except RunnerGateError as exc:
        if args.parent_join_resume and args.run_dir.is_dir():
            _retain_parent_join_blocker(
                args, exc.code, exc.detail, traceback_text=traceback.format_exc(),
            )
            return 3
        if args.direct_terminal and args.run_dir.is_dir():
            _retain_full_run_blocker(
                args, exc.code, exc.detail, traceback_text=traceback.format_exc(),
            )
            return 3
        profile_runtime_handler = globals().get("PROFILE_RUNTIME_BLOCKER_HANDLER")
        if callable(profile_runtime_handler) and args.run_dir.is_dir():
            return int(profile_runtime_handler(args, exc, False))
        raise
    except Exception as exc:
        if args.parent_join_resume and args.run_dir.is_dir():
            _retain_parent_join_blocker(
                args,
                "PARENT_JOIN_UNEXPECTED_EXCEPTION",
                f"{type(exc).__name__}: {exc}",
                traceback_text=traceback.format_exc(),
            )
            return 3
        if args.direct_terminal and args.run_dir.is_dir():
            _retain_full_run_blocker(
                args,
                "FULL_18H_UNEXPECTED_EXCEPTION",
                f"{type(exc).__name__}: {exc}",
                traceback_text=traceback.format_exc(),
            )
            return 3
        profile_runtime_handler = globals().get("PROFILE_RUNTIME_BLOCKER_HANDLER")
        if callable(profile_runtime_handler) and args.run_dir.is_dir():
            return int(profile_runtime_handler(args, exc, True))
        raise


if os.environ.get("GPUWRF_V10_PINNED_REPLAY", "") == "1":
    from scripts.v0234_v10_pinned_replay_profile import apply_profile

    apply_profile(sys.modules[__name__])
elif os.environ.get("GPUWRF_S1_RESIDUAL_VALIDATION", "") == "1":
    from scripts.v0234_s1_residual_gpu_arm_profile import apply_profile

    apply_profile(sys.modules[__name__])
elif os.environ.get("GPUWRF_S1_DIFF6_VALIDATION", "") == "1":
    from scripts.v0234_s1_diff6_gpu_arm_profile import apply_profile

    apply_profile(sys.modules[__name__])
elif os.environ.get("GPUWRF_DYCORE_SUBOPERATOR_LADDER", "") == "1":
    from scripts.v0234_dycore_suboperator_kimi_gpu_ladder import apply_profile

    apply_profile(sys.modules[__name__])
elif os.environ.get("GPUWRF_FIRST_INTERVAL_MOMENTUM", "") == "1":
    from scripts.v0234_first_interval_momentum_runner_profile import apply_profile

    apply_profile(sys.modules[__name__])
elif os.environ.get("GPUWRF_DETERMINISTIC_WAKE_CLOSURE", "") == "1":
    from scripts.v0234_deterministic_wake_closure_runner_profile import apply_profile

    apply_profile(sys.modules[__name__])
elif os.environ.get("GPUWRF_NESTED_BUNDLE_APPROVED_SHA", "") == (
    "470e6111d516479bed4bc0c3b2be1007bb082afd"
):
    from scripts.v0234_stage_omega_transport_runner_profile import apply_profile

    apply_profile(sys.modules[__name__])
elif os.environ.get("GPUWRF_NESTED_BUNDLE_APPROVED_SHA", "") == (
    "395fb800df0bb619462db8e312f734ca0c538161"
):
    from scripts.v0234_nested_h_sca_order_runner_profile import apply_profile

    apply_profile(sys.modules[__name__])
elif os.environ.get("GPUWRF_NESTED_BUNDLE_APPROVED_SHA", "") == (
    "18d97595c59ca01840081f11109780c291dfcae8"
):
    from scripts.v0234_nested_scalar_sixth_order_runner_profile import apply_profile

    apply_profile(sys.modules[__name__])
elif os.environ.get("GPUWRF_NESTED_BUNDLE_APPROVED_SHA", "") == (
    "044783549697caf8c80d094f2b3e73f5ef340537"
):
    from scripts.v0234_nested_theta_sixth_order_runner_profile import apply_profile

    apply_profile(sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
