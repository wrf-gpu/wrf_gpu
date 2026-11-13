"""Standalone (not pytest: the suite guards <DATA_ROOT>/alisios): verify wt1_pins.py against the WT1 sources.

  PYTHONDONTWRITEBYTECODE=1 python tests/v50_extract/verify_pins.py   -> prints PINS_OK or raises
"""
import ast
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
pins_text = (HERE / "wt1_pins.py").read_text()
mod = ast.parse(pins_text)
consts = {}
for node in mod.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in (
            "TEACHER_DIR", "TEACHER_SHA", "BLOCKS", "FRAME_FIELDS_LINES"):
        consts[node.targets[0].id] = ast.literal_eval(node.value)
src = {}
for name, sha in consts["TEACHER_SHA"].items():
    data = (Path(consts["TEACHER_DIR"]) / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha, name
    src[name] = data.decode().splitlines()
for name, lo, hi in consts["BLOCKS"]:
    block = "\n".join(src[name][lo - 1:hi])
    assert f"# ---- {name} {lo}-{hi}\n{block}\n" in pins_text, (name, lo, hi)
ff = pins_text[pins_text.index("def frame_profiles(src):"):pins_text.index("    return profiles")]
body = [line for line in ff.splitlines()[4:]]
assert body == [src["teacher_extract_v2.py"][n - 1] for n in consts["FRAME_FIELDS_LINES"]], "frame_fields lines"
print("PINS_OK", {k: v[:12] for k, v in consts["TEACHER_SHA"].items()}, len(consts["BLOCKS"]), "blocks")
sys.exit(0)
