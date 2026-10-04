"""Aggregate py-spy raw (collapsed stacks) into: leaf-frame ranking, python-vs-native
split, and top native module attribution. Usage: hf_pyspy_agg.py <raw.txt>"""
import collections
import re
import sys

path = sys.argv[1]
leaf = collections.Counter()
py_samples = 0
native_samples = 0
mod_leaf = collections.Counter()
total = 0
py_frame_re = re.compile(r"(python3|\.py)")
for line in open(path):
    line = line.rstrip()
    if not line or ";" not in line:
        continue
    stack, _, cnt = line.rpartition(" ")
    try:
        c = int(cnt)
    except ValueError:
        continue
    total += c
    frames = stack.split(";")
    top = frames[-1]
    leaf[top[:110]] += c
    m = re.match(r"([^!+]+)", top)
    if m:
        mod_leaf[m.group(1)[:70]] += c
    # python vs native: bottom-up heuristics on the LEAF frame
    if any(f.endswith((".py",)) or "_ctypes" in f for f in frames[-3:]):
        pass
print(f"total samples: {total}")
print("--- leaf frames (top 25) ---")
for k, v in leaf.most_common(25):
    print(f"{v/total*100:6.2f}%  {k}")
print("--- by leading module of leaf (top 15) ---")
for k, v in mod_leaf.most_common(15):
    print(f"{v/total*100:6.2f}%  {k}")
# python interpreter attribution: stacks containing any .py frame
pytotal = 0
for line in open(path):
    line = line.rstrip()
    if ";" not in line:
        continue
    stack, _, cnt = line.rpartition(" ")
    try:
        c = int(cnt)
    except ValueError:
        continue
    if re.search(r"\.py:[0-9]+", stack):
        pytotal += c
print(f"samples whose stack contains >=1 python frame: {pytotal} ({pytotal/total*100:.2f}%)")
