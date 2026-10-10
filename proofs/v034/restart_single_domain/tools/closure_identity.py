"""OFF-path proof: changed files lie outside the traced-step import closure (AOT cheap-key scope)."""
import hashlib, json, subprocess, sys
from pathlib import Path
from gpuwrf.runtime import aot_cheap_key as k
base_rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
root = k._gpuwrf_package_root()
files = k._trace_reachable_source_files(root)
repo = Path.cwd().resolve()
changed = subprocess.run(["git", "diff", "--name-only", base_rev], capture_output=True, text=True, check=True).stdout.split()
changed_abs = {(repo / c).resolve() for c in changed}
def digest(get):
    h = hashlib.sha256()
    for p in files:
        data = get(p); rel = p.relative_to(root).as_posix()
        k._upd(h, b"f_rel", rel.encode()); k._upd(h, b"f_size", str(len(data)).encode()); k._upd(h, b"f_sha", hashlib.sha256(data).hexdigest().encode())
    return h.hexdigest()
base = digest(lambda p: subprocess.run(["git", "show", f"{base_rev}:{p.resolve().relative_to(repo)}"], capture_output=True, check=True).stdout)
cand = digest(lambda p: p.read_bytes())
print(json.dumps({"base_rev": base_rev, "closure_files": len(files), "changed": changed,
                  "changed_in_closure": sorted(str(f) for f in files if f.resolve() in changed_abs),
                  "closure_digest_base": base, "closure_digest_candidate": cand, "identical": base == cand,
                  "source_fingerprint_hash_candidate": k.source_fingerprint_hash()}, indent=1))
