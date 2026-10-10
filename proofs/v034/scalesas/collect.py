"""Parse CU_SCALESAS oracle stdout into arrays (and optionally a savepoint npz)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

def parse(path, n=None, kx=None):
    lines = Path(path).read_text().split("\n")
    lines = [ln for ln in lines if ln.strip()]
    hdr, rth, rqv, rqc, rqi = [], [], [], [], []
    for c in range(len(lines) // 5):
        h = lines[5 * c].split()
        hdr.append([float(x) for x in h[1:]])
        rth.append([float(x) for x in lines[5 * c + 1].split()])
        rqv.append([float(x) for x in lines[5 * c + 2].split()])
        rqc.append([float(x) for x in lines[5 * c + 3].split()])
        rqi.append([float(x) for x in lines[5 * c + 4].split()])
    hdr = np.array(hdr)
    return dict(RAINCV=hdr[:, 0], PRATEC=hdr[:, 1], HBOT=hdr[:, 2], HTOP=hdr[:, 3],
                SCALEFUN=hdr[:, 4], SIGMU=hdr[:, 5], RTHCUTEN=np.array(rth), RQVCUTEN=np.array(rqv),
                RQCCUTEN=np.array(rqc), RQICUTEN=np.array(rqi))

if __name__ == "__main__":
    out = parse(sys.argv[1])
    if len(sys.argv) > 2:
        np.savez_compressed(sys.argv[2], **out)
