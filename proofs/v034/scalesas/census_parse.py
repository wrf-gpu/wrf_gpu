import numpy as np, sys
sys.path.insert(0, "<USER_HOME>/src/wrf_gpu2_wt/o1-sas/proofs/v034/scalesas")
from collect import parse
def load(prefix):
    st = {}; extra = {}
    for ln in open(prefix + ".census").read().split("\n"):
        if not ln.strip(): continue
        t = ln.split()
        if t[0] == "STAGE": st[int(t[1])] = np.array(list(map(int, t[2:]))).astype(bool)
        elif t[0] == "RNRAW": extra[t[0]] = np.array(list(map(float, t[1:])))
        else: extra[t[0]] = np.array(list(map(int, t[1:])))
    o = parse(prefix + ".out")
    n = len(o["RAINCV"])
    kill = np.zeros(n, int)
    alive = np.ones(n, bool)
    for k in sorted(st):
        kill[alive & ~st[k]] = k
        alive = st[k]
    act = o["RAINCV"] > 0
    paths = {
        "kill_kbcon": kill == 1, "kill_cinpcr": kill == 2, "kill_dthk": kill == 3, "kill_cina": kill == 4,
        "kill_cthk": kill == 5, "kill_jmin_aa1": kill == 6, "kill_wc_ddaa1": kill == 7, "kill_closure": kill == 8,
        "restore_rn<=0": alive & ~act, "ktconn": extra["KTCONN"] > 1, "ktconn_active": (extra["KTCONN"] > 1) & act,
        "evapcap": extra["FLGEVAP"] == 0, "evapcap_active": (extra["FLGEVAP"] == 0) & act,
        "active_QE": act & extra["ASQEC"].astype(bool), "active_nonQE": act & ~extra["ASQEC"].astype(bool),
        "htop_km": act & (o["HTOP"] == 44),
    }
    return paths, o
if __name__ == "__main__":
    for pre in sys.argv[1:]:
        paths, o = load(pre)
        print(pre, {k: int(v.sum()) for k, v in paths.items()})
