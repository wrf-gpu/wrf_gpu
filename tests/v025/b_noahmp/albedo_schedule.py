"""Per-step Noah-MP forcing from hourly CPU-WRF frames (shared by the pristine-driver input and the port test, NF10)."""
import numpy as np

FIELDS = ("sfctmp", "sfcprs", "psfc", "uu", "vv", "q2", "qc", "soldn", "lwdn",
          "prcpconv", "prcpnonc", "prcpsnow", "prcpgrpl", "prcphail", "cosz")
PRECIP = slice(9, 14)


def step_forcing(hourly, dt, nsteps):
    """(nsteps, 15) float32. Step k (1-based) sees the forcing at t = (k-1) dt, linear between the hourly frames
    (SWDOWN stays exactly 0 between two dark frames); precipitation rates are the hourly-interval means."""
    H = np.asarray(hourly, dtype=np.float64)
    t = np.arange(nsteps, dtype=np.float64) * float(dt) / 3600.0
    h = np.minimum(np.floor(t).astype(int), H.shape[0] - 2)
    fr = (t - h)[:, None]
    out = H[h] * (1.0 - fr) + H[h + 1] * fr
    out[:, PRECIP] = H[h, PRECIP]
    return out.astype(np.float32)
