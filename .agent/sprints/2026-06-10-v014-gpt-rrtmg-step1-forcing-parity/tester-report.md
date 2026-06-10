# Tester Report: V0.14 GPT RRTMG Step-1 Forcing Parity

Decision: PASS FOR LOCALIZATION, NOT A RELEASE-CLOSING FIX.

Commands reported by the worker and recorded in
`.agent/reviews/2026-06-10-v014-gpt-rrtmg-step1-forcing-parity.md`:

```bash
python -m py_compile proofs/v014/rrtmg_step1_forcing_parity.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/rrtmg_step1_forcing_parity.py
python -m json.tool proofs/v014/rrtmg_step1_forcing_parity.json >/tmp/rrtmg_step1_forcing_parity.validated.json
git diff --check
```

All reported gates passed in the worker session. Manager-side lightweight
acceptance gates must still validate the checked-in proof/report schema and
sprint closeout before commit.

Proof objects:

- `proofs/v014/rrtmg_step1_forcing_parity.json`
- `proofs/v014/rrtmg_step1_forcing_parity.md`

Coverage judgment:

- The proof is sufficient to keep the next NoahMP-focused strict attempt
  unblocked.
- It is insufficient to close final v0.14 RRTMG parity because the exact WRF
  derived RRTMG column inputs are not yet dumped.

Residual risk:

- The first divergent derived RRTMG quantity is still unnamed.
- A production edit before adding the WRF RRTMG forcing hook would be guesswork.

