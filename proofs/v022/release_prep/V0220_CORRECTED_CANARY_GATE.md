# v0.22.0 Corrected Canary Gate

## Verdict

PASS. The v0.22.0 default canary digest matches a freshly paired v0.21.1
matched-env/AOT warm baseline:

- Fresh v0.21.1 matched baseline: `9709039c987ab45a6cfd4bc2bccf403aa0eabb0f361b7d8b61c39574fb32f4e4`
- v0.22.0 default candidate: `9709039c987ab45a6cfd4bc2bccf403aa0eabb0f361b7d8b61c39574fb32f4e4`
- Leaves: `178`
- Bytes: `367594272`

No new GPU or cold-compile run was required for this correction. The proof reuses
the already captured v0.22 candidate digest and replaces the stale hardcoded
reference with the fresh v0.21.1 matched baseline.

## Why The 519 Gate Was Invalid

The hardcoded `519cd3e5c69561d750189ad694e1300bc729df4a0e6348d670d055eb0ab6db8e`
reference came from an older cold-compiled/autotuned executable blob. Diagnostic
comparison found the same source, same XLA flags, and same HLO hashes, but
different serialized executable blobs across independent cold compiles. That
means the fixed byte-digest gate was measuring a stale autotune/serialization
artifact, not a source regression.

The corrected rule is:

- prefer a fresh v0.21.1 reference captured under matched env/AOT;
- compare v0.22 default against that paired reference;
- if independent cold-compiled blobs differ under unchanged source/HLO/XLA flags,
  use the ratified operational-identity fallback instead of treating stale blob
  bytes as a release blocker.

## Proof Objects

- Fresh v0.21.1 digest: `proofs/v022/release_prep/v0211_ref_digest.json`
- v0.22 corrected digest: `proofs/v022/release_prep/v0220_136fix_digest_confirm.json`
- Blocker resolution summary: `proofs/v022/release_prep/v0220_bit_identity_blocker_resolution.json`
- Warm-determinism logs:
  - `proofs/v022/release_prep/logs/fresh_v0211_warmdet_5c4_1_20250121.log`
  - `proofs/v022/release_prep/logs/fresh_v0211_warmdet_5c4_2_20250121.log`
  - `proofs/v022/release_prep/logs/fresh_v0211_warmdet_5c4_3_20250121.log`
