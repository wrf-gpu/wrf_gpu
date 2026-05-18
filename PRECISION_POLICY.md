# Precision Policy

- Default precision is undecided until the M2 bakeoff.
- Dycore precision must be validated before performance tuning is accepted.
- Mixed precision requires explicit validation by variable and scheme.
- Do not use BF16 or FP16 in physics unless acceptance tests prove safety.
- Tolerances are documented per variable, per fixture, and per validation tier.
- Debug modes may use stricter precision or bitwise controls, but production speed claims must use production precision.
