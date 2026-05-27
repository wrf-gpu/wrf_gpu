# Theta Saturation Diagnosis

Summary: 1-hour instrumented forecast around `_physics_boundary_step` using run `20260521_18z_l3_24h_20260522T133443Z`.

- No lower-30 cell reached the 400 K post-cap value during the instrumented hour.
- The 400 K lower-30 post-cap value appears on 0 of 360 steps.
- Steps clipping finite pre-cap theta from above 400 K: 0 of 360.
- Max pre-cap lower-30 theta: 399.999237 K.
- Max post-cap lower-30 theta: 399.999237 K.
- Max absolute kinematic theta flux: 0.624180477.
- Max absolute PBL theta delta per step: 11.6799927 K.

Decision: the 400 K lower-30 guard is too tight as an operational maximum statistic because the first hour sits within 0.001 K of the ceiling and the predecessor 24h run showed the max pinned there. AC2 widens only the lower-30 ceiling to 450 K, keeps the 200 K floor, and treats remaining skill failure as possible downstream physics/surface coupling rather than as solved by the envelope change.

Proof: per-step pre-cap and post-cap theta histograms are in `/tmp/wrf_gpu2_skillfix2/.agent/sprints/2026-05-27-m7-skill-fix-iter2/theta_saturation_diagnosis.json`.
