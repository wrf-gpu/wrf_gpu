# V0.14 Step-1 QVAPOR Pre-Call Truth Schema

Opened 2026-06-09.

Purpose: establish whether authoritative same-boundary WRF pre-call `QVAPOR`
truth exists for the Step-1 live-nest theta proof.

Do not use post-RK or different-boundary `QVAPOR` artifacts as pre-call truth
without explicit boundary proof. If same-boundary truth is missing, the next
step is a minimal WRF savepoint around the live-nest `adjust_tempqv` / first
RK handoff boundary.
