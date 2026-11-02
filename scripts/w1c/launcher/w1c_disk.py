"""W1c-1L disk gate (ALISIOS decision (b) 2026-10-31T03:12Z): per-member DATA floor with bounded waiting. Pure, no I/O."""
import time

GPU_SECONDS_TOTAL_CAP, SIC_PAIR_CAP, MEMBER_CAP = 1800, 900, 360   # GO draft 028240e5 budget; SCREEN4 member cap 360 s
# ALISIOS decision (b) 2026-10-31T03:12Z: entry and EVERY member need 430 + one member raw (5.16 GiB [M]) + 1 GiB margin,
# rounded up -> 437 GiB; otherwise the launcher WAITS (no waiver) for per-member raw releases, bounded by deadline/window.
MEMBER_RAW_GIB, MARGIN_GIB = 5.16, 1.0
MEMBER_FLOOR_GIB = 437
assert MEMBER_FLOOR_GIB >= 430 + MEMBER_RAW_GIB + MARGIN_GIB
PRIVATE_PEAK_GIB = 24
MEMBER_SETUP_S = 60          # registry + checks before the model clock of one member
WAIT_POLL_S = 20
RAW_JOURNAL = 'RAW_DELETE_JOURNAL_W1C_1L.jsonl'   # canonical name (launcher contract; w1c_raw_release.py writes it)


class DiskFloorTimeout(Exception):
    pass


def wait_for_floor(free_gib, need_gib, remaining_s, *, sleep=time.sleep, poll=WAIT_POLL_S, log=print):
    """Block until free_gib() >= need_gib. remaining_s() is the time still usable for the NEXT member's work
    (deadline/window clamp); refuse when it cannot fit a member any more. Returns waited seconds."""
    waited = 0.0
    while True:
        free = free_gib()
        if free >= need_gib:
            return waited
        left = remaining_s()
        if left < MEMBER_CAP + MEMBER_SETUP_S + poll:
            raise DiskFloorTimeout(f'DATA {free:.2f} GiB < {need_gib} GiB and only {left:.0f} s left for a member')
        log(f'W1C_DISK_WAIT free={free:.2f}GiB need={need_gib}GiB left={left:.0f}s')
        sleep(poll); waited += poll


def run_rows(rows, execute, before_member):
    """screen4.run_four semantics (STOP_ON_FIRST_FAIL, NOT_ATTEMPTED rows recorded, no registry start) plus a
    pre-member disk gate: a floor timeout halts the run and records this and later rows NOT_ATTEMPTED."""
    results = []; stop_reason = None
    for member in rows:
        base = {'ordinal': member['ordinal'], 'row_id': member['row_id'], 'registry_run_id': member['registry_run_id'],
                'case_id': member['case_id'], 'arm_id': member['arm_id'], 'NO_RESUME': True, 'raw_deleted': False,
                'wrfout_files': [], 'registry_started': False}
        if stop_reason is None:
            try:
                base['disk_wait_s'] = before_member(member)
            except DiskFloorTimeout as exc:
                stop_reason = f"HALT_DISK_FLOOR ordinal={member['ordinal']}: {exc}"
            except BaseException as exc:  # post-wait window/deadline/lease refusal or a signal: record, never escape
                stop_reason = f"HALT_PRE_MEMBER ordinal={member['ordinal']}: {type(exc).__name__}: {exc}"
                base['global_abort'] = not isinstance(exc, Exception)
        if stop_reason is not None:
            results.append(dict(base, status='NOT_ATTEMPTED', reason=stop_reason, attempted=False))
            continue
        try:
            result = execute(member)
        except BaseException as exc:
            result = dict(base, status='FAIL', failure_type=type(exc).__name__, failure=str(exc))
        result['attempted'] = True; result.setdefault('disk_wait_s', base['disk_wait_s'])
        if result['status'] != 'PASS':
            stop_reason = f"STOP_ON_FIRST_FAIL ordinal={member['ordinal']} row_id={member['row_id']}"
        results.append(result)
    return results
