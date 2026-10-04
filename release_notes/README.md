# Current release notes

The current **shipped** release is **wrf_gpu v0.23.3**.

Read the prepared v0.23.4 candidate notes here:

- [`RELEASE_NOTES_v0.23.4.md`](RELEASE_NOTES_v0.23.4.md)

Read the current published release notes here:

- [`RELEASE_NOTES_v0.23.3.md`](RELEASE_NOTES_v0.23.3.md)

**v0.23.4 is prepared, pending publication.** Its correctness content (nine-nest chain: V10,
late-Ni, nine-nest replay) is fully closed and green. Its performance content is partial: the
scalar-batching dispatch fix is validated for 2-domain configurations (no regression), but a real
~51.67% regression for 3-domain/9-domain configurations that engage d03 remains — the investigation
into it is closed (every quick-fix candidate falsified/rejected, hardware-measured as occupancy/
latency-bound, not bandwidth-bound), but the regression itself is not fixed. See
[`RELEASE_NOTES_v0.23.4.md`](RELEASE_NOTES_v0.23.4.md) for the full, current status.

Historical release notes remain in the versioned `RELEASE_NOTES_v*.md` files.
