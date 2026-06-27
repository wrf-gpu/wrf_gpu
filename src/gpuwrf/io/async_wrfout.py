"""Background wrfout writer (v0.2.0 wall-clock win #3): double-buffered output.

The daily product loop currently blocks the GPU compute loop on the synchronous
NetCDF write of every hour (``execute_daily_pipeline`` -> ``write_wrfout_netcdf``).
The GPT wall-clock analysis (``.agent/reviews/2026-06-01-gpt-wallclock-
optimization.md`` idea #3) measured this synchronous output/CPU work at ~110 s on
the d02 24h path and ~176 s on d03 24h -- all of it serialized in front of the
next forecast hour.

This writer overlaps that work: the main thread eagerly pulls hour N to host
(``prepare_wrfout_payload`` -- a :class:`PreparedWrfout` with NO device refs),
hands it to a SINGLE background thread, and immediately advances the GPU toward
hour N+1. The background thread writes the NetCDF while the GPU computes.

DESIGN / SAFETY
- One worker thread, ``maxsize``-bounded queue (default 2): bounds host RAM and
  guarantees back-pressure if the writer falls behind the GPU.
- A single writer thread => writes are serialized => deterministic file ordering.
  Callers that also read NetCDF during the integration loop must keep those reads
  out of the async window (or use the synchronous fallback), because NetCDF4/HDF5
  builds are not universally safe for concurrent handles across threads.
- The device->host pull happens on the MAIN thread before submit, so the
  background thread never touches a device array that the GPU might reuse.
- The NetCDF bytes are byte-for-byte identical to the synchronous path; only the
  wall-clock timing of the write changes.
- ``join()`` re-raises the first writer error so a failed write still fails the
  run (fail-closed), and must be called before any consumer reads the wrfouts
  (validation/inventory/scoring) or before pipeline exit.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gpuwrf.io.wrfout_writer import PreparedWrfout, write_prepared_wrfout


@dataclass(frozen=True)
class _WriteJob:
    """One queued NetCDF write.

    ``variable_subset``/``target`` (both ``None`` for the main wrfout stream) carry
    a secondary WRF ``auxhist`` stream's variable restriction and output path, so a
    single background writer thread serializes BOTH streams' writes (deterministic
    ordering, no concurrent ``Dataset`` handles).
    """

    prepared: PreparedWrfout
    variable_subset: Optional[frozenset[str]] = None
    target: Optional[Path] = None
    include_mandatory_coords: bool = False
    compress: bool = False


class AsyncWrfoutWriter:
    """Single-thread, bounded-queue background NetCDF writer.

    Usage::

        with AsyncWrfoutWriter() as writer:
            for hour in ...:
                state = advance(state)
                prepared = prepare_wrfout_payload(state, ...)
                writer.submit(prepared)   # returns immediately; GPU continues
        # __exit__ joins outstanding writes and re-raises any writer error.
    """

    def __init__(self, max_pending: int = 2) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self._queue: "queue.Queue[Optional[_WriteJob]]" = queue.Queue(maxsize=max_pending)
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._written: list[Path] = []
        self._written_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker, name="wrfout-writer", daemon=True
        )
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:  # sentinel: shut down
                    return
                # Skip new writes once an earlier write has failed; still drain
                # the queue so producers blocked on put() are released.
                with self._error_lock:
                    failed = self._error is not None
                if not failed:
                    try:
                        path = write_prepared_wrfout(
                            item.prepared,
                            variable_subset=item.variable_subset,
                            target_override=item.target,
                            include_mandatory_coords=item.include_mandatory_coords,
                            compress=item.compress,
                        )
                        with self._written_lock:
                            self._written.append(path)
                    except BaseException as exc:  # noqa: BLE001 - record & keep draining
                        with self._error_lock:
                            if self._error is None:
                                self._error = exc
            finally:
                self._queue.task_done()

    def submit(self, prepared: PreparedWrfout) -> None:
        """Enqueue a prepared payload (full main wrfout stream) for background writing.

        Blocks (back-pressure) if ``max_pending`` writes are already queued.
        Re-raises a prior writer error promptly so the pipeline fails closed.
        """

        self._enqueue(_WriteJob(prepared=prepared))

    def submit_subset(
        self,
        prepared: PreparedWrfout,
        *,
        variable_subset: frozenset[str] | tuple[str, ...],
        target: Path,
        include_mandatory_coords: bool = False,
        compress: bool = False,
    ) -> None:
        """Enqueue a secondary-stream (WRF ``auxhist`` / training) subset write.

        Reuses the same host-materialized ``prepared`` payload but writes only
        ``variable_subset`` to ``target`` -- so a subset frame costs no extra
        device->host pull and is serialized on the same background writer thread as
        the main stream. ``include_mandatory_coords`` / ``compress`` (both OFF by
        default, preserving the auxhist surface-stream behaviour) enable the #122
        self-contained, lossless-compressed training stream.
        """

        self._enqueue(
            _WriteJob(
                prepared=prepared,
                variable_subset=frozenset(variable_subset),
                target=Path(target),
                include_mandatory_coords=include_mandatory_coords,
                compress=compress,
            )
        )

    def _enqueue(self, job: _WriteJob) -> None:
        if self._closed:
            raise RuntimeError("AsyncWrfoutWriter is closed")
        self._raise_if_failed()
        self._queue.put(job)

    def _raise_if_failed(self) -> None:
        with self._error_lock:
            err = self._error
        if err is not None:
            raise err

    def join(self) -> list[Path]:
        """Block until all queued writes finish; re-raise the first writer error.

        Idempotent. Must be called before reading the produced wrfouts.
        """

        if not self._closed:
            self._queue.put(None)  # sentinel
            self._closed = True
        self._thread.join()
        self._raise_if_failed()
        with self._written_lock:
            return list(self._written)

    def __enter__(self) -> "AsyncWrfoutWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Always drain/join so the writer thread cannot outlive the pipeline.
        # If the body raised, still try to flush queued writes but let the body's
        # exception propagate (do not mask it with a writer error).
        try:
            self.join()
        except BaseException:
            if exc_type is None:
                raise
