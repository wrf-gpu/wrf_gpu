"""WRF auxiliary history (``auxhist``) secondary output stream.

WRF supports up to 24 *auxiliary history* streams alongside the primary
``wrfout`` history stream. Each stream ``N`` is driven by four namelist controls
(``&time_control``):

* ``auxhist{N}_outname``     -- filename pattern; default ``"auxhist{N}_d<domain>_<date>"``
* ``auxhist{N}_interval``    -- output interval **in minutes** (``auxhist{N}_interval_m``)
* ``frames_per_auxhist{N}``  -- history frames packed per output file (1 => one
                                file per frame, the operational default here)
* ``io_form_auxhist{N}``     -- output format; ``2`` == NetCDF (the only form this
                                port writes)

The classic operational use is a high-frequency surface-diagnostic stream: e.g.
write ``U10/V10/T2/Q2/PSFC/RAINNC/SWDOWN`` every 15 minutes while the main
``wrfout`` stays hourly with the full 3-D field list.

This module is pure host-side stream metadata + cadence logic. It holds NO model
state and performs NO device work: the actual NetCDF bytes are emitted by the
existing wrfout writer (:func:`gpuwrf.io.wrfout_writer.write_prepared_wrfout`
with a ``variable_subset``), so the auxhist file carries exactly the same
schema/attrs/time coordinates as the main stream -- it is a genuine, schema-valid
WRF history file restricted to the configured variable subset.

The stream is OFF by default everywhere: a pipeline with ``auxhist=None`` writes
only the main ``wrfout`` stream, byte-for-byte unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import gcd
from typing import Iterable, Sequence

# WRF io_form codes (share_io_module / module_io.F). Only NetCDF is supported by
# this port's writer; auxhist accepts the same codes for namelist faithfulness.
IO_FORM_NETCDF = 2

# A sensible WRF-style high-frequency *surface* diagnostic subset. These names are
# all real wrfout mass-point surface fields the writer already produces. Callers
# may override ``variables`` with any subset of the writer's emitted names.
DEFAULT_SURFACE_AUXHIST_VARIABLES: tuple[str, ...] = (
    "U10",
    "V10",
    "T2",
    "Q2",
    "PSFC",
    "RAINC",
    "RAINNC",
    "SWDOWN",
    "GLW",
    "PBLH",
    "HFX",
    "LH",
    "TSK",
)


@dataclass(frozen=True)
class AuxhistStreamConfig:
    """Configuration for one WRF auxiliary-history (``auxhist``) output stream.

    Mirrors the WRF ``&time_control`` auxhist namelist controls one-to-one:

    * ``stream_id``         -> the auxhist stream number ``N`` (1..24); selects the
                               ``auxhist{N}_*`` namelist group and the default
                               filename prefix.
    * ``interval_minutes``  -> ``auxhist{N}_interval`` (minutes between frames).
    * ``variables``         -> the variable subset emitted to this stream (WRF
                               selects auxhist fields via the Registry i/o stream
                               bitmask; here it is an explicit name set).
    * ``frames_per_file``   -> ``frames_per_auxhist{N}`` (history frames per file).
    * ``io_form``           -> ``io_form_auxhist{N}`` (2 == NetCDF).
    * ``outname``           -> ``auxhist{N}_outname`` filename pattern with the WRF
                               ``<domain>`` / ``<date>`` tokens. ``None`` uses the
                               WRF default ``"auxhist{N}_d<domain>_<date>"``.
    """

    stream_id: int = 1
    interval_minutes: int = 15
    variables: tuple[str, ...] = field(default=DEFAULT_SURFACE_AUXHIST_VARIABLES)
    frames_per_file: int = 1
    io_form: int = IO_FORM_NETCDF
    outname: str | None = None

    def __post_init__(self) -> None:
        if not (1 <= int(self.stream_id) <= 24):
            raise ValueError(f"auxhist stream_id must be in 1..24, got {self.stream_id}")
        if int(self.interval_minutes) <= 0:
            raise ValueError(f"auxhist interval_minutes must be > 0, got {self.interval_minutes}")
        if int(self.frames_per_file) < 1:
            raise ValueError(f"frames_per_auxhist must be >= 1, got {self.frames_per_file}")
        if int(self.io_form) != IO_FORM_NETCDF:
            raise ValueError(
                f"auxhist io_form {self.io_form} unsupported; this port writes NetCDF (io_form=2)"
            )
        if not self.variables:
            raise ValueError("auxhist stream needs at least one variable")
        # Normalize to a tuple of unique names preserving declared order.
        seen: dict[str, None] = {}
        for name in self.variables:
            seen.setdefault(str(name), None)
        object.__setattr__(self, "variables", tuple(seen))

    @property
    def variable_subset(self) -> frozenset[str]:
        """The emitted-variable set, in the form the writer's subset hook expects."""

        return frozenset(self.variables)

    @property
    def outname_pattern(self) -> str:
        """The effective filename pattern (WRF default when ``outname`` is unset)."""

        if self.outname:
            return self.outname
        return f"auxhist{int(self.stream_id)}_d<domain>_<date>"

    def filename(self, valid_time: datetime, domain: str) -> str:
        """Resolve the auxhist filename for ``valid_time`` on ``domain``.

        Follows WRF ``auxhist{N}_outname`` token substitution: ``<domain>`` ->
        the 2-digit domain index, ``<date>`` -> the WRF time string
        ``YYYY-MM-DD_HH:MM:SS``. ``domain`` may be given either as the digits
        (``"02"``) or in the pipeline's ``"d02"`` form; both resolve to ``02``.
        """

        digits = _domain_digits(domain)
        date_token = valid_time.strftime("%Y-%m-%d_%H:%M:%S")
        return self.outname_pattern.replace("<domain>", digits).replace("<date>", date_token)

    def fires_at(self, lead_minutes: float, *, tol_seconds: float = 1.0) -> bool:
        """Return ``True`` when ``lead_minutes`` is an auxhist output boundary.

        WRF writes auxhist frame ``k`` at lead times ``k * auxhist{N}_interval``
        (``k >= 1``); the model state at ``t=0`` (the IC) is not an auxhist frame.
        ``lead_minutes`` is the elapsed forecast time, in minutes, at the current
        output boundary. ``tol_seconds`` guards against float roundoff at fractional
        leads (e.g. a 10 s step landing 0.1666.. min off an exact multiple).
        """

        lead_minutes = float(lead_minutes)
        if lead_minutes <= 0.0:
            return False
        interval = float(self.interval_minutes)
        remainder = lead_minutes % interval
        tol_minutes = float(tol_seconds) / 60.0
        return remainder <= tol_minutes or (interval - remainder) <= tol_minutes

    def frame_index(self, lead_minutes: float) -> int:
        """The 1-based auxhist frame number for an output boundary ``lead_minutes``."""

        return int(round(float(lead_minutes) / float(self.interval_minutes)))


def auxhist_output_boundaries(
    run_start: datetime,
    total_hours: float,
    config: AuxhistStreamConfig,
) -> list[tuple[int, float, datetime]]:
    """Enumerate the auxhist output boundaries over a forecast of ``total_hours``.

    Returns ``(frame_index, lead_minutes, valid_time)`` for every frame the stream
    would emit -- i.e. every multiple of ``interval_minutes`` strictly within
    ``(0, total_hours]``. Pure metadata; used by the proof and any driver that
    wants to plan the auxhist cadence ahead of the run.
    """

    interval = int(config.interval_minutes)
    total_minutes = int(round(float(total_hours) * 60.0))
    boundaries: list[tuple[int, float, datetime]] = []
    lead = interval
    while lead <= total_minutes:
        valid_time = run_start + timedelta(minutes=lead)
        boundaries.append((lead // interval, float(lead), valid_time))
        lead += interval
    return boundaries


# --------------------------------------------------------------------------- #
# Multi-stream support: WRF drives auxhist1..auxhist24 INDEPENDENTLY, each with #
# its own interval / outname / frames / variable subset. The pipeline accepts a #
# LIST of configs and fires each at its own cadence; the empty list is the      #
# OFF-by-default case (no second stream, main wrfout byte-unchanged). A single  #
# config (or None) is accepted for back-compat and coerced to the list form.    #
# --------------------------------------------------------------------------- #


def coerce_auxhist_streams(
    auxhist: AuxhistStreamConfig | Iterable[AuxhistStreamConfig] | None,
) -> tuple[AuxhistStreamConfig, ...]:
    """Normalize the pipeline ``auxhist`` setting to a tuple of stream configs.

    Accepts the back-compatible forms and the new multi-stream form:

    * ``None``                         -> ``()`` (OFF; no second stream).
    * a single :class:`AuxhistStreamConfig` -> ``(config,)`` (legacy single-stream).
    * any iterable of configs          -> a tuple of those configs (multi-stream).

    Enforces WRF's constraint that each auxhist stream number is configured at
    most once (``stream_id`` unique across the list) so two configs never collide
    on the same ``auxhist{N}_*`` group / filename prefix. Preserves declared order.
    """

    if auxhist is None:
        return ()
    if isinstance(auxhist, AuxhistStreamConfig):
        streams: tuple[AuxhistStreamConfig, ...] = (auxhist,)
    else:
        streams = tuple(auxhist)
    seen_ids: dict[int, None] = {}
    for stream in streams:
        if not isinstance(stream, AuxhistStreamConfig):
            raise TypeError(
                f"auxhist streams must be AuxhistStreamConfig, got {type(stream).__name__}"
            )
        sid = int(stream.stream_id)
        if sid in seen_ids:
            raise ValueError(f"duplicate auxhist stream_id {sid}; each stream number is unique")
        seen_ids[sid] = None
    return streams


def auxhist_substeps_per_hour(
    streams: Sequence[AuxhistStreamConfig] | AuxhistStreamConfig | None,
) -> int:
    """Number of equal sub-hour forecast segments needed to serve all streams.

    The forecast hour is advanced in equal segments fine enough that BOTH the
    hourly main-wrfout boundary AND every stream's ``interval``-minute boundary
    land on a genuine model-state snapshot. With no streams this is ``1`` (a
    single full-hour advance, byte-for-byte the legacy hourly path). With one or
    more streams the segment length is ``gcd(60, i1, i2, ...)`` minutes, so each
    stream's frames are real GPU output at the requested lead, never interpolated.

    The gcd is taken with 60 so the hourly main boundary is always a segment
    boundary too (the main wrfout cadence is unchanged regardless of substeps).
    """

    configs = coerce_auxhist_streams(streams)
    if not configs:
        return 1
    step_minutes = 60
    for stream in configs:
        step_minutes = gcd(step_minutes, int(stream.interval_minutes))
    # gcd(60, ...) divides 60, so 60 // step_minutes is always a positive integer.
    return max(1, 60 // step_minutes)


def _domain_digits(domain: str) -> str:
    """Normalize a domain identifier (``"d02"`` / ``"02"`` / ``2``) to 2 digits."""

    text = str(domain).strip().lower()
    if text.startswith("d"):
        text = text[1:]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot extract domain index from {domain!r}")
    return f"{int(digits):02d}"


__all__ = [
    "AuxhistStreamConfig",
    "DEFAULT_SURFACE_AUXHIST_VARIABLES",
    "IO_FORM_NETCDF",
    "auxhist_output_boundaries",
    "auxhist_substeps_per_hour",
    "coerce_auxhist_streams",
]
