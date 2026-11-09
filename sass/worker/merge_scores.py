"""Folding score files together while scoring is still running."""

import logging
from pathlib import Path
from typing import Any

from sass.io import dump_score_file, load_score_file
from sass.selection.dedup import deduplicate_scores, score_cutoff

logger = logging.getLogger(__name__)

#: Measured peak resident memory of `merge_score_batch_task`, in GB per 1e6 of `top_m`.
#: The task holds at most about `3 * top_m` rows, independent of fan-in.
MERGE_GB_PER_1E6_TOP_M = 1.18


def estimate_peak_merge_gb(top_m: int) -> float:
    """Peak worker memory the score merge will need for this `top_m`.

    Used to warn before a run starts.
    """
    return MERGE_GB_PER_1E6_TOP_M * top_m / 1e6


def merge_score_batch_task(
    files: list[Path],
    out_file: Path,
    top_m: int,
    warts_separator: str,
    keep_isomer_warts: bool,
    tag: Any = None,
) -> tuple:
    """Merge a batch of score files into one and delete the inputs.

    Submitted while scoring is still running, so score files do not accumulate.

    Parameters
    ----------
    files
        Score files to merge. Deleted once the merged file is written.
    out_file
        Destination for the merged file.
    top_m
        Number of top rows to keep.
    warts_separator, keep_isomer_warts
        Deduplication settings.
    tag
        Opaque value echoed back to the caller, such as the component count.

    Returns
    -------
        ``("merge", tag, out_file, n_rows, n_bytes, cutoff)`` where ``cutoff`` is the
        merged file's ``top_m``-th score, or ``None`` if it holds fewer than ``top_m`` rows.
    """
    freed = sum(Path(f).stat().st_size for f in files if Path(f).is_file())

    def trim(rows: list) -> list:
        return deduplicate_scores(
            rows,
            sort_scores=True,
            limit=top_m,
            warts_separator=warts_separator,
            keep_isomer_warts=keep_isomer_warts,
        )

    rows = []
    for f in files:
        rows.extend(load_score_file(f))
        # Trim as we go, so peak memory stays near `3 * top_m` rows regardless of fan-in.
        if len(rows) > 2 * top_m:
            rows = trim(rows)
    rows = trim(rows)
    dump_score_file(out_file, rows)
    for f in files:
        Path(f).unlink(missing_ok=True)

    return (
        "merge",
        tag,
        out_file,
        len(rows),
        out_file.stat().st_size - freed,  # Net change in bytes on disk.
        score_cutoff(rows, top_m),
    )
