"""Byte accounting for the transient files a phase writes."""

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class DiskTracker:
    """Track bytes written and freed by dask tasks, and report the peak.

    Each task returns the byte delta of the files it wrote or removed, and the head node sums
    them. The peak matters most, since these directories are transient.
    """

    def __init__(self, label: str, worker_count: Callable[[], str] | None = None) -> None:
        self.label = label
        self.live = 0
        self.peak = 0
        #: Returns the live worker count for the progress line, e.g. "812/1000".
        self.worker_count = worker_count

    def add(self, delta: int) -> None:
        self.live += delta
        self.peak = max(self.peak, self.live)

    def reset(self, label: str) -> None:
        self.label = label
        self.live = 0
        self.peak = 0

    @property
    def live_gb(self) -> float:
        return self.live / 1e9

    @property
    def peak_gb(self) -> float:
        return self.peak / 1e9

    def log_progress(self, n_done: int, extra: str = "") -> None:
        # Include the live worker count, since slow progress can mean missing workers.
        logger.info(
            "Finished %s %s tasks. Workers: %s. %s on disk: %.2f GB (peak %.2f GB).%s",
            n_done,
            self.label,
            self.worker_count() if self.worker_count else "n/a",
            self.label,
            self.live_gb,
            self.peak_gb,
            extra,
        )

    def log_peak(self) -> None:
        logger.info("Peak %s disk usage: %.2f GB.", self.label, self.peak_gb)
