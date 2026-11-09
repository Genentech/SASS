"""dask worker plugins: per-worker logging, and the OpenEye error handler."""

import logging
import sys
from pathlib import Path

from dask.distributed import Worker, WorkerPlugin

from sass.logging_setup import StreamToLogger, set_logger

logger = logging.getLogger(__name__)


class WorkerLoggingPlugin(WorkerPlugin):
    """Plugin to write worker logs, stdout, and stderr to file."""

    def __init__(self, log_dir: Path | str) -> None:
        self.log_dir = Path(log_dir)

    def setup(self, worker: Worker):
        worker_id = worker.id
        log_file = self.log_dir / f"{worker_id}.log"
        logger = set_logger(log_file)
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)


class OEChemSetupPlugin(WorkerPlugin):
    """Proactively import oechem package for `uv` environments to avoid ModuleNotFoundError."""

    def setup(self, worker: Worker):  # noqa: ARG002 # `worker` arg must be present per dask requirements.
        try:
            from openeye import oechem  # noqa: PLC0415

            logger.info(f"Successfully imported oechem from {oechem.__file__}")
        except ImportError:
            logger.exception("Failed to import oechem. Downstream worker tasks may fail.")
            raise
