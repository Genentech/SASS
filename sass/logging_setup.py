"""Logging setup: run log files, stdout/stderr redirection, and the OpenEye error handler."""

# Standard Library
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Union

# Third Party Library
from openeye import oechem

logger = logging.getLogger(__name__)


class FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after every emit for immediate log visibility."""

    def emit(self, record: logging.LogRecord):
        super().emit(record)
        self.flush()


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.

    https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    """

    def __init__(self, logger: logging.Logger, level: int) -> None:
        """
        Instantiate the logger.

        Parameters
        ----------
        logger
            Name/Path to save the log to.
        level
            Level of the logs.
        """
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf: str) -> None:
        # Buffer to newlines, so a line split across several writes becomes one log record.
        self.linebuf += buf
        while "\n" in self.linebuf:
            line, self.linebuf = self.linebuf.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line.rstrip())

    def flush(self) -> None:
        if self.linebuf.strip():
            self.logger.log(self.level, self.linebuf.rstrip())
        self.linebuf = ""


def set_logger(
    log_file: Union[Path, str], log_level: int = logging.INFO, logger_name: str | None = None
) -> logging.Logger:

    if logger_name is not None:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()

    logger.setLevel(log_level)  # Set logger level (not just handler level)
    fileHandler = FlushingFileHandler(log_file)
    logFormatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(log_level)
    logger.addHandler(fileHandler)
    return logger


class CustomOEErrorHandler(oechem.OEErrorHandlerImplBase):
    """
    Custom error handler for writing to both STDOUT and log file.

    Based on code from Kriszti Boda.
    """

    #: Collapses quoted names and numbers, so repeats of one warning share a signature.
    _VARIABLE_PART = re.compile(r"'[^']*'|\d+")

    def __init__(
        self,
        log_file: str | None = None,
        logger: logging.Logger | None = None,
        max_repeats: int = 1,
        warning_counts: Counter | None = None,
    ) -> None:
        """Initialize the custom error handler.

        Parameters
        ----------
        log_file
            Path to the log file for OpenEye error output.
        logger
            Logger instance for logging errors.
        max_repeats
            How many times to write each distinct warning before suppressing it. Set to 0 to
            drop warnings entirely.
        warning_counts
            Counter shared with the copies made by `CreateCopy`, so all copies use one tally.
        """
        oechem.OEErrorHandlerImplBase.__init__(self)
        self._log_file: oechem.oeofstream = oechem.oeofstream()
        self._log_file.append(log_file)
        self._logger = logger
        self._log_file_path = log_file
        self._max_repeats = max_repeats
        self._warning_counts: Counter = Counter() if warning_counts is None else warning_counts

    def Msg(self, level: int, msg: str):
        if level in (oechem.OEErrorLevel_Error, oechem.OEErrorLevel_Fatal):
            self._log_file.write("Preventing call to exit: {0}\n".format(msg))
            self._log_file.flush()
            if self._logger:
                self._logger.error(f"{oechem.OEErrorLevelToString(level)}: {msg}")
            sys.exit(1)
        # elif level == oechem.OEErrorLevel_Verbose:
        #     self._log_file.write(f"Verbose: {msg}\n")
        elif level == oechem.OEErrorLevel_Warning:
            signature = self._VARIABLE_PART.sub("*", msg)
            self._warning_counts[signature] += 1
            seen = self._warning_counts[signature]
            if seen <= self._max_repeats:
                self._log_file.write(f"Warning: {msg}\n")
                if seen == self._max_repeats:
                    self._log_file.write(
                        "         (further warnings of this kind suppressed; "
                        "totals are logged at the end of the run)\n"
                    )
                # `oeofstream` buffers, so flush to make warnings visible during the run.
                self._log_file.flush()
        # elif level == oechem.OEErrorLevel_Info:
        #     self._log_file.write(f"Info: {msg}\n")
        # else:
        #     self._log_file.write(f"{level}: {msg}\n")

    def log_warning_summary(self) -> None:
        """Log how many times each suppressed warning fired. Call once at the end of a run."""
        suppressed = {sig: n for sig, n in self._warning_counts.items() if n > self._max_repeats}
        if not suppressed or self._logger is None:
            return
        self._logger.info("OpenEye warning totals (message pattern -> times raised):")
        for signature, count in sorted(suppressed.items(), key=lambda kv: -kv[1]):
            self._logger.info("  %8d x %s", count, signature)

    def CreateCopy(self):
        return CustomOEErrorHandler(
            self._log_file_path,
            self._logger,
            max_repeats=self._max_repeats,
            warning_counts=self._warning_counts,
        ).__disown__()


def report_failure(exc: BaseException | None = None) -> None:
    """Put the reason a run failed on the real stderr, and say where the traceback is.

    Needed because `StreamToLogger` redirects `sys.stderr` into the log file.
    """
    # The most recently attached file handler is this run's.
    log_file = next(
        (h.baseFilename for h in reversed(logging.getLogger().handlers) if isinstance(h, logging.FileHandler)),
        None,
    )
    lines = [f"SASS failed: {exc}" if exc is not None else "SASS run failed."]
    if log_file:
        lines.append(f"Full traceback: {log_file}")
    # Write to the original stderr, since `sys.stderr` points into the log.
    print("\n".join(lines), file=sys.__stderr__ or sys.stderr, flush=True)
