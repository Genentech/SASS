"""General utils."""

# Standard Library
import functools
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Union

import yaml

# Third Party Library
from openeye import oechem


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.

    https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    """

    def __init__(self, logger: Union[Path, str], level: str) -> None:
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

    def write(self, buf: str):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def set_logger(log_file: Union[Path, str]):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S"
    )

    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_file)
    logFormatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.INFO)
    rootLogger.addHandler(fileHandler)
    return rootLogger


def load_config(conf_file: Union[Path, str]):
    with Path(conf_file).open("r") as f:
        return yaml.safe_load(f)


class CustomOEErrorHandler(oechem.OEErrorHandlerImplBase):
    """
    Custom error handler for writing to both STDOUT and log file.

    Based on code from Kriszti Boda.
    """

    def __init__(self, log_file: str | None = None) -> None:
        """Write something here to satisfy ruff D107."""
        oechem.OEErrorHandlerImplBase.__init__(self)
        self._log_file: oechem.oeofstream = oechem.oeofstream()
        self._log_file.append(log_file)

    def Msg(self, level: int, msg: str):
        if level in (oechem.OEErrorLevel_Error, oechem.OEErrorLevel_Fatal):
            self._log_file.write("Preventing call to exit: {0}\n".format(msg))
            logging.error(f"{oechem.OEErrorLevelToString(level)}: {msg}")
            sys.exit(1)
        # elif level == oechem.OEErrorLevel_Verbose:
        #     self._log_file.write(f"Verbose: {msg}\n")
        elif level == oechem.OEErrorLevel_Warning:
            self._log_file.write(f"Warning: {msg}\n")
            # sys.stdout.write(f"Warning: {msg}\n")
        # elif level == oechem.OEErrorLevel_Info:
        #     self._log_file.write(f"Info: {msg}\n")
        # else:
        #     self._log_file.write(f"{level}: {msg}\n")

    def CreateCopy(self):
        return CustomOEErrorHandler().__disown__()


def extract_base_id(mol_id: str, warts_separator: str = "_", keep_isomer_warts: bool = False) -> str:
    """
    Extract base id from input id strings that contain isomer/conformer warts.

    Parameters
    ----------
    mol_id
        Input id string.
    warts_separator
        String separator used to identify id parts.
    keep_isomer_warts
        Whether to keep isomer warts in the id string.
    """
    if keep_isomer_warts is False:
        return remove_isomer_warts(mol_id, warts_separator)
    else:
        return mol_id


def remove_isomer_warts(mol_id: str, warts_separator: str = "_") -> str:
    """Remove isomer warts from input id strings.

    Note that this does not work when there's also conformer warts in the id string.
    However, currently there is no cases where conformer warts are stored.
    """
    if warts_separator not in mol_id:
        return mol_id
    else:
        id_parts = mol_id.split(warts_separator)
        return warts_separator.join(id_parts[:-1])


def wait(files: Iterable, wait_time: int = 30, final_delay: int = 10):
    """Wait for all specified files to exist."""
    while not all(Path(f).is_file() for f in files):
        time.sleep(wait_time)

    time.sleep(final_delay)  # To give time for files to fully write out. For small files, 10 sec should be enough.


def make_temp_copy(file: Path | str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as tmp:
        shutil.copyfile(file, tmp.name)
        return Path(tmp.name)


def delay_return(func: Callable | None = None, delay: int = 10) -> callable:
    """Add a delay to a function to ensure it runs for minimally x seconds.

    This is useful for dask tasks that finish too quickly and may cause issues when
    communicating with the head node.
    """
    if func is None:
        return lambda func: delay_return(func, delay=delay)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Add a delay to a function to ensure it runs for at least x seconds."""
        start = time.time()
        res = func(*args, **kwargs)
        _delay = max(0, delay - (time.time() - start))
        logging.info(f"Sleeping for {_delay} seconds.")
        time.sleep(_delay)
        return res

    return wrapper
