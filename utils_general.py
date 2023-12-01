# Standard Library
import logging
import os
from pathlib import Path
import sys
import time
from typing import Iterable, Union

# Third Party Library
from openeye import oechem
import yaml


class StreamToLogger:
    """
    Imitate file-like stream object that redirects writes to a logger instance.

    https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
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


def load_config(conf_file):
    with open(conf_file, "r") as f:
        config = yaml.safe_load(f)
    return config


class CustomOEErrorHandler(oechem.OEErrorHandlerImplBase):
    """
    Custom error handler for writing to both STDOUT and log file.

    Based on code from Kriszti Boda.
    """

    def __init__(self, log_file: Union[str, Path]):
        oechem.OEErrorHandlerImplBase.__init__(self)
        self._log_file: oechem.oeofstream = oechem.oeofstream()
        self._log_file.append(log_file)

    def Msg(self, level: int, msg: str):
        if level == oechem.OEErrorLevel_Error or level == oechem.OEErrorLevel_Fatal:
            self._log_file.write("Preventing call to exit: {0}\n".format(msg))
            print("{}: {}".format(oechem.OEErrorLevelToString(level), msg))
            sys.exit(1)
        elif level == oechem.OEErrorLevel_Verbose:
            self._log_file.write(f"Verbose: {msg}\n")
        elif level == oechem.OEErrorLevel_Warning:
            self._log_file.write(f"Warning: {msg}\n")
        elif level == oechem.OEErrorLevel_Info:
            self._log_file.write(f"Info: {msg}\n")
        else:
            self._log_file.write(f"{level}: {msg}\n")

    def CreateCopy(self):
        return CustomOEErrorHandler().__disown__()


def extract_base_id(mol_id: str, warts_separator: str = "_") -> str:
    """
    Extract base id from input id strings that contain isomer/conformer warts.
    231110: Infer from the `mol_id` on which parts is warts by using a different separator
    for joining based_ids, e.g. "&". Now, base_ids is assumed to be whatever string that
    comes before the 1st "_" character.

    Args:
        mol_id: Input id.
        separator: String separator used to identify id parts.
    """

    return mol_id.split(warts_separator)[0]


def wait(files: Iterable, wait_time: int = 30, final_delay: int = 10):
    """
    Wait for all specified files to exist.
    """

    while not all(os.path.isfile(f) for f in files):
        time.sleep(wait_time)

    time.sleep(final_delay)
