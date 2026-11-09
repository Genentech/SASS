"""Load a run config from disk. Validation lives in `sass.preflight`."""

import logging
from pathlib import Path
from typing import Union

import yaml

logger = logging.getLogger(__name__)


def load_config(conf_file: Union[Path, str]):
    with Path(conf_file).open("r") as f:
        return yaml.safe_load(f)
