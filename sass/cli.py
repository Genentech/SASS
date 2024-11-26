#!/usr/bin/env python3
"""
(C) 2024 Genentech. All rights reserved.

Console script for sass.
"""

import os
import subprocess
import sys
from typing import Annotated

import typer

from sass import SRC_DIR

app = typer.Typer(help="Provides a basic command-line interface for running SASS.")


@app.command("run", help="Run SASS workflow.")
def run(
    config_file: Annotated[
        str,
        typer.Option(
            ...,
            "--config",
            "-c",
            help=(
                "Path to the config file for a SASS run. For an example config file, "
                "see ./sass/config_template.yml."
            ),
        ),
    ]
):
    """Launches a SASS run."""
    cmd = [sys.executable, SRC_DIR / "query_main.py", "--config", config_file]
    subprocess.run(cmd, check=True, env=os.environ.copy())


@app.command(hidden=True)
def _hidden_fn() -> None:
    raise NotImplementedError


# if __name__ == '__main__':
#     app()
