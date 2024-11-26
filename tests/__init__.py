"""
(C) 2023 Genentech. All rights reserved.

Test file for sass module
"""
import pathlib
import sys
import typing

import pytest_check as check
import scripttest


def run_script(*args: typing.Any, **kwargs: typing.Any) -> scripttest.ProcResult:
    """Runs Python script in test environment."""
    env = scripttest.TestFileEnvironment("./test-outputs")
    return env.run(sys.executable, *args, **kwargs)


def run_help(pyscript: str) -> None:
    """Runs Python script with '--help' and generates output to be included into documentation."""
    script = run_script(pyscript, "--help", expect_stderr=True)
    check.equal(0, script.returncode)
    help_desc = script.stdout.split("\n")

    script_base_name = pathlib.Path(pyscript).name[:-3]
    help_file_name = pathlib.Path("docs", "text", script_base_name + "-help.txt")
    with pathlib.Path(help_file_name).open("w", encoding="utf8") as help_file:
        for line in help_desc:
            help_file.write(line + "\n")
