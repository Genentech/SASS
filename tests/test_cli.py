"""Test file for sass/cli.py interface."""

import os

from . import run_help


class TestCli:
    """Test class for sass/cli.py."""

    @classmethod
    def setup_class(cls):
        """Set-up method for test class."""
        cls._pyscript = os.path.join("..", "sass", "cli.py")

    def test_run_help(self):
        run_help(self._pyscript)
