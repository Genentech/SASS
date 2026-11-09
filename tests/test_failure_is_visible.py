"""A failed run has to say why on the terminal, not only in the log file.

`build_context` redirects `sys.stderr` to the log, so `report_failure` writes to the real stderr.
"""

import logging
import subprocess
import types

import pytest
import typer

from sass.logging_setup import StreamToLogger, report_failure


@pytest.fixture()
def log_file(tmp_path):
    """Attach a file handler to the root logger, as `set_logger` does, and take it away after."""
    path = tmp_path / "sass_run.log"
    handler = logging.FileHandler(path)
    logging.getLogger().addHandler(handler)
    yield path
    logging.getLogger().removeHandler(handler)
    handler.close()


class TestTheReasonReachesTheTerminal:
    def test_the_exception_message_is_printed(self, capfd):
        report_failure(ValueError("`cluster_name` is required"))
        assert "`cluster_name` is required" in capfd.readouterr().err

    def test_it_goes_to_the_real_stderr(self, capfd):
        """The message reaches the real stderr file descriptor, which `capfd` sees and `capsys` does not."""
        report_failure(ValueError("boom"))
        captured = capfd.readouterr()
        assert "boom" in captured.err
        assert "boom" not in captured.out

    def test_the_log_file_is_named(self, capfd, log_file):
        """The terminal message names the log file that holds the full traceback."""
        report_failure(ValueError("boom"))
        assert str(log_file) in capfd.readouterr().err

    def test_it_works_before_logging_is_set_up(self, capfd, monkeypatch):
        """The message prints even before a log file exists, as when config validation fails."""
        root = logging.getLogger()
        monkeypatch.setattr(root, "handlers", [h for h in root.handlers if not isinstance(h, logging.FileHandler)])
        report_failure(ValueError("boom"))
        err = capfd.readouterr().err
        assert "boom" in err
        assert "Full traceback" not in err, "there is no log file to point at"

    def test_a_mid_run_failure_reports_without_an_exception(self, capfd, log_file):
        """Called with no exception, it still reports the failure and names the log file."""
        report_failure()
        err = capfd.readouterr().err
        assert "failed" in err.lower()
        assert str(log_file) in err


class TestTracebacksSurviveTheLogRedirect:
    """`StreamToLogger` buffers writes into one log record per line.

    The traceback printer can split one line across several `write()` calls.
    """

    @staticmethod
    def records(caplog):
        return [r.getMessage() for r in caplog.records]

    def test_a_line_split_across_writes_is_one_record(self, caplog):
        stream = StreamToLogger(logging.getLogger("sass.test.stream"), logging.ERROR)
        with caplog.at_level(logging.ERROR, logger="sass.test.stream"):
            stream.write("ValueError")
            stream.write(":")
            stream.write(" `cluster_name` is required\n")
        assert self.records(caplog) == ["ValueError: `cluster_name` is required"]

    def test_several_lines_in_one_write_stay_separate(self, caplog):
        stream = StreamToLogger(logging.getLogger("sass.test.stream"), logging.ERROR)
        with caplog.at_level(logging.ERROR, logger="sass.test.stream"):
            stream.write("first\nsecond\n")
        assert self.records(caplog) == ["first", "second"]

    def test_a_trailing_partial_line_is_flushed(self, caplog):
        """`flush()` logs a buffered line that has no trailing newline."""
        stream = StreamToLogger(logging.getLogger("sass.test.stream"), logging.ERROR)
        with caplog.at_level(logging.ERROR, logger="sass.test.stream"):
            stream.write("no newline here")
            stream.flush()
        assert self.records(caplog) == ["no newline here"]

    def test_flushing_twice_does_not_repeat_it(self, caplog):
        stream = StreamToLogger(logging.getLogger("sass.test.stream"), logging.ERROR)
        with caplog.at_level(logging.ERROR, logger="sass.test.stream"):
            stream.write("once")
            stream.flush()
            stream.flush()
        assert self.records(caplog) == ["once"]

    def test_blank_lines_are_not_logged(self, caplog):
        stream = StreamToLogger(logging.getLogger("sass.test.stream"), logging.ERROR)
        with caplog.at_level(logging.ERROR, logger="sass.test.stream"):
            stream.write("\n\n  \nreal\n")
        assert self.records(caplog) == ["real"]


class TestTheCliDoesNotBuryTheChildsMessage:
    def test_a_failing_run_exits_with_the_childs_code(self, monkeypatch):
        """A failing child run exits with the child's return code."""
        from sass import cli

        monkeypatch.setattr(
            cli.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=1, args=a)  # noqa: ARG005
        )
        with pytest.raises(typer.Exit) as exc:
            cli.run(query="q.sdf")
        assert exc.value.exit_code == 1

    def test_the_wrapper_raises_nothing_of_its_own(self, monkeypatch):
        """The wrapper raises `typer.Exit`, not `CalledProcessError`."""
        from sass import cli

        monkeypatch.setattr(
            cli.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=2, args=a)  # noqa: ARG005
        )
        with pytest.raises(typer.Exit):
            cli.run(query="q.sdf")
        assert not isinstance(pytest.raises(typer.Exit, cli.run, query="q.sdf").value, subprocess.CalledProcessError)

    def test_a_successful_run_exits_cleanly(self, monkeypatch):
        from sass import cli

        monkeypatch.setattr(
            cli.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, args=a)  # noqa: ARG005
        )
        assert cli.run(query="q.sdf") is None
