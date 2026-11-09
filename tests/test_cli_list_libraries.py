"""`sass list-libraries` shows what `synthon_library` accepts and whether each library resolves on this host."""

import pytest
from typer.testing import CliRunner

import sass.cli
from sass.cli import app

runner = CliRunner()


@pytest.fixture()
def registry():
    return {
        "data_roots": [{"path": "/nonexistent", "host": "cluster_b"}],
        "aliases": {"lib": "lib_202503"},
        "libraries": {
            "lib_202503": {
                "description": "A test library.",
                "hosts": ["cluster_a"],
                "synthon_conf_dir": "conformers",
                "synthon_data_file": "data.pkl",
                "grouped_synthon_file": "grouped.pkl",
                "synthon_file": "synthons.txt",
                "reaction_file": "reactions.txt",
            },
            "other_202401": {
                "synthon_conf_dir": "other/conformers",
                "synthon_data_file": "other/data.pkl",
                "grouped_synthon_file": "other/grouped.pkl",
                "synthon_file": "other/synthons.txt",
                "reaction_file": "other/reactions.txt",
            },
        },
    }


@pytest.fixture()
def complete_root(tmp_path):
    """A data root where `lib_202503` is fully present and `other_202401` is not."""
    (tmp_path / "conformers").mkdir()
    for name in ("data.pkl", "grouped.pkl", "synthons.txt", "reactions.txt"):
        (tmp_path / name).touch()
    return tmp_path


def run(monkeypatch, registry, root, host):
    # `cli` imports these by name, so patch them there rather than in `registry`.
    monkeypatch.setattr(sass.cli, "load_registry", lambda: registry)
    monkeypatch.setattr(sass.cli, "detect_data_root", lambda _r=None: (root, host))
    result = runner.invoke(app, ["list-libraries"])
    assert result.exit_code == 0, result.output
    return result.output


def test_lists_names_aliases_and_descriptions(monkeypatch, registry, complete_root):
    out = run(monkeypatch, registry, complete_root, "cluster_a")
    assert "lib_202503" in out
    assert "(lib)" in out, "the alias is what most people will type"
    assert "A test library." in out


def test_marks_a_complete_library_available(monkeypatch, registry, complete_root):
    out = run(monkeypatch, registry, complete_root, "cluster_a")
    assert "lib_202503" in out
    assert "available" in out.split("lib_202503")[1].split("\n")[0]


def test_marks_a_library_missing_files(monkeypatch, registry, complete_root):
    """`other_202401` declares no `hosts:`, so it is expected on every host, but its files are absent."""
    out = run(monkeypatch, registry, complete_root, "cluster_a")
    assert "MISSING FILES" in out.split("other_202401")[1].split("\n")[0]


def test_marks_a_library_not_registered_for_this_host(monkeypatch, registry, complete_root):
    """The data may exist elsewhere, but it is not declared for this cluster."""
    out = run(monkeypatch, registry, complete_root, "cluster_b")
    assert "not on cluster_b" in out.split("lib_202503")[1].split("\n")[0]


def test_off_cluster_says_so_rather_than_guessing(monkeypatch, registry):
    """Without a mounted data root, the command reports that rather than a status per library."""
    out = run(monkeypatch, registry, None, None)
    assert "No SASS data root mounted" in out
    assert "/nonexistent" in out, "say which roots were tried"
    assert "MISSING FILES" not in out, "absence of a mount is not evidence of a broken library"


def test_explains_how_to_use_an_unlisted_library(monkeypatch, registry, complete_root):
    out = run(monkeypatch, registry, complete_root, "cluster_a")
    assert "library keys" in out


def test_the_shipped_registry_lists_without_error():
    """Smoke test against the shipped `libraries.yml`, whatever host this is."""
    result = runner.invoke(app, ["list-libraries"])
    assert result.exit_code == 0, result.output


def test_it_is_reachable_as_a_subcommand():
    result = runner.invoke(app, ["--help"])
    assert "list-libraries" in result.output


def test_no_stale_paths_in_the_help_text():
    """The command does not take a config file."""
    result = runner.invoke(app, ["list-libraries", "--help"])
    assert result.exit_code == 0
    assert "--config" not in result.output
