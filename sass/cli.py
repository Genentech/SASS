#!/usr/bin/env python3
"""Console script for sass."""

import subprocess
import sys
from typing import Annotated, Optional

import typer

import sass
from sass import SRC_DIR
from sass.library.registry import LIBRARY_KEYS, detect_data_root, load_registry

app = typer.Typer(help="Provides a basic command-line interface for running SASS.")


@app.command("run", help="Run SASS workflow.")
def run(
    query: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--query",
            "-q",
            help="Query molecule file -- a 3D conformer, SDF V2000 or oeb. Overrides the config.",
        ),
    ] = None,
    config_file: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--config",
            "-c",
            help=(
                "Config file for the run. Optional -- without it the packaged template's "
                "defaults are used. See sass/config_template.yml."
            ),
        ),
    ] = None,
    library: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--library",
            help="Synthon library name from sass/libraries.yml. See `sass list-libraries`.",
        ),
    ] = None,
    setting: Annotated[
        Optional[list[str]],
        typer.Option(
            ...,
            "--set",
            metavar="KEY=VALUE",
            help="Override any config key, e.g. --set top_m=500000. Repeatable.",
        ),
    ] = None,
):
    """Launch a SASS run."""
    if config_file is None and query is None:
        raise typer.BadParameter("give --query (and optionally --config), or a --config that sets query_molecule.")

    base_cmd = [sys.executable, str(SRC_DIR / "main.py")]
    if config_file is not None:
        base_cmd += ["--config", config_file]
    if query is not None:
        base_cmd += ["--query", query]
    if library is not None:
        base_cmd += ["--library", library]
    for override in setting or []:
        base_cmd += ["--set", override]
    # The child has already reported its error, so exit without a traceback.
    result = subprocess.run(base_cmd, check=False)
    if result.returncode:
        raise typer.Exit(result.returncode)


@app.command("list-libraries", help="List the synthon libraries `synthon_library` accepts.")
def list_libraries():
    """Show every registered library and whether it is usable here."""
    registry = load_registry()
    root, host = detect_data_root(registry)
    if root is None:
        tried = ", ".join(e["path"] for e in registry.get("data_roots", []))
        typer.echo(f"No SASS data root mounted (tried: {tried}). Availability is unknown.\n")
    else:
        typer.echo(f"Data root: {root}" + (f"  (host: {host})" if host else "") + "\n")

    aliases = registry.get("aliases", {})
    by_target: dict[str, list[str]] = {}
    for alias, target in aliases.items():
        by_target.setdefault(target, []).append(alias)

    for name, entry in sorted(registry.get("libraries", {}).items()):
        if root is None:
            status = "?"
        elif entry.get("hosts") is not None and host not in entry["hosts"]:
            status = f"not on {host}"
        elif all((root / entry[k]).exists() for k in LIBRARY_KEYS if k in entry):
            status = "available"
        else:
            status = "MISSING FILES"

        names = ", ".join(sorted(by_target.get(name, [])))
        typer.echo(f"  {name:<24} {status:<14} {'(' + names + ')' if names else ''}")
        if entry.get("description"):
            typer.echo(f"      {' '.join(str(entry['description']).split())}")
        # Omitted artifacts do not change the status, but are listed.
        unnamed = [k for k in LIBRARY_KEYS if k not in entry]
        if unnamed:
            typer.echo(f"      does not provide: {', '.join(unnamed)}")
        # The settings a single-fragment run has to match.
        geometry = entry.get("single_frag_geometry")
        if geometry:
            settings = ", ".join(f"{k}: {v}" for k, v in geometry.items())
            typer.echo(f"      single-fragment queries need -- {settings}")

    typer.echo(
        "\nUse either the alias in parentheses or the full name. An alias is a moving pointer "
        "to the current release;\nthe full name pins one. For a library not listed here, set "
        "the library keys in the config explicitly."
    )


@app.command("version", help="Print the version of SASS.")
def version():
    """Prints the version of SASS."""
    print(sass.__version__)  # noqa: T201


if __name__ == "__main__":
    app()
