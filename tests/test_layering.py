"""The package import layering, enforced, plus checks that tasks, config keys and docs agree.

`selection/` stays free of OpenEye and dask so it can be tested in isolation, and `worker/` avoids
heavy imports because every worker process loads it.
"""

import ast
from pathlib import Path

import pytest

from sass import SRC_DIR

#: package -> modules it may not import (prefix match on the top-level name).
#: Only packages that exist are checked.
FORBIDDEN = {
    "selection": ("openeye", "dask", "dask_jobqueue", "pandas", "matplotlib", "seaborn", "sklearn"),
    "worker": ("dask", "dask_jobqueue", "matplotlib", "seaborn", "sklearn"),
    "chem": ("dask", "dask_jobqueue", "matplotlib", "seaborn", "sklearn"),
}

#: package -> `sass.*` submodules it may not import.
FORBIDDEN_INTERNAL = {
    "selection": ("sass.chem", "sass.library", "sass.cluster", "sass.worker", "sass.tasks", "sass.analysis"),
    "worker": ("sass.cluster", "sass.tasks", "sass.analysis"),
    "chem": ("sass.cluster", "sass.worker", "sass.tasks", "sass.analysis"),
}


def imported_modules(path: Path) -> set[str]:
    """Every module name imported by `path`, including inside functions."""
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def modules_in(package: str) -> list[Path]:
    d = SRC_DIR / package
    return sorted(d.rglob("*.py")) if d.is_dir() else []


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_does_not_import_forbidden_third_party(package):
    files = modules_in(package)
    if not files:
        pytest.skip(f"sass/{package}/ does not exist yet")
    violations = [
        (f.relative_to(SRC_DIR).as_posix(), name)
        for f in files
        for name in imported_modules(f)
        if name.split(".")[0] in FORBIDDEN[package]
    ]
    assert not violations, f"sass/{package}/ must not import these: {violations}"


@pytest.mark.parametrize("package", sorted(FORBIDDEN_INTERNAL))
def test_package_does_not_import_forbidden_sass_layers(package):
    files = modules_in(package)
    if not files:
        pytest.skip(f"sass/{package}/ does not exist yet")
    banned = FORBIDDEN_INTERNAL[package]
    violations = [
        (f.relative_to(SRC_DIR).as_posix(), name)
        for f in files
        for name in imported_modules(f)
        if any(name == b or name.startswith(b + ".") for b in banned)
    ]
    assert not violations, f"sass/{package}/ must not import these sass layers: {violations}"


@pytest.mark.parametrize("module", ["ids", "io"])
def test_leaf_modules_import_nothing_from_sass(module):
    """`ids` and `io` are the bottom of the stack; everything may import them."""
    path = SRC_DIR / f"{module}.py"
    if not path.is_file():
        pytest.skip(f"sass/{module}.py does not exist")
    sass_imports = {n for n in imported_modules(path) if n == "sass" or n.startswith("sass.")}
    assert not sass_imports, f"sass/{module}.py must not import from sass: {sass_imports}"


def test_every_config_task_has_a_module_exposing_run():
    """`preflight.VALID_TASKS` matches `main.TASK_MODULES`, and each task module exposes `run(ctx, client)`."""
    import inspect
    from importlib import import_module

    from sass.main import TASK_MODULES
    from sass.preflight import valid_task_names

    assert set(valid_task_names()) == set(
        TASK_MODULES
    ), f"config and TASK_MODULES disagree: {set(valid_task_names()) ^ set(TASK_MODULES)}"
    for name in TASK_MODULES:
        module = import_module(f"sass.tasks.{name}")
        assert hasattr(module, "run"), f"sass/tasks/{name}.py has no run()"
        assert list(inspect.signature(module.run).parameters) == [
            "ctx",
            "client",
        ], f"sass/tasks/{name}.py: run() must take (ctx, client)"


def test_every_config_key_is_documented():
    """Every key in the config template appears in `guide/configuration.md`."""
    import yaml

    config = yaml.safe_load((SRC_DIR / "config_template.yml").read_text())
    reference = (SRC_DIR.parent / "guide" / "configuration.md").read_text()

    undocumented = [key for key in config if f"`{key}`" not in reference]
    assert not undocumented, f"not in guide/configuration.md: {undocumented}"


def test_the_config_template_stays_terse():
    """Comment lines stay below `max_comment_fraction` of the config template.

    Long-form guidance belongs in `guide/`.
    """
    max_comment_fraction = 0.35
    lines = (SRC_DIR / "config_template.yml").read_text().splitlines()
    comment_only = sum(1 for line in lines if line.strip().startswith("#"))
    assert (
        comment_only / len(lines) < max_comment_fraction
    ), f"{comment_only}/{len(lines)} lines are comments; move the prose to guide/"
