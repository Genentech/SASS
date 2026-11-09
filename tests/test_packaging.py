"""The wheel `python -m build` produces must ship every file `sass` reads by path.

`config_template.yml` and `libraries.yml` are resolved relative to the installed package, so a
wheel without them installs but fails on the first run.
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from sass import SRC_DIR

#: Building a wheel takes longer than the project-wide unit-test timeout.
pytestmark = pytest.mark.timeout(600)

REPO_ROOT = SRC_DIR.parent

#: Every file resolved by path relative to the installed `sass/` package directory at run time.
#: Keep in sync with `[tool.setuptools.package-data]` in pyproject.toml.
REQUIRED_PACKAGE_FILES = ("sass/config_template.yml", "sass/libraries.yml")


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    """Build the wheel with the real build backend. `--no-isolation` lets it run offline."""
    out_dir = tmp_path_factory.mktemp("wheel")
    # `build` also writes these next to `pyproject.toml`; they are removed afterward.
    egg_info = REPO_ROOT / "sass_public.egg-info"
    build_dir = REPO_ROOT / "build"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", str(out_dir)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"`python -m build` failed with exit {result.returncode}\n"
                f"--- stdout ---\n{result.stdout[-3000:]}\n--- stderr ---\n{result.stderr[-3000:]}"
            )
        wheels = list(out_dir.glob("*.whl"))
        assert len(wheels) == 1, f"expected exactly one wheel in {out_dir}, found {wheels}"
        return wheels[0]
    finally:
        shutil.rmtree(egg_info, ignore_errors=True)
        shutil.rmtree(build_dir, ignore_errors=True)


def test_no_package_yaml_is_left_out_of_the_wheel(built_wheel):
    """Every YAML file under `sass/` is in the wheel.

    The package data lists YAML files by name, so a new one is easy to leave out.
    """
    on_disk = {f"sass/{path.name}" for path in SRC_DIR.glob("*.yml")}
    assert on_disk, "no package YAML found at all; this test has stopped looking in the right place"

    with zipfile.ZipFile(built_wheel) as wheel:
        shipped = set(wheel.namelist())
    missing = sorted(on_disk - shipped)
    assert not missing, (
        f"{missing} exist under sass/ but are not in the wheel -- add them to "
        "[tool.setuptools.package-data] in pyproject.toml and to MANIFEST.in"
    )


def test_required_package_files_are_in_the_wheel(built_wheel):
    with zipfile.ZipFile(built_wheel) as wheel:
        names = set(wheel.namelist())
        missing = [f for f in REQUIRED_PACKAGE_FILES if f not in names]
        assert not missing, (
            f"{missing} missing from {built_wheel.name} -- check MANIFEST.in and "
            "[tool.setuptools.package-data] in pyproject.toml"
        )

        # Present-but-empty would pass a plain membership check and still break every run.
        sizes = {info.filename: info.file_size for info in wheel.infolist()}
        empty = [f for f in REQUIRED_PACKAGE_FILES if sizes.get(f, 0) == 0]
        assert not empty, f"{empty} are in {built_wheel.name} but zero bytes"
