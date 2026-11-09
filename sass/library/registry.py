"""Resolve `synthon_library` into concrete artifact paths.

A library is a set of artifacts that must all come from the same release. Mixing releases gives
wrong synthons without an error. A library is given in one of two forms:

1. **A registry name**, `synthon_library: my_library`, looked up in `sass/libraries.yml`.
2. **The keys, spelled out**, for a library the registry does not list.

Any key set explicitly overrides the registry and is logged. Every required path is checked
before the run starts.
"""

import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

#: The artifacts a library is made of.
LIBRARY_KEYS = (
    "synthon_conf_dir",
    "synthon_data_file",
    "grouped_synthon_file",
    "synthon_file",
    "reaction_file",
    "synthon_id_index",
    "single_frag_synthon_conf_dir",
)

#: Library keys whose config value is used as a write destination. `group_synthon_data` always
#: writes its two tables to `exp_dir`.
WRITE_HONORED_KEYS = frozenset({"synthon_conf_dir", "synthon_id_index", "single_frag_synthon_conf_dir"})

#: Library keys resolved only if present, since a library is usable without them.
OPTIONAL_KEYS = ("synthon_id_index", "single_frag_synthon_conf_dir")

#: Overrides the detected data root. For tests, and for unusual mounts.
DATA_ROOT_ENV = "SASS_DATA_ROOT"

_REGISTRY_FILE = Path(__file__).resolve().parent.parent / "libraries.yml"


def load_registry(path: Path | None = None) -> dict[str, Any]:
    """Read `libraries.yml`."""
    with (path or _REGISTRY_FILE).open() as f:
        return yaml.safe_load(f)


def detect_data_root(registry: dict[str, Any] | None = None) -> tuple[Path | None, str | None]:
    """Return the first data root that exists, and the host it identifies.

    `$SASS_DATA_ROOT` takes precedence. Returns `(None, None)` when no root is mounted.
    """
    override = os.environ.get(DATA_ROOT_ENV)
    if override:
        logger.info("Data root from $%s: %s", DATA_ROOT_ENV, override)
        return Path(override), None

    registry = registry if registry is not None else load_registry()
    for entry in registry.get("data_roots", []):
        root = Path(entry["path"])
        if root.is_dir():
            return root, entry.get("host")
    return None, None


def detect_host() -> str | None:
    """Which cluster we are on, or `None` if no data root is visible."""
    return detect_data_root()[1]


def _known_names(registry: dict[str, Any]) -> list[str]:
    return sorted(registry.get("aliases", {})) + sorted(registry.get("libraries", {}))


def _entry_for(name: str, registry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Resolve an alias to a concrete registry entry."""
    concrete = registry.get("aliases", {}).get(name, name)
    libraries = registry.get("libraries", {})
    if concrete not in libraries:
        raise ValueError(
            f"Unknown synthon_library {name!r}. Known names: {', '.join(_known_names(registry))}. "
            "For a library the registry does not list, set the library keys explicitly."
        )
    return concrete, libraries[concrete]


def _paths_from_registry(name: str, registry: dict[str, Any]) -> tuple[dict[str, str], str | None]:
    """Resolve a registry name to paths, plus a note if it is not declared for this host.

    When an entry has `hosts:`, the library counts as available only on those hosts.
    """
    concrete, entry = _entry_for(name, registry)
    root, host = detect_data_root(registry)
    if root is None:
        tried = ", ".join(e["path"] for e in registry.get("data_roots", []))
        logger.warning(
            "No SASS data root found (tried: %s), so synthon_library=%r cannot be resolved. "
            "Set $%s, or give every library path explicitly. This is expected off-cluster.",
            tried,
            name,
            DATA_ROOT_ENV,
        )
        return {}, None

    # Log the concrete entry, not the alias, so the log records which library ran.
    logger.info("Resolved synthon_library=%r -> %r under %s%s.", name, concrete, root, f" ({host})" if host else "")
    if entry.get("description"):
        logger.info("  %s", " ".join(str(entry["description"]).split()))

    undeclared = None
    hosts = entry.get("hosts")
    if hosts is not None and host is not None and host not in hosts:
        undeclared = (
            f"Library {concrete!r} is not declared as available on {host!r} "
            f"(registered for: {', '.join(hosts)}). Add {host!r} to its `hosts:` in "
            "sass/libraries.yml once its files are in place."
        )

    return {key: str(root / entry[key]) for key in LIBRARY_KEYS if key in entry}, undeclared


def resolve_library_paths(
    config: dict[str, Any],
    registry: dict[str, Any] | None = None,
    produced: Iterable[str] = (),
    required: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fill unset library paths from `synthon_library`, then check the ones we need exist.

    When `synthon_library` is unset, the configured paths are only checked.

    `produced` names artifacts this run builds rather than reads, which need not exist yet.
    `required` maps the `OPTIONAL_KEYS` this run reads to the reason each is required.
    """
    name = config.get("synthon_library")
    undeclared = None
    if name:
        if os.sep in str(name):
            raise ValueError(
                f"synthon_library must be a library name, not a path ({name!r}). Known names: "
                f"{', '.join(_known_names(registry if registry is not None else load_registry()))}. "
                "For a library the registry does not list, set the library keys explicitly."
            )
        registry = registry if registry is not None else load_registry()
        candidates, undeclared = _paths_from_registry(str(name), registry)

        produced = set(produced)
        overridden = []
        for key, path in candidates.items():
            if config.get(key):
                # An explicit path wins, but is logged, since it may not match the release.
                if str(Path(config[key])) != path:
                    overridden.append(key)
                continue
            if key in produced:
                # Never resolve a build output into the shared library directory.
                remedy = (
                    f"set `{key}` to the path you want written, or unset `synthon_library`"
                    if key in WRITE_HONORED_KEYS
                    # `group_synthon_data` always writes these to `exp_dir`.
                    else "unset `synthon_library` and collect the artifact from `exp_dir`"
                )
                raise ValueError(
                    f"This run builds `{key}`, but its output path is unset and `synthon_library: {name}` would "
                    f"resolve it to {path} -- a shared library this run would write into. Either {remedy}."
                )
            config[key] = path

        if overridden:
            logger.warning(
                "These keys override synthon_library and may not match it: %s. "
                "Library artifacts must all come from the same release.",
                ", ".join(sorted(overridden)),
            )

    _check_library_files(config, undeclared, produced, required)
    return config


def _check_library_files(
    config: dict[str, Any],
    undeclared: str | None = None,
    produced: Iterable[str] = (),
    required: Mapping[str, str] | None = None,
) -> None:
    """Stat every library path that is set, and raise listing all that are missing."""
    produced = set(produced)
    required = required or {}
    missing = []
    for key in LIBRARY_KEYS:
        if key in produced:
            continue  # This run builds it; it is an output, not an input.
        value = config.get(key)
        if value and Path(value).exists():
            continue
        if key in required:
            # Optional for the library, but required by this run.
            missing.append(f"{key}: {value or 'not set'} -- required by {required[key]}")
            continue
        if key in OPTIONAL_KEYS:
            # Unset a missing optional path, so nothing downstream tries to open it.
            if value:
                logger.info("No %s at %s; continuing without it.", key, value)
                config[key] = None
            continue
        missing.append(f"{key}: {value or 'not set'}")

    if missing:
        raise ValueError(
            "Synthon library is incomplete -- these are missing:\n  "
            + "\n  ".join(missing)
            + f"\nCheck `synthon_library` (currently {config.get('synthon_library')!r}) and any explicit overrides."
            + (f"\n\n{undeclared}" if undeclared else "")
        )

    if undeclared:
        # The files are present, so continue, but the registry may need updating.
        logger.warning("%s Continuing: every required artifact is present.", undeclared)
