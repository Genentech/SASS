"""Resolving `synthon_library` into concrete paths.

Covers the accepted forms (registry name, explicit keys, nothing) and the startup check that
every required path exists before a cluster is created.
"""

import logging
from pathlib import Path
from typing import ClassVar

import pytest
import yaml

from sass.library.registry import (
    DATA_ROOT_ENV,
    LIBRARY_KEYS,
    OPTIONAL_KEYS,
    detect_data_root,
    load_registry,
    resolve_library_paths,
)
from sass.preflight import (
    _BUILDS,
    QUERY_INPUT,
    SCORING_TASKS,
    TASK_REQUIREMENTS,
    VALID_TASKS,
    optional_artifacts_this_run_reads,
)


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    """A data root with a library's artifacts split across two directories."""
    root = tmp_path / "data"
    (root / "lib_a" / "synthon_conformers").mkdir(parents=True)
    (root / "lib_b").mkdir(parents=True)
    for name in ("synthon_id_index.pkl", "synthon_data.pkl", "grouped_synthons.pkl"):
        (root / "lib_a" / name).touch()
    (root / "lib_b" / "synthons.txt").touch()
    (root / "lib_b" / "reactions.txt").touch()
    monkeypatch.setenv(DATA_ROOT_ENV, str(root))
    return root


@pytest.fixture()
def registry():
    return {
        "data_roots": [{"path": "/nonexistent/cluster_b", "host": "cluster_b"}],
        "aliases": {"lib": "lib_202503"},
        "libraries": {
            "lib_202503": {
                "description": "test library",
                "synthon_conf_dir": "lib_a/synthon_conformers",
                "synthon_id_index": "lib_a/synthon_id_index.pkl",
                "synthon_data_file": "lib_a/synthon_data.pkl",
                "grouped_synthon_file": "lib_a/grouped_synthons.pkl",
                "synthon_file": "lib_b/synthons.txt",
                "reaction_file": "lib_b/reactions.txt",
            }
        },
    }


class TestRegistryName:
    def test_an_alias_resolves_to_every_path_it_names(self, fake_root, registry):
        config = {"synthon_library": "lib"}
        resolve_library_paths(config, registry)
        assert {key: config[key] for key in LIBRARY_KEYS if key in config} == {
            "synthon_conf_dir": str(fake_root / "lib_a/synthon_conformers"),
            "synthon_id_index": str(fake_root / "lib_a/synthon_id_index.pkl"),
            "synthon_data_file": str(fake_root / "lib_a/synthon_data.pkl"),
            "grouped_synthon_file": str(fake_root / "lib_a/grouped_synthons.pkl"),
            "synthon_file": str(fake_root / "lib_b/synthons.txt"),
            "reaction_file": str(fake_root / "lib_b/reactions.txt"),
        }

    @pytest.mark.usefixtures("fake_root")
    def test_artifacts_may_live_in_different_directories(self, registry):
        """A library's artifacts may live in different directories under the data root."""
        config = {"synthon_library": "lib"}
        resolve_library_paths(config, registry)
        assert "lib_a" in config["synthon_conf_dir"]
        assert "lib_b" in config["synthon_file"]

    @pytest.mark.usefixtures("fake_root")
    def test_the_concrete_entry_is_logged_not_the_alias(self, registry, caplog):
        """The log names the concrete library an alias points to."""
        with caplog.at_level(logging.INFO, logger="sass.library.registry"):
            resolve_library_paths({"synthon_library": "lib"}, registry)
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "'lib_202503'" in messages, messages

    @pytest.mark.usefixtures("fake_root")
    def test_a_concrete_name_works_without_going_through_an_alias(self, registry):
        config = {"synthon_library": "lib_202503"}
        resolve_library_paths(config, registry)
        assert config["synthon_data_file"].endswith("synthon_data.pkl")

    @pytest.mark.usefixtures("fake_root")
    def test_an_unknown_name_lists_what_is_known(self, registry):
        with pytest.raises(ValueError, match="Unknown synthon_library") as exc:
            resolve_library_paths({"synthon_library": "misspelled_library"}, registry)
        assert "lib" in str(exc.value)
        assert "lib_202503" in str(exc.value)


class TestPathsAreRejected:
    """`synthon_library` is a name, never a path. The error points to the explicit library keys."""

    @pytest.mark.parametrize("value", ["/data/mylib", "./mylib", "some/dir"])
    def test_a_path_is_refused_with_a_route_forward(self, value, registry):
        with pytest.raises(ValueError, match="must be a library name, not a path") as exc:
            resolve_library_paths({"synthon_library": value}, registry)
        assert "set the library keys explicitly" in str(exc.value)
        assert "lib" in str(exc.value), "the error should list what names are available"

    def test_an_unregistered_library_works_via_explicit_keys(self, tmp_path):
        """The supported route for a library the registry does not list."""
        lib = tmp_path / "mylib"
        (lib / "conformers").mkdir(parents=True)
        for name in ("data.pkl", "grouped.pkl", "odd-name_synthons.txt", "v2_reactions.txt"):
            (lib / name).touch()

        config = {
            "synthon_conf_dir": str(lib / "conformers"),
            "synthon_data_file": str(lib / "data.pkl"),
            "grouped_synthon_file": str(lib / "grouped.pkl"),
            "synthon_file": str(lib / "odd-name_synthons.txt"),
            "reaction_file": str(lib / "v2_reactions.txt"),
        }
        resolve_library_paths(config)
        assert config["synthon_file"].endswith("odd-name_synthons.txt")


class TestStartupCheck:
    def test_every_missing_path_is_reported_at_once(self, fake_root, registry):
        """All missing paths are reported in one error."""
        (fake_root / "lib_b" / "reactions.txt").unlink()
        (fake_root / "lib_a" / "synthon_data.pkl").unlink()

        with pytest.raises(ValueError, match="Synthon library is incomplete") as exc:
            resolve_library_paths({"synthon_library": "lib"}, registry)
        assert "reaction_file" in str(exc.value)
        assert "synthon_data_file" in str(exc.value)

    def test_a_missing_id_index_is_allowed(self, fake_root, registry):
        """A string-id library has no id index, so a missing one is unset rather than an error."""
        (fake_root / "lib_a" / "synthon_id_index.pkl").unlink()
        config = {"synthon_library": "lib"}
        resolve_library_paths(config, registry)
        assert not config["synthon_id_index"], "a missing optional artifact must be unset, not left dangling"
        assert config["synthon_file"], "the rest of the library still resolves"

    def test_hand_written_configs_are_checked_too(self, fake_root):
        """Explicit paths without `synthon_library` are checked too."""
        config = {key: str(fake_root / "lib_a" / "nope") for key in LIBRARY_KEYS}
        with pytest.raises(ValueError, match="Synthon library is incomplete"):
            resolve_library_paths(config)


class TestOverrides:
    def test_an_explicit_path_wins_and_is_flagged(self, fake_root, registry, caplog):
        """An explicit path overrides the library entry and logs a warning."""
        other = fake_root / "lib_a" / "other_synthons.txt"
        other.touch()
        config = {"synthon_library": "lib", "synthon_file": str(other)}

        with caplog.at_level(logging.WARNING, logger="sass.library.registry"):
            resolve_library_paths(config, registry)

        assert config["synthon_file"] == str(other)
        assert any("override" in r.getMessage() for r in caplog.records)

    @pytest.mark.usefixtures("fake_root")
    def test_building_an_unset_artifact_will_not_write_into_the_named_library(self, registry):
        """A build task with no explicit output path raises rather than writing into the named library."""
        config = {"synthon_library": "lib"}
        with pytest.raises(ValueError, match="shared library this run would write into") as exc:
            resolve_library_paths(config, registry, produced=["synthon_conf_dir"])
        assert "synthon_conf_dir" in str(exc.value)

    @pytest.mark.usefixtures("fake_root")
    def test_building_into_an_explicit_path_is_fine(self, registry, tmp_path):
        """A build task with an explicit output path resolves normally."""
        mine = tmp_path / "my_conformers"
        config = {"synthon_library": "lib", "synthon_conf_dir": str(mine)}
        resolve_library_paths(config, registry, produced=["synthon_conf_dir"])
        assert config["synthon_conf_dir"] == str(mine)


class TestDataRootDetection:
    def test_the_first_existing_root_wins(self, tmp_path, monkeypatch):
        monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
        present = tmp_path / "second"
        present.mkdir()
        reg = {
            "data_roots": [
                {"path": str(tmp_path / "first"), "host": "cluster_b"},
                {"path": str(present), "host": "cluster_a"},
            ]
        }
        assert detect_data_root(reg) == (present, "cluster_a")

    def test_off_cluster_returns_nothing_rather_than_raising(self, tmp_path, monkeypatch):
        """Without a mounted root, detection returns `(None, None)` rather than raising."""
        monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
        reg = {"data_roots": [{"path": str(tmp_path / "absent"), "host": "cluster_b"}]}
        assert detect_data_root(reg) == (None, None)

    def test_the_env_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
        reg = {"data_roots": [{"path": "/", "host": "cluster_b"}]}
        assert detect_data_root(reg)[0] == tmp_path


class TestShippedRegistry:
    """Structural checks on the shipped `libraries.yml`."""

    def test_it_parses_and_every_alias_points_somewhere_real(self):
        reg = load_registry()
        libraries = reg["libraries"]
        for alias, target in reg["aliases"].items():
            assert target in libraries, f"alias {alias!r} -> unknown library {target!r}"

    def test_every_entry_names_the_required_artifacts(self):
        """Every registry entry names each required artifact. `OPTIONAL_KEYS` are exempt."""
        required = set(LIBRARY_KEYS) - set(OPTIONAL_KEYS)
        for name, entry in load_registry()["libraries"].items():
            assert required <= set(entry), f"{name} is missing {sorted(required - set(entry))}"

    def test_data_roots_are_absolute(self):
        for entry in load_registry()["data_roots"]:
            assert entry["path"].startswith("/"), entry

    def test_it_is_valid_yaml_on_disk(self):
        from sass.library.registry import _REGISTRY_FILE

        yaml.safe_load(_REGISTRY_FILE.read_text())

    def test_entry_paths_are_relative_and_stay_under_the_root(self):
        """Entry paths are relative and stay under the data root."""
        for name, entry in load_registry()["libraries"].items():
            for key in LIBRARY_KEYS:
                value = entry.get(key)
                if value is None:
                    continue
                assert not value.startswith("/"), f"{name}.{key} is absolute: {value}"
                assert ".." not in Path(value).parts, f"{name}.{key} escapes the data root: {value}"

    def test_every_declared_path_exists_on_this_host(self):
        """Every path in the shipped registry exists under the mounted data root.

        Only entries expected on this host are checked. An entry with no `hosts:` list is expected
        everywhere.
        """
        registry = load_registry()
        root, host = detect_data_root(registry)
        if root is None:
            pytest.skip("no SASS data root mounted; nothing to check paths against")

        problems = []
        checked = 0
        for name, entry in registry["libraries"].items():
            hosts = entry.get("hosts")
            if hosts is not None and host not in hosts:
                continue
            checked += 1
            for key in LIBRARY_KEYS:
                value = entry.get(key)
                if value is not None and not (root / value).exists():
                    problems.append(f"{name}.{key}: {root / value}")

        assert not problems, "registry paths that do not exist:\n  " + "\n  ".join(problems)
        if not checked:
            pytest.skip(f"no registry entry is expected on {host!r}")


class TestClusterDetection:
    """Which SLURM cluster this is, for partition and QoS defaults.

    The mounted data root is checked first, and SLURM's `ClusterName` is the fallback.
    """

    @staticmethod
    def _resolve(config, monkeypatch, data_root_host=None, slurm_host=None):
        import logging as _logging

        from sass import preflight as sass_config

        monkeypatch.setattr(sass_config, "DASK_SLURM_CONFIG", {"cluster_a": {}, "cluster_b": {}})
        monkeypatch.setattr(sass_config, "detect_host", lambda: data_root_host)
        monkeypatch.setattr(sass_config, "detect_host_from_slurm", lambda: slurm_host)
        return sass_config.resolve_cluster_name(config, _logging.getLogger("sass.test.cluster"))

    def test_explicit_always_wins(self, monkeypatch):
        """An explicit `cluster_name` wins over detection."""
        assert self._resolve({"cluster_name": "cluster_b"}, monkeypatch, data_root_host="cluster_a") == "cluster_b"

    def test_the_data_root_is_preferred_over_slurm(self, monkeypatch):
        assert self._resolve({}, monkeypatch, data_root_host="cluster_a", slurm_host="cluster_b") == "cluster_a"

    def test_slurm_is_used_when_no_data_root_is_mounted(self, monkeypatch):
        """SLURM's `ClusterName` is used when no data root is mounted."""
        assert self._resolve({}, monkeypatch, data_root_host=None, slurm_host="cluster_b") == "cluster_b"

    def test_neither_detector_raises_naming_the_options(self, monkeypatch):
        with pytest.raises(ValueError, match="Could not detect which cluster") as exc:
            self._resolve({}, monkeypatch)
        assert "cluster_a" in str(exc.value)
        assert "cluster_b" in str(exc.value)

    def test_an_unknown_explicit_name_is_refused(self, monkeypatch):
        with pytest.raises(ValueError, match="Invalid cluster name"):
            self._resolve({"cluster_name": "nope"}, monkeypatch)


class TestSlurmClusterName:
    def test_known_names_map_to_a_config_key(self):
        from sass.cluster.slurm import DASK_SLURM_CONFIG, SLURM_CLUSTER_NAMES

        for reported, host in SLURM_CLUSTER_NAMES.items():
            assert host in DASK_SLURM_CONFIG, f"ClusterName {reported!r} maps to unknown host {host!r}"

    def test_an_unknown_cluster_name_warns_rather_than_guessing(self, monkeypatch, caplog):
        """An unmapped `ClusterName` logs a warning and returns `None`."""
        import subprocess

        from sass.cluster import slurm

        def fake_run(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, stdout="ClusterName             = other_cluster\n", stderr="")

        monkeypatch.setattr(slurm.subprocess, "run", fake_run)
        with caplog.at_level(logging.WARNING, logger="sass.cluster.slurm"):
            assert slurm.detect_host_from_slurm() is None
        assert any("other_cluster" in r.getMessage() for r in caplog.records), caplog.records

    def test_a_mapped_name_is_recognized(self, monkeypatch):
        import subprocess

        from sass.cluster import slurm

        def fake_run(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, stdout="ClusterName             = site_cluster\n", stderr="")

        monkeypatch.setattr(slurm.subprocess, "run", fake_run)
        monkeypatch.setattr(slurm, "SLURM_CLUSTER_NAMES", {"site_cluster": "cluster_a"})
        assert slurm.detect_host_from_slurm() == "cluster_a"

    def test_a_missing_or_broken_scontrol_returns_none(self, monkeypatch):
        """Without a working `scontrol`, detection returns `None` rather than raising."""
        from sass.cluster import slurm

        def boom(*_args, **_kwargs):
            raise FileNotFoundError("scontrol")

        monkeypatch.setattr(slurm.subprocess, "run", boom)
        assert slurm.detect_host_from_slurm() is None


class TestHostAuthorization:
    """A library must be declared for the host in its `hosts:` list to count as available."""

    @staticmethod
    def _registry_for(host, hosts):
        return {
            "data_roots": [{"path": "/unused", "host": host}],
            "aliases": {},
            "libraries": {
                "lib_202503": {
                    "hosts": hosts,
                    "synthon_conf_dir": "lib_a/synthon_conformers",
                    "synthon_data_file": "lib_a/synthon_data.pkl",
                    "grouped_synthon_file": "lib_a/grouped_synthons.pkl",
                    "synthon_file": "lib_b/synthons.txt",
                    "reaction_file": "lib_b/reactions.txt",
                }
            },
        }

    @pytest.fixture()
    def as_cluster_b(self, fake_root, monkeypatch):
        """Resolve against the fixture tree, but report the host as cluster_b."""
        from sass.library import registry as reg_mod

        monkeypatch.setattr(reg_mod, "detect_data_root", lambda _registry=None: (fake_root, "cluster_b"))
        return fake_root

    @pytest.mark.usefixtures("as_cluster_b")
    def test_an_undeclared_host_warns_when_the_data_is_all_there(self, caplog):
        """An undeclared host only logs a warning when every file is present."""
        registry = self._registry_for("cluster_b", ["cluster_a"])
        with caplog.at_level(logging.WARNING, logger="sass.library.registry"):
            resolve_library_paths({"synthon_library": "lib_202503"}, registry)

        warnings = [r.getMessage() for r in caplog.records]
        assert any("not declared as available on 'cluster_b'" in w for w in warnings), warnings
        assert any("registered for: cluster_a" in w for w in warnings), warnings
        assert any("hosts:" in w for w in warnings), "the warning must say how to fix it"

    def test_an_undeclared_host_with_missing_data_explains_both(self, as_cluster_b):
        """With files missing on an undeclared host, the error names both problems."""
        (as_cluster_b / "lib_b" / "reactions.txt").unlink()
        registry = self._registry_for("cluster_b", ["cluster_a"])

        with pytest.raises(ValueError, match="Synthon library is incomplete") as exc:
            resolve_library_paths({"synthon_library": "lib_202503"}, registry)

        message = str(exc.value)
        assert "reaction_file" in message, "the missing artifact is still named"
        assert "not declared as available on 'cluster_b'" in message, "and so is why it is likely missing"

    @pytest.mark.usefixtures("as_cluster_b")
    def test_a_declared_host_is_silent(self, caplog):
        registry = self._registry_for("cluster_b", ["cluster_b", "cluster_a"])
        with caplog.at_level(logging.WARNING, logger="sass.library.registry"):
            resolve_library_paths({"synthon_library": "lib_202503"}, registry)
        assert not [r for r in caplog.records if "not declared" in r.getMessage()]

    @pytest.mark.usefixtures("as_cluster_b")
    def test_no_hosts_key_means_expected_everywhere(self, caplog):
        registry = self._registry_for("cluster_b", None)
        registry["libraries"]["lib_202503"].pop("hosts")
        with caplog.at_level(logging.WARNING, logger="sass.library.registry"):
            resolve_library_paths({"synthon_library": "lib_202503"}, registry)
        assert not [r for r in caplog.records if "not declared" in r.getMessage()]


@pytest.mark.usefixtures("fake_root")
class TestOptionalArtifactsThatThisRunReads:
    """An optional artifact is required when one of the run's tasks reads it.

    The `registry` fixture omits `single_frag_synthon_conf_dir`.
    """

    def test_a_run_that_does_not_read_it_is_unaffected(self, registry):
        """A run that does not read single-fragment conformers does not require them."""
        config = {"synthon_library": "lib"}
        resolve_library_paths(config, registry, required=optional_artifacts_this_run_reads(["score_synthons"]))
        assert not config.get("single_frag_synthon_conf_dir")

    def test_scoring_single_fragments_without_them_is_refused(self, registry):
        with pytest.raises(ValueError, match="Synthon library is incomplete") as exc:
            resolve_library_paths(
                {"synthon_library": "lib"},
                registry,
                required=optional_artifacts_this_run_reads(["score_single_fragments"]),
            )
        assert "single_frag_synthon_conf_dir" in str(exc.value)
        assert "score_single_fragments" in str(exc.value), "name the task that needs it"
        assert "generate_single_fragment_conformers" in str(exc.value), "and the task that builds it"

    def test_a_named_but_unbuilt_directory_is_refused_too(self, registry):
        """A registry path that was never built is refused too."""
        registry["libraries"]["lib_202503"]["single_frag_synthon_conf_dir"] = "lib_a/single_frag"
        with pytest.raises(ValueError, match="single_frag_synthon_conf_dir"):
            resolve_library_paths(
                {"synthon_library": "lib"},
                registry,
                required=optional_artifacts_this_run_reads(["write_single_fragment_poses"]),
            )

    def test_the_build_task_is_not_blocked_by_its_own_output(self):
        """The task that builds the artifact does not require it."""
        assert not optional_artifacts_this_run_reads(["generate_single_fragment_conformers"])
        assert not optional_artifacts_this_run_reads(["generate_single_fragment_conformers", "score_single_fragments"])


class TestTaskRequirementsIsOneTable:
    """The query and the library artifacts a task needs are declared in one place."""

    def test_the_scoring_task_set_is_unchanged_by_being_derived(self):
        """`SCORING_TASKS`, derived from the table, has the expected members."""
        assert set(SCORING_TASKS) == {
            "score_full_library",
            "score_synthons",
            "rescore_products",
            "write_product_poses",
            "write_single_fragment_poses",
            "score_single_fragments",
        }

    def test_every_task_named_is_a_real_task(self):
        assert set(TASK_REQUIREMENTS) <= set(VALID_TASKS)

    def test_every_required_artifact_is_optional_and_buildable(self):
        """Every artifact a task requires is an optional key with a task that builds it."""
        artifacts = {n for needs in TASK_REQUIREMENTS.values() for n in needs if n != QUERY_INPUT}
        assert artifacts, "the table should still name at least one artifact"
        assert artifacts <= set(OPTIONAL_KEYS), "a mandatory key here would be dead configuration"
        assert artifacts <= set(_BUILDS), "every required artifact needs a task that builds it"


class TestArtifactsThisRunBuilds:
    """`group_synthon_data` creates two of the library artifacts, so it cannot require them."""

    @staticmethod
    def _args(group_synthons=False, map_id=False):
        import argparse

        return argparse.Namespace(group_synthons=group_synthons, map_id=map_id)

    def test_group_synthon_data_produces_both_tables(self):
        from sass.preflight import artifacts_this_run_builds

        produced = artifacts_this_run_builds(["group_synthon_data"], self._args())
        assert produced == {"grouped_synthon_file", "synthon_data_file"}

    def test_the_cli_flags_count_too(self):
        """`--group-synthons` / `--map_id` rebuild the tables without naming the task."""
        from sass.preflight import artifacts_this_run_builds

        assert artifacts_this_run_builds([], self._args(group_synthons=True)) == {"grouped_synthon_file"}
        assert artifacts_this_run_builds([], self._args(map_id=True)) == {"synthon_data_file"}

    def test_conformer_generation_produces_the_conformer_directory(self):
        from sass.preflight import artifacts_this_run_builds

        assert artifacts_this_run_builds(["generate_synthon_conformers"], self._args()) == {"synthon_conf_dir"}

    def test_a_scoring_run_builds_nothing_and_needs_everything(self):
        from sass.preflight import artifacts_this_run_builds

        assert artifacts_this_run_builds(["score_synthons", "rescore_products"], self._args()) == set()

    def test_produced_artifacts_are_exempt_from_the_existence_check(self, fake_root, registry, tmp_path):
        """Building a library must not require the library to exist.

        The destinations are named explicitly, since a produced artifact is not resolved from
        `synthon_library`.
        """
        (fake_root / "lib_a" / "grouped_synthons.pkl").unlink()
        (fake_root / "lib_a" / "synthon_data.pkl").unlink()

        with pytest.raises(ValueError, match="Synthon library is incomplete"):
            resolve_library_paths({"synthon_library": "lib"}, registry)

        config = {
            "synthon_library": "lib",
            "grouped_synthon_file": str(tmp_path / "does_not_exist_yet.pkl"),
            "synthon_data_file": str(tmp_path / "nor_this_one.pkl"),
        }
        resolve_library_paths(config, registry, produced={"grouped_synthon_file", "synthon_data_file"})
        assert config["synthon_file"], "the artifacts we do read are still resolved and checked"


class TestColorForceFieldsAreLazy:
    """Color force fields are written on first access, so a run that never overlays writes none."""

    @staticmethod
    def _context(tmp_path):
        import argparse
        import logging as _logging

        from sass.config import load_config
        from sass.context import RunContext

        config = load_config(Path(__file__).resolve().parents[1] / "sass" / "config_template.yml")
        return RunContext(
            config=config,
            args=argparse.Namespace(),
            logger=_logging.getLogger("sass.test.colorff"),
            exp_dir=tmp_path,
            top_m=config["top_m"],
            dedup_limit=5e6,
            reactions=None,
            max_stereocenter=3,
            conf_format="oez",
            synthon_connectors=config["synthon_connectors"],
            frag_connectors=config["frag_connectors"],
            warts_id_sep="_",
            product_id_sep="&",
            opt_shape_func="GetTanimoto",
            opt_color_func="GetColorTanimoto",
            overlap_shape_func="GetTanimoto",
            overlap_color_func="GetColorTanimoto",
            rocs_key_opt=None,
            rocs_key_score=None,
            synthon_comb_limit=None,
            pseudo_res_dir=tmp_path / "pseudo_res",
            synthon_conf_dir=tmp_path / "conformers",
            rescore_file_w=tmp_path / "a.pkl",
            rescore_file_r=tmp_path / "a.pkl",
            combined_pseudo_res_file_w=tmp_path / "b.pkl",
            combined_pseudo_res_file_r=tmp_path / "b.pkl",
            synthon_data_w=tmp_path / "c.pkl",
            synthon_data_r=tmp_path / "c.pkl",
            grouped_synthon_w=tmp_path / "d.pkl",
            grouped_synthon_r=tmp_path / "d.pkl",
            rescore_res_dir_w=tmp_path / "rescore",
            full_enumeration_product_dir_w=tmp_path / "e",
            full_enumeration_product_dir_r=tmp_path / "e",
            full_enumeration_conf_dir_w=tmp_path / "f",
            full_enumeration_conf_dir_r=tmp_path / "f",
            full_enumeration_score_dir_w=tmp_path / "g",
            single_frag_synthon_conf_dir=tmp_path / "h",
            single_frag_pseudo_res_file_w=tmp_path / "i.pkl",
            single_frag_pseudo_res_file_r=tmp_path / "i.pkl",
        )

    def test_building_a_context_writes_nothing(self, tmp_path):
        self._context(tmp_path)
        assert not (tmp_path / "color_ffs").exists(), "a context that never scores must not write color FFs"

    def test_first_access_writes_them(self, tmp_path):
        ctx = self._context(tmp_path)
        out = ctx.color_ff_dir
        assert out.is_dir()
        assert sorted(p.name for p in out.glob("*.txt")), "no force field files written"

    def test_they_are_written_once(self, tmp_path):
        ctx = self._context(tmp_path)
        assert ctx.color_ff_dir is ctx.color_ff_dir

    def test_an_explicit_directory_is_used_as_is(self, tmp_path):
        ctx = self._context(tmp_path)
        ctx.config["color_ff_dir"] = str(tmp_path / "mine")
        assert ctx.color_ff_dir == tmp_path / "mine"
        assert not (tmp_path / "color_ffs").exists()


class TestSingleFragmentPoseTitles:
    """Single-fragment poses get the original library ids back, like product poses do."""

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("0_1[U]", "aaa:111:1_1[U]"),
            ("2_3[Np]", "ccc:333:3_3[Np]"),
            ("1_1", "bbb:222:2_1"),
        ],
    )
    def test_a_single_fragment_title_round_trips(self, title, expected):
        """The `[connector]` suffix rides along inside the wart, untouched."""
        from sass.ids import restore_original_synthon_ids

        index = ["aaa:111:1", "bbb:222:2", "ccc:333:3"]
        assert restore_original_synthon_ids(title, index, "&", "_") == expected

    def test_a_string_id_title_passes_through_unchanged(self):
        """A string-id library's titles are not integers, so there is nothing to restore."""
        from sass.ids import restore_original_synthon_ids

        title = "synthonid-00000000+06A_1[U]"
        assert restore_original_synthon_ids(title, ["aaa"], "&", "_") == title

    def test_the_task_wires_the_restore_in(self):
        """`write_single_fragment_poses` restores ids the same way the product pose task does."""
        import inspect

        from sass.tasks import write_single_fragment_poses

        source = inspect.getsource(write_single_fragment_poses.run)
        assert "restore_synthon_ids_in_poses" in source
        assert "retitle=retitle" in source


class TestConnectorTagSurvivesIdRestoration:
    """Single-fragment titles carry a trailing `[<connector>]`, and it must not break parsing.

    The tag is kept and the id translated whether or not the title has an isomer wart.
    """

    INDEX: ClassVar[list[str]] = ["aaaFIRST", "bbbSECOND", "cccTHIRD"]

    def restore(self, title):
        from sass.ids import restore_original_synthon_ids

        return restore_original_synthon_ids(title, self.INDEX, "&", "_")

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("0_1[U]", "aaaFIRST_1[U]"),  # the shape that exists today
            ("0[U]", "aaaFIRST[U]"),  # unwarted
            ("0_1[Np]", "aaaFIRST_1[Np]"),  # a two-connector synthon's other entry
            ("0_1", "aaaFIRST_1"),  # no tag at all
            ("0", "aaaFIRST"),
            ("0&2_1", "aaaFIRST&cccTHIRD_1"),  # a product id is unaffected
        ],
    )
    def test_the_tag_is_preserved_and_the_id_translated(self, title, expected):
        assert self.restore(title) == expected

    def test_a_non_integer_title_still_passes_through_with_its_tag(self):
        """The query molecule and any pre-existing string-id output must survive untouched."""
        assert self.restore("already-a-string_1[U]") == "already-a-string_1[U]"

    def test_an_out_of_range_id_keeps_its_tag(self):
        assert self.restore("99_1[U]") == "99_1[U]"


class TestBuildSynthonIdIndex:
    """Building the synthon id index must be reproducible."""

    @staticmethod
    def _synthon_file(tmp_path, ids):
        f = tmp_path / "synthons.txt"
        rows = "\n".join(f"CC[U]\t{i}\t1\trxn\ttest\t{i}" for i in ids)
        f.write_text("SMILES\toriginal_synton_id\tsynton#\treaction_id\trelease\tsynton_id\n" + rows + "\n")
        return f

    def test_ids_are_sorted_and_deduplicated(self, tmp_path):
        from sass.library.synthons import build_synthon_id_index

        f = self._synthon_file(tmp_path, ["zzz", "aaa", "mmm", "aaa"])
        assert build_synthon_id_index(f) == ["aaa", "mmm", "zzz"]

    def test_row_order_does_not_change_the_index(self, tmp_path):
        """The index depends only on the set of ids, not on row order."""
        from sass.library.synthons import build_synthon_id_index

        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        a = build_synthon_id_index(self._synthon_file(tmp_path / "a", ["c", "a", "b"]))
        b = build_synthon_id_index(self._synthon_file(tmp_path / "b", ["b", "c", "a"]))
        assert a == b == ["a", "b", "c"]

    def test_a_null_id_is_refused(self, tmp_path):
        """A null synthon id raises rather than being numbered like a real one."""
        from sass.library.synthons import build_synthon_id_index

        f = tmp_path / "synthons.txt"
        f.write_text("SMILES\tsynton_id\nCC[U]\taaa\nCC[U]\t\n")
        with pytest.raises(ValueError, match="null value"):
            build_synthon_id_index(f)

    def test_separator_collisions_warn_but_do_not_fail(self, tmp_path, caplog):
        """Ids containing the title separators `_` or `&` log a warning but are still indexed."""
        from sass.library.synthons import build_synthon_id_index

        f = self._synthon_file(tmp_path, ["aaa", "b_b", "c&c"])
        with caplog.at_level(logging.WARNING, logger="sass.library.synthons"):
            assert len(build_synthon_id_index(f)) == 3
        assert any("'_' or '&'" in r.getMessage() for r in caplog.records)
