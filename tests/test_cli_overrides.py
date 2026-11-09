"""`--query`, `--library` and `--set key=value` folded into a loaded config.

CLI values take precedence over the config file, which takes precedence over the template default.
"""

import argparse

import pytest

from sass import SRC_DIR
from sass.config import load_config
from sass.preflight import apply_cli_overrides


def make_args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**{"query": None, "library": None, "overrides": [], **kwargs})


@pytest.fixture()
def config() -> dict:
    return load_config(SRC_DIR / "config_template.yml")


class TestNamedFlags:
    def test_query_overrides_the_config(self, config):
        config["query_molecule"] = "from_config.sdf"
        apply_cli_overrides(config, make_args(query="from_cli.sdf"))
        assert config["query_molecule"] == "from_cli.sdf"

    def test_library_overrides_the_config(self, config):
        apply_cli_overrides(config, make_args(library="my_library"))
        assert config["synthon_library"] == "my_library"

    def test_omitting_them_leaves_the_config_alone(self, config):
        config["query_molecule"] = "from_config.sdf"
        apply_cli_overrides(config, make_args())
        assert config["query_molecule"] == "from_config.sdf"
        assert config["synthon_library"] is None, "the template default stands"


class TestSet:
    def test_it_overrides_any_key(self, config):
        apply_cli_overrides(config, make_args(overrides=["top_m=500000"]))
        assert config["top_m"] == 500000

    def test_values_are_parsed_as_yaml_not_left_as_strings(self, config):
        """`--set num_workers=200` must give an int, exactly as the config file would."""
        apply_cli_overrides(
            config,
            make_args(overrides=["num_workers=200", "compress_pkl_files=false", "exp_dir=null", "n_fragments=[2, 3]"]),
        )
        assert config["num_workers"] == 200
        assert config["compress_pkl_files"] is False
        assert config["exp_dir"] is None
        assert config["n_fragments"] == [2, 3]

    def test_a_value_containing_equals_survives(self, config):
        apply_cli_overrides(config, make_args(overrides=["slurm_partition=a=b"]))
        assert config["slurm_partition"] == "a=b"

    def test_repeated_flags_all_apply(self, config):
        apply_cli_overrides(config, make_args(overrides=["top_m=10", "num_workers=3"]))
        assert config["top_m"] == 10
        assert config["num_workers"] == 3

    def test_an_unknown_key_is_refused(self, config):
        """An unknown `--set` key raises, unlike an unknown key in a config file, which only warns."""
        with pytest.raises(ValueError, match="not a config key"):
            apply_cli_overrides(config, make_args(overrides=["not_a_real_key=1"]))

    def test_a_near_miss_gets_a_suggestion(self, config):
        with pytest.raises(ValueError, match="Did you mean: top_m"):
            apply_cli_overrides(config, make_args(overrides=["top_M=5"]))

    def test_a_missing_equals_is_refused(self, config):
        with pytest.raises(ValueError, match="KEY=VALUE"):
            apply_cli_overrides(config, make_args(overrides=["top_m"]))

    def test_set_can_reach_the_keys_that_also_have_flags(self, config):
        """`--set` also reaches keys that have a dedicated flag."""
        apply_cli_overrides(config, make_args(overrides=["query_molecule=q.sdf"]))
        assert config["query_molecule"] == "q.sdf"


class TestPrecedence:
    def test_a_named_flag_beats_set(self, config):
        """A named flag wins over `--set` for the same key, regardless of order."""
        apply_cli_overrides(config, make_args(query="flag.sdf", overrides=["query_molecule=set.sdf"]))
        assert config["query_molecule"] == "flag.sdf"
