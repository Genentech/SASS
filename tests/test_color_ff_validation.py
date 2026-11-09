"""The color force fields a run will overlay with, checked before a cluster exists.

There is one file per ROCS key, from `0` through `len(synthon_connectors)`. Setting `color_ff_dir`
replaces the generated set, so the keys that build it are warned about when also set.
"""

import logging

import pytest

from sass import SRC_DIR
from sass.chem.overlay import COLOR_FF_FILE
from sass.config import load_config
from sass.preflight import _GENERATED_COLOR_FF_KEYS, validate_color_force_fields

logger = logging.getLogger("sass.test.colorff")

SCORING = ["score_synthons"]


@pytest.fixture()
def config():
    return load_config(SRC_DIR / "config_template.yml")


def write_ffs(directory, keys):
    directory.mkdir(parents=True, exist_ok=True)
    for key in keys:
        (directory / COLOR_FF_FILE.format(key=key)).write_text("# color force field\n")
    return directory


class TestASuppliedDirectoryMustBeComplete:
    def test_a_complete_directory_passes(self, config, tmp_path):
        config["color_ff_dir"] = str(write_ffs(tmp_path / "ffs", range(5)))
        validate_color_force_fields(config, SCORING, logger)

    def test_a_missing_connector_count_is_refused(self, config, tmp_path):
        """Files for 0-2 exist, and a three-connector synthon arrives later."""
        config["color_ff_dir"] = str(write_ffs(tmp_path / "ffs", [0, 1, 2]))
        with pytest.raises(ValueError, match="missing 2 of the 5") as exc:
            validate_color_force_fields(config, SCORING, logger)
        assert COLOR_FF_FILE.format(key=3) in str(exc.value)
        assert COLOR_FF_FILE.format(key=4) in str(exc.value)

    def test_the_zero_key_counts_too(self, config, tmp_path):
        """`align_connector_atom: false` scores every synthon with key 0."""
        config["color_ff_dir"] = str(write_ffs(tmp_path / "ffs", [1, 2, 3, 4]))
        with pytest.raises(ValueError, match="missing 1 of the 5") as exc:
            validate_color_force_fields(config, SCORING, logger)
        assert COLOR_FF_FILE.format(key=0) in str(exc.value)

    def test_the_expected_count_follows_synthon_connectors(self, config, tmp_path):
        """Two connector atoms means three files, not five."""
        config["synthon_connectors"] = ["U", "Np"]
        config["custom_color_ff"] = ["[U]", "[Np]"]
        config["color_ff_dir"] = str(write_ffs(tmp_path / "ffs", range(3)))
        validate_color_force_fields(config, SCORING, logger)

    def test_a_path_that_is_not_a_directory_is_refused(self, config, tmp_path):
        not_a_dir = tmp_path / "ffs.txt"
        not_a_dir.write_text("oops")
        config["color_ff_dir"] = str(not_a_dir)
        with pytest.raises(ValueError, match="not a directory"):
            validate_color_force_fields(config, SCORING, logger)

    def test_the_error_says_how_to_get_out_of_it(self, config, tmp_path):
        config["color_ff_dir"] = str(write_ffs(tmp_path / "ffs", [0]))
        with pytest.raises(ValueError, match="unset `color_ff_dir`") as exc:
            validate_color_force_fields(config, SCORING, logger)
        assert "earlier run" in str(exc.value), "reusing a run's color_ffs/ is the intended route"


class TestTheDisabledKeysAreNamed:
    """Keys that `color_ff_dir` makes inert are named in a warning when set."""

    def test_an_overridden_key_is_warned_about(self, config, tmp_path, caplog):
        config["color_ff_dir"] = str(write_ffs(tmp_path / "ffs", range(5)))
        config["custom_color_weight"] = -42
        with caplog.at_level(logging.WARNING, logger=logger.name):
            validate_color_force_fields(config, SCORING, logger)
        assert "custom_color_weight" in caplog.text
        assert "no effect" in caplog.text

    def test_every_disabled_key_is_listed(self, config, tmp_path, caplog):
        config["color_ff_dir"] = str(write_ffs(tmp_path / "ffs", range(5)))
        config["base_cff_file"] = "/some/base.txt"
        config["additional_color_patterns"] = [{"pattern": "[OX1]", "weight": -5, "radius": 2}]
        config["no_default_color"] = True
        with caplog.at_level(logging.WARNING, logger=logger.name):
            validate_color_force_fields(config, SCORING, logger)
        for key in ("base_cff_file", "additional_color_patterns", "no_default_color"):
            assert key in caplog.text, key

    def test_a_default_config_does_not_warn(self, config, tmp_path, caplog):
        """Setting only `color_ff_dir` logs no warnings."""
        config["color_ff_dir"] = str(write_ffs(tmp_path / "ffs", range(5)))
        with caplog.at_level(logging.WARNING, logger=logger.name):
            validate_color_force_fields(config, SCORING, logger)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_align_on_bond_is_not_called_dead(self, config, tmp_path, caplog):
        """Its color features are skipped, but it still drives the query transform."""
        config["color_ff_dir"] = str(write_ffs(tmp_path / "ffs", range(5)))
        config["align_on_bond"] = True
        with caplog.at_level(logging.WARNING, logger=logger.name):
            validate_color_force_fields(config, SCORING, logger)
        assert "align_on_bond" not in caplog.text
        assert "align_on_bond" not in _GENERATED_COLOR_FF_KEYS

    def test_every_named_key_is_a_real_config_key(self, config):
        assert set(_GENERATED_COLOR_FF_KEYS) <= set(config)


class TestTheGeneratedSetMustCoverEveryKey:
    def test_too_few_patterns_for_the_connectors_is_refused(self, config):
        """Generated files would stop at key 1, and a 2-connector synthon asks for key 2."""
        config["custom_color_ff"] = ["[U]"]
        with pytest.raises(ValueError, match="custom_color_ff") as exc:
            validate_color_force_fields(config, SCORING, logger)
        assert "synthon_connectors" in str(exc.value)

    def test_the_template_defaults_are_consistent(self, config):
        """The shipped template passes its own check."""
        validate_color_force_fields(config, SCORING, logger)

    def test_more_patterns_than_connectors_is_fine(self, config):
        """Extra files are harmless; only a missing one fails."""
        config["custom_color_ff"] = ["[U]", "[Np]", "[Pu]", "[Am]", "[Cm]"]
        validate_color_force_fields(config, SCORING, logger)

    def test_it_does_not_apply_without_connector_colors(self, config):
        """`align_on_dummy_atom: false` scores everything with key 0, so one file is enough."""
        config["custom_color_ff"] = ["[U]"]
        config["align_on_dummy_atom"] = False
        validate_color_force_fields(config, SCORING, logger)


class TestOnlyRunsThatOverlayAreChecked:
    @pytest.mark.parametrize("tasks", [["generate_synthon_conformers"], ["group_synthon_data"], []])
    def test_a_run_that_reads_no_color_force_field_is_not_checked(self, config, tmp_path, tasks):
        """Runs with no overlaying task skip the check."""
        config["color_ff_dir"] = str(tmp_path / "does_not_exist")
        validate_color_force_fields(config, tasks, logger)

    @pytest.mark.parametrize(
        "task",
        ["score_synthons", "rescore_products", "write_product_poses", "score_single_fragments"],
    )
    def test_every_overlaying_task_is_checked(self, config, tmp_path, task):
        config["color_ff_dir"] = str(tmp_path / "does_not_exist")
        with pytest.raises(ValueError, match="not a directory"):
            validate_color_force_fields(config, [task], logger)
