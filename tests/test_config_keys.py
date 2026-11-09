"""Every config key is checked at startup, not whenever the task that reads it runs."""

import pytest
import yaml

from sass import SRC_DIR
from sass.config import load_config
from sass.preflight import validate_config_keys


@pytest.fixture()
def template() -> dict:
    return load_config(SRC_DIR / "config_template.yml")


class TestMissingKeys:
    def test_the_template_itself_passes(self, template):
        validate_config_keys(template)

    def test_a_missing_key_is_refused(self, template):
        del template["clean_up_intermediate_files"]
        with pytest.raises(ValueError, match="missing 1 key") as exc:
            validate_config_keys(template)
        assert "clean_up_intermediate_files" in str(exc.value)

    def test_every_missing_key_is_listed_at_once(self, template):
        """All missing keys are reported in one error."""
        for key in ("clean_up_intermediate_files", "dask_worker_memory", "top_m"):
            del template[key]
        with pytest.raises(ValueError, match="missing 3 key") as exc:
            validate_config_keys(template)
        for key in ("clean_up_intermediate_files", "dask_worker_memory", "top_m"):
            assert key in str(exc.value)

    def test_the_message_is_pasteable_yaml(self, template):
        """The error lists the missing keys as YAML that can be pasted into the config."""
        del template["dask_worker_memory"]
        del template["clean_up_intermediate_files"]
        with pytest.raises(ValueError, match="Add them to continue") as exc:
            validate_config_keys(template)

        block = str(exc.value).split("Add them to continue:\n", 1)[1]
        assert "..." not in block, "a YAML document-end marker would not be pasteable"
        parsed = yaml.safe_load(block)
        assert parsed == {"dask_worker_memory": 4, "clean_up_intermediate_files": True}


class TestUnknownKeys:
    def test_an_unknown_key_warns_but_runs(self, template, caplog):
        """An unknown key logs a warning but does not raise."""
        template["clean_up_intermediat_files"] = True
        with caplog.at_level("WARNING", logger="sass.preflight"):
            validate_config_keys(template)
        assert "clean_up_intermediat_files" in caplog.text

    def test_missing_wins_over_unknown(self, template):
        """A misspelled key still raises for the correct key it leaves missing."""
        template["clean_up_intermediat_files"] = template.pop("clean_up_intermediate_files")
        with pytest.raises(ValueError, match="missing 1 key"):
            validate_config_keys(template)


class TestTemplateIsAuthoritative:
    def test_every_key_the_package_reads_is_in_the_template(self):
        """Every key the package reads as `config[...]` or `ctx.config[...]` is in the template."""
        import ast
        import pathlib

        template = load_config(SRC_DIR / "config_template.yml")
        read_keys: set[str] = set()
        for path in pathlib.Path(SRC_DIR).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Constant):
                    continue
                if not isinstance(node.slice.value, str):
                    continue
                target = node.value
                # `config[...]` or `<anything>.config[...]`
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name == "config":
                    read_keys.add(node.slice.value)

        assert read_keys, "the scan found nothing; it has stopped working"
        assert not (read_keys - set(template)), (
            "these keys are read from the config but absent from config_template.yml, so "
            f"`validate_config_keys` cannot catch them: {sorted(read_keys - set(template))}"
        )
