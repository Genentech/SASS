"""The conformers a single-fragment run scores must have been built the way the config says.

The bond-extension distance is measured from a sampled conformer file and compared with
`align_on_bond` and `extend_bond_distance`. The geometry recorded in `libraries.yml` must agree.
"""

import math

import pytest
from openeye import oechem

from sass import EXTEND_ATOM_ISOTOPE, EXTEND_ATOM_SYMBOL
from sass.library.registry import LIBRARY_KEYS, load_registry
from sass.preflight import probe_single_fragment_geometry, validate_single_fragment_geometry

CONNECTOR = "U"


def write_conformers(directory, distance, n=3):
    """A single-fragment conformer file: one connector each, optionally bond-extended.

    `distance=None` writes no extension atom, as `align_on_bond: false` produces.
    """
    directory.mkdir(parents=True, exist_ok=True)
    ofs = oechem.oemolostream(str(directory / "reaction_1_synthon_0_conf_single_frag_0.oez"))
    for i in range(n):
        mol = oechem.OEGraphMol()
        carbon = mol.NewAtom(oechem.OEElemNo_C)
        connector = mol.NewAtom(oechem.OEGetAtomicNum(CONNECTOR))
        mol.NewBond(carbon, connector, 1)
        mol.SetCoords(carbon, oechem.OEFloatArray([0.0, 0.0, 0.0]))
        mol.SetCoords(connector, oechem.OEFloatArray([1.5, 0.0, 0.0]))
        if distance is not None:
            ext = mol.NewAtom(oechem.OEGetAtomicNum(EXTEND_ATOM_SYMBOL))
            ext.SetIsotope(EXTEND_ATOM_ISOTOPE)
            # Along the carbon->connector vector, which is the x axis here.
            mol.SetCoords(ext, oechem.OEFloatArray([1.5 + distance, 0.0, 0.0]))
        mol.SetTitle(f"{i}_1[U]")
        oechem.OEWriteMolecule(ofs, mol)
    ofs.close()
    return directory


def config_for(directory, align_on_bond, extend_bond_distance):
    return {
        "single_frag_synthon_conf_dir": str(directory),
        "conf_file_format": "oez",
        "synthon_connectors": [CONNECTOR, "Np", "Pu", "Am"],
        "align_on_bond": align_on_bond,
        "extend_bond_distance": extend_bond_distance,
    }


class TestProbe:
    def test_it_measures_the_distance_that_was_written(self, tmp_path):
        _, measured = probe_single_fragment_geometry(write_conformers(tmp_path / "c", 3.0), "oez", CONNECTOR)
        assert measured == pytest.approx(3.0, abs=1e-3)

    def test_a_non_default_distance_is_read_back_too(self, tmp_path):
        """The value is not assumed to be 3 anywhere; it comes out of the file."""
        _, measured = probe_single_fragment_geometry(write_conformers(tmp_path / "c", 1.75), "oez", CONNECTOR)
        assert measured == pytest.approx(1.75, abs=1e-3)

    def test_no_extension_atom_reads_as_none(self, tmp_path):
        _, measured = probe_single_fragment_geometry(write_conformers(tmp_path / "c", None), "oez", CONNECTOR)
        assert measured is None, "no extension atom means the set was built with align_on_bond: false"

    def test_it_names_the_file_it_sampled(self, tmp_path):
        """The error quotes it, so a user can go and look at the same molecule."""
        sample, _ = probe_single_fragment_geometry(write_conformers(tmp_path / "c", 3.0), "oez", CONNECTOR)
        assert sample.name == "reaction_1_synthon_0_conf_single_frag_0.oez"

    def test_an_empty_directory_probes_to_nothing(self, tmp_path):
        """Not an error here: the caller warns rather than failing on a cross-check."""
        (tmp_path / "empty").mkdir()
        assert probe_single_fragment_geometry(tmp_path / "empty", "oez", CONNECTOR) is None

    def test_a_real_silicon_atom_is_not_mistaken_for_the_extension(self, tmp_path):
        """Silicon-containing building blocks exist; only SASS's atom carries the isotope."""
        d = tmp_path / "c"
        d.mkdir()
        ofs = oechem.oemolostream(str(d / "reaction_1_synthon_0_conf_single_frag_0.oez"))
        mol = oechem.OEGraphMol()
        silicon = mol.NewAtom(oechem.OEGetAtomicNum("Si"))  # isotope 0: part of the molecule
        connector = mol.NewAtom(oechem.OEGetAtomicNum(CONNECTOR))
        mol.NewBond(silicon, connector, 1)
        mol.SetCoords(silicon, oechem.OEFloatArray([0.0, 0.0, 0.0]))
        mol.SetCoords(connector, oechem.OEFloatArray([1.9, 0.0, 0.0]))
        mol.SetTitle("1_1[U]")
        oechem.OEWriteMolecule(ofs, mol)
        ofs.close()

        _, measured = probe_single_fragment_geometry(d, "oez", CONNECTOR)
        assert measured is None, "a bare Si is part of the synthon, not a bond extension"


class TestMismatchIsRefused:
    def test_a_matching_config_passes(self, tmp_path):
        validate_single_fragment_geometry(config_for(write_conformers(tmp_path / "c", 3.0), True, 3))

    def test_a_matching_config_without_extension_passes(self, tmp_path):
        validate_single_fragment_geometry(config_for(write_conformers(tmp_path / "c", None), False, 3))

    def test_the_wrong_distance_is_refused(self, tmp_path):
        """A distance mismatch raises and quotes both the measured and configured values."""
        with pytest.raises(ValueError, match="extend_bond_distance") as exc:
            validate_single_fragment_geometry(config_for(write_conformers(tmp_path / "c", 3.0), True, 2))
        assert "extend_bond_distance: 3" in str(exc.value), "the measured value"
        assert "extend_bond_distance: 2" in str(exc.value), "and the configured one"

    def test_extending_a_query_the_conformers_were_not_extended_for_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no 33Si extension atom"):
            validate_single_fragment_geometry(config_for(write_conformers(tmp_path / "c", None), True, 3))

    def test_not_extending_a_query_when_the_conformers_are_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="align_on_bond: true"):
            validate_single_fragment_geometry(config_for(write_conformers(tmp_path / "c", 3.0), False, 3))

    def test_a_float_round_trip_is_not_a_mismatch(self, tmp_path):
        """The distance goes through a conformer file; it must not fail on the last decimal."""
        validate_single_fragment_geometry(config_for(write_conformers(tmp_path / "c", 3.0000001), True, 3))

    def test_the_message_is_pasteable_yaml(self, tmp_path):
        """The suggested setting is written as lowercase YAML."""
        with pytest.raises(ValueError, match="align_on_bond") as exc:
            validate_single_fragment_geometry(config_for(write_conformers(tmp_path / "c", 3.0), False, 3))
        assert "align_on_bond: True" not in str(exc.value)
        assert "align_on_bond: false" in str(exc.value), "the run's own setting, lowercased"

    def test_an_unreadable_directory_warns_rather_than_blocking(self, tmp_path, caplog):
        """An empty conformer directory logs a warning instead of raising."""
        (tmp_path / "empty").mkdir()
        with caplog.at_level("WARNING", logger="sass.preflight"):
            validate_single_fragment_geometry(config_for(tmp_path / "empty", True, 3))
        assert "were not checked" in caplog.text


class TestRegistryRecordsTheSameGeometry:
    """`libraries.yml` tells a user what to set. It must agree with what the check will measure."""

    @pytest.fixture()
    def registry(self):
        return load_registry()

    def test_every_library_with_conformers_declares_its_geometry(self, registry):
        for name, entry in registry["libraries"].items():
            if "single_frag_synthon_conf_dir" in entry:
                assert "single_frag_geometry" in entry, f"{name} ships single-fragment conformers but says nothing"

    def test_the_declared_keys_are_the_ones_a_user_sets(self, registry):
        for name, entry in registry["libraries"].items():
            geometry = entry.get("single_frag_geometry")
            if geometry:
                assert set(geometry) == {"align_on_bond", "extend_bond_distance"}, name

    def test_geometry_is_not_mistaken_for_a_path(self, registry):
        """It sits alongside the artifact keys, so it must not be resolved as one of them."""
        assert "single_frag_geometry" not in LIBRARY_KEYS
        for entry in registry["libraries"].values():
            assert not isinstance(entry.get("single_frag_geometry"), str)

    def test_the_declaration_matches_the_data_on_disk(self, registry):
        """The geometry declared in `libraries.yml` matches the conformers on disk."""
        from sass.library.registry import detect_data_root

        root, host = detect_data_root(registry)
        if root is None:
            pytest.skip("no SASS data root mounted")

        checked = 0
        for name, entry in registry["libraries"].items():
            geometry = entry.get("single_frag_geometry")
            hosts = entry.get("hosts")
            if not geometry or (hosts is not None and host is not None and host not in hosts):
                continue
            conf_dir = root / entry["single_frag_synthon_conf_dir"]
            if not conf_dir.is_dir():
                continue
            probed = probe_single_fragment_geometry(conf_dir, "oez", "U")
            assert probed is not None, f"{name}: could not sample {conf_dir}"
            _, measured = probed
            if geometry["align_on_bond"]:
                assert measured is not None, f"{name} declares align_on_bond but the conformers carry no extension"
                assert math.isclose(measured, geometry["extend_bond_distance"], abs_tol=0.01), (
                    f"{name} declares extend_bond_distance {geometry['extend_bond_distance']} "
                    f"but the conformers measure {measured:.4f}"
                )
            else:
                assert measured is None, f"{name} declares align_on_bond: false but the conformers are extended"
            checked += 1
        if not checked:
            pytest.skip("no registered library ships single-fragment conformers here")
