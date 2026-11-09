"""Tests for reading chunked synthon conformer libraries.

Chunking must not change the molecule stream a reader sees. Also covers the conformer manifest,
id translation, and OpenEye warning suppression.
"""

from typing import ClassVar

import pytest
from openeye import oechem

from sass.library.conformers import build_conformer_manifest, resolve_conformer_files, stream_conformers

CONF_FORMAT = "oez"


def write_mols(path, mols):
    ostream = oechem.oemolostream(str(path))
    for mol in mols:
        oechem.OEWriteMolecule(ostream, mol)
    ostream.close()


@pytest.fixture()
def mols():
    """A handful of distinct molecules with SASS-style titles."""
    out = []
    for i, smi in enumerate(["CCO", "CCCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC", "CNC"]):
        mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(mol, smi)
        mol.SetTitle(f"{1000 + i}_1")
        out.append(oechem.OEMol(mol))
    return out


def test_manifest_groups_chunks_by_reaction_and_component(tmp_path, mols):
    write_mols(tmp_path / f"reaction_11_synthon_0_conf.chunk0.{CONF_FORMAT}", mols[:3])
    write_mols(tmp_path / f"reaction_11_synthon_0_conf.chunk1.{CONF_FORMAT}", mols[3:])
    write_mols(tmp_path / f"reaction_11_synthon_1_conf.chunk0.{CONF_FORMAT}", mols[:2])
    write_mols(tmp_path / f"reaction_2_synthon_0_conf.{CONF_FORMAT}", mols[:1])  # unchunked

    manifest = build_conformer_manifest(tmp_path, CONF_FORMAT)

    assert set(manifest) == {("reaction_11", 0), ("reaction_11", 1), ("reaction_2", 0)}
    assert len(manifest[("reaction_11", 0)]) == 2
    assert len(manifest[("reaction_2", 0)]) == 1, "unchunked layout must still resolve"
    # Chunk order must be deterministic, or tie-breaking stops being reproducible.
    assert manifest[("reaction_11", 0)] == sorted(manifest[("reaction_11", 0)])


def test_reaction_ids_containing_underscores_are_parsed_correctly(tmp_path, mols):
    """Reaction ids that contain the separator, such as `reaction_4bbb`, are parsed whole."""
    write_mols(tmp_path / f"reaction_4bbb_synthon_2_conf.chunk0.{CONF_FORMAT}", mols[:2])
    write_mols(tmp_path / f"reaction_1_synthon_0_conf.chunk0.{CONF_FORMAT}", mols[:1])

    manifest = build_conformer_manifest(tmp_path, CONF_FORMAT)

    assert ("reaction_4bbb", 2) in manifest
    assert ("reaction_1", 0) in manifest


def test_chunked_and_unchunked_yield_the_same_molecules(tmp_path, mols):
    """Chunked and unchunked files yield the same molecule stream."""
    single = tmp_path / f"reaction_5_synthon_0_conf.{CONF_FORMAT}"
    write_mols(single, mols)

    chunk_dir = tmp_path / "chunked"
    chunk_dir.mkdir()
    for k, start in enumerate(range(0, len(mols), 2)):
        write_mols(chunk_dir / f"reaction_5_synthon_0_conf.chunk{k}.{CONF_FORMAT}", mols[start : start + 2])

    from_single = [m.GetTitle() for m in stream_conformers([single])]
    chunk_files = build_conformer_manifest(chunk_dir, CONF_FORMAT)[("reaction_5", 0)]
    from_chunks = [m.GetTitle() for m in stream_conformers(chunk_files)]

    assert from_chunks == from_single
    assert len(chunk_files) == 4


def test_resolve_falls_back_without_a_manifest(tmp_path, mols):
    write_mols(tmp_path / f"reaction_7_synthon_0_conf.chunk0.{CONF_FORMAT}", mols[:2])
    write_mols(tmp_path / f"reaction_7_synthon_0_conf.chunk1.{CONF_FORMAT}", mols[2:4])
    write_mols(tmp_path / f"reaction_8_synthon_0_conf.{CONF_FORMAT}", mols[:1])

    assert len(resolve_conformer_files(tmp_path, "reaction_7", 0, CONF_FORMAT)) == 2
    assert len(resolve_conformer_files(tmp_path, "reaction_8", 0, CONF_FORMAT)) == 1
    assert resolve_conformer_files(tmp_path, "reaction_9", 0, CONF_FORMAT) == []


def test_manifest_ignores_unrelated_files(tmp_path, mols):
    write_mols(tmp_path / f"reaction_3_synthon_0_conf.{CONF_FORMAT}", mols[:1])
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / f"reaction_3_products.{CONF_FORMAT}").write_bytes(b"")
    write_mols(tmp_path / f"reaction_3_synthon_x_conf.{CONF_FORMAT}", mols[:1])  # non-numeric component

    manifest = build_conformer_manifest(tmp_path, CONF_FORMAT)

    assert set(manifest) == {("reaction_3", 0)}


class TestRestoreOriginalSynthonIds:
    """Translating integer product ids back to library ids for the final pose SDF."""

    INDEX: ClassVar[list[str]] = ["aaaFIRST", "bbbSECOND", "cccTHIRD", "ddd+FOURTH-x"]

    def restore(self, title):
        from sass.ids import restore_original_synthon_ids

        return restore_original_synthon_ids(title, self.INDEX, title_separator="&", warts_separator="_")

    def test_translates_ids_and_keeps_the_isomer_wart(self):
        """Poses are specific stereoisomers, so the wart must survive the translation."""
        assert self.restore("0&2_1") == "aaaFIRST&cccTHIRD_1"

    def test_handles_ids_containing_the_separators_they_replaced(self):
        """The original ids may contain `+`/`-`; only `&` and `_` are structural here."""
        assert self.restore("3&1_7") == "ddd+FOURTH-x&bbbSECOND_7"

    def test_single_component_and_no_wart(self):
        assert self.restore("2") == "cccTHIRD"
        assert self.restore("2_3") == "cccTHIRD_3"

    def test_non_numeric_titles_pass_through(self):
        """The query molecule and any string-id output must be left alone."""
        assert self.restore("my_query_molecule") == "my_query_molecule"
        assert self.restore("synthonid000000000001A&synthonid000000000002A_1") == (
            "synthonid000000000001A&synthonid000000000002A_1"
        )

    def test_out_of_range_id_is_left_alone_rather_than_crashing(self):
        """An id outside the index is left unchanged."""
        assert self.restore("99&0_1") == "99&0_1"


class TestProductIdToSids:
    """Normalizing the two product-id shapes that reach instantiation."""

    def convert(self, pid):
        from sass.ids import product_id_to_sids

        return product_id_to_sids(pid, warts_separator="_", title_separator="&")

    def test_integer_tuple_passes_through(self):
        """Pseudo-product ids from `score_synthons` are already integer tuples."""
        assert self.convert((12345, 67890)) == (12345, 67890)

    def test_numeric_oe_title_is_cast_and_wart_stripped(self):
        """Rescore rows carry the OE molecule title, which must be a string."""
        assert self.convert("12345&67890_1") == (12345, 67890)
        assert self.convert("12345") == (12345,)

    def test_string_ids_are_rejected_with_an_actionable_message(self):
        """String-id conformer libraries raise an error that names `synthon_id_index`."""
        with pytest.raises(ValueError, match="must be built with a `synthon_id_index`"):
            self.convert("synthonid000000000003A&synthonid000000000001A_1")


class TestOEWarningSuppression:
    """Repeated OpenEye warnings are logged once and counted, not once per occurrence."""

    FLAT = (
        "OE3DToAtomStereo is unable to perceive atom stereo from a flat geometry on atom "
        "{atom} of molecule 'mol_1 (2)'"
    )
    PROBLEM = "OE3DToAtomStereo had a problem during OEMolToSmiles when writing '{name}'"

    def make_handler(self, tmp_path, max_repeats=1):
        from sass.logging_setup import CustomOEErrorHandler

        return CustomOEErrorHandler(str(tmp_path / "oe.log"), logger=None, max_repeats=max_repeats)

    def written(self, tmp_path):
        return (tmp_path / "oe.log").read_text()

    def test_repeated_warning_is_written_once(self, tmp_path):
        from openeye import oechem

        h = self.make_handler(tmp_path)
        for i in range(500):
            h.Msg(oechem.OEErrorLevel_Warning, self.FLAT.format(atom=i))
        text = self.written(tmp_path)

        assert text.count("unable to perceive") == 1, "one line per distinct warning"
        assert "further warnings of this kind suppressed" in text
        assert sum(h._warning_counts.values()) == 500, "every occurrence must still be counted"

    def test_distinct_warnings_are_each_kept(self, tmp_path):
        from openeye import oechem

        h = self.make_handler(tmp_path)
        for i in range(50):
            h.Msg(oechem.OEErrorLevel_Warning, self.FLAT.format(atom=i))
            h.Msg(oechem.OEErrorLevel_Warning, self.PROBLEM.format(name=f"mol{i}"))
        text = self.written(tmp_path)

        assert text.count("unable to perceive") == 1
        assert text.count("had a problem") == 1, "a different message must not be suppressed by the first"

    def test_copies_share_one_tally(self, tmp_path):
        """Copies made by `CreateCopy` share one tally of repeated warnings."""
        from openeye import oechem

        h = self.make_handler(tmp_path)
        copy = h.CreateCopy()
        h.Msg(oechem.OEErrorLevel_Warning, self.FLAT.format(atom=1))
        for i in range(20):
            copy.Msg(oechem.OEErrorLevel_Warning, self.FLAT.format(atom=i))

        assert self.written(tmp_path).count("unable to perceive") == 1

    def test_summary_reports_totals(self, tmp_path, caplog):
        import logging as _logging

        from openeye import oechem

        from sass.logging_setup import CustomOEErrorHandler

        log = _logging.getLogger("oe_summary_test")
        h = CustomOEErrorHandler(str(tmp_path / "oe.log"), logger=log, max_repeats=1)
        for i in range(1234):
            h.Msg(oechem.OEErrorLevel_Warning, self.FLAT.format(atom=i))

        with caplog.at_level(_logging.INFO, logger="oe_summary_test"):
            h.log_warning_summary()

        assert any("1234 x" in r.getMessage() for r in caplog.records), "totals must survive suppression"


def test_task_payload_carries_only_its_own_conformer_files(tmp_path):
    """Each scoring task gets its own file slice, not the whole directory manifest."""
    import pickle

    # A manifest spanning many reactions, of which one task may reference only two entries.
    manifest = {}
    for r in range(200):
        for s_idx in (0, 1):
            files = [tmp_path / f"reaction_{r}_synthon_{s_idx}_conf.chunk{k}.oez" for k in range(24)]
            manifest[(f"reaction_{r}", s_idx)] = files
    whole = len(pickle.dumps(manifest, protocol=5))

    rxn_slice = {s_idx: files for (rxn, s_idx), files in manifest.items() if rxn == "reaction_7"}
    per_task = len(pickle.dumps(rxn_slice, protocol=5))

    assert (
        per_task * 20 < whole
    ), f"per-task payload {per_task} B is not materially smaller than the whole manifest {whole} B"

    # The task generator attaches the slice, keyed by component. A real query is fragmented
    # because toy molecules do not satisfy the fragment matching.
    from sass.chem.fragment import fragment_molecule
    from sass.chem.mol import load_first_mol
    from sass.tasks.score_synthons import _generate_reaction_scoring_tasks

    query = load_first_mol("tests/data/test_query_4.sdf", mol_cleanup=False)
    frag_sets = fragment_molecule(
        mol=oechem.OEMol(query),
        n_fragments=2,
        heavy_atom_limit=4,
        cleave_acyclic_bonds=True,
        cleave_cyclic_bonds=True,
        connector_isotope=1,
        connector_atoms=["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"],
    )
    tasks = list(
        _generate_reaction_scoring_tasks(
            rxn_id="reaction_7",
            n_component=2,
            query_fragments_dict={2: frag_sets[:2]},
            grouped_synthons={"reaction_7": {0: [(1, "[U]NCc1ccccc1")], 1: [(2, "[U]C(=O)CC")]}},
            synthon_connectors=["U", "Np", "Pu", "Am"],
            frag_connectors=["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"],
            cross_score=True,
            rxn_conf_files=rxn_slice,
        )
    )
    assert tasks, "generator produced no tasks"
    for task in tasks:
        assert len(task) == 7, "task tuple must carry the conformer slice and the fragment keys"
        conf_files, frag_keys = task[5], task[6]
        assert set(conf_files) == set(task[3]), "one entry per scored component"
        assert all(f.name.startswith("reaction_7_") for files in conf_files.values() for f in files)
        # Each fragment key is (canonical SMILES, shape hash). See tests/test_fragment_memo_key.py.
        assert len(frag_keys) == len(task[3]), "one fragment key per scored component"
        for smiles, shape in frag_keys:
            assert isinstance(smiles, str)
            assert smiles
            assert isinstance(shape, str)
            assert shape


class TestMergeChunksOrdering:
    """Merging must be a no-op on the molecule sequence, not a reshuffle.

    Chunks merge in sorted order, which matches what `resolve_conformer_files` gives a streaming read.
    """

    @staticmethod
    def _write(path, titles):
        ostream = oechem.oemolostream(str(path))
        for title in titles:
            mol = oechem.OEGraphMol()
            oechem.OESmilesToMol(mol, "CCO")
            mol.SetTitle(title)
            oechem.OEWriteMolecule(ostream, oechem.OEMol(mol))
        ostream.close()

    @staticmethod
    def _titles(path):
        return [m.GetTitle() for m in oechem.oemolistream(str(path)).GetOEMols()]

    def test_stems_are_enumerated_for_fan_out(self, tmp_path):
        """`conformer_chunk_stems` lists each chunked stem once, in sorted order."""
        from sass.library.conformers import conformer_chunk_stems

        for stem in ("reaction_1_synthon_0_conf", "reaction_1_synthon_1_conf", "reaction_2_synthon_0_conf"):
            self._write(tmp_path / f"{stem}.chunk0.oez", ["x"])
            self._write(tmp_path / f"{stem}.chunk1.oez", ["y"])
        assert conformer_chunk_stems(tmp_path, "oez") == [
            "reaction_1_synthon_0_conf",
            "reaction_1_synthon_1_conf",
            "reaction_2_synthon_0_conf",
        ]

    def test_per_stem_merging_matches_the_serial_whole_directory_merge(self, tmp_path):
        """Merging stem by stem gives the same files as merging the whole directory at once."""
        from sass.library.conformers import (
            conformer_chunk_stems,
            merge_conformer_chunks_for_stem,
            merge_synthon_conformer_chunks,
        )

        layout = {
            "reaction_1_synthon_0_conf": {0: ["a"], 10: ["b"], 2: ["c"]},
            "reaction_2_synthon_0_conf": {1: ["d", "e"], 0: ["f"]},
        }

        def build(root):
            root.mkdir()
            for stem, chunks in layout.items():
                for idx, titles in chunks.items():
                    self._write(root / f"{stem}.chunk{idx}.oez", titles)
            return root

        serial, distributed = build(tmp_path / "serial"), build(tmp_path / "distributed")

        merge_synthon_conformer_chunks(serial, "oez")
        # Reversed, so the result cannot depend on the order stems are visited.
        for stem in reversed(conformer_chunk_stems(distributed, "oez")):
            merge_conformer_chunks_for_stem(stem, distributed, "oez")

        for stem in layout:
            assert self._titles(serial / f"{stem}.oez") == self._titles(distributed / f"{stem}.oez")
        assert sorted(f.name for f in serial.iterdir()) == sorted(f.name for f in distributed.iterdir())

    def test_chunks_merge_in_sorted_order(self, tmp_path):
        """Chunks merge in the reader's order, where `chunk10` sorts before `chunk2`."""
        from sass.library.conformers import merge_synthon_conformer_chunks, resolve_conformer_files

        stem = "reaction_1_synthon_0_conf"
        for idx, titles in ((0, ["a"]), (10, ["b"]), (2, ["c"])):
            self._write(tmp_path / f"{stem}.chunk{idx}.oez", titles)

        expected = [t for f in resolve_conformer_files(tmp_path, "reaction_1", 0, "oez") for t in self._titles(f)]
        merge_synthon_conformer_chunks(tmp_path, "oez")

        merged = tmp_path / f"{stem}.oez"
        assert merged.is_file()
        assert self._titles(merged) == expected, "merging must not reorder the molecule stream"
        assert not list(tmp_path.glob("*chunk*")), "chunks should be gone once merged"

    def test_a_single_chunk_is_renamed(self, tmp_path):
        from sass.library.conformers import merge_synthon_conformer_chunks

        stem = "reaction_1_synthon_0_conf"
        self._write(tmp_path / f"{stem}.chunk0.oez", ["only"])
        merge_synthon_conformer_chunks(tmp_path, "oez")
        assert self._titles(tmp_path / f"{stem}.oez") == ["only"]

    def test_rerunning_does_not_consume_its_own_output(self, tmp_path):
        """A second merge leaves the merged file unchanged."""
        from sass.library.conformers import merge_synthon_conformer_chunks

        stem = "reaction_1_synthon_0_conf"
        self._write(tmp_path / f"{stem}.chunk0.oez", ["x"])
        self._write(tmp_path / f"{stem}.chunk1.oez", ["y"])
        merge_synthon_conformer_chunks(tmp_path, "oez")
        before = self._titles(tmp_path / f"{stem}.oez")

        merge_synthon_conformer_chunks(tmp_path, "oez")
        assert self._titles(tmp_path / f"{stem}.oez") == before == ["x", "y"]
