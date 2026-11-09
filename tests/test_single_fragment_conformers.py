"""Generating single-fragment scoring conformers from regular synthon conformers.

A synthon with N connectors becomes N molecules, each keeping one connector (substituted to a
single target atom) and titled `<synthon title>[<original connector>]`.
"""

import pytest
from openeye import oechem

from sass.worker.generate_conformers import generate_single_frag_synthon_conformers

CONNECTORS = ["U", "Np", "Pu", "Am"]


def write_synthons(path, entries):
    """Write conformer-bearing synthons. `entries` is [(title, smiles_with_placeholders), ...].

    OMEGA cannot parameterize the connector elements, so conformers are built with Br/I
    stand-ins that are then substituted to U/Np.
    """
    from openeye import oeomega

    from sass.chem.confgen import omega_opt_builder
    from sass.chem.mol import substitute_atoms

    omega = oeomega.OEOmega(omega_opt_builder(max_conf=3, max_search_time=20, energy_window=10))
    ostream = oechem.oemolostream(str(path))
    for title, smiles in entries:
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smiles)
        mol.SetTitle(title)
        code = omega.Build(mol)
        assert code == oeomega.OEOmegaReturnCode_Success, f"{smiles}: {oeomega.OEGetOmegaError(code)}"
        substitute_atoms(mol=mol, sub_dict={("Br", 0): ("U", 0), ("I", 0): ("Np", 0)}, delete_H=False)
        for conf in mol.GetConfs():
            conf.SetTitle(title)
        oechem.OEWriteMolecule(ostream, mol)
    ostream.close()


def generate(src, out_dir, chunk_size=300, extend_bond=False):
    generate_single_frag_synthon_conformers(
        synthon_file=src,
        output_dir=out_dir,
        synthon_connectors=CONNECTORS,
        target_connector_atom="U",
        extend_bond=extend_bond,
        extend_atom_symbol="Si",
        extend_atom_isotope=33,
        extend_bond_length=3,
        chunk_size=chunk_size,
    )
    return [
        (mol.GetTitle(), oechem.OEMolToSmiles(mol))
        for f in sorted(out_dir.glob("*_single_frag_*.oez"))
        for mol in oechem.oemolistream(str(f)).GetOEMols()
    ]


def test_a_two_connector_synthon_becomes_two_molecules(tmp_path):
    """One per connector, each keeping only its own."""
    src = tmp_path / "reaction_1_synthon_0_conf.oez"
    write_synthons(src, [("12345_1", "BrCCCCOI")])

    out = generate(src, tmp_path / "out")

    assert sorted(t for t, _ in out) == ["12345_1[Np]", "12345_1[U]"]
    for _title, smiles in out:
        assert smiles.count("[U]") == 1, f"exactly one connector should survive: {smiles}"
        assert "Np" not in smiles, f"the other connector must be deleted: {smiles}"


def test_titles_inherit_the_synthon_id_verbatim(tmp_path):
    """Output titles keep the synthon id and wart unchanged, followed by the connector tag."""
    src = tmp_path / "reaction_1_synthon_0_conf.oez"
    write_synthons(src, [("987654_2", "BrCCCCO")])

    titles = [t for t, _ in generate(src, tmp_path / "out")]

    assert titles == ["987654_2[U]"]
    base = titles[0].split("[")[0]
    assert base == "987654_2", "the id and wart must survive untouched for later restoration"


def test_a_string_id_synthon_also_passes_through(tmp_path):
    """String synthon ids pass through unchanged."""
    src = tmp_path / "reaction_1_synthon_0_conf.oez"
    write_synthons(src, [("synthonid-00000000+06A_1", "BrCCCCO")])

    assert [t for t, _ in generate(src, tmp_path / "out")] == ["synthonid-00000000+06A_1[U]"]


def test_every_input_molecule_is_represented(tmp_path):
    src = tmp_path / "reaction_1_synthon_0_conf.oez"
    write_synthons(src, [("1_1", "BrCCCCO"), ("2_1", "BrCCNI"), ("3_1", "Brc1ccccc1")])

    bases = {t.split("[")[0] for t, _ in generate(src, tmp_path / "out")}

    assert bases == {"1_1", "2_1", "3_1"}


def test_output_is_chunked_for_downstream_parallelism(tmp_path):
    """Output is split into files of about `chunk_size` molecules."""
    src = tmp_path / "reaction_1_synthon_0_conf.oez"
    write_synthons(src, [(f"{i}_1", "BrCCCCO") for i in range(8)])

    out_dir = tmp_path / "out"
    generate(src, out_dir, chunk_size=3)

    chunks = sorted(out_dir.glob("*_single_frag_*.oez"))
    assert len(chunks) > 1, "8 molecules at chunk_size=3 should span several files"
    counts = [len(list(oechem.oemolistream(str(c)).GetOEMols())) for c in chunks]
    assert sum(counts) == 8
    assert max(counts) <= 3 + 1, f"chunks should respect the requested size: {counts}"


def test_a_chunked_input_leaks_its_chunk_number_into_the_output_name(tmp_path):
    """The output name comes from the input file stem, so a `.chunk3` input keeps that suffix.

    Callers merge chunked inputs first to avoid this.
    """
    src = tmp_path / "reaction_1_synthon_0_conf.chunk3.oez"
    write_synthons(src, [("1_1", "BrCCCCO")])

    out_dir = tmp_path / "out"
    generate(src, out_dir)

    names = [f.name for f in out_dir.glob("*_single_frag_*.oez")]
    assert names == ["reaction_1_synthon_0_conf.chunk3_single_frag_0.oez"], names


@pytest.mark.parametrize("extend_bond", [False, True])
def test_bond_extension_is_optional_and_adds_one_atom(tmp_path, extend_bond):
    """`extend_bond` adds one marker atom along the connection vector."""
    src = tmp_path / "reaction_1_synthon_0_conf.oez"
    write_synthons(src, [("1_1", "BrCCCCO")])

    out = generate(src, tmp_path / "out", extend_bond=extend_bond)

    assert len(out) == 1
    has_marker = "Si" in out[0][1]
    assert has_marker is extend_bond, f"extend_bond={extend_bond} but SMILES is {out[0][1]}"
