"""Tests for sass.utils.utils_confgen module."""

import sys

import pytest
import pytest_check as check
from openeye import oechem

from sass import TEST_DIR
from sass.utils.utils_confgen import get_reaction_ring_atoms, sample_general_mixed_synthon_conf
from sass.utils.utils_libgen import generate_library_products
from sass.utils.utils_mol import (
    find_connected_atoms,
    get_conn_symbols_freq,
    label_synthon_ring_connector,
)

DATA_DIR = TEST_DIR / "data"


@pytest.mark.parametrize(
    ("smirks", "synthon_ring_connector_dict", "current_synthons", "current_synthon_idx", "other_synthons"),
    [
        (  # s_248, syn_1 (2) as `current_synthons`
            "[U]-[*:2]-[Np].([U]-[*:1].[*:3]-[Np])>>[*:1]-[*:2]-[*:3]",
            {"U": str(True), "Np": str(True)},
            [(141203, "CC(NC([U])=S)C([Np])=O"), (117929, "CSCCC(NC([U])=S)C([Np])=O")],
            1,
            {0: "Fc1c(Cl)c(Cl)ccc1N([U])[Np]"},
        ),
        (  # s_248, syn_0 (1) as `current_synthons`
            "[U]-[*:2]-[Np].([U]-[*:1].[*:3]-[Np])>>[*:1]-[*:2]-[*:3]",
            {"U": str(True), "Np": str(True)},
            [(18362204, "COC(=O)C1C(CN([U])[Np])C1c1ccccc1"), (18360582, "CS[C@@H]1C[C@H](CN([U])[Np])C1")],
            0,
            {1: "CSCCC(NC([U])=S)C([Np])=O"},
        ),
        (  # s_2, 3-component reaction, all ring-forming synthons,
            "[U]-[*:1]-[*:2]-[Pu].[Np]-[*:3]-[Pu].[U]-[*:5]=[*:4]-[Np]>>[*:2]-1-[*:3]-[*:4]=[*:5]-[*:1]-1",
            {"U": str(True), "Np": str(True), "Pu": str(True)},
            [
                (86557, "CCOc1ccccc1N=C([Pu])S[U]"),
                (86559, "FC(F)Sc1ccc(cc1)N=C([Pu])S[U]"),
                (86571, "Cc1cccc(C)c1N=C([Pu])S[U]"),
            ],
            0,
            {1: "[Np]N([Pu])C[C@H]1CCCCN1Cc1ccccc1", 2: "[U]C=C([Np])c1ccc2OCCOc2c1"},
        ),
        (  # s_2, syn_1 as `current_synthons`
            "[U]-[*:1]-[*:2]-[Pu].[Np]-[*:3]-[Pu].[U]-[*:5]=[*:4]-[Np]>>[*:2]-1-[*:3]-[*:4]=[*:5]-[*:1]-1",
            {"U": str(True), "Np": str(True), "Pu": str(True)},
            [(12629030, "Cc1cccc(CN2CCOC(CN([Np])[Pu])C2)c1"), (12629266, "Cc1ccc(CN2CCOC(CN([Np])[Pu])C2)cc1")],
            1,
            {0: "Cc1cccc(C)c1N=C([Pu])S[U]", 2: "[U]C=C([Np])c1ccccc1"},
        ),
        (  # s_2, syn_2 as `current_synthons`
            "[U]-[*:1]-[*:2]-[Pu].[Np]-[*:3]-[Pu].[U]-[*:5]=[*:4]-[Np]>>[*:2]-1-[*:3]-[*:4]=[*:5]-[*:1]-1",
            {"U": str(True), "Np": str(True), "Pu": str(True)},
            [
                (86580, "[U]C=C([Np])c1ccccc1"),
                (86581, "Brc1cccc(c1)C([Np])=C[U]"),
                (86583, "[U]C=C([Np])c1ccc2OCCOc2c1"),
            ],
            2,
            {0: "FC(F)Sc1ccc(cc1)N=C([Pu])S[U]", 1: "Cc1cccc(CN2CCOC(CN([Np])[Pu])C2)c1"},
        ),
        (  # s_1, syn 0 as `current_sythons`
            "([U]-[*:1].[Np]-[*:2]-[*:3]=[Pu]).[Pu]=[*:4].[Np]-[*:5]=[*:6]-[U]>>[*:1]-[*:6]=[*:5]-[*:2]-[*:3]=[*:4]",
            {"U": "True", "Np": "True", "Pu": "False"},
            [(86335, "COCC(C)N=C(S[U])N([Np])N=[Pu]")],
            0,
            {1: "COc1cc(C=[Pu])ccc1OCc1ccccc1", 2: "CCn1ncc(n1)C([Np])=C[U]"},
        ),
        (
            # m_280190_0, syn_1 as `current_synthons`
            "[U]-[*:1].([Pu]-[N:2]=[N:3]-[N:4]-[Np].[*:5][U]).[Pu]-[#6:6]=[#6:7]-[Np]>>([*:1]-[*:5].[#6:6]=1-[N:2]=[N:3]-[N:4]-[#6:7]=1)",
            {"U": "False", "Np": "True", "Pu": "True"},
            [("VpQuIth7qxFlzj_1cO2o7Q", "O=C([U])c1ccc(CN([Np])N=N[Pu])cc1")],
            1,
            {0: "O=C(O)C1CCN([U])CC1", 2: "[Np]C=C([Pu])c1nccs1"},
        ),
    ],
    # TODO: use SynthonHandler to automatically sample synthons for testing.
    # TODO: hypothetical cyclopropane forming rxn, an extreme case, each conn_nei is connected to 2 conn,
    # and each other_conn_nei also connected to 2 conn (but that's okay!).
    # TODO: Another case: one nei is connected to 1 lin conn and 1 ring conn. (or more!)
)
def test_sample_general_mixed_synthon_conf(
    tmp_path,
    smirks,
    synthon_ring_connector_dict,
    current_synthons,
    current_synthon_idx,
    other_synthons,
):
    """Function runs without exception, connector atoms properly restored."""
    tmp_path.mkdir(exist_ok=True)
    ofile_format = "oez"
    rxn_name = "test_rxn"
    atom_label = "self_label"
    conf_file = tmp_path / f"{rxn_name}_synthon_{current_synthon_idx}_conf.{ofile_format}"

    sample_general_mixed_synthon_conf(
        smirks=smirks,
        synthon_ring_connector_dict=synthon_ring_connector_dict,
        current_synthons=current_synthons,
        current_synthon_idx=current_synthon_idx,
        other_synthons=other_synthons,
        output_file=conf_file,
        omega_max_conf=50,
        omega_max_time=120,
        omega_e_window=10,
    )

    check.is_true(conf_file.is_file())

    # Check if the symbol and frequency of connectors are the same between input and output.
    for (_, input_smi), mc_mol in zip(current_synthons, oechem.oemolistream(str(conf_file)).GetOEMols()):
        input_mol = oechem.OEMol()
        oechem.OESmilesToMol(input_mol, input_smi)
        connector_set = set(synthon_ring_connector_dict.keys())
        mc_freq = get_conn_symbols_freq(mc_mol, connector_set)
        input_freq = get_conn_symbols_freq(input_mol, connector_set)
        for conn in connector_set:
            check.equal(mc_freq.get(conn, 0), input_freq.get(conn, 0))
            check.is_true(mc_freq.get(conn, 0) <= 1)  # Allow only up to 1 of each connector per molecule.
        # Labels on the resulting molecule atoms are "self".
        for a in mc_mol.GetAtoms():
            if a.HasData(atom_label):
                check.equal(a.GetData(atom_label), "self")


@pytest.mark.parametrize(
    ("reactant_smis", "smirks", "expected"),
    [
        (
            # s_1
            {
                0: "COCC(C)N=C(S[U])N([Np])N=[Pu]",
                1: "[O-][N+](=O)c1ccc(C=[Pu])o1",
                2: "Clc1ccccc1C([Np])=C[U]",
            },
            "([U]-[*:1].[Np]-[*:2]-[*:3]=[Pu]).[Pu]=[*:4].[Np]-[*:5]=[*:6]-[U]>>[*:1]-[*:6]=[*:5]-[*:2]-[*:3]=[*:4]",
            5,
        ),
        (
            # s_2
            {
                0: "CCOc1ccccc1N=C([Pu])S[U]",
                1: "[Np]N([Pu])C1CCCCC1",
                2: "Cc1cc(C([Np])=C[U])c(C)n1-c1ccc(C)c(C)c1",
            },
            "[U]-[*:1]-[*:2]-[Pu].[Np]-[*:3]-[Pu].[U]-[*:5]=[*:4]-[Np]>>[*:2]-1-[*:3]-[*:4]=[*:5]-[*:1]-1",
            5,
        ),
        (
            # s_41
            {
                0: "CCCCc1ccc(NC(S[U])=NN=[Np])cc1",
                1: "Cc1ccc(cc1)C(=[Np])C[U]",
            },
            "([U]~[*:1].[*:4]~[Np]).[U]~[*:2]-[*:3]~[Np]>>[*:1]-[*:2]-[*:3]=[*:4]",
            6,
        ),
        (
            # s_22
            {
                0: "CN(C)CC(O)COc1ccc(N[U])cc1",
                1: "[U]C(=O)c1cn(CCNC(=O)OCC2c3ccccc3-c3ccccc23)cn1",
            },
            "[U]-[*:1].[U]-[*:2]>>[*:1]-[*:2]",
            0,
        ),
    ],
)
def test_get_reaction_ring_atoms(reactant_smis: dict[int, str], smirks: str, expected: int):
    """Correctly finds the size of the ring formed during the reaction.

    Parameters
    ----------
    reactant_smis
        Reactants of this reaction with the reagent order as the keys.
    smirks
        SMIRKS string of the reaction.
    expected
        Expected number of atoms in the ring formed during the reaction.
    """
    atom_tag = "self_label"
    other_label = "other"
    connector_symbols = ["U", "Np", "Pu", "Am"]
    start_atom_tag = "start_atom"
    start_atom_found = False
    # Label reactants, find the connector atom neighbor (as start atom).
    # Because `get_reaction_ring_atoms` only checks for `other_label`, label all reactant atoms with `other_label`.

    ring_connector_dict = label_synthon_ring_connector(reactant_smis, connector_symbols, smirks)
    print(ring_connector_dict)

    reactants = {}
    for idx, smi in reactant_smis.items():
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smi)
        for atom in mol.GetAtoms():
            atom.SetData(atom_tag, other_label)
            atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
            if atom_symbol in ring_connector_dict:
                is_ring_conn = ring_connector_dict[atom_symbol]
                if start_atom_found is False and (is_ring_conn == str(True) or expected == 0):
                    # If expected is 0, this is a non-ring forming reaction.
                    reactant_start_atom = next(find_connected_atoms(atom))
                    reactant_start_atom.SetData(start_atom_tag, "")
        reactants[idx] = [mol]

    # Form product
    product = next(
        generate_library_products(
            reactants=reactants,
            smirks=smirks,
            sort_title=False,
            title_separator="&",
        )
    )

    # Because the atom data are copied but atom themselves are not copied when forming products,
    # need to label atom and find it in the product.
    for atom in product.GetAtoms():
        if atom.HasData(start_atom_tag):
            start_atom = atom
            break

    ring_size = len(
        get_reaction_ring_atoms(
            mol=product,
            start_atom=start_atom,
            label_tag=atom_tag,
            other_label=other_label,
        )
    )

    check.equal(ring_size, expected)


if __name__ == "__main__":
    sys.exit(pytest.main())
