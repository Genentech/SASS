"""Tests for sass.chem.synthon module."""

import sys

import pytest
import pytest_check as check
from openeye import oechem

from sass.chem.mol import get_conn_symbols, smiles_to_mol
from sass.chem.synthon import order_and_substitute_fragments
from sass.library.synthons import map_synthon_ids
from sass.selection.dedup import deduplicate_scores


@pytest.mark.parametrize(
    "scores",
    [
        [(1.0, f"pid{n}") for n in range(100)] * 2,
        [(1.1, f"pid{n}", "other_info") for n in range(90)] * 3,
        [(500, f"pid{n}_{n*2}_{n+5}") for n in range(99)],
    ],
)
@pytest.mark.parametrize(
    "sort_scores",
    [True, False],
)
@pytest.mark.parametrize(
    "limit",
    [10, 20, 50, 500],
)
@pytest.mark.parametrize("keep_isomer_warts", [True, False])
def test_deduplicate_scores(scores, sort_scores, limit, keep_isomer_warts):
    """Correct size of tuple in each `scores` element is returned, `limit` is observed."""
    warts_separator = "_"
    out = deduplicate_scores(scores, warts_separator, keep_isomer_warts, sort_scores, limit)
    check.equal(len(scores[0]), len(out[0]))
    check.less_equal(len(out), limit)
    if scores[0][1].count(warts_separator) == 0:
        expected_warts = 0
    else:
        expected_warts = scores[0][1].count(warts_separator) - int(not keep_isomer_warts)
    check.equal(out[0][1].count(warts_separator), expected_warts)


@pytest.mark.skip(reason="Not implemented yet.")
def test_synthon_selection(): ...


@pytest.mark.parametrize(
    "synthon_smiles_eg",
    [
        {2: "[U]CC[Pu]", 1: "[U]CC(C(C(F)(F)F)[Am])[Pu]", 4: "[Am]CCN"},
        {3: "[U]CC[Np]", 0: "[U]CC(C(C(F)(F)F)[Xe])[Pu]", 6: "[Pu]CC(CC[Xe])[Np]"},  # cyclic, Xe is not a connector
    ],
)
@pytest.mark.parametrize(
    "fragment_smiles",
    [
        ["[He]C(C)C(F)[Ne]", "[He]C(C)C(C(C(F)(F)F)[Ar])[Ne]", "[Ar]CC(F)N"],
        ["[He]C(C)C(C(C(F)(F)F)[Ar])[Ne]", "[Ar]CC(F)N", "[He]C(C)C(F)[Ne]"],
        ["[Ar]CC[He]", "[Ar]CC(C(C(F)(F)F)[Xe])[Kr]", "[Kr]CC(CC[Xe])[He]"],  # cyclic, Xe is not a connector
    ],
)
def test_order_and_substitute_fragments(synthon_smiles_eg: dict, fragment_smiles):
    """Workflow works without error, fragments are substituted correctly."""
    fragments = [smiles_to_mol(smiles) for smiles in fragment_smiles]
    synthon_connectors = ["U", "Np", "Pu", "Am"]
    frag_connectors = ["He", "Ne", "Ar", "Kr"]
    syn_order, ordered_fragments = order_and_substitute_fragments(
        synthon_smiles_eg,
        synthon_connectors,
        fragments,
        frag_connectors,
        cross_score=False,
    )
    # check the ordered and substituted fragments contain the correct connector atoms.
    for frags in ordered_fragments:
        for syn_idx, frag in zip(syn_order, frags):
            syn_conn = get_conn_symbols(synthon_smiles_eg[syn_idx], synthon_connectors)
            frag_conn = get_conn_symbols(oechem.OEMolToSmiles(frag), synthon_connectors)
            print(synthon_smiles_eg[syn_idx], oechem.OEMolToSmiles(frag))
            check.equal(set(syn_conn), set(frag_conn))


@pytest.fixture()
def map_synthon_ids_test_data():
    return (
        ["rxn1", "rxn2"],
        {
            "rxn1": {
                1: [("sid1", "CCCC"), ("sid2", "CCCC")],
                2: [
                    ("sid3", "CCCC"),
                ],
            },
            "rxn2": {
                1: [
                    ("sid1", "CCCC"),
                ],
            },
        },
        {
            "sid1": {
                "SMILES": "CCCC",
                "heavy_atom_count": 4,
                "rxn1": {"s_idx": 1},
                "rxn2": {"s_idx": 1},
            },
            "sid2": {
                "SMILES": "CCCC",
                "heavy_atom_count": 4,
                "rxn1": {"s_idx": 1},
            },
            "sid3": {
                "SMILES": "CCCC",
                "heavy_atom_count": 4,
                "rxn1": {"s_idx": 2},
            },
        },
    )


def test_map_synthon_ids(map_synthon_ids_test_data):
    """Synthon data are correctly organized."""
    reactions, grouped_synthons, expected = map_synthon_ids_test_data
    result = map_synthon_ids(reactions, grouped_synthons, [])
    check.equal(result, expected)


def test_map_synthon_ids_different_smiles(map_synthon_ids_test_data):
    """Error is raised if encountering different SMILES for same synthon id."""
    reactions, grouped_synthons, _ = map_synthon_ids_test_data
    grouped_synthons["rxn1"][1][0] = ("sid1", "CCCCO")
    with pytest.raises(ValueError) as excinfo:  # noqa: PT011
        map_synthon_ids(reactions, grouped_synthons, [])
    assert "Synthon sid1 has different" in str(excinfo.value)


def test_map_synthon_ids_duplicate_sid(map_synthon_ids_test_data):
    """Error is raised if encountering duplicate synthon ids in the same reaction."""
    reactions, grouped_synthons, _ = map_synthon_ids_test_data
    grouped_synthons["rxn1"][2].append(("sid1", "CCCC"))
    with pytest.raises(ValueError) as excinfo:  # noqa: PT011
        map_synthon_ids(reactions, grouped_synthons, [])
    assert "Duplicate synthon id sid1" in str(excinfo.value)


if __name__ == "__main__":
    sys.exit(pytest.main())
