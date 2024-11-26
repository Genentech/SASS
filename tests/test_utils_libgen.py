"""Tests for sass.utils.utils_libgen.py module."""

import pytest
import pytest_check as check

from sass.utils.utils_libgen import generate_library_products, generate_stereoisomers, instantiate_singleton_product
from sass.utils.utils_mol import smiles_to_mol


@pytest.mark.parametrize(
    ("reactants", "smirks", "expected_title", "should_raise"),
    [
        (
            [
                (333, "COCC(C)N=C(S[U])N([Np])N=[Pu]", 0),
                (222, "Clc1ccccc1C([Np])=C[U]", 2),
                (111, "[O-][N+](=O)c1ccc(C=[Pu])o1", 1),
            ],
            "([U]-[*:1].[Np]-[*:2]-[*:3]=[Pu]).[Pu]=[*:4].[Np]-[*:5]=[*:6]-[U]>>[*:1]-[*:6]=[*:5]-[*:2]-[*:3]=[*:4]",
            "111&222&333",
            False,
        ),
        (  # This example has the synthon position index incorrect and should raise an exception.
            [
                (333, "COCC(C)N=C(S[U])N([Np])N=[Pu]", 0),
                (222, "Clc1ccccc1C([Np])=C[U]", 1),
                (111, "[O-][N+](=O)c1ccc(C=[Pu])o1", 2),
            ],
            "([U]-[*:1].[Np]-[*:2]-[*:3]=[Pu]).[Pu]=[*:4].[Np]-[*:5]=[*:6]-[U]>>[*:1]-[*:6]=[*:5]-[*:2]-[*:3]=[*:4]",
            "111&222&333",
            True,
        ),
    ],
)
def test_instantiate_singleton_product(reactants, smirks, expected_title, should_raise):
    """Correct number of products are generated, titles set corrected."""
    title_sep = "&"
    if should_raise:
        with check.raises(ValueError):
            instantiate_singleton_product(reactants, smirks, True, title_sep)
    else:
        prod = instantiate_singleton_product(reactants, smirks, True, title_sep)
        check.equal(prod.GetTitle(), expected_title)


@pytest.mark.parametrize(
    ("reactants", "smirks"),
    [
        (
            {
                0: [(111, "COCC(C)N=C(S[U])N([Np])N=[Pu]")],
                1: [(222, "[O-][N+](=O)c1ccc(C=[Pu])o1")],
                2: [(333, "Clc1ccccc1C([Np])=C[U]")],
            },
            "([U]-[*:1].[Np]-[*:2]-[*:3]=[Pu]).[Pu]=[*:4].[Np]-[*:5]=[*:6]-[U]>>[*:1]-[*:6]=[*:5]-[*:2]-[*:3]=[*:4]",
        ),
        (
            {
                0: [(111, "CC(C)N[U]")],
                1: [(222, "CC([U])=O")],
            },
            "[U]-[*:1].[U]-[*:2]>>[*:1]-[*:2]",
        ),
    ],
)
@pytest.mark.parametrize("data_repeats", [1, 2, 4])
def test_generate_library_products(reactants: dict, data_repeats: int, smirks: str):
    """Correct number of products are generatd."""
    # Multiply the reactants
    reactants = {key: val * data_repeats for key, val in reactants.items()}
    prod = list(generate_library_products(reactants, smirks, True, "&"))
    check.equal(len(prod), data_repeats ** len(reactants))


@pytest.mark.parametrize(
    ("smiles", "n_expected"),
    [
        ("c1ccccc1", 1),
        ("C#CC1CC1(N)C(O)=O", 4),
        ("CC1CC1C", 3),  # Meso molecule
        ("CC(C)(OC(NC(C(O)=O)CC#C)=O)C", 2),
    ],
)
def test_generate_stereoisomers(smiles, n_expected):
    """Correct number of stereoisomers are generated."""
    mol = smiles_to_mol(smiles)
    isomers = list(generate_stereoisomers(mol, None))  # Use default FLIPPER options.
    check.equal(len(isomers), n_expected)
