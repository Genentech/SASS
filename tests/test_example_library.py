"""The bundled example library uses the column names SASS reads."""

import inspect

import pandas as pd

from sass import DATA_DIR
from sass.library.synthons import SynthonHandler

EXAMPLE = DATA_DIR / "example_library"


def _default(param: str):
    return inspect.signature(SynthonHandler.__init__).parameters[param].default


def _columns(name: str) -> set[str]:
    return set(pd.read_csv(EXAMPLE / name, sep="\t", nrows=0).columns)


def test_synthon_file_columns():
    columns = _columns("synthons.txt")
    for param in ("syn_smiles_col", "syn_id_col", "syn_num_col", "rxn_id_col"):
        assert _default(param) in columns, f"{param}={_default(param)!r} missing from synthons.txt"


def test_reaction_file_columns():
    columns = _columns("reactions.txt")
    for param in ("rxn_id_col", "rxn_component_col", "rxn_smirks_col"):
        assert _default(param) in columns, f"{param}={_default(param)!r} missing from reactions.txt"
    assert set(_default("rxn_smarts_cols")) <= columns


def test_it_loads():
    SynthonHandler(reaction_file=EXAMPLE / "reactions.txt", synthon_file=EXAMPLE / "synthons.txt")
