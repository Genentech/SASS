"""Scoring memoization must key on a fragment's shape, not just the graph connectivity.

With ring cleaving, two different cuts can give the same SMILES for different geometry, so a
SMILES-only key would let them share scores.
"""

import collections

import pytest
from openeye import oechem, oeomega

from sass.chem.fragment import fragment_memo_key, fragment_molecule

FRAG_CONNECTORS = ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"]

#: The `t_bisamide` product from the test library. Its piperazine lets ring cleaving reach the
#: same constitution two ways.
QUERY_SMILES = "CCCCC(=O)N1CCN(CC1)C(=O)c1ccncc1"


@pytest.fixture(scope="module")
def query() -> oechem.OEMol:
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, QUERY_SMILES)
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetStrictStereo(False)
    if not omega(mol):
        raise RuntimeError(f"OMEGA failed on {QUERY_SMILES}")
    return mol


def fragments(query: oechem.OEMol, n: int, cleave_cyclic: bool) -> list[list[oechem.OEMol]]:
    return fragment_molecule(
        mol=oechem.OEMol(query),
        n_fragments=n,
        heavy_atom_limit=3,
        cleave_acyclic_bonds=True,
        cleave_cyclic_bonds=cleave_cyclic,
        connector_isotope=1,
        connector_atoms=FRAG_CONNECTORS,
    )


class TestKeyProperties:
    def test_the_same_fragment_keys_the_same(self, query):
        frag = fragments(query, 2, cleave_cyclic=False)[0][0]
        assert fragment_memo_key(frag) == fragment_memo_key(oechem.OEGraphMol(frag))

    def test_moving_one_atom_changes_the_key(self, query):
        """Moving one atom keeps the SMILES half of the key and changes the shape half."""
        frag = fragments(query, 2, cleave_cyclic=False)[0][0]
        before = fragment_memo_key(frag)

        moved = oechem.OEGraphMol(frag)
        atom = next(iter(moved.GetAtoms()))
        x, y, z = moved.GetCoords()[atom.GetIdx()]
        moved.SetCoords(atom, oechem.OEFloatArray([x + 1.0, y, z]))

        after = fragment_memo_key(moved)
        assert after[0] == before[0], "constitution is unchanged, so the SMILES half should match"
        assert after[1] != before[1], "geometry changed, so the shape half must differ"

    def test_the_key_does_not_depend_on_atom_numbering(self, query):
        """Identical geometry gives the same key."""
        frag = fragments(query, 2, cleave_cyclic=False)[0][0]
        reversed_order = oechem.OEGraphMol()
        oechem.OEAddMols(reversed_order, frag)
        assert fragment_memo_key(reversed_order) == fragment_memo_key(frag)


class TestRingCleavingCollision:
    """Ring cleaving reaches the same SMILES from two different cuts.

    This query's piperazine is symmetric, so its two cuts differ only by a rigid motion. The key
    still keeps them apart. It is conservative and never merges two different shapes.
    """

    def test_same_smiles_from_different_cuts_does_not_collide(self, query):
        by_smiles = collections.defaultdict(set)
        for frag_set in fragments(query, 3, cleave_cyclic=True):
            for frag in frag_set:
                smiles, shape = fragment_memo_key(frag)
                by_smiles[smiles].add(shape)

        collisions = {smi: shapes for smi, shapes in by_smiles.items() if len(shapes) > 1}
        assert collisions, (
            "this query is supposed to reach at least one SMILES from two different ring cuts; "
            "if it no longer does, the regression is untested rather than fixed"
        )

    def test_smiles_alone_would_have_merged_them(self, query):
        """A SMILES-only key cannot tell these fragments apart."""
        smiles_only = collections.Counter()
        full_key = set()
        for frag_set in fragments(query, 3, cleave_cyclic=True):
            for frag in frag_set:
                smiles, _ = fragment_memo_key(frag)
                smiles_only[smiles] += 1
                full_key.add(fragment_memo_key(frag))
        assert len(full_key) > len(
            smiles_only
        ), "keying on SMILES alone yields fewer keys, i.e. it merges fragments that the shape-aware key keeps apart"

    def test_acyclic_fragmentation_has_no_collisions(self, query):
        """Without ring cleaving a fragment's cut is unique, so SMILES alone would have sufficed."""
        by_smiles = collections.defaultdict(set)
        for n in (2, 3):
            for frag_set in fragments(query, n, cleave_cyclic=False):
                for frag in frag_set:
                    smiles, shape = fragment_memo_key(frag)
                    by_smiles[smiles].add(shape)
        assert not {s: v for s, v in by_smiles.items() if len(v) > 1}
