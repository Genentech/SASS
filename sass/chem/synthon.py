"""Synthon-specific molecule surgery: connector atoms, ring labeling, bond extension.

A synthon is a molecule whose connector atoms (`U`, `Np`, `Pu`, `Am`) stand in for the bonds it
will form. The functions here substitute connectors, label the ones that sit in rings, and
extend a dummy bond so an overlay can align along it.
"""

# Standard Library
import logging
from collections import defaultdict

# Third Party Library
from openeye import oechem

from sass.chem.fragment import FragNodeMatcher, enumerate_fragment_connector_substitution, map_mol_frag_graph
from sass.chem.mol import (
    find_atoms,
    find_connected_atoms,
    replace_atoms_with_centroid,
    set_atom_type,
    smiles_to_mol,
)

logger = logging.getLogger(__name__)


def generate_heavy_isotope_atoms(mol: oechem.OEMol) -> list[oechem.OEAtomBase]:
    """Generator that yields atom pointers of non-standard isotopic atoms."""
    for atom in mol.GetAtoms():
        if atom.GetIsotope() != 0:
            yield atom


def special_substitute_atoms(mol: oechem.OEMol, sub_map: dict[str, list[str]], isotope: int = 0) -> oechem.OEMol:
    """
    Substitute connector atoms in the input molecule.

    In the substitution mapping, if two keys map to the same destination atom, the two
    source atoms are substituted and merged into a geometric centroid. If a key maps to
    multiple destination atoms, multiple instances of the destination atoms are placed at
    the coordinate of the source atom. This is done to ensure matching of the connector
    atoms during ROCS scoring.

    Parameters
    ----------
    mol
        Input molecule to be substituted.
    sub_map
        Mapping of substitution, e.g. {'He': ['U'], 'Ne': ['Np', 'Pu'], 'Ar': ['U']}.
    isotope
        The desired isotope number of the substituted atom.

    Returns
    -------
        The substituted molecule.
    """
    freq = defaultdict(int)  # Counting the duplicate sub_atoms.
    mol = oechem.OEMol(mol)

    # Step 1: Substitute atoms based on `sub_map`.
    for atom in mol.GetAtoms():
        atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
        if atom_symbol in sub_map:
            sub_atom = sub_map[atom_symbol][0]  # should have at least one target.
            set_atom_type(atom, sub_atom, isotope)
            freq[sub_atom] += 1
            if len(sub_map[atom_symbol]) > 1:
                # Place additional connector atoms (no bond needed), for cross-score.
                for i in range(1, len(sub_map[atom_symbol])):
                    sub_atom = sub_map[atom_symbol][i]
                    new_atom = mol.NewAtom(atom)
                    set_atom_type(new_atom, sub_atom, isotope)
                    freq[sub_atom] += 1

    # Step 2: Swap any duplicate sub_atom with centroids.
    for key, val in freq.items():
        if val >= 2:
            mol = replace_atoms_with_centroid(
                mol,
                atoms=[(key, isotope)] * val,
                centroid_isotope=isotope,
                centroid_symbol=key,
                delete_original_atoms=True,
            )

    return mol


def label_synthon_ring_connector(
    synthon_smiles: dict[int, str],
    connector_atoms: list[str],
    smirks: str,
) -> dict[str, "bool"]:
    """
    Determine whether a synthon connector forms a ring bond, or an acyclic bond.

    Label the atoms connected to the connector atoms (nei atoms) by the connector symbol,
    react by SMIRKS rule, check the bond type of the newly formed bonds, and mark the
    connector atoms accordingly.

    For example, for a reaction of [U]-[*:1].[U]-[*:2]>>[*:1]-[*:2], the only connector
    "U" is not a ring connector, so the return is {"U": "False"}.
    For a reaction of ([U]-[*:1].[Np]-[*:2]-[*:3]=[Pu]).[Pu]=[*:4].[Np]-[*:5]=[*:6]-[U]>>
    [*:1]-[*:6]=[*:5]-[*:2]-[*:3]=[*:4], the return is {"U": "True", "Np": "True", "Pu": "False"}.

    Parameters
    ----------
    synthon_smiles
        SMILES or SMARTS of synthons labeled by their reaction position index in SMIRKS.
    connector_atoms
        All possible connector atom symbols on synthons.
    smirks
        Reaction SMIRKS to react the synthons.

    Returns
    -------
    dict
        A mapping of connector atom symbol to a boolean value indicating whether it forms a ring connector.
    """
    connector_atoms = set(connector_atoms)
    connector_nei_tag = "connected_to"
    connector_nei_dict = defaultdict(list)  # e.g. {'U': [atom1, atom2]}

    synthon_mols = {}
    for key, smi in synthon_smiles.items():
        mol = smiles_to_mol(smi)
        synthon_mols[key] = mol

    for mol in synthon_mols.values():  # `mol` objects are mutable
        for atom in mol.GetAtoms():
            atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
            if atom_symbol in connector_atoms:
                for _atom in find_connected_atoms(atom):
                    cur_data = []
                    if _atom.HasData(connector_nei_tag):
                        cur_data = _atom.GetData(connector_nei_tag)
                    _atom.SetData(connector_nei_tag, [*cur_data, atom_symbol])
                    # Set the data as a list, in case multiple connector atom share the same neighbor atom.
                    # Caution: a list object won't survive pickling, or read/write to file.

    # React and check whether the formed bonds are in a ring.
    libgen = oechem.OELibraryGen(smirks)
    for i, rxnt in synthon_mols.items():
        ret_code = libgen.SetStartingMaterial(rxnt, i, True)  # Use unique matches
        if ret_code == 0:  # 0 means failed to add. > 1 means multiple matches.
            raise ValueError(f"Failed to add synthon {rxnt} as {i}-th reactant.")
        if ret_code > 1:
            logger.info(f"Multiple matches found for synthon {rxnt} as {i}-th reactant.")
    prod = oechem.OEMol(next(libgen.GetProducts()))

    oechem.OEFindRingAtomsAndBonds(prod)
    for atom in prod.GetAtoms():
        if atom.HasData(connector_nei_tag):
            for connector_symbol in atom.GetData(connector_nei_tag):
                connector_nei_dict[connector_symbol].append(atom)

    res = {}
    for key, val in connector_nei_dict.items():
        atom1, atom2 = val  # One connector symbol can only correspond to one bond formed / two atoms.
        bond = mol.GetBond(atom1, atom2)
        res[key] = str(bond.IsInRing())  # Use str(bool) to match the frag_connector labels.

    return res


def extend_dummy_bond(
    mol: oechem.OEMol,
    dummy_atom_symbol: str,
    dummy_isotope: int,
    new_dummy_symbol: str,
    new_isotope: int,
    distance: float,
    explicit_bond: bool = False,
) -> oechem.OEMol:
    """
    Extend the dummy bond vector on each conformer.

    Place a second dummy atom along the line formed by the original dummy atom and the
    atom connected to that dummy atom. The new dummy bond vector is defined as the line
    formed by the two dummy atoms.

    Parameters
    ----------
    mol
        Input molecule.
    dummy_atom_symbol
        Symbol of the original dummy atom in `mol`.
    dummy_isotope
        Isotope number of the original dummy atom.
    new_dummy_symbol
        Symbol of the 2nd dummy atom to be placed.
    new_isotope
        Isotope number of the 2nd dummy atom.
    distance
        Directional distance from the original dummy atom to the new dummy atom. New
        dummy atom is placed alonged the line formed by the original dummy atom and its
        neighbor atom. Positive distance means the new dummy atom is placed in the same
        direction as the (neighbor atom -> original dummy), negative distance means the
        opposite direction.
    explicit_bond
        Whether to create an explicit bond between the original dummy atom and the new
        dummy atom. Having a bond or not has no effect on ROCS.

    Returns
    -------
    oechem.OEMol
        A new molecule with the extended dummy bond vector.
    """
    mol = oechem.OEMol(mol)
    new_atom = mol.NewAtom(oechem.OEGetAtomicNum(new_dummy_symbol))
    new_atom.SetIsotope(new_isotope)
    if explicit_bond:
        dummy_atom = next(find_atoms(mol, dummy_atom_symbol, dummy_isotope))
        mol.NewBond(new_atom, dummy_atom, 1)
    new_mol = oechem.OEMol(mol)
    new_mol.DeleteConfs()

    for conf in mol.GetConfs():
        dummy_atom = next(find_atoms(conf, dummy_atom_symbol, dummy_isotope))
        connected_atom = next(find_connected_atoms(dummy_atom))
        dummy_coord = conf.GetCoords(dummy_atom)
        connected_coord = conf.GetCoords(connected_atom)
        dummy_bond_length = oechem.OEGetDistance(conf, dummy_atom, connected_atom)
        scaled_distance_factor = distance / dummy_bond_length
        new_coord = [b + scaled_distance_factor * (b - a) for a, b in zip(connected_coord, dummy_coord, strict=False)]
        conf.SetCoords(new_atom, new_coord)
        new_mol.NewConf(conf)

    return new_mol


def order_and_substitute_fragments(
    synthon_smiles_eg: dict[int, str],
    synthon_connectors: list[str],
    fragments: list[oechem.OEMol],
    frag_connectors: list[str],
    cross_score: bool,
) -> list[list[oechem.OEMol]]:
    """Generate all ordered fragment lists for a set of synthons.

    Given the synthon order, enumerate all ways to order the fragments, and all ways to
    map the fragment connector atoms to synthon connector atoms, and substitute those
    fragment connector atoms. The synthon index order is returned, since the representative
    synthons stand in for whole synthon lists. It is used to look up synthon conformer files.

    Parameters
    ----------
    synthon_smiles_eg
        Representative SMILES of synthons, indexed by the reaction order.
    synthon_connectors
        Connector atoms found in synthons.
    fragments
        Unordered query fragments.
    frag_connectors
        Connector atoms found in query fragments.
    cross_score
        Whether to allow scoring of synthon and fragment pairs that have different number
        of connector atoms. See docstring of `enumerate_fragment_connector_substitution`
        for more details.

    Returns
    -------
        The order of the synthon indices, and lists of ordered and substituted fragments.
    """
    synthon_order, synthon_smiles = zip(*synthon_smiles_eg.items(), strict=False)
    syn_nodes = map_mol_frag_graph(synthon_smiles, set(synthon_connectors))
    frag_smiles = [oechem.OEMolToSmiles(f) for f in fragments]
    frag_nodes = map_mol_frag_graph(frag_smiles, set(frag_connectors))

    matches = FragNodeMatcher.match_all_graphs(syn_nodes, frag_nodes)
    ordered_fragments = []
    for match in matches:
        # Sort the fragments by synthon order. Frag node indices are stored as f'{idx}_2'
        # (see `match_all_graphs`).
        frag_order = [int(match[syn_node].index.split("_")[0]) for syn_node in syn_nodes]
        ordered_frags = [fragments[idx] for idx in frag_order]
        for res in enumerate_fragment_connector_substitution(syn_nodes, match, cross_score):
            if res:
                temp = [special_substitute_atoms(f, res) for f in ordered_frags]
                ordered_fragments.append(temp)

    return synthon_order, ordered_fragments
