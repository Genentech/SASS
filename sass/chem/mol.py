"""Generic OpenEye molecule operations: loading, cleanup, atom and bond surgery.

Nothing here knows about synthons, fragments, or the SASS pipeline.
"""

# Standard Library
import logging
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Union

# Third Party Library
from openeye import oechem, oequacpac

logger = logging.getLogger(__name__)


@contextmanager
def read_mols(file: Union[Path, str], graph: bool = False) -> Iterator[Iterator[oechem.OEMolBase]]:
    """Iterate the molecules in `file`, closing the stream afterwards.

    Prefer this to a bare `oemolistream(path).GetOEMols()`, which leaves the stream open in
    long-lived worker processes.
    """
    istream = oechem.oemolistream(str(file))
    if not istream.IsValid():
        raise ValueError(f"Cannot read molecules from {file}. Check the path and the file extension.")
    try:
        yield istream.GetOEGraphMols() if graph else istream.GetOEMols()
    finally:
        istream.close()


def load_first_mol(
    file: Union[str, Path],
    clear_title: bool = True,
    mol_cleanup: bool = True,
    load_GraphMol: bool = False,
) -> oechem.OEMol:
    """Load the first molecule from a file.

    Parameters
    ----------
    file
        Path to the file containing molecules.
    clear_title
        Whether to clear the title of the molecule.
    mol_cleanup
        Whether to clean up the molecule.
    load_GraphMol
        Whether to load the molecule as a `GraphMol` instead of `OEMol`.
    """
    istream = oechem.oemolistream(str(file))
    if not istream.IsValid():
        raise ValueError(f"Cannot read molecules from {file}. Check the path and the file extension.")
    try:
        reader = istream.GetOEGraphMols() if load_GraphMol else istream.GetOEMols()
        try:
            first = next(reader)
        except StopIteration:
            raise ValueError(f"No molecule found in {file}.") from None
        mol = oechem.OEGraphMol(first) if load_GraphMol else oechem.OEMol(first)
    finally:
        istream.close()

    if mol.NumAtoms() == 0:
        raise ValueError(f"The first molecule in {file} has no atoms.")
    if clear_title:
        mol.SetTitle("")
    if mol_cleanup:
        cleanup_mol(mol)
    return mol


def smiles_to_mol(smiles: str, title: str = "") -> oechem.OEGraphMol:
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smiles)
    mol.SetTitle(str(title))
    return mol


def substitute_atoms(
    mol: oechem.OEMolBase,
    sub_dict: dict,
    delete_H: bool = False,
) -> oechem.OEMolBase:
    """
    Substitute atoms according to the input mapping.

    This function modifies the input mol AND returns the modified mol. The sub_dict
    should not contain any chain substitutions, e.g. C -> U, U -> Np, ...

    Parameters
    ----------
    mol
        Input molecule.
    sub_dict
        Mapping of which atom to substitute to what. Format:
        {(<src_atom_symbol>, <src_isotope>): (<dst_atom_sybmol>, <dst_isotope>)},
        e.g.: {('U', 0): ('C', 13), ('Xe', 22): ('Np', 0)}.
    delete_H
        Whether to delete the H atoms on the substituted atoms.

    Returns
    -------
    oechem.OEMolBase
        The modified molecule after atom substitution.

    """
    atom_subbed = False
    for a in mol.GetAtoms():
        a_symbol = oechem.OEGetAtomicSymbol(a.GetAtomicNum())
        a_isotope = a.GetIsotope()
        key = (a_symbol, a_isotope)
        if key in sub_dict:
            set_atom_type(a, *sub_dict[key])
            atom_subbed = True
            if delete_H:
                for nei in a.GetAtoms():
                    if nei.GetAtomicNum() == 1:
                        mol.DeleteAtom(nei)

    if atom_subbed is False:
        logger.warning("No atom was substituted.")
    return mol


def cleanup_smiles(smiles: str) -> oechem.OEMol:
    """Clean up input SMILES using `cleanup_mol`."""
    mol = smiles_to_mol(smiles)
    cleanup_mol(mol)
    return oechem.OEMolToSmiles(mol)


def implicit_hydrogen(mol: oechem.OEMolBase) -> oechem.OEMolBase:
    """
    Use `ImplicitHydrogens` method to clean up mol.

    Assign correct valence and formal charge, while allowing future enumeration
    of stereocenters. Modifies the input molecule in-place AND returns the mol.
    """
    oechem.OEAssignImplicitHydrogens(mol)
    oechem.OEAssignFormalCharges(mol)
    return mol


# TODO: expand to a more comprehensive filter/cleanup, with options for deep-clean.


def cleanup_mol(
    mol: oechem.OEMolBase,
    valence_method: Callable = oechem.OEAssignMDLHydrogens,
    protonation: bool = False,
) -> oechem.OEMol:
    """
    Clean up input molecule.

    Modifies input `mol` in-place AND returns the `mol`.

    Parameters
    ----------
    mol
        Input molecule.
    valence_method
        Method for completing the valence of all atoms.
    protonation
        Whether to return a protonated or neutral molecule.
    """
    valence_method(mol)

    # `OEGetReasonableProtomer` sets BOTH tautomer and charges. Remove charges if needed downstream.
    if not oequacpac.OEGetReasonableProtomer(mol):
        # This function protonates the `mol` in-place and gets a reasonable tautomer.
        logger.error(f"Unable to process mol {mol.GetTitle()}, SMILES: {oechem.OEMolToSmiles(mol)}.")
    if protonation is False:
        oequacpac.OERemoveFormalCharge(mol)

    oechem.OEAssignAromaticFlags(mol)

    return mol


def count_non_H_atom(mol: oechem.OEMolBase, exclude_symbols: set[str] = set()) -> int:
    """Count the number non-H atoms, excluding those in the `exclude_symbols` set."""
    n = 0
    exclude_symbols.add("H")
    for a in mol.GetAtoms():
        a_num = a.GetAtomicNum()
        if oechem.OEGetAtomicSymbol(a_num) not in exclude_symbols:
            n += 1
    return n


def get_conn_symbols(smi: str, connector_set: set[str]) -> list[str]:
    """Return a list of connector atoms in a given SMILES string based on string matching."""
    return [char for char in connector_set if char in smi]


def get_conn_symbols_freq(mol: oechem.OEMolBase, connector_set: set[str]) -> dict[str, int]:
    """Return the connector atom frequency in the input mol."""
    res = defaultdict(int)
    for a in mol.GetAtoms():
        a_symbol = oechem.OEGetAtomicSymbol(a.GetAtomicNum())
        if a_symbol in connector_set:
            res[a_symbol] += 1
    return res


def get_shortest_path_atoms(
    mol: oechem.OEMolBase,
    start_atom_symbol: str,
    start_isotope: int,
    end_atom_symbol: str,
    end_isotope: int,
    include_start_end: bool = True,
) -> Optional[list[oechem.OEAtomBase]]:
    """
    Get the atoms along the shortest path between two atoms in a molecule using BFS.

    The returned path optionally includes the start and end atom.

    Parameters
    ----------
    mol
        Input molecule.
    start_atom_symbol
        Symbol of the atom on one terminus of the path.
    start_isotope
        Isotope of the atom on one terminus of the path.
    end_atom_symbol
        Symbol of the atom on the other terminus of the path.
    end_isotope
        Isotope of the atom on the other terminus of the path.
    include_start_end
        Whether to include the start and end atom in the return.
    """
    s_atoms = list(find_atoms(mol, start_atom_symbol, start_isotope))
    if len(s_atoms) == 0:
        raise ValueError(f"Start atom {start_isotope}{start_atom_symbol} is not found.")
    if len(s_atoms) > 1:
        raise ValueError(f"Multiple start atoms {start_isotope}{start_atom_symbol} are found. Likely ambiguity.")
    s_atom = s_atoms[0]
    # Returns the shortest path to any matching end atom.

    visited_atoms = {s_atom}  # add to `visited` when enqueuing, not when visiting
    q = deque([[s_atom]])  # queue contains a list of paths (list[Atom])
    while q:
        cur_path = q.popleft()
        cur_atom = cur_path[-1]
        for b in cur_atom.GetBonds():
            nei = b.GetNbr(cur_atom)
            if nei not in visited_atoms:
                visited_atoms.add(nei)
                new_path = [*cur_path, nei]
                nei_symbol = oechem.OEGetAtomicSymbol(nei.GetAtomicNum())
                nei_isotope = nei.GetIsotope()
                if nei_symbol == end_atom_symbol and nei_isotope == end_isotope:
                    if include_start_end is False:
                        for a in (s_atom, nei):
                            new_path.remove(a)
                    return new_path
                q.append(new_path)

    logger.error(f"Unable to find a path between specified start {start_isotope} and end {end_isotope} atoms.")
    return None


def get_same_ring_bonds(
    bond: oechem.OEBondBase,
    include_self: bool,
) -> list[oechem.OEBondBase]:
    """
    Get all bonds in the same ring as the input bond.

    Note: the mol containing the bond should already have ring atoms and bonds assigned
    with the `oechem.OEFindRingAtomsAndBonds` function.

    Parameters
    ----------
    bond
        Input bond object.
    include_self
        Whether to include the input bond in the output.

    Returns
    -------
        List of bonds in the same ring as the input bond.
    """
    visited_bonds = {bond}
    start_atom = bond.GetBgn()  # Arbitrarily choosing one of the two atoms as start.
    end_atom = bond.GetEnd()
    q = deque([[bond, start_atom]])  # Last element is the ending atom of the last bond.
    while q:
        cur_path = q.popleft()
        s_atom = cur_path.pop()
        for nei_bond in s_atom.GetBonds():
            if nei_bond.IsInRing() and nei_bond not in visited_bonds:
                visited_bonds.add(nei_bond)
                next_start = nei_bond.GetNbr(s_atom)
                if next_start == end_atom:
                    if include_self is False:
                        return [*cur_path, nei_bond][1:]
                    else:
                        return [*cur_path, nei_bond]
                q.append([*cur_path, nei_bond, next_start])
    return []


def set_atom_type(atom: oechem.OEAtomBase, target_symbol: str, target_isotope: int) -> None:
    """
    Set the atomic symbol and isotope of an atom.

    Modifies the atom in-place.

    Parameters
    ----------
    atom
        The atom object to be modified.
    target_symbol
        Symbol of the atom type to be set.
    target_isotope
        Isotope of the atom to be set.
    """
    atom.SetAtomicNum(oechem.OEGetAtomicNum(target_symbol))
    atom.SetIsotope(target_isotope)


def find_atoms(
    mol: oechem.OEMolBase,
    atom_symbol: str,
    isotope: int = 0,
) -> Iterator[oechem.OEAtomBase]:
    """Yield atoms in the input molecule that matches the atomic symbol and isotope number."""
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == oechem.OEGetAtomicNum(atom_symbol) and a.GetIsotope() == isotope:
            yield a


def find_connected_atoms(
    atom: oechem.OEAtomBase,
    return_H: bool = False,
) -> Iterator[oechem.OEAtomBase]:
    """Generator that yields atoms connected to the input atom (neighbors)."""
    for nei in atom.GetAtoms():
        if return_H or nei.GetAtomicNum() != 1:  # Optionally yield H atoms.
            yield nei


def unset_stereo(atom: oechem.OEAtomBase):
    """
    Remove the stereochemical information from an atom.

    Works only on 2D molecules. On 3D molecules, the stereochemistry is inferred from
    the 3D coordinates, and unsetting has no effect on the actual stereochemistry of the atom.
    """
    neis = list(atom.GetAtoms())

    ret_code = atom.SetStereo(neis, oechem.OEAtomStereo_Tetra, oechem.OEAtomStereo_Undefined)
    if ret_code is False:
        logger.warning("Unsetting stereocenter failed.")


def delete_atom(mol: oechem.OEMol, atom_symbol: str, isotope: int):
    """Delete all atoms in a molecule that match the specified atom symbol and isotope."""
    for a in find_atoms(mol, atom_symbol, isotope):
        oechem.OESuppressHydrogens(a)  # Make H implicit.
        mol.DeleteAtom(a)  # This will also remove the connected bonds.


def set_conf_centroid_coord(
    conf: oechem.OEConfBase,
    atoms: Iterable[tuple[str, int]],
    centroid_atom: oechem.OEAtomBase,
) -> oechem.OEMol:
    """
    Set the centroid atom to be at the geometric center of input atoms on a conformer.

    Parameters
    ----------
    conf
        Input conformer.
    atoms
        List of atoms to calculate the centroid from.
    centroid_atom
        The atom to set the centroid coordinate to.
    """
    atoms = set(atoms)

    # Take average of the dummy atoms coordinates.
    x = y = z = 0
    n = 0
    for symbol, isotope in atoms:
        atoms = list(find_atoms(conf, symbol, isotope))
        for atom in atoms:
            _x, _y, _z = conf.GetCoords(atom)
            x += _x
            y += _y
            z += _z
            n += 1
    if n == 0:
        raise ValueError("Input conf does not contain any of the specified atom types.")
    if n == 1:
        logger.warning("Only one matching atom was found in total. Centroid set at the position of that atom.")
    new_coord = [coord / n for coord in (x, y, z)]
    conf.SetCoords(centroid_atom, new_coord)
    return conf


def replace_atoms_with_centroid(
    mol: oechem.OEMolBase,
    atoms: Iterable[tuple[str, int]],
    centroid_symbol: str,
    centroid_isotope: int = 0,
    delete_original_atoms: bool = True,
) -> oechem.OEMol:
    """
    Replace the connector atoms with centroids in each conformer of a multi-conformer mol.

    Works with any number of connector atoms.
    """
    # Use an unlikely atom type as the temp centroid, in case the desired centroid type
    # is among the `dummies`.
    _symbol, _isotope = "Xe", 777
    if (_symbol, _isotope) in atoms:
        raise ValueError(f"{_symbol}_{_isotope} is of the same type as one of the existing connector atoms.")

    mol = oechem.OEMol(mol)
    new_atom = mol.NewAtom(oechem.OEGetAtomicNum(_symbol))
    new_atom.SetIsotope(_isotope)
    # Change the coordinate on each conformer of the multi-conformer molecule.
    new_mol = oechem.OEMol(mol)
    new_mol.DeleteConfs()
    for conf in mol.GetConfs():
        new_conf = set_conf_centroid_coord(conf, atoms, new_atom)
        new_mol.NewConf(new_conf)

    if delete_original_atoms:  # TODO: rewrite to loop through atoms just once.
        for symbol, isotope in atoms:
            delete_atom(new_mol, symbol, isotope)

    # Change the temp centroid into the desired final centroid type. Atom type change
    # can be done on the mol object level instead on each conformer.
    substitute_atoms(new_mol, sub_dict={(_symbol, _isotope): (centroid_symbol, centroid_isotope)})

    return new_mol


def get_distance(mol1: oechem.OEMol, mol2: oechem.OEMol, symbol: str, isotope: int) -> float:
    """Calculate the distance between two atoms on two different molecules."""
    atom1 = next(find_atoms(mol1, symbol, isotope))
    atom2 = next(find_atoms(mol2, symbol, isotope))
    return oechem.OEGetDistance(mol1, atom1, mol2, atom2)
