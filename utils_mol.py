"""
Functions for molecule handling.
"""

# Standard Library
from collections import defaultdict, deque
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Set, Tuple, Union

# Third Party Library
from openeye import oechem, oeomega, oequacpac, oeshape


def split_n_chunks(
    input_file: Union[Path, str],
    chunk_n: int,
    chunk_size: int,
    output_folder: Union[Path, str],
    file_prefix: str,
    file_suffix: str,
):
    """
    Split one input file into chunks, given either target number of chunks, or size of each chunk.
    """

    os.makedirs(output_folder, exist_ok=True)

    db = oechem.OEMolDatabase(input_file)
    chunk = 0
    if chunk_size is None:
        n_mol = db.NumMols()
        chunk_size = math.ceil(n_mol / chunk_n)
    ostream = oechem.oemolostream()
    for i, mol in enumerate(db.GetOEMols()):
        if i % chunk_size == 0:
            ostream.close()
            ostream = oechem.oemolostream(os.path.join(output_folder, f"{file_prefix}_chunk_{chunk}.{file_suffix}"))
            chunk += 1
        oechem.OEWriteMolecule(ostream, mol)
    ostream.close()

    logging.info(f"{input_file} split in to {chunk_n} chunks.")


# Molecule manipulation


def load_first_mol(
    file: Union[str, Path],
    clear_title: bool = True,
    mol_cleanup: bool = True,
):
    for mol in oechem.oemolistream(file).GetOEMols():
        ref_mol = oechem.OEMol(mol)
        if clear_title:
            ref_mol.SetTitle("")
        if mol_cleanup:
            cleanup_mol(ref_mol)
        return ref_mol  # Return the 1st molecule.


def smiles_to_mol(smiles: str, title: str = "") -> oechem.OEMol:
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    mol.SetTitle(str(title))
    return mol


def substitute_atoms(
    mol: oechem.OEMolBase,
    sub_dict: dict,
    delete_H: bool = False,
) -> oechem.OEMolBase:
    """
    Substitute atoms in mol objects according to the input substitution dictionary.
    Modifies the input mol, AND return the mol.

    Args:
        mol: Input molecule.
        sub_dict: Lookup table on which atom to substitute to what. Format:
            {('U', 0): ('C', 13), ('Xe', 22): ('Np', 0)}.
        delete_H: Whether to delete the H atoms on the substituted atoms.
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
        logging.warning("No atom was substituted.")
    return mol


def cleanup_smiles(smiles: str):
    """
    Clean up input SMILES using `cleanup_mol`.
    """

    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    _ = cleanup_mol(mol)
    return oechem.OEMolToSmiles(mol)


def implicit_hydrogen(mol: oechem.OEMolBase):
    """
    Assign correct valence and formal charge, while allowing future enumeration
    of stereocenters. Changes the input in-place.
    """

    oechem.OEAssignImplicitHydrogens(mol)
    oechem.OEAssignFormalCharges(mol)

    return mol


def cleanup_mol(
    mol: oechem.OEMolBase,
    valence_method: Callable = oechem.OEAssignMDLHydrogens,
    protonation: bool = False,
) -> oechem.OEMol:
    """
    Clean up input molecule.

    Args:
        mol: Input molecule.
        valence_method: Method for completing the valence of all atoms.
    """

    valence_method(mol)

    # `OEGetReasonableProtomer` sets BOTH tautomer and charges. Remove charges if needed downstream.
    if not oequacpac.OEGetReasonableProtomer(mol):
        # This function protonates the `mol` in-place and gets a reasonable tautomer.
        logging.error(f"Unable to process mol {mol.GetTitle()}, SMILES: {oechem.OEMolToSmiles(mol)}.")
    if protonation is False:
        oequacpac.OERemoveFormalCharge(mol)

    oechem.OEAssignAromaticFlags(mol)

    return mol


def fragment_molecule(
    mol: oechem.OEMol,
    n_fragments: int,
    heavy_atom_limit: int,
    cleave_acyclic_bonds: bool = True,
    cleave_cyclic_bonds: bool = True,
    connector_isotope: int = 1,
    connector_atoms: List[str] = ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"],
    # connector_idx: int = 0,
    connector_tag: str = "is_ring_connector",
    deduplicate: bool = True,
) -> List[List[oechem.OEMol]]:
    """
    Recursively fragment a molecule into desired number of fragments.

    Args:
        mol: Input molecule to be fragmented.
        n_fragments: Desired number of fragments.
        heavy_atom_limit: Minimum number of heavy atoms to be considered a valid fragment.
        cleave_acyclic_bonds: Whether to cleave acyclic bonds during fragmentation.
        cleave_cyclic_bonds: Whether to cleave cyclic bonds during fragmentation.
        connector_atoms: Atoms used to cap the cleaved bonds. By default use the noble
            gases, since they are usually not found in organic reactants, and are different
            from the transuranium atoms used in Enamine synthon files.
        connector_isotope: The isotope number of the connector atoms. Use a non-0 isotope
            number to not count connector atoms as heavy atoms when checking fragment sizes.
        connector_tag: Name of the atom label to indicate whether a connector atom is on
            a formerly ring or non-ring bond.
        deduplicate: Whether to deduplicate the resulting fragment sets. Due to the need
            to recurse into each of the fragment from a split step, there are deduplicate
            fragment sets where the fragments are the same when not considering the
            connector atoms.

    Returns:
        A list of list of fragments.
    """

    def get_frags_key(fragments: List[oechem.OEMol]) -> str:
        """
        "Hash" the fragments by replacing the connector atoms with C, generate SMILES,
        and sort and concat the strings.
        """

        smis = []
        for frag in fragments:
            _frag = oechem.OEMol(frag)
            substitute_atoms(_frag, sub_dict=sub_dict)
            smis.append(oechem.OEMolToSmiles(_frag))
        smis.sort()
        return "&".join(smis)

    def backtrack(fragments: List[oechem.OEMol], mol: oechem.OEMol, c_idx: int):
        """
        For all frags generated from `mol`, recurse into each frag, and append the rest
        of the frags to `fragments`.
        """

        nonlocal frags_list

        if len(fragments) == n_fragments - 1:
            res = fragments + [mol]
            if not deduplicate:
                frags_list.append(res)
            else:
                frags_key = get_frags_key(res)
                if frags_key not in frags_key_set:
                    frags_key_set.add(frags_key)
                    frags_list.append(res)
            return

        if len(fragments) > n_fragments - 1:
            return

        # cut 1 acyclic bond, or 2 ring bonds
        oechem.OEFindRingAtomsAndBonds(mol)  # call this each time to re-assign ring membership
        for bond in mol.GetBonds():
            if cleave_acyclic_bonds and not bond.IsInRing():
                b_idx = bond.GetIdx()
                _mol = oechem.OEMol(mol)  # Do not cleave the original mol, otherwise hard to backtrack.
                if connector_tag is not None and bond.HasData(ring_bond_label):
                    _connector_tag = (connector_tag, bond.GetData(ring_bond_label))
                else:
                    _connector_tag = None
                _frags = get_n_fragments(
                    mol=_mol,
                    b_indices=[b_idx],
                    cap_atom_list=[(connector_atoms[c_idx], connector_isotope)],
                    connector_atom_tag=_connector_tag,
                )
                if len(_frags) == 2:
                    if heavy_atom_limit is None or check_fragment_size(_frags, count_non_H_atom, heavy_atom_limit):
                        for i in range(len(_frags)):
                            backtrack(fragments + _frags[:i] + _frags[i + 1 :], _frags[i], c_idx + 1)
            elif cleave_cyclic_bonds and bond.IsInRing():  # is ringbond
                for bond2 in mol.GetBonds():
                    if bond2 != bond and bond2.IsInRing():
                        _mol = oechem.OEMol(mol)
                        b_idx1 = bond.GetIdx()
                        b_idx2 = bond2.GetIdx()
                        if connector_tag is not None:
                            _connector_tag = (connector_tag, bond.GetData(ring_bond_label))
                        else:
                            _connector_tag = None
                        _frags = get_n_fragments(
                            mol=_mol,
                            b_indices=[b_idx1, b_idx2],
                            cap_atom_list=[
                                (connector_atoms[c_idx], connector_isotope),
                                (connector_atoms[c_idx + 1], connector_isotope),
                            ],
                            connector_atom_tag=_connector_tag,
                        )
                        if len(_frags) == 2:
                            ring_label = "prev_ring_atom"
                            for _frag in _frags:
                                for atom in get_shortest_path_atoms(
                                    _frag,
                                    connector_atoms[c_idx],
                                    connector_isotope,
                                    connector_atoms[c_idx + 1],
                                    connector_isotope,
                                    include_start_end=False,
                                ):
                                    atom.SetData(ring_label, True)

                            if heavy_atom_limit is None or check_fragment_size(
                                _frags,
                                count_non_ring_atom,
                                heavy_atom_limit,
                                ring_label=ring_label,
                            ):
                                for i in range(len(_frags)):
                                    backtrack(fragments + _frags[:i] + _frags[i + 1 :], _frags[i], c_idx + 2)

    ring_bond_label = "parent_ring_bond"
    oechem.OEFindRingAtomsAndBonds(mol)
    for bond in mol.GetBonds():
        bond.SetData(
            ring_bond_label, str(bond.IsInRing())
        )  # Convert to str, to prevent this data being dropped when pickling..

    # For deduplication
    sub_atom = "C"
    sub_isotope = 0
    sub_dict = {(conn, connector_isotope): (sub_atom, sub_isotope) for conn in connector_atoms}
    frags_key_set = set()

    frags_list = []
    backtrack([], mol, 0)
    return frags_list


def get_n_fragments(
    mol: oechem.OEMol,
    b_indices: List[int],
    cap_atom_map: dict = None,
    cap_atom_list: List[Tuple[str, int]] = None,
    bond_order: int = 1,
    cap_atom_3D: bool = True,
    clear_cap_atom_data: bool = True,
    connector_atom_tag: Tuple[str, Any] = None,
) -> List[oechem.OEMolBase]:
    """
    Split input molecule along the specified bond, and cap the bond cleaved with
    desired atom type/isotope.

    It uses bond indices instead of bond object to specify which bonds to cleave, since
    often the input `mol` is a copy of the original mol to be cleaved, so the bonds are
    not the same objects as the bonds in the original mol, but the bond indices are the
    same in the copied `mol`.

    Args:
        mol: Molecule to be fragmented.
        b_indices: A list of bond indices of the bonds to be broken.
        cap_atom_map: Mapping of bond idx to the type of capping dummy atom for that bond.
            Format: {b_idx: {'symbol': 'C', 'isotope': 13}}.
        cap_atom_3D: Whether to position the cap (dummy) atom at the original
            position of the atom across the bond cleaved, or at the center of
            mass of molecule (default location for new atoms).
        cap_atom_list: Custom, secondary input for cap-atoms. Use when `cap_atom_map` is None.
        bond_order: Bond order of the new bond between the connector atom and the fragment.
        clear_cap_atom_data: Whether to clear the label of the cap (dummy) atom, if
            copied from template. Also clears all tagged data. Necessary to avoid mixing
            up when selecting fragments based on tags (only the original atom should
            keep the tag, not the copied atom.)
        connector_atom_tag: Tag name and value to assign to the connector atoms.
    """

    # TODO: deprecate this cap_atom_map part. User should specify the cap_atom_list.
    if cap_atom_map is None:
        if cap_atom_list is None:
            cap_atom_map = {idx: {"symbol": "C", "isotope": 13 + n} for n, idx in enumerate(b_indices)}
        else:
            cap_atom_map = {
                idx: {"symbol": symbol, "isotope": isotope}
                for idx, (symbol, isotope) in zip(b_indices, cap_atom_list)
                # TODO this requires cap_atom_list to be the same length as bonds to cut.
            }

    for bond in mol.GetBonds():
        b_idx = bond.GetIdx()
        if b_idx in b_indices:  # Technially should take a set here. But this list is very short.
            a1, a2 = bond.GetBgn(), bond.GetEnd()
            mol.DeleteBond(bond)
            cap_atom = cap_atom_map[b_idx]["symbol"]
            cap_isotope = cap_atom_map[b_idx]["isotope"]

            for atom, template in ((a1, a2), (a2, a1)):
                if cap_atom_3D:
                    # Use the other atom as template for the dummy (to copy the position)
                    dummy = mol.NewAtom(template)
                else:
                    # Create a new atom at the center of mass.
                    dummy = mol.NewAtom(oechem.OEGetAtomicNum(cap_atom))

                if clear_cap_atom_data:
                    dummy.SetName("")
                    dummy.Clear()

                dummy.SetAtomicNum(oechem.OEGetAtomicNum(cap_atom))
                dummy.SetIsotope(cap_isotope)
                mol.NewBond(atom, dummy, order=bond_order)
                if connector_atom_tag is not None:
                    key, val = connector_atom_tag
                    dummy.SetData(key, val)

    # Split `mol` into components
    return list(generate_components(mol))


def generate_components(mol: oechem.OEMol) -> Iterator[oechem.OEMol]:
    """
    Splits components of a multi-component molecule into separate molecules.

    Copied from https://docs.eyesopen.com/toolkits/python/_downloads/9d7132664fe135a989141fdc9aef0af7/parts2mols.py
    """

    numparts, partlist = oechem.OEDetermineComponents(mol)
    pred = oechem.OEPartPredAtom(partlist)
    for i in range(1, numparts + 1):
        pred.SelectPart(i)
        partmol = oechem.OEGraphMol()
        oechem.OESubsetMol(partmol, mol, pred)
        yield partmol


def get_heavy_isotopes(mol: oechem.OEMol) -> List[int]:
    """
    Return a sorted list of non-standard isotope numbers in the mol.
    For use when sorting the output from the `generate_n_fragment` function to match to
    3/3+ component reaction synthons, assuming that the dummy atoms are substituted with
    13C, 14C, 15C, etc.
    """

    res = []
    for heavy_atom in generate_heavy_isotope_atoms(mol):
        res.append(heavy_atom.GetIsotope())
    return sorted(res)


def generate_heavy_isotope_atoms(mol: oechem.OEMol) -> List[oechem.OEAtomBase]:
    """
    Yield atom pointers of non-standard isotope atoms.
    """

    for atom in mol.GetAtoms():
        if atom.GetIsotope() != 0:
            yield atom


def count_non_H_atom(mol) -> int:
    """
    Count the number non-H atoms with default isotope number (i.e. not including connector atoms).
    However, if connector atoms have default isotopes (0), this function will cause error.
    """

    return sum(atom.GetAtomicNum() != 1 and atom.GetIsotope() == 0 for atom in mol.GetAtoms())


def count_non_connector_heavy_atoms(mol, connector_symbols: Set[str]) -> int:
    """
    Count the number of non-H, non-connector atoms in input `mol`.
    Connector atoms are designated by having non-0 isotope numbers.
    """

    n = 0
    for a in mol.GetAtoms():
        a_num = a.GetAtomicNum()
        if a_num != 1:
            if oechem.OEGetAtomicSymbol(a_num) not in connector_symbols:
                n += 1
    return n


def check_fragment_size(
    fragments: List[oechem.OEMol],
    filter_fn: Callable,
    heavy_atom_count: int,
    **kwargs,
):
    """
    Check whether all fragments pass the heavy-atom filter.

    Args:
        fragments: An array of mol objects. E.g. generated from fragmenting a parent mol.
        filter_fn: Function used to count the size (heavy-atom) of each fragment.
        heavy_atom_count: The minimum size of fragments allowed.
        kwargs: Keyword args for the `filter_fn`.
    """

    return all(filter_fn(frag, **kwargs) >= heavy_atom_count for frag in fragments)


def count_non_ring_atom(mol: oechem.OEMol, ring_label: str) -> int:
    """
    Count the number of heavy atoms that are not part of the ring-fragment.
    This is NOT simply counting atoms that are not .IsRing()! The ring atoms need to be
    labeled with `ring_label` prior to this step. Ring atoms are usually the atoms that
    are along the shortest path between two ring connectors.

    Args:
        mol: Input mol object.
        ring_label: Text label designating an atom in ring-fragments. Label is entered
            as an oechem object tag.
    """

    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            if not atom.HasData(ring_label):
                count += 1

    return count


def get_conn_symbols(smi, connector_set):
    """
    Return a list of connector atoms in a given SMILES string based on string matching.
    """

    return [char for char in connector_set if char in smi]


def enumerate_syn_frag_orders(
    smis: List[str],
    ring_connector_dict: dict,
) -> List[List[int]]:
    """
    Enumerate valid orders of synthons/query fragments based on connector information.
    e.g. [R1-U, R2-Np, Np-R3-U], a valid order of the indices is 0, 2, 1, i.e. [R1-U, U-R3-Np, Np-R2].
    Works by finding a "terminal" synthon/fragment and follow the connector atom information.
    For reactions that generate a ring with >2 pieces, all pieces could be a "terminal" piece.

    Args:
        smis: SMILES of the input synthons/query fragments.
        ring_connector_dict: A mapping of connector atoms to whether they are ring connectors.
            e.g. {'U': 'True', 'Np': 'True', 'Pu': 'False'.}

    Returns:
        A list of list of ordered indices.
    """

    connector_set = set(ring_connector_dict.keys())

    def backtrack(temp):
        nonlocal res

        if len(temp) == len(smis):
            res.append(temp[:])
            return

        if len(temp) == 0:  # find the terminal piece
            for i, smi in enumerate(smis):
                conn_symbols = get_conn_symbols(smi, connector_set)
                if len(conn_symbols) == 1 or (
                    len(conn_symbols) == 2 and all(ring_connector_dict[conn] == str(True) for conn in conn_symbols)
                ):
                    # use str(True) to prevent data being dropped during pickling.
                    backtrack(temp + [i])
        else:
            prev_smi = smis[temp[-1]]
            conn_symbols = get_conn_symbols(prev_smi, connector_set)
            # find any synthon that can connect to the previous fragment.
            next_idx = set()
            for i, smi in enumerate(smis):
                if i not in temp:  # exclude ones already used
                    if any(conn in smi for conn in conn_symbols):
                        next_idx.add(i)
            for _idx in next_idx:
                backtrack(temp + [_idx])

    res = []
    backtrack([])
    return res


def enumerate_fragment_mol_orders(
    smis: List[str],
    frag_mols: List[oechem.OEMol],
    cross_score: bool,
    frag_connector_symbols: set,
    synthon_connector_symbols: set,
):
    """
    Given an ordered synthon list and an terminus-to-terminus fragments list, return all valid sets
    of query fragment molecules to be scored against the synthons.

    Args:
        smis: ORDERED synthon SMILES list.
        frag_mols: A list of query fragments in order, starting from a terminal fragment.
        cross_score: Whether to enable cross-scoring, i.e. scoring ring connector portions
            against linear connector portions.
        frag_connector_symbols: Connector atoms on fragments.
        synthon_connector_symbols: Connector atoms on synthons.

    Returns:
        Lists of list of mol objects, in the order corresponding to input synthons.
    """

    def order_connector_map(conns1, conns2):
        """
        Create combinations of synthon-fragment mapping when there are more than 1 connectors.
        e.g. synthon connector: [U, Np], fragment connector: [He, Ne], there are 2 possible
        mapping: U-He/Np-Ne, U-Ne/Np-He.
        """

        def backtrack(temp: list, used_conns):
            nonlocal result

            if len(temp) == len(conns1):
                result.append(temp[:])  # more ideally use dict, but harder to copy

            for ele in conns2:
                if ele not in used_conns:
                    used_conns.add(ele)
                    backtrack(
                        temp + [(conns1[len(temp)], ele)], used_conns
                    )  # i.e. always take the ele from `syn_conns` in the same order.
                    used_conns.remove(ele)

        assert len(conns1) == len(conns1)

        result = []
        backtrack([], set())
        return result

    def special_substitute_atoms(mol, sub_map, isotope: int = 0):
        """
        Substitute connector atoms in `mol`. If subsituted mol contains the same connector
        atom more than once, merge two into a centroid. If a frag connector atom map to
        mulitple synthon connectors, place all at the frag connector atom coordinate.


        Args:
            mol: Input molecule to be substituted.
            sub_map: Mapping of substitution, e.g. {'He': ['U'], 'Ne': ['Np', 'Pu'], 'Ar': ['U']}.
            isotope: The desired isotope number of the substituted atom.
        """

        freq = defaultdict(int)
        mol = oechem.OEMol(mol)

        # Step 1: Substitute atoms based on `sub_map`.
        for atom in mol.GetAtoms():
            atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
            if atom_symbol in sub_map:
                sub_atom = sub_map[atom_symbol][0]  # should have at least one target.
                set_atom_type(atom, sub_atom, isotope)
                freq[sub_atom] += 1
                if len(sub_map[atom_symbol]) > 1:
                    # Set additional connector atoms (no bond), for cross-score.
                    for i in range(1, len(sub_map[atom_symbol])):
                        sub_atom = sub_map[atom_symbol][i]
                        new_atom = mol.NewAtom(atom)
                        set_atom_type(new_atom, sub_atom, isotope)
                        freq[sub_atom] += 1

        # Step 2: Swap any duplicate sub_atom with centroids.
        # Assuming that the freq of sub_atom is at most 2 in each fragment. i.e. no reaction form 3 bonds.
        # This is also constrained by the fragmentation fn (only cut 1 lin bond or 1 ring twice).
        for val, key in freq.items():
            if key == 2:
                mol = replace_dummies_with_centroid(
                    mol,
                    dummies=[(val, 0)] * 2,
                    centroid_isotope=isotope,
                    centroid_symbol=val,
                    delete_original_dummies=True,
                )

        return mol

    # If need to deal with non-standard isotopes, use tuples as keys for `known_F_map`.
    def backtrack(temp: List[oechem.OEMol], i, known_S_set, known_F_map):
        """
        Enumerate valid and properly substituted fragment molecules based on the order of the synthons.
        There can be multiple valid connector atom substitution pattern due to ring connectors,
        which can match in two ways.

        Args:
            temp: Temporary result holder for backtracking.
            i: Index of the synthon/fragment being compared.
            known_S_set: Tracking which synthon connector atoms have been "matched".
            known_F_map: Tracking which fragment connector atoms have been "matched", and
                what atoms they should be changed into. One fragment connector may map to
                2 synthon connectors (cross-scoring).
        """

        nonlocal res

        if i == len(smis):
            sub_map = {
                key: list(val) for key, val in known_F_map.items()
            }  # convert mapped symbols to list for compatibility with `special_substitute_atoms` fn.
            substituted_frags = [special_substitute_atoms(frag, sub_map) for frag in temp]
            res.append(substituted_frags)
            return

        # match i-th S with i-th F
        all_synthon_conns = get_conn_symbols(smis[i], synthon_connector_symbols)
        all_frag_conns = get_conn_symbols(oechem.OEMolToSmiles(frag_mols[i]), frag_connector_symbols)

        # Get unseen synthon and fragment connector atom types.
        synthon_conns = [conn for conn in all_synthon_conns if conn not in known_S_set]
        frag_conns = [conn for conn in all_frag_conns if conn not in known_F_map.keys()]

        # Assign the mapped connectors
        frag = oechem.OEMol(frag_mols[i])
        if len(synthon_conns) == len(frag_conns) == 0:  # Last synthon/frag
            backtrack(temp + [frag], i + 1, known_S_set, known_F_map)
        elif len(synthon_conns) == len(frag_conns):  # Compare only unmatched synthon and frag conns.
            connector_map = order_connector_map(frag_conns, synthon_conns)
            for conn_map in connector_map:  # For 2 connectors on ring synthons, try different mappings.
                sub_dict = {key: val for (key, val) in conn_map}
                for key, val in sub_dict.items():
                    known_F_map[key].add(val)
                for s_conn in synthon_conns:
                    known_S_set.add(s_conn)
                backtrack(temp + [frag], i + 1, known_S_set, known_F_map)
                # backtrack
                for key, val in sub_dict.items():
                    known_F_map[key].remove(val)
                    if len(known_F_map[key]) == 0:
                        del known_F_map[key]
                for s_conn in synthon_conns:
                    known_S_set.remove(s_conn)
        elif cross_score:
            if len(synthon_conns) == 1 and len(frag_conns) == 2:
                # 2 frag conns map to the same synthon conn.
                for f_conn in frag_conns:
                    known_F_map[f_conn].add(synthon_conns[0])
                for s_conn in synthon_conns:
                    known_S_set.add(s_conn)
                backtrack(temp + [frag], i + 1, known_S_set, known_F_map)
                for f_conn in frag_conns:
                    known_F_map[f_conn].remove(synthon_conns[0])
                    if len(known_F_map[f_conn]) == 0:
                        del known_F_map[f_conn]
                for s_conn in synthon_conns:
                    known_S_set.remove(s_conn)
            elif len(synthon_conns) == 2 and len(frag_conns) == 1:
                # 1 frag conn maps to two synthon conns.
                for s_conn in synthon_conns:
                    known_F_map[frag_conns[0]].add(s_conn)
                for s_conn in synthon_conns:
                    known_S_set.add(s_conn)
                backtrack(temp + [frag], i + 1, known_S_set, known_F_map)
                # backtrack
                for s_conn in synthon_conns:
                    known_F_map[frag_conns[0]].remove(s_conn)
                    if len(known_F_map[frag_conns[0]]) == 0:
                        del known_F_map[frag_conns[0]]
                for s_conn in synthon_conns:
                    known_S_set.remove(s_conn)

    assert len(smis) == len(frag_mols)

    res = []
    backtrack([], 0, set(), defaultdict(set))
    return res


def label_synthon_ring_connector(
    synthon_smiles: Dict[int, str],
    connector_atoms: List[str],
    smirks: str,
) -> dict:
    """
    Determine whether a synthon connector forms a ring bond, or an acyclic bond.

    Process: Label the atoms connected to the connector atoms (nei atoms) by the connector symbol,
    react by SMIRKS rule, find the nei atoms in the product. If nei atoms IsInRing in the
    product, increment the res[connector_symbol] by 1. If both nei atoms of the same
    connector atom is in ring, the connector atoms are ring connectors.

    Args:
        synthon_smiles: SMILES or SMARTS of synthons in the order of SMIRKS reactants.
        connector_atoms: All possible connector atom symbols on synthons.
        smirks: Reaction SMIRKS to react the synthons.

    Returns:
        A mapping of connector atom symbol: True/False (True = is a ring connector).
    """

    connector_atoms = set(connector_atoms)
    connector_nei_tag = "connected_to"
    ring_connector_dict = defaultdict(int)

    synthon_mols = {}
    for key, smi in synthon_smiles.items():
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smi)
        synthon_mols[key] = mol

    for mol in synthon_mols.values():  # `mol` objects are mutable
        for atom in mol.GetAtoms():
            atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
            if atom_symbol in connector_atoms:
                for _atom in find_connected_atoms(atom):
                    cur_data = []
                    if _atom.HasData(connector_nei_tag):
                        cur_data = _atom.GetData(connector_nei_tag)
                    _atom.SetData(connector_nei_tag, cur_data + [atom_symbol])
                    # Set the data as a list, in case multiple connector atom share the same neighbor atom.

    # React and find out which connector is in ring.
    libgen = oechem.OELibraryGen(smirks)
    for i, rxnt in synthon_mols.items():
        libgen.SetStartingMaterial(rxnt, i)

    for p in libgen.GetProducts():
        prod = oechem.OEMol(p)
        break

    oechem.OEFindRingAtomsAndBonds(prod)
    for atom in prod.GetAtoms():
        if atom.HasData(connector_nei_tag):
            for connector_symbol in atom.GetData(connector_nei_tag):
                ring_connector_dict[connector_symbol] += int(atom.IsInRing())

    res = {key: "True" if val == 2 else "False" for key, val in ring_connector_dict.items()}

    return res


def get_frag_ring_connector_labels(frag_mols: List[oechem.OEMol], connector_tag: str):
    """
    Return a dict of atom_symbol: True/False from a list of fragment molecules. The function
    assumes that the frags are generate with `fragment_molecule` and the ring-connectors already labeled.
    """

    res = {}

    for mol in frag_mols:
        for atom in mol.GetAtoms():
            if atom.HasData(connector_tag):
                atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
                res[atom_symbol] = atom.GetData(connector_tag)

    return res


def label_connected_atoms(
    mol: oechem.OEMol,
    atom_symbol: str,
    isotope: int,
    tag: Any,
    data: Any,
):
    """
    Label the atoms connected to the input atoms.

    Args:
        mol: Molecule to be operated on.
        atom_symbol: Symbol of the input atom used for identification.
        isotope: Isotope number of the input atom used for identification.
        tag: Name of the tag of the data associated with the connected (neighboring) atoms.
        data: Content of the data associated with the connected atoms.
    """

    # TODO: consider also accepting atom pointer for more precise identification?

    for atom in find_atoms(mol, atom_symbol, isotope):
        for nei in find_connected_atoms(atom):
            nei.SetData(tag, data)


def get_shortest_path_atoms(
    mol: oechem.OEMolBase,
    start_atom_symbol: str,
    start_isotope: int,
    end_atom_symbol: str,
    end_isotope: int,
    include_start_end: bool = True,
) -> List[oechem.OEAtomBase]:
    """
    Get the atoms along the shortest path (BFS) between two atoms in a molecule.
    The path includes the start and end atom.

    Args:
        mol: Input molecule.
        start_atom_symbol: Symbol of the atom on one terminus of the path.
        start_isotope: Isotope of the atom on one terminus of the path.
        end_atom_symbol: Symbol of the atom on the other terminus of the path.
        end_isotope: Isotope of the atom on the other terminus of the path.
    """

    s_atom = list(find_atoms(mol, start_atom_symbol, start_isotope))[0]  # assuming only one matching atom.

    visited_atoms = set([s_atom])  # add to set when enqueuing, not when visiting
    q = deque([[s_atom]])  # queue contains a list of paths (List[Atom])
    while q:
        cur_path = q.popleft()
        cur_atom = cur_path[-1]
        for b in cur_atom.GetBonds():
            nei = b.GetNbr(cur_atom)
            if nei not in visited_atoms:
                visited_atoms.add(nei)
                new_path = cur_path + [nei]
                nei_symbol = oechem.OEGetAtomicSymbol(nei.GetAtomicNum())
                nei_isotope = nei.GetIsotope()
                if nei_symbol == end_atom_symbol and nei_isotope == end_isotope:
                    if include_start_end is False:
                        for a in (s_atom, nei):
                            new_path.remove(a)
                    return new_path
                    # ensures that it doesn't find the start_atom as the end,
                    # if happend to be labeled the same atom-type and isotope.
                q.append(new_path)

    logging.error(f"Unable to find a path between specified start {start_isotope} and end {end_isotope} atoms.")


def get_atom_bond_set(mol: oechem.OEMolBase, atoms: List[oechem.OEAtomBase]) -> oechem.OEAtomBondSet:
    """
    Return an oechem.OEAtomBondSet object containing atoms and bonds of a sub-molecule,
    given a list of input atoms. Bonds connecting input atoms are detected and included.

    Args:
        mol: Input molecule.
        atoms: Array of atoms to be included in the atom-bond set.
    """

    ab_set = oechem.OEAtomBondSet()
    ab_set.AddAtoms(atoms)
    for bond in mol.GetBonds():
        if ab_set.HasAtom(bond.GetBgn()) and ab_set.HasAtom(bond.GetEnd()):
            ab_set.AddBond(bond)

    return ab_set


def get_shortest_path_submol(
    mol: oechem.OEMol,
    start_atom_symbol: str,
    start_isotope: int,
    end_atom_symbol: str,
    end_isotope: int,
) -> oechem.OEMol:
    """
    Return a sub-molecule along the shortest path between two atoms.

    Args:
        See docstring of `get_shortest_path_atoms`.
    """

    path_atoms = get_shortest_path_atoms(mol, start_atom_symbol, start_isotope, end_atom_symbol, end_isotope)
    ab_set = get_atom_bond_set(mol, path_atoms)
    out = oechem.OEMol()
    oechem.OESubsetMol(out, mol, ab_set)
    return out


def set_atom_type(atom: oechem.OEAtomBase, atom_symbol: str, isotope: int):
    """
    Set the atomic symbol and isotope of an atom.

    Args:
        atom: Pointer to an oechem.OEAtomBase object (atom).
    """

    atom.SetAtomicNum(oechem.OEGetAtomicNum(atom_symbol))
    atom.SetIsotope(isotope)


def find_atoms(
    mol: oechem.OEMolBase,
    atom_symbol: str,
    isotope: int = 0,
) -> Iterator[oechem.OEAtomBase]:
    """
    Generator that yields atoms in the input molecule that matches the atomic symbol and
    isotope number.
    """

    for a in mol.GetAtoms():
        if a.GetAtomicNum() == oechem.OEGetAtomicNum(atom_symbol) and a.GetIsotope() == isotope:
            yield a


def find_connected_atoms(
    atom: oechem.OEAtomBase,
    return_H: bool = False,
) -> Iterator[oechem.OEAtomBase]:
    """
    Generator that yields atoms connected to the input atom (neighbors).
    """

    for nei in atom.GetAtoms():
        if return_H or nei.GetAtomicNum() != 1:
            yield nei


def extend_dummy_bond(
    mol: oechem.OEMol,
    dummy_atom_symbol: str,
    dummy_isotope: int,
    new_dummy_symbol: str = "Si",
    new_isotope: int = 33,
    distance_factor: int = 1,
) -> oechem.OEMol:
    """
    Place a second dummy atom along the line formed by the original dummy atom and the
    atom connected to that dummy atom. The new dummy bond vector is defined as the line
    formed by the two dummy atoms.

    Args:
        mol: Input molecule.
        dummy_atom_symbol: Symbol of the original dummy atom in `mol`.
        dummy_isotope: Isotope number of the original dummy atom.
        new_dummy_symbol: Symbol of the 2nd dummy atom to be placed.
        new_isotope: Isotope number of the 2nd dummy atom.
        distance_factor: If connected_atom - dummy_atom - new_atom are lined up, this
            arg controls the distance between new_atom-dummy_atom compared to dummy_atom-
            connected_atom. `1` is same distance, >1 is longer, <1 is shorter, 0 means
            new_atom will superimpose on dummy_atom, -1 means new_atom will superimpose
            on connected_atom, <-1 means new_atom will be further "inside" the molecule.
            # 230808: not good to based off dummy bond length, since different dummy bonds
            can have different lengths. Use a constant base, like 1.5 A.
    """

    mol = oechem.OEMol(mol)
    new_atom = mol.NewAtom(oechem.OEGetAtomicNum(new_dummy_symbol))
    new_atom.SetIsotope(new_isotope)
    new_mol = oechem.OEMol(mol)
    new_mol.DeleteConfs()

    for conf in mol.GetConfs():
        dummy_atom = list(find_atoms(conf, dummy_atom_symbol, dummy_isotope))[0]
        connected_atom = list(find_connected_atoms(dummy_atom))[0]
        dummy_coord = conf.GetCoords(dummy_atom)
        connected_coord = conf.GetCoords(connected_atom)
        dummy_bond_length = oechem.OEGetDistance(conf, dummy_atom, connected_atom)
        distance_factor = (
            1.5 * distance_factor / dummy_bond_length
        )  # Scale the factor such the final distance is 1.5 * input distance_factor.
        new_coord = [b + distance_factor * (b - a) for a, b in zip(connected_coord, dummy_coord)]
        conf.SetCoords(new_atom, new_coord)
        new_mol.NewConf(conf)

    return new_mol


def unset_stereo(atom: oechem.OEAtomBase):
    """
    Remove the stereochemical information from an atom. Works only on 2D molecules. On
    3D molecules, the stereochemistry is inferred from the 3D coordinates, and unsetting
    here has no effect on the actualy stereochemistry of the atom.
    """

    neis = []
    for bond in atom.GetBonds():
        neis.append(bond.GetNbr(atom))

    ret_code = atom.SetStereo(neis, oechem.OEAtomStereo_Tetra, oechem.OEAtomStereo_Undefined)
    if ret_code is False:
        logging.warning("Unsetting stereocenter failed.")


def delete_atom(mol: oechem.OEMol, atom_symbol: str, isotope: int) -> oechem.OEMol:
    """
    Delete all atoms in a molecule that match the specified atom symbol and isotope.
    """

    mol = oechem.OEMol(mol)  # make a copy
    for a in find_atoms(mol, atom_symbol, isotope):
        oechem.OESuppressHydrogens(a)  # Make H implicit.
        mol.DeleteAtom(a)  # This will also remove the connected bonds.

    return mol


def set_conf_centroid_coord(
    conf: oechem.OEConfBase,
    connectors: Iterable[Tuple[str, int]],
    atom_to_set: oechem.OEAtomBase,
) -> oechem.OEMol:
    """
    Set the centroid atom to be at the geometric center of connector atoms.

    Args:
        conf: A conformer of a OEMol object.
        connectors: Connector atoms expressed as (<atom symbol>, <isotope>).
        atom_to_set: Reference to the centroid atom for which to set the coordinate.
    """

    conf = oechem.OEMol(conf)  # Need to make a copy, but cannot instantiate OEConfBase directly.
    connectors = set(connectors)

    # Take average of the dummy atoms coordinates.
    x = y = z = 0
    n = 0
    for symbol, isotope in connectors:
        atoms = list(find_atoms(conf, symbol, isotope))
        if len(atoms) > 0:
            for atom in atoms:
                _x, _y, _z = conf.GetCoords(atom)
                x += _x
                y += _y
                z += _z
                n += 1
    new_coord = [coord / n for coord in (x, y, z)]
    conf.SetCoords(atom_to_set, new_coord)
    return conf


def replace_dummies_with_centroid(
    mol: oechem.OEMolBase,
    dummies: Iterable[Tuple[str, int]],
    centroid_symbol: str,
    centroid_isotope: int = 0,
    delete_original_dummies: bool = True,
) -> oechem.OEMol:
    """
    For multi-conf mol; modifies the coordinates of each conformer.
    This is compatible with arbitrary number of connector atoms. Although synthons have
    at most 2 connector atoms. But it's useful to accept arbitrary # of atoms, such that
    when calling this function, input all possible connector atom labels.
    """

    # Use an unlikely atom type as the temp centroid, in case the desired centroid type
    # is among the `dummies`.
    _symbol, _isotope = "Xe", 777
    assert (
        _symbol,
        _isotope,
    ) not in dummies, f"{_symbol}_{_isotope} is of the same type as one of the existing connector atoms."

    mol = oechem.OEMol(mol)
    new_atom = mol.NewAtom(oechem.OEGetAtomicNum(_symbol))
    new_atom.SetIsotope(_isotope)
    new_mol = oechem.OEMol(mol)
    new_mol.DeleteConfs()

    for conf in mol.GetConfs():
        new_conf = set_conf_centroid_coord(conf, dummies, new_atom)
        new_mol.NewConf(new_conf)

    if delete_original_dummies:
        for symbol, isotope in dummies:
            new_mol = delete_atom(new_mol, symbol, isotope)

    substitute_atoms(new_mol, sub_dict={(_symbol, _isotope): (centroid_symbol, centroid_isotope)})

    return new_mol


def get_singleton_product(
    reactants: List[Tuple[int, str, int]],
    smirks: str,
    sort_title: bool,
    title_separator: str,
) -> oechem.OEMol:
    """
    Generate a single product from one set of reactants.

    Args:
        reactants: List of reactants containing the id, SMILES, and reaction position index.
        smirks: SMIRKS string of the reaction.
        check_reactant_order: Whether to try out different ordering of the reactants
            such that all reactants can be set to the correct position of `libGen`.
        sort_title: Whether to name the product based on sorted ids of the reactants.
        title_separator: String separator for joining the reactant ids to form product ids.
    """

    libGen = oechem.OELibraryGen(smirks)
    libGen.SetExplicitHydrogens(False)
    libGen.SetClearCoordinates(True)
    # Clear StartingMaterial cooridnates to prevent assignment of unspecified stereocenters.
    libGen.SetTitleSeparator(title_separator)

    for reactant in reactants:
        sid, smi, sorder = reactant
        mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(mol, smi)
        mol.SetTitle(str(sid))
        ret_code = libGen.SetStartingMaterial(mol, sorder)
        assert ret_code == 1, "Reactants are not in correct order!"

    for prod in libGen.GetProducts():
        prod_mol = oechem.OEMol(prod)
        break
    prod_mol = cleanup_mol(prod)  # output is an OEMolBase object.
    prod_mol = oechem.OEGraphMol(prod_mol)  # Convert to OEGraphMol.

    if sort_title:
        sids = prod_mol.GetTitle().split(title_separator)
        if len(sids) > 1:
            title = title_separator.join(sorted(sids))
            prod_mol.SetTitle(title)
    return prod_mol


def generate_library_products(
    reactants: Dict[int, List[Tuple[int, str]]],
    smirks: str,
    sort_title: bool,
    title_separator: str,
) -> Iterator[oechem.OEMol]:
    """
    A generator that yields the pairwise product between two lists of synthons.

    Args:
        reactants: Dict of reactants, grouped by the reaction position index. e.g.
            {0: [(rid1, smi1), (rid2, smi2), ...], 1: [(rid10, smi10), (rid11, smi11), ...]}
        smirks: SMIRKS string of a reaction.
        sort_title: Whether to rename products by sorting the individual synthon ids (by
            default, a new product is named <reac1_id>_<reac2_id>).
        title_separator: String separator for joining the reactant ids to form product ids.
    """

    libGen = oechem.OELibraryGen(smirks)
    libGen.SetExplicitHydrogens(False)  # Suppress adding H to [13C] dummy atom. May not be needed.
    libGen.SetClearCoordinates(True)
    # Clears StartingMaterial cooridnates to prevent assignment of unspecified
    # stereochemistry based on 3D information.
    libGen.SetTitleSeparator(title_separator)

    for s_idx, data in reactants.items():
        for d in data:
            if isinstance(d, oechem.OEMolBase):
                mol = oechem.OEMol(d)
            else:
                sid, smi = d
                mol = oechem.OEMol()
                oechem.OESmilesToMol(mol, smi)
                mol.SetTitle(str(sid))
            libGen.AddStartingMaterial(mol, s_idx)

    for prod in libGen.GetProducts():
        prod_mol = cleanup_mol(prod)
        prod_mol = oechem.OEGraphMol(prod)  # Set to OEGraphMol to not having to deal with conformer titles.

        if sort_title:
            sids = prod_mol.GetTitle().split(title_separator)
            if len(sids) > 1:
                title = title_separator.join(sorted(sids))
                prod_mol.SetTitle(title)
        yield prod_mol


def generate_stereoisomers(
    mols: Union[Iterable[oechem.OEMolBase], oechem.oemolistream],
    flipper_options: oeomega.OEFlipperOptions = None,
) -> Iterator[oechem.OEMolBase]:
    """
    A generator that yields stereoisomers from input molecules. Each stereoisomer
    of an input molecule is written to file as a separate molecule.

    Args:
        mols: An array of oechem molecules, or an oechem file stream.
        flipper_options: Options for the Flipper module.
    """

    if isinstance(mols, oechem.oemolistream):
        mols = mols.GetOEMols()

    for mol in mols:
        for isomer in _generate_stereoisomers(mol, flipper_options):
            yield isomer


def _generate_stereoisomers(
    mol: oechem.OEMolBase,
    flipper_options: oeomega.OEFlipperOptions,
) -> Iterator[oechem.OEMolBase]:

    """
    A generator that yields stereoisomers from the input molecule. Each stereoisomer
    is written to file as a separate molecule.

    Args:
        mol: Input molecule.
        flipper_options: Options for the Flipper module. If set to `None`, uses
            the default options. Some options such as `SetWarts` (OEConfGen::OEFlipperOptions::SetWarts)
            are important.
    """

    # Load default Flipper options if not provided.
    if flipper_options is None:
        # logging.warning('Flipper option should be set explicitly by the user!')
        flipper_options = oeomega.OEFlipperOptions()

        # Copy/pasted from https://docs.eyesopen.com/toolkits/python/omegatk/omegaexamples.html#generating-stereoisomers
        opts = oechem.OESimpleAppOptions(
            flipper_options, "stereo_and_torsion", oechem.OEFileStringType_Mol, oechem.OEFileStringType_Mol
        )
        flipper_options.UpdateValues(opts)

    for isomer in oeomega.OEFlipper(mol, flipper_options):
        fmol = oechem.OEMol(isomer)  # Create a new mol, safe from mutating pointers.
        yield fmol


def generate_conformers(
    mols: Union[Iterable[oechem.OEMolBase], oechem.oemolistream],
    omega: oeomega.OEOmega,
) -> Iterator[oechem.OEMCMolBase]:
    """
    A generator that yields multi-conformer molecules.

    Args:
        mols: An array of oechem molecules, or an oechem file stream.
        omega: OMEGA instances to be used for generating conformers.
    """

    if isinstance(mols, oechem.oemolistream):
        mols = mols.GetOEMols()
    for mol in mols:
        yield _get_conformer(mol, omega)


def _get_conformer(
    mol: oechem.OEMol,
    omega: oeomega.OEOmega,
) -> oechem.OEMCMolBase:
    """ """

    ret_code = omega.Build(mol)
    if ret_code == oeomega.OEOmegaReturnCode_Success:  # i.e. 0
        return mol
    else:
        logging.error(
            (
                f"Conformer generation failed for {mol.GetTitle()}, "
                f"{oechem.OEMolToSmiles(mol)}, "
                f"error code: {ret_code}, "
                f"message: {oeomega.OEGetOmegaError(ret_code)}"
            )
        )


def build_custom_FF(
    cff: oeshape.OEColorForceField,
    smarts_pattern: str,
    smarts_pattern2: str,
    weight: int,
    radius: int,
    interaction: str,
):
    """
    Create a custom color force field, for ROCS scoring.

    Args:
        cff: An OEColorForceField object. If `None`, starts from default.
        smarts_pattern: SMARTS pattern of the substructure to be assigned a
            particular "color". Be sure to enclose patterns in [].
        weight: Weight of the color in scoring. See:
            https://docs.eyesopen.com/toolkits/python/shapetk/shape_theory.html#color-features
        radius: Radius of the color interaction.
        interaction: Type of color feature interaction, i.e. "gaussian" or "discrete".
    """
    if cff is None:
        cff = oeshape.OEColorForceField()
        cff.Init(oeshape.OEColorFFType_ImplicitMillsDean)

    new_type = f"new_type_{smarts_pattern}"
    cff.AddType(new_type)
    int_new_type = cff.GetType(new_type)
    cff.AddColorer(int_new_type, smarts_pattern)

    if smarts_pattern2 is not None:
        new_type2 = f"new_type_{smarts_pattern2}"
        cff.AddType(new_type2)
        int_new_type2 = cff.GetType(new_type2)
        cff.AddColorer(int_new_type2, smarts_pattern2)

    if smarts_pattern2 is None:
        cff.AddInteraction(int_new_type, int_new_type, interaction, weight, radius)
    else:
        cff.AddInteraction(int_new_type, int_new_type2, interaction, weight, radius)

    return cff


def add_multi_custom_FF(
    color_ff: oeshape.OEColorForceField,
    patterns: Iterable[str],
    weight: int,
    radius: int,
    interaction: str,
) -> oeshape.OEColorForceField:
    """
    Add self-interaction for patterns to the custom color FF.

    Args:
        color_ff: Input color FF to be modified.
        patterns: An iterable of SMARTS patterns.
        weight: Weight of the color FF for a pattern.
        radius: Radius of the color FF for a pattern.
    """

    for pattern in patterns:
        color_ff = build_custom_FF(
            color_ff,
            smarts_pattern=pattern,
            smarts_pattern2=None,
            weight=weight,
            radius=radius,
            interaction=interaction,
        )

    return color_ff
