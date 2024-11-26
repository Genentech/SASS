"""Util functions for generating conformers on whole molecules and synthons."""

import logging
import time
from collections import deque
from pathlib import Path

from openeye import oechem, oeomega

from sass.utils.utils_libgen import flipper_opt_builder, generate_library_products, generate_stereoisomers
from sass.utils.utils_mol import (
    cleanup_mol,
    find_connected_atoms,
    generate_components,
    get_conn_symbols_freq,
    load_first_mol,
    smiles_to_mol,
)


# Vanilla OMEGA
def build_conformer(
    mol: oechem.OEMol,
    omega: oeomega.OEOmega,
) -> oechem.OEMCMolBase | None:
    """Build conformers of the input molecule."""
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
        return None


def write_conformers(
    in_file: Path | None,  # isomer file
    out_file: Path,  # conformer file
    omega_max_conf: int,
    omega_max_time: int,
    omega_e_window: int,
    input_mols: list[oechem.OEMolBase] | None = None,
) -> None:
    omega_opt = omega_opt_builder(
        max_conf=omega_max_conf,
        max_search_time=omega_max_time,
        energy_window=omega_e_window,
    )
    omega = oeomega.OEOmega(omega_opt)
    if not input_mols:
        istream = oechem.oemolistream(str(in_file))
        input_mols = istream.GetOEMols()
    ostream = oechem.oemolostream(str(out_file))
    n = 0
    start = time.time()
    for mol in input_mols:
        omega.Build(mol)
        oechem.OEWriteMolecule(ostream, mol)
        n += 1
    if not input_mols:
        istream.close()
    ostream.close()

    finish = time.time()
    logging.info(f"Generated conformers on {n} isomers. Time elapsed: {finish - start:.1f} s.")


# Synthon conformer generation that involves forming minimum products.
def sample_general_mixed_synthon_conf(
    smirks: str,
    synthon_ring_connector_dict: dict[str, str],
    current_synthons: list[tuple[int, str]],
    current_synthon_idx: int,
    other_synthons: dict[int, str],
    output_file: Path,
    omega_max_conf: int,
    omega_max_time: int,
    omega_e_window: int,
):
    """
    Sample conformers of synthons that contains ring and linear connectors.

    Direct conformer sampling could lead to unrealistic conformations due to unconstrained
    geometry of the partial ring atom/bonds. Instead, this workflow generates minimum
    products from the synthon by completing the ring-portion of ring-forming synthons
    or adding one extra atom to non-ring-forming synthons, does conformer sampling, and
    then deletes the added atom/bonds.

    Another benefit of using minimum products is avoiding having to deal with special
    connector atoms like U, Np, etc. that are not parametrized in OMEGA.

    Parameters
    ----------
    smirks
        SMIRKS to react the synthons.
    synthon_ring_connector_dict
        Dict of {atom: str(bool)} indicating whether a connector atom is a ring-atom or not.
    current_synthons
        The synthon set to generate conformers on.
    current_synthon_idx
        The position index of the synthon (for product generation).
    other_synthons
        Single examples of other synthons needed to complete a product.
    output_file
        Path to the output file.
    omega_max_conf
        Maximum number of conformers to generate.
    omega_max_time
        Maximum time (in s) to spend on one conformer generation.
    omega_e_window
        Energy above the minimum conformer energy to reject conformers.
    """
    reactants = {}

    label = "self_label"  # Label tag for atoms indicating whether "self" or "other" synthons.
    nei_label_self = "self_connector_nei"  # Label tag for atoms adjacent to ring connectors on self synthons.
    nei_label_other = "other_connector_nei"  # Label tag for atoms adjacent to ring connectors on other synthons.

    flat_mols = []
    # Label "self" synthons as "self". And store connector data on connector neis.
    for sid, smi in current_synthons:
        mol = smiles_to_mol(smi, str(sid))
        for atom in mol.GetAtoms():
            atom.SetData(label, "self")
            atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
            if atom_symbol in synthon_ring_connector_dict:
                # Label the adjacent atom.
                nei = next(find_connected_atoms(atom))  # connector atom always attached to only one atom.
                if nei.HasData(nei_label_self):
                    data = nei.GetData(nei_label_self) + f"_{atom_symbol}"  # use str to be read/write-safe.
                else:  # first time tagging, don't start with "_".
                    data = atom_symbol
                nei.SetData(nei_label_self, data)
        flat_mols.append(mol)
    reactants[current_synthon_idx] = flat_mols

    # For cases where two ring-synthon connectors are connected to the same atom (i.e.
    # atom.GetData(nei_label_self) gives more than one symbols), label the ring_connector_neis
    # on the complementary synthon to keep the information on the relative position of
    # those two synthons. (Previously if just use the info stored on self `atom`, won't
    # be able to tell where each connector should go.)
    connectors_sharing_neis = set()  # connector symbols that share a common nei atom.
    for atom in flat_mols[0].GetAtoms():
        if atom.HasData(nei_label_self):
            nei_symbols = atom.GetData(nei_label_self).split("_")
            if len(nei_symbols) > 1:
                for nei_symbol in nei_symbols:
                    connectors_sharing_neis.add(nei_symbol)

    # Label other synthons as "other". Store connector data on applicable connector neis.
    for s_idx, smi in other_synthons.items():
        mol = smiles_to_mol(smi)
        for atom in mol.GetAtoms():
            atom.SetData(label, "other")
            # Label atoms that connect to those `connectors_sharing_neis`.
            atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
            if atom_symbol in connectors_sharing_neis:
                nei = next(find_connected_atoms(atom))
                if nei.HasData(nei_label_other):
                    msg = (
                        "This atom should not have been labeled. This suggests that there are multiple "
                        "conn atoms connecting to this atom, which should not be the case."
                    )
                    raise ValueError(msg)
                nei.SetData(nei_label_other, atom_symbol)
        reactants[s_idx] = [mol]

    # React to form products.
    title_sep = "&"
    complete_mols = list(
        generate_library_products(
            reactants=reactants,
            smirks=smirks,
            sort_title=False,
            title_separator=title_sep,
        )
    )

    # Cleave off irrelevant parts (exocyclic portions of the new ring formed (except double bonds),
    # or anything formed from linear bonds).
    part_mols = []
    for mol in complete_mols:
        oechem.OEFindRingAtomsAndBonds(mol)  # why?
        mol.SetTitle(mol.GetTitle().strip(title_sep))  # Since other synthon titles are "", product title
        # may look like "&abcdc&", or "&&abcded".
        seen_connector_atoms = set()  # Avoid double processing of pairs of ring connectors.

        for atom in mol.GetAtoms():
            if atom.HasData(nei_label_self):  # Only `self` mol has the `nei_label_self`.
                connector_symbols = atom.GetData(nei_label_self).split("_")
                for nei in list(
                    find_connected_atoms(atom)
                ):  # Eagerly evaluates to avoid infinite loop with changing neis.
                    if nei.GetData(label) == "other":  # i.e. from other synthon, to delete
                        if len(connector_symbols) == 1:
                            # One conn per nei, use the data stored on self_conn_nei.
                            stored_conn_symbol = atom.GetData(nei_label_self)
                        else:
                            # If `self_conn_nei` has > 1 conn, use data on `other_conn_nei`,
                            # which is guaranteed to have data for 1 conn.
                            stored_conn_symbol = nei.GetData(nei_label_other)

                        if stored_conn_symbol not in seen_connector_atoms:
                            seen_connector_atoms.add(stored_conn_symbol)

                            # Check whether connector is ring or non-ring
                            if synthon_ring_connector_dict[stored_conn_symbol] == str(False):
                                # 240116 new: treat lin connector just like ring connector.
                                # 240205 new: keep all substituent if any double/triple bond.
                                # TODO: abstract this cleave nei-bonds to a function.
                                has_double_bond = False
                                for _nei in find_connected_atoms(nei):
                                    bond = nei.GetBond(_nei)
                                    if _nei.GetData(label) == "other" and bond.GetOrder() > 1:
                                        has_double_bond = True
                                        break

                                for _nei in find_connected_atoms(nei):
                                    bond = nei.GetBond(_nei)
                                    if _nei.GetData(label) == "other":
                                        if has_double_bond is False:
                                            mol.DeleteBond(bond)
                                        else:
                                            for b in _nei.GetBonds():
                                                if b != bond:
                                                    mol.DeleteBond(b)
                            else:
                                # Cleave off ring substituents.
                                # Find the new ring formed in the reaction. Not just the smallest ring.
                                # Returned atoms must contain both 'self' and 'other' labels.
                                new_ring_atoms = get_reaction_ring_atoms(
                                    mol,
                                    label_tag=label,
                                    other_label="other",
                                    start_atom=atom,
                                )  # make sure to return the actual atom obj or pointers, not copies.
                                for _atom in new_ring_atoms:
                                    # Add both ring connector symbols to seen_set.
                                    for atom_tag in (nei_label_self, nei_label_other):
                                        if _atom.HasData(atom_tag):
                                            seen_connector_atoms.add(_atom.GetData(atom_tag))

                                    if _atom.HasData(label) and _atom.GetData(label) == "other":
                                        # Cleave off substituents on ring atoms from `other` synthons.

                                        # 2024-2-5 re-write. If any bond to nei is double bond, keep all `other`
                                        # nei atom
                                        has_double_bond = False
                                        for _nei in find_connected_atoms(_atom):
                                            bond = _atom.GetBond(_nei)
                                            if (
                                                _nei not in new_ring_atoms
                                                and _nei.GetData(label) == "other"
                                                and bond.GetOrder() > 1
                                            ):
                                                has_double_bond = True
                                                break

                                        for _nei in find_connected_atoms(_atom):
                                            bond = _atom.GetBond(_nei)
                                            if (
                                                _nei not in new_ring_atoms and _nei.GetData(label) == "other"
                                            ):  # checking for `_nei.GetData(label) == "other"` unnecessary.
                                                if has_double_bond is False:  # single bonds don't indicate aromaticity
                                                    mol.DeleteBond(bond)
                                                else:
                                                    # Contains at least one double bond nei, keep all.
                                                    for b in _nei.GetBonds():
                                                        # delete all except the incident bond.
                                                        if b != bond:
                                                            mol.DeleteBond(b)

        # Split the disjoint molecule, and get the "self" fragment (contains both "self" and "other" atoms).
        for frag in generate_components(mol):
            if any(a.GetData(label) == "self" for a in frag.GetAtoms() if a.HasData(label)):
                _mol = cleanup_mol(frag)  # cleanup the cut bonds, fill with H.
                part_mols.append(oechem.OEMol(_mol))
                break

    # Generate stereoisomer.
    flipper_opt = flipper_opt_builder()
    stereo_mols = []
    for mol in part_mols:
        stereo_mols.extend(list(generate_stereoisomers(mol, flipper_opt)))

    # Generate conformers.
    synthon_omega_opt = omega_opt_builder(
        max_conf=omega_max_conf,
        max_search_time=omega_max_time,
        energy_window=omega_e_window,
    )
    synthon_omega = oeomega.OEOmega(synthon_omega_opt)
    for mol in stereo_mols:
        build_conformer(mol, synthon_omega)

    # Remove added ring atoms/bonds.
    ostream = oechem.oemolostream(str(output_file))
    for mol in stereo_mols:
        out_mol = cleave_off_parts(
            mol,
            atom_label=label,
            nei_label_self=nei_label_self,
            nei_label_other=nei_label_other,
            other_label_val="other",
            restore_connector=True,
        )
        oechem.OEWriteMolecule(ostream, out_mol)
    ostream.close()

    # Check whether the connector atoms are properly restored. Similar to in `test_utils_confgen.py`.
    input_eg_mol = flat_mols[0]
    output_eg_mol = load_first_mol(output_file, False, False)
    input_conn_freq = get_conn_symbols_freq(input_eg_mol, list(synthon_ring_connector_dict.keys()))
    output_conn_freq = get_conn_symbols_freq(output_eg_mol, list(synthon_ring_connector_dict.keys()))
    if input_conn_freq != output_conn_freq:
        raise ValueError("Connector atoms are not properly restored.")


def get_reaction_ring_atoms(
    mol: oechem.OEMol,
    start_atom: oechem.OEAtomBase,
    label_tag: str,
    other_label: str,
) -> list[oechem.OEAtomBase]:
    """
    Find atoms in the ring formed during the reaction.

    Defined as the smallest ring formed that contains the `start_atom` (with `self_label`)
    and atoms that are labeled as `other_label`.

    This is not strictly correct for all graphs, e.g. if a reaction forms multiple rings,
    and one of the smaller paths containing `other_label` is not the ring formed by this
    `start_atom`, but by some other start_atoms.
    However, for sensible molecules and reactions, this should be sufficient.

    Parameters
    ----------
    mol : oechem.OEMol
        The molecule to search for the ring.
    start_atom : oechem.OEAtomBase
        The connector neighbor atom on the self reactant to start the search.
    label_tag : str
        The data tag name used for labeling atoms as self/other.
    other_label : str
        The label used for identifying atoms from other reactants.

    Returns
    -------
    list[oechem.OEAtomBase]
        A list of atoms in the ring formed during the reaction.

    """
    oechem.OEFindRingAtomsAndBonds(mol)
    n_bonds = mol.NumBonds()
    paths: list[list] = []

    parent_atom = None
    q = deque([(start_atom, parent_atom, {start_atom})])
    while q:
        cur_atom, prev_atom, cur_path = q.popleft()
        if len(cur_path) <= n_bonds:  # upper bound of the traversal.
            # check for returning to `start_atom` when enqueuing child.
            for nei in find_connected_atoms(cur_atom):
                if nei.IsInRing() and nei != prev_atom:
                    if nei == start_atom:
                        # Check if any of the atoms in the set/list contains the `other_label`,
                        # in which case it's a ring formed by the reaction (contains both `self` and `other`).
                        # By default, the `start_atom` is from self, so no need to check for `self_label` here.
                        if any(a.GetData(label_tag) == other_label for a in cur_path):
                            return cur_path
                        paths.append(list(cur_path))  # need to make a copy
                    elif nei not in cur_path:  # not returning to another atom in path (sub-ring)
                        next_path = list(cur_path)
                        # Still need to copy the path each time.. no better than using a list..
                        next_path.append(nei)
                        q.append((nei, cur_atom, next_path))
    return []


def cleave_off_parts(
    mol: oechem.OEMol,
    atom_label: str,
    nei_label_self: str,
    nei_label_other: str,
    other_label_val: str,
    restore_connector: bool,
) -> oechem.OEMolBase:
    """
    Cleave off parts that do not belong to the original synthon, and restore the original connectors.

    Parameters
    ----------
    mol
        Input molecule after conformer generation.
    atom_label : str
        Name of atom data tag that indicates which atoms to keep/remove.
    other_label_val : str
        Value of the atom data tag for removing an atom (i.e. not self synthon).
    nei_label_self : str
        Name of the atom data tag that contains information on the
        type of connector atoms that were connected to this atom on "self" synthon.
    nei_label_other : str
        Same as `nei_label_self`, but for atoms on "other" synthon.
    restore_connector : bool
        Whether to restore the original connector atom types. The atom type
        information is stored in the neighbor atoms.
    """
    out_mol = oechem.OEMol()  # Instantiate an empty mol, add Conf to it.
    out_mol.DeleteConfs()
    for conf in mol.GetConfs():
        temp_mol = oechem.OEMol(conf)  # make a copy to prevent mutating the conf/mol
        if restore_connector:
            for atom in temp_mol.GetAtoms():
                if atom.HasData(nei_label_self):
                    connector_symbols = atom.GetData(nei_label_self).split("_")
                    for nei in list(find_connected_atoms(atom)):
                        bond = atom.GetBond(nei)
                        if nei.GetData(atom_label) == other_label_val:  # i.e. from other synthons, to be deleted.
                            # restore the original connector. Connector symbol data are stored either on
                            # `self_conn_nei` or `other_conn_nei`.
                            new_atom = temp_mol.NewAtom(nei)
                            if len(connector_symbols) == 1:
                                # If `self_conn_nei` has 1 connector, use the data on `self_conn_nei`.
                                stored_conn_symbol = atom.GetData(nei_label_self)
                            else:
                                # If `self_conn_nei` has > 1 conn, use data on `other_conn_nei`,
                                # which is guaranteed to have data for 1 conn.
                                stored_conn_symbol = nei.GetData(nei_label_other)
                            new_atom.SetAtomicNum(oechem.OEGetAtomicNum(stored_conn_symbol))
                            new_atom.SetData(atom_label, "self")  # set to anything other than `other_label_val`
                            temp_mol.NewBond(atom, new_atom, bond.GetOrder())  # set to same bond order.
                            # conf.DeleteBond(bond)  # Optional... the bond will be deleted when nei_atom is deleted.

        if out_mol.NumConfs() == 0:
            # For the first conformer, copy the `conf` to also set the correct number/type of atoms.
            out_mol = oechem.OEMol(temp_mol)
        else:
            out_mol.NewConf(temp_mol)

    for atom in out_mol.GetAtoms():
        if atom.HasData(atom_label) and atom.GetData(atom_label) == other_label_val:
            oechem.OESuppressHydrogens(atom)
            out_mol.DeleteAtom(atom)

    return out_mol


def omega_opt_builder(
    max_conf: int,
    max_search_time: int,
    energy_window: int,
) -> oeomega.OEOmegaOptions:
    """
    Build OMEGA options.

    Parameters
    ----------
    max_conf
        Maximum number of conformers to generate.
    max_search_time
        Maximum time (in s) to spend on one conformer generation.
    energy_window
        Energy above the minimum conformer energy to reject conformers.
    """
    omega_options = oeomega.OEOmegaOptions()
    mol_builder_options = oeomega.OEMolBuilderOptions()
    mol_builder_options.SetEnumNitrogen(oeomega.OENitrogenEnumeration_Unspecified)
    omega_options.SetMolBuilderOptions(mol_builder_options)
    omega_options.SetMaxConfs(max_conf)
    omega_options.GetTorDriveOptions().SetMaxSearchTime(max_search_time)
    omega_options.SetEnergyWindow(energy_window)
    # Copied from https://docs.eyesopen.com/toolkits/python/omegatk/omegaexamples.html#generating-conformers
    opts = oechem.OESimpleAppOptions(
        omega_options,
        "Omega",
        oechem.OEFileStringType_Mol,
        oechem.OEFileStringType_Mol3D,
    )
    omega_options.UpdateValues(opts)

    return omega_options
