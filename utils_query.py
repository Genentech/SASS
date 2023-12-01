"""
Workflow utils for `query_dev.py`. Contains multiprocessing functions that use global
variables from `mp_global_param`.
"""

# Standard Library
from collections import deque
from itertools import repeat
import logging
import math
import multiprocessing
import os
from pathlib import Path
import pickle
from statistics import mean
import subprocess
import time
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union
from uuid import uuid1

# Third Party Library
from openeye import oechem, oeomega, oeshape
from ray.util.multiprocessing import Pool as Ray_Pool

# Genentech Library
from constants import SCRIPT_DIR
import mp_global_param
from utils_data import Synthon_handler
from utils_general import extract_base_id, wait
from utils_mol import (
    _generate_stereoisomers,
    _get_conformer,
    cleanup_mol,
    enumerate_fragment_mol_orders,
    enumerate_syn_frag_orders,
    find_atoms,
    find_connected_atoms,
    generate_components,
    generate_heavy_isotope_atoms,
    generate_library_products,
    generate_stereoisomers,
    get_conn_symbols,
    get_frag_ring_connector_labels,
    get_shortest_path_atoms,
    get_singleton_product,
    label_synthon_ring_connector,
    split_n_chunks,
    substitute_atoms,
)


def get_num_atoms_between_dummies(mol: oechem.OEMol) -> int:
    """
    Current not in use. For limiting ROCS to only synthons and query fragments with
    the same number of atoms between dummy atoms (partial-ring-size matching).
    """

    # Find the two heavy isotope atoms.
    dummies = []
    for heavy_atom in generate_heavy_isotope_atoms(mol):
        symbol = oechem.OEGetAtomicSymbol(heavy_atom.GetAtomicNum())
        isotope = heavy_atom.GetIsotope()
        dummies.append((symbol, isotope))

    return len(get_shortest_path_atoms(mol, *dummies[0], *dummies[1]))  # Should only have two heavy isotopes.


def merge_score_files(
    files,
    top_n,
    warts_separator,
    is_binary: bool = True,
    score_size_limit: int = 5e6,
) -> list:
    """
    Merge chunks of ROCS scores.

    Args:
        files: Score files to be merged.
        top_n: Desired output size of the score file.
        id_parts: Number of id parts (separated by '_') to use for deduplication.
        is_binary: Whether the files are binary or not.
        score_size_limit: Limit of the size of the combined scores, beyond which the
            list will deduplicate and trim to `top_n`. This is to prevent memory overflow
            and avoid sorting a huge list in the end.
    """

    scores = []
    open_flag = "rb" if is_binary else "r"
    for file in files:
        with open(file, open_flag) as f:
            scores.extend(pickle.load(f))
        if len(scores) > score_size_limit:
            scores = deduplicate_scores(
                scores,
                sort_scores=True,
                limit=top_n,
                warts_separator=warts_separator,
            )

    scores = deduplicate_scores(
        scores,
        sort_scores=True,
        limit=top_n,
        warts_separator=warts_separator,
    )

    return scores


def deduplicate_scores(
    scores: List[Tuple[float, str, Any]],
    sort_scores: bool = False,
    limit: int = None,
    **kwargs,
) -> List[Tuple[float, str]]:
    """
    Deduplicate score list by mol id.

    Args:
        scores: Array of score data, e.g. [(score1, id1, ...), (score2, id2, ...), ...].
        sort_scores: Whether to sort the array by the scores. Most times the
            input array is already sorted by descending scores. However, use this
            option with caution, especially when score information is not
            present (i.e. scores are all None or ''), in which case sorting will
            cause incorrect behavior (sorting by id).
        limit: Number of top scores to output.
        kwargs: Keyword args for `extract_base_id` function. i.e. `warts_separator`.

    Returns:
        Array of [(score, id), ...] where all ids are unique.
    """

    dedup_res = []
    seen = set()
    if sort_scores:
        scores.sort(reverse=True)

    n = 0
    for data in scores:
        score, sid = data[:2]
        if limit is not None and n >= limit:
            break
        sid = extract_base_id(mol_id=sid, **kwargs)
        if sid not in seen:
            seen.add(sid)
            new_entry = [score, sid] + list(data[2:])  # Retain additional information.
            if isinstance(data, tuple):
                new_entry = tuple(new_entry)
            dedup_res.append(new_entry)
            n += 1

    return dedup_res


def deduplicate_by_title(
    mols: Union[List, oechem.oemolistream],
) -> Iterator[oechem.OEMol]:
    """
    Yield unique molecules based on the titles.
    """

    id_set = set()
    if isinstance(mols, oechem.oemolistream):
        mols = mols.GetOEMols()

    for mol in mols:
        m_title = mol.GetTitle()
        if m_title not in id_set:
            id_set.add(m_title)
            yield mol


def _top_synthon_idx(array, top_s_frac: float):
    return max(1000, int(len(array) * top_s_frac))
    # return int(len(array) * top_s_frac)


def mp_frag_select_synthon(
    score_file: Union[str, Path],
    rxn_id: str,
    top_m: int,
    top_s_frac: float,
    ncpu: int,
    warts_separator: str,
    sass_separator: str,
    ha_map: dict,
) -> list:
    """
    Synthon selection parallelized at the fragment sets level.
    i.e. for scores dict with structure of {rxn_id: {frag_set_x: {f_order_y: {s1: [...]}}}},
    each process handles one `frag_set_x`.

    Args:
        score_file: Path to the score file.
        rxn_id: ID of the reaction.
        top_m: Top m scores to return from the `score_file`.
        top_s_frac: Top s fraction to consider from each score list.
        ncpu: Number of cpu to parallelize on.
        warts_separator: Isomer/conformer title separator.
        sass_separator: Title separator when joining synthon ids.
        ha_map: Dict of number of heavy atoms for each synthon.

    Returns:
        A list of `top_m` synthon-combinations with score, id, and reaction information.
    """

    # Load score dict
    with open(score_file, "rb") as f:
        scores = pickle.load(f)

    logging.info("Start mp_frag synthon selection.")

    a_fset_dict = scores[list(scores.keys())[0]]
    a_forder_dict = a_fset_dict[list(a_fset_dict.keys())[0]]
    n_component = len(a_forder_dict.keys())

    products = []
    if ncpu > 1:
        with multiprocessing.Pool(ncpu) as pool:
            for prod in pool.imap(
                _sp_frag_select_synthon,
                zip(
                    scores.items(),
                    repeat(top_m),
                    repeat(top_s_frac),
                    repeat(rxn_id),
                    repeat(n_component),
                    repeat(warts_separator),
                    repeat(sass_separator),
                    repeat(ha_map),
                ),
            ):
                products.extend(prod)
    else:
        for _score in scores.items():
            prod = _sp_frag_select_synthon(
                (_score, top_m, top_s_frac, rxn_id, n_component, warts_separator, sass_separator)
            )
            products.extend(prod)

    products.sort(reverse=True)
    products = deduplicate_scores(products, limit=top_m, warts_separator=warts_separator)

    # pool.terminate()

    return products


def _sp_frag_select_synthon(data) -> list:
    """
    Operate on a dict of {f_order1: {s1: [(score, id)], s2: []}, f_order2:...} to generate
    all combinations of scores for arbitrary number of components.
    """

    def simple_average(array):
        return mean(array)

    def min_score(array):
        return min(array)

    def ha_weighted_avg(array, sids: list):
        ha_counts = [ha_map[int(sid)] for sid in sids]
        return get_weighted_average(ha_counts, array)

    (score_dict_item, top_m, top_s_frac, rxn_id, n_component, warts_separator, sass_separator, ha_map) = data
    dedup_limit = max(1e6, 3 * top_m)

    f_set_idx, score_dict = score_dict_item

    aggregation_fn = min_score

    def backtrack(scores: list, sid_list, f_order, rxn_id):
        nonlocal res

        if len(res) > dedup_limit:
            res = deduplicate_scores(res, True, top_m, warts_separator=warts_separator)

        if len(sid_list) == n_component:
            # comb_score = aggregation_fn(scores)
            comb_score = ha_weighted_avg(scores, sid_list)
            score_data = (comb_score, sass_separator.join(sorted(sid_list)), rxn_id, f"{f_set_idx}_{f_order}")
            res.append(score_data)
            return

        cur_s_idx = len(sid_list)  # use result list to track which s-score is on.
        cur_scores = score_dict[f_order][score_keys[cur_s_idx]]
        cur_s_limit = _top_synthon_idx(cur_scores, top_s_frac)

        for i in range(len(score_dict[f_order][score_keys[cur_s_idx]])):
            if i > cur_s_limit:
                break
            else:
                _score, _sid = score_dict[f_order][score_keys[cur_s_idx]][i]
                scores.append(_score)
                sid_list.append(_sid)
                backtrack(scores, sid_list, f_order, rxn_id)
                scores.pop()
                sid_list.pop()

    products = []

    for f_order in score_dict.keys():
        res = []
        score_keys = list(score_dict[f_order].keys())
        backtrack([], [], f_order, rxn_id)
        res.sort(reverse=True)
        products.extend(res[:top_m])

    products.sort(reverse=True)
    products = deduplicate_scores(products, False, top_m, warts_separator=warts_separator)

    return products


def get_weighted_average(n_heavy_atoms: List[int], scores: List[float]) -> float:
    return sum(n_ha * score for n_ha, score in zip(n_heavy_atoms, scores)) / sum(n_heavy_atoms)


def mp_write_final_products(
    products: List[Tuple[float, str, str]],
    output_folder: Union[str, Path],
    outfile_prefix: str,
    id_map: dict,
    shandler: Synthon_handler,
    ncpu: int,
    chunk_size: int,
    title_separator: str,
):
    """
    Enumerate library products from the given score list and write products to file.

    Args:
        products: Array of (score, product_id, rxn_id) where product_id is the '_'-joined
            synthon ids used to identify the synthons that make up this product.
        chunk_size: The size of each output file. Breaking into smaller files during
            instantiation faciliates single-cpu OMEGA/ROCS.
    """

    def _write_products(data) -> List[oechem.OEMol]:
        (_reactants, _smirks, title_separator) = data
        mol = get_singleton_product(_reactants, _smirks, sort_title=True, title_separator=title_separator)
        return list(_generate_stereoisomers(mol, mp_global_param.flipper_option))

    os.makedirs(output_folder, exist_ok=True)

    ostream = oechem.oemolostream()

    n = 0
    chunk_n = 0

    if mp_global_param.ray_head is not None:
        pool = Ray_Pool(ray_address=mp_global_param.ray_head)
    else:
        pool = multiprocessing.Pool(ncpu)

    logging.info("Start instantiating top-m products.")

    # Pre-generate the lists to avoid copying large files to each cpu.
    reactants = []
    smirks = []
    for _, pid, rxn_id in products:
        sids = [int(x) for x in pid.split(title_separator)]
        _reactants = []
        for sid in sids:
            smi, sorder = id_map[sid]
            _reactants.append((sid, smi, sorder))
        reactants.append(_reactants)
        smirks.append(shandler.get_reaction_smirks_by_id(rxn_id))

    for isomers in pool.imap(
        _write_products,
        zip(
            reactants,
            smirks,
            repeat(title_separator),
        ),
    ):
        for isomer in isomers:
            if n % chunk_size == 0:
                ostream.close()
                ostream = oechem.oemolostream(os.path.join(output_folder, f"{outfile_prefix}_chunk_{chunk_n}.oeb.gz"))
                chunk_n += 1
            oechem.OEWriteMolecule(ostream, isomer)
            n += 1
            if n % 10000 == 0:
                logging.info(f"Instantiated {n} products.")
    ostream.close()
    pool.terminate()

    logging.info(f"Finished product enumeration. Total {n} products in {chunk_n} chunks.")


def sample_general_ring_synthon_conf(
    smirks: str,
    synthon_ring_connector_dict: Dict[str, str],
    current_synthons: List[Tuple[int, str]],
    current_synthon_idx: int,
    other_synthons: Dict[int, str],
    exp_dir: str,
    ofile_format: str,
    rxn_name: str,
    ncpu: int,
):
    """
    Sample conformers of synthons that contains ring-connectors.
    Direct conformer sampling could lead to unrealistic conformations due to unconstrained
    geometry of the partial ring atom/bonds. Instead, this workflow completes the ring-
    portion of the synthons, does conformer sampling, and then delete the added atom/bonds.

    Args:
        smirks: Reaction SMIRKS to react the synthons.
        synthon_ring_connector_dict: Dict of {atom: str(bool)} indicating whether a
            connector atom is a ring-atom or not.
        current_synthons: The synthon set to generate conformers on. Should only be synthons
            containing ring-connectors.
        current_synthon_idx: The position index of the synthon (for product generation).
        other_synthons: Single exampels of other synthons needed to complete a product.
        exp_dir: Directory to write the output conformers to.
        ofile_format: File extension of the conformer files. Usually "oez".
        rxn_name: ID of the reaction.
        ncpu: Number of cpu for parallelization.
    """

    reactants = {}  # For complete synthons.

    # Label all current synthons. For detecting which atoms to keep when restoring to original synthons.
    label = "self_label"
    nei_label = "connector_nei_in_ring"
    flat_mols = []
    for sid, smi in current_synthons:
        mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(mol, smi)
        mol.SetTitle(str(sid))
        for atom in mol.GetAtoms():
            atom.SetData(label, "self")
            atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
            if atom_symbol in synthon_ring_connector_dict:
                for nei in find_connected_atoms(atom):
                    if nei.HasData(nei_label):
                        data = nei.GetData(nei_label) + f"_{atom_symbol}"
                    else:
                        data = atom_symbol
                    nei.SetData(nei_label, data)
        flat_mols.append(mol)
    reactants[current_synthon_idx] = flat_mols

    # Label other synthons.
    for s_idx, smi in other_synthons.items():
        mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(mol, smi)
        for atom in mol.GetAtoms():
            atom.SetData(label, "other")
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
        oechem.OEFindRingAtomsAndBonds(mol)
        mol.SetTitle(mol.GetTitle().strip(title_sep))
        unprocessed_connector_atoms = set(synthon_ring_connector_dict.keys())
        # Cleave the bonds.
        for atom in mol.GetAtoms():
            if atom.HasData(nei_label):
                connector_symbol = atom.GetData(nei_label).split("_")[0]
                if connector_symbol in unprocessed_connector_atoms:
                    unprocessed_connector_atoms.remove(connector_symbol)
                    if synthon_ring_connector_dict[connector_symbol] == str(False):
                        # cleave the bond directly
                        for bond in atom.GetBonds():
                            nei = bond.GetNbr(atom)
                            if nei.GetData(label) == "other":  # i.e. the connector atom.
                                # switch to a special C atom.
                                new_atom = mol.NewAtom(nei)
                                new_atom.SetAtomicNum(oechem.OEGetAtomicNum("C"))
                                new_atom.SetData("previous_connector", connector_symbol)
                                mol.NewBond(atom, new_atom, bond.GetOrder())  # set to same bond order.
                                mol.DeleteBond(bond)
                                break
                    else:
                        # Find the new ring formed in the reaction. Not just the smallest ring.
                        # Returned atoms must contain both 'self' and 'other' labels.
                        new_ring_atoms = get_reaction_ring_atoms(
                            mol,
                            label_tag=label,
                            other_label="other",
                            start_atom=atom,
                        )
                        for atom in new_ring_atoms:
                            atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
                            if atom_symbol in unprocessed_connector_atoms:
                                unprocessed_connector_atoms.remove(atom_symbol)
                            if atom.GetData(label) == "other":
                                for bond in atom.GetBonds():
                                    nei = bond.GetNbr(atom)
                                    if nei not in new_ring_atoms:
                                        # 231117: keep =O, =N, etc, that are needed to indicate aromaticity.
                                        if bond.GetOrder() <= 1:
                                            mol.DeleteBond(bond)
                                        else:
                                            for b in nei.GetBonds():
                                                # delete all except the incident bond.
                                                if b != bond:
                                                    mol.DeleteBond(b)
        # Split, and get the "self" fragment (desired fragment will contain some "other" atoms,
        # but undesired fragments only contain "other" atoms.)
        for frag in generate_components(mol):
            if any(a.GetData(label) == "self" for a in frag.GetAtoms() if a.HasData(label)):
                _mol = cleanup_mol(frag)  # cleanup the cut bonds, fill valence.
                part_mols.append(oechem.OEMol(_mol))
                break

    # Stereoisomer
    stereo_mols = []
    for mol in part_mols:
        for isomer in generate_stereoisomers([mol], mp_global_param.flipper_option):
            stereo_mols.append(isomer)

    raw_conf_file = os.path.join(exp_dir, f"{rxn_name}_synthon_{current_synthon_idx}_conf_whole_ring.{ofile_format}")
    conf_file = os.path.join(exp_dir, f"{rxn_name}_synthon_{current_synthon_idx}_conf.{ofile_format}")

    mp_get_conf(
        input_mols=stereo_mols,
        output_file=raw_conf_file,
        omega_opt_key="synthon",
        ncpu=ncpu,
        sub_dict=None,
    )

    mp_cleave_off_parts(
        raw_conf_file,
        conf_file,
        atom_label=label,
        keep_label_val="self",
        remove_label_val="other",
        restore_connector=True,
        connector_nei_label=nei_label,
        ncpu=ncpu,
    )

    os.remove(raw_conf_file)


def get_reaction_ring_atoms(
    mol: oechem.OEMol,
    start_atom: oechem.OEAtomBase,
    label_tag: str,  # data tag name
    other_label: str,
):
    """
    Defined as the smallest ring formed that contains the `start_atom` (with `self_label`)
    and atoms that are labeled as `other_label`.

    This is not strictly correct for any graph, e.g. if a reaction forms multiple rings,
    and one of the smaller paths containing `other_label` is not the ring formed by this
    `start_atom`, but by some other start_atoms.
    However, for sensible molecules and reactions, this should be sufficient.
    """

    oechem.OEFindRingAtomsAndBonds(mol)
    n_bonds = mol.NumBonds()
    paths: List[set] = []

    parent_atom = None
    q = deque([(start_atom, parent_atom, set([start_atom]))])
    while q:
        cur_atom, prev_atom, cur_path = q.popleft()
        if len(cur_path) <= n_bonds:  # upper bound of the traversal.
            # check for returning to `start_atom` when enqueuing child.
            for nei in find_connected_atoms(cur_atom):
                if nei.IsInRing() and nei != prev_atom:
                    if nei == start_atom:
                        # Check if any of the atoms in the set/list contains the `other_label`,
                        # in which case it's a ring formed by the reaction (contains both `self` and `other`).
                        if any(a.GetData(label_tag) == other_label for a in cur_path):
                            return cur_path
                        paths.append(set(cur_path))
                    elif nei not in cur_path:  # not returning to another ring
                        next_path = set(cur_path)
                        # Still need to copy the path each time.. no better than using a list..
                        next_path.add(nei)
                        q.append((nei, cur_atom, next_path))


def mp_cleave_off_parts(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    atom_label: str,
    keep_label_val: str,
    remove_label_val: str,
    restore_connector: bool,
    connector_nei_label: str,
    ncpu: int,
):
    """
    Cleave off the connectors, and restore the special "C" atoms to the original connectors.

    Args:
        input_file: Input conformer file.
        output_file: Name of the output conformer file.
        atom_label: Name of atom data tag that indicates which atoms to keep/remove.
        keep_label_val: Value of the atom data tag for keeping an atom.
        remove_label_val: Value of the atom data tag for removing an atom.
        restore_connector: Whether to restore the original connector atom types. The
            information is stored in the neighbor atoms, or the special "C" atom.
        connector_nei_label: Name of the atom data tag that contains information on the
            type of connector atoms that were connected to this atom.
        ncpu: Number of processors for parallelization.
    """

    def _sp_cleave_off_parts(data):
        """ """

        (mol, connector_nei_label, atom_label, keep_label_val, remove_label_val, restore_connector) = data
        mol = oechem.OEMol(mol)
        if restore_connector:
            for atom in mol.GetAtoms():
                if atom.HasData(connector_nei_label):
                    connector_symbols = iter(atom.GetData(connector_nei_label).split("_"))
                    for nei in find_connected_atoms(atom):
                        if nei.GetData(atom_label) == remove_label_val:
                            # instead of create new atom, just mutate that 'other' atom in-place.
                            nei.SetData(atom_label, "previous_connector")
                            # Trick it into a not other atom to not be deleted.
                            nei.SetAtomicNum(oechem.OEGetAtomicNum(next(connector_symbols)))
                            nei.SetIsotope(0)
                            # in principle there should be equal # of non H "other" as the connector_symbol list.
                        if nei.HasData("previous_connector"):
                            prev_connector = nei.GetData("previous_connector")
                            nei.SetAtomicNum(oechem.OEGetAtomicNum(prev_connector))
                            nei.SetIsotope(0)

        for atom in mol.GetAtoms():
            if atom.HasData(atom_label) and atom.GetData(atom_label) == remove_label_val:
                oechem.OESuppressHydrogens(atom)
                mol.DeleteAtom(atom)

        return mol

    istream = oechem.oemolistream(input_file)
    ostream = oechem.oemolostream(output_file)
    with multiprocessing.Pool(ncpu) as pool:
        for mol in pool.imap(
            _sp_cleave_off_parts,
            zip(
                istream.GetOEMols(),
                repeat(connector_nei_label),
                repeat(atom_label),
                repeat(keep_label_val),
                repeat(remove_label_val),
                repeat(restore_connector),
            ),
        ):
            oechem.OEWriteMolecule(ostream, mol)

    istream.close()
    ostream.close()


def sample_synthon_conformers(
    rxn_id: str,
    rxn_synthons: dict,
    ncpu: int,
    sub_dict: dict,
    ofile_format: str,
    connector_atoms: List[str],
    smirks: str,
    exp_dir: str,
):
    """
    Sample conformations of both non-ring and ring synthons.
    """

    eg_synthon_smiles = {}
    for key, syns in rxn_synthons.items():
        _, smi = syns[0]
        eg_synthon_smiles[key] = smi

    synthon_ring_connector_dict = label_synthon_ring_connector(
        synthon_smiles=eg_synthon_smiles,
        connector_atoms=connector_atoms,
        smirks=smirks,
    )

    for s_idx, synthons in rxn_synthons.items():
        logging.info(f"Start omega on {rxn_id} synthon {s_idx}")
        current_connectors = get_conn_symbols(eg_synthon_smiles[s_idx], connector_atoms)
        if all(synthon_ring_connector_dict[c_atom] == str(False) for c_atom in current_connectors):
            print("sampling non-ring synthons")
            sample_non_ring_synthons_conf(
                rxn_id=rxn_id,
                current_synthons=synthons,
                current_synthon_idx=s_idx,
                ncpu=ncpu,
                sub_dict=sub_dict,
                ofile_format=ofile_format,
                reverse_atom_substitution=True,
                exp_dir=exp_dir,
            )
        else:
            print("sampling ring synthons")
            other_synthons = {key: val for key, val in eg_synthon_smiles.items() if key != s_idx}
            sample_general_ring_synthon_conf(
                smirks=smirks,
                synthon_ring_connector_dict=synthon_ring_connector_dict,
                current_synthons=synthons,
                current_synthon_idx=s_idx,
                other_synthons=other_synthons,
                exp_dir=exp_dir,
                ofile_format=ofile_format,
                rxn_name=rxn_id,
                ncpu=ncpu,
            )


def sample_non_ring_synthons_conf(
    rxn_id: str,
    current_synthons: List[Tuple[int, str]],
    current_synthon_idx: int,
    ncpu: int,
    sub_dict: dict,
    ofile_format: str,
    reverse_atom_substitution: bool,
    exp_dir: str,
):
    """
    Simple conformer sampling of input isomers.

    Args:
        rxn_id: Reaction id, for naming the conformer file.
        current_synthons: Input synthons for conformer generation.
        current_synthon_idx: Synthon order in from the synthon file. For naming the
            output conformer file.
        ncpu: Number of processors for parallelization.
        sub_dict: Mapping of atom substitution.
        ofile_format: Conformer file format. Usually "oez".
        reverse_atom_substitution: Whether to revert the substituted atoms back to original
            on the conformers.
        exp_dir: Directory to write the conformer files to.
    """

    ofile = os.path.join(exp_dir, f"{rxn_id}_synthon_{current_synthon_idx}_conf.{ofile_format}")
    flat_mols = []

    for sid, smiles in current_synthons:
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smiles)
        mol.SetTitle(str(sid))
        substitute_atoms(mol, sub_dict)  # Need to substitute U/Np etc. before cleanup.
        cleanup_mol(mol)
        flat_mols.append(mol)

    mols = []
    for mol in flat_mols:
        for isomer in generate_stereoisomers([mol], mp_global_param.flipper_option):
            mols.append(isomer)

    if reverse_atom_substitution:
        reverse_sub_dict = {val: key for key, val in sub_dict.items()}
    else:
        reverse_sub_dict = None

    mp_get_conf(
        mols,
        ofile,
        "synthon",
        ncpu,
        sub_dict=reverse_sub_dict,
    )


def mp_get_conf(
    input_mols: Iterable[oechem.OEMolBase],
    output_file: Union[str, Path],
    omega_opt_key: str,
    ncpu: int,
    sub_dict: dict,
):
    """
    Generate conformers of input molecules (OMEGA) using multiprocessing.
    Write the multi-conformer molecules to the `output_file`.
    """

    def _get_conf(data):
        """
        Sub-function for multi-processing `mp_get_conf` function.
        """

        mol, omega_opt_key = data

        try:
            omega_opt = mp_global_param.omega_opts[omega_opt_key]
            omega = oeomega.OEOmega(omega_opt)
            return _get_conformer(mol, omega)
        except Exception as e:
            logging.error(f"Conformer generation failed for {mol.GetTitle()}")
            logging.error(e)

    logging.info("Start omega.")
    ostream = oechem.oemolostream(output_file)
    n = 0
    start_time = time.time()

    if ncpu == 1:
        omega_opt = mp_global_param.omega_opts[omega_opt_key]
        omega = oeomega.OEOmega(omega_opt)
        for mol in input_mols:
            mc_mol = _get_conformer(mol, omega)
            if mc_mol is not None:
                if sub_dict is not None:
                    substitute_atoms(mc_mol, sub_dict=sub_dict)
                oechem.OEWriteMolecule(ostream, mc_mol)
                n += 1
                if n % 100 == 0:
                    logging.info(f"OMEGA processed {n} molecules.")
    else:
        if mp_global_param.ray_head is not None:
            pool = Ray_Pool(ray_address=mp_global_param.ray_head)
        else:
            pool = multiprocessing.Pool(ncpu)

        for mc_mol in pool.imap(
            _get_conf,
            zip(input_mols, repeat(omega_opt_key)),
        ):
            if mc_mol is not None:
                if sub_dict is not None:
                    substitute_atoms(
                        mc_mol,
                        sub_dict=sub_dict,
                        delete_H=True,
                    )
                oechem.OEWriteMolecule(ostream, mc_mol)
                n += 1
                if n % 1000 == 0:
                    logging.info(f"OMEGA Iter: {n}")

        pool.terminate()

    finish_time = time.time()
    logging.info(f"Omega finished, operated on a total of {n} isomers. Time took: {finish_time - start_time} s.")
    ostream.close()


def mp_frag_rocs_score(
    frag_sets,
    smirks,
    synthon_smarts,
    synthon_connectors,
    frag_connectors,
    connector_tag,
    cross_score,
    synthon_conf_dir,
    rxn_id,
    conf_format,
    grouped_rxn_synthon_lens,
    conf_chunk_dir,
    exp_dir,
    warts_id_sep,
    ncpu,
) -> dict:
    """
    Synthon scoring, parallelized on the query fragment level.

    Args:
        frag_sets: Fragment sets to be scores. Each fragment set will be handled by one
            process.
        smirks: Reaction SMIRKS for labeling synthon connectors.
        synthon_smarts: Examples of synthons for a given reaction. For determining synthon
            connectors and number of reaction components.
        synthon_connectors: All possible synthon connector atom types.
        frag_connectors: All possible fragment connector atom types.
        connector_tag: Name of atom data tag on query fragments to indicate a ring/non-
            ring connector atom.
        cross_score: Whether to score ring parts against non-ring parts.
        synthon_conf_dir: Directory containing synthon conformers.
        rxn_id: Reaction id, for naming the score files.
        conf_format: Format of the input synthon conformer files.
        grouped_rxn_synthon_lens: Lengths of each synthon list for a given reaction.
        conf_chunk_dir: Temp directory for storing chunked conformer files.
        exp_dir: Directory to write the results to.
        warts_id_sep: Separator for conformer ids. For extracting base id for deduplication.
        ncpu: Number of processors for parallelization.
    """

    res = {}
    i = 0
    with multiprocessing.Pool(ncpu) as pool:
        for out in pool.map(
            _sp_frag_rocs_score,
            zip(
                frag_sets,
                range(len(frag_sets)),
                repeat(synthon_smarts),
                repeat(smirks),
                repeat(synthon_connectors),
                repeat(frag_connectors),
                repeat(connector_tag),
                repeat(cross_score),
                repeat(synthon_conf_dir),
                repeat(rxn_id),
                repeat(conf_format),
                repeat(grouped_rxn_synthon_lens),
                repeat(conf_chunk_dir),
                repeat(exp_dir),
                repeat(warts_id_sep),
            ),
        ):
            if out:  # i.e. not an empty dict
                res[f"frag_set_{i}"] = out
            i += 1
    return res


def _sp_frag_rocs_score(data):
    """
    Single process function for `mp_frag_rocs_score`.
    """

    (
        frags,
        fset_idx,
        synthon_smarts,
        smirks,
        synthon_connectors,
        frag_connectors,
        connector_tag,
        cross_score,
        synthon_conf_dir,
        rxn_id,
        conf_format,
        grouped_rxn_synthon_lens,
        conf_chunk_dir,
        exp_dir,
        warts_id_sep,
    ) = data
    res = {}

    logging.info(f"Scoring {fset_idx}-th fragment sets.")
    (synthon_order, ordered_frags,) = order_fragments(  # synthon_order is 0-indexed!
        synthon_smarts=synthon_smarts,  # This contains info on how many component-reaction.
        smirks=smirks,
        synthon_connectors=synthon_connectors,
        fragments=frags,
        frag_connectors=frag_connectors,
        frag_ring_connector_label=connector_tag,
        cross_score=cross_score,
    )

    if len(ordered_frags) == 0:
        logging.warning(f"No valid ordering of fragments for frag set {fset_idx}.")

    for order_idx, ordered_f in enumerate(ordered_frags):
        res.setdefault(f"forder_{order_idx}", {})
        for score_idx, (s_idx, _frag) in enumerate(zip(synthon_order, ordered_f)):
            conf_f = os.path.join(synthon_conf_dir, f"{rxn_id}_synthon_{s_idx}_conf.{conf_format}")
            assert os.path.isfile(conf_f), logging.error(f"{conf_f} does not exist.")
            num_connector = len(get_conn_symbols(oechem.OEMolToSmiles(_frag), synthon_connectors))

            n_chunks = math.ceil(grouped_rxn_synthon_lens[s_idx] / 5e3)

            if n_chunks > 1:
                scores = chunk_rocs_scores(
                    ref_mol=_frag,
                    frag_idx=score_idx,
                    input_conf_file=conf_f,
                    temp_conf_dir=conf_chunk_dir,
                    output_folder=exp_dir,
                    n_chunks=n_chunks,
                    outfile_prefix=f"{rxn_id}_synthon_{s_idx}",
                    conf_file_suffix=conf_format,
                    rocs_key=num_connector,
                    rocs_best_hits=3 * grouped_rxn_synthon_lens[s_idx],
                    use_self_cpu=True,
                    query_cleanup=False,
                    warts_separator=warts_id_sep,
                )
            else:
                scores = sp_simple_rocs_scores((_frag, conf_f, num_connector))
                scores = deduplicate_scores(scores, sort_scores=True, warts_separator=warts_id_sep)
            res[f"forder_{order_idx}"][f"score_{score_idx}"] = scores.copy()

    return res


def sp_simple_rocs_scores(data) -> List[Tuple[float, str]]:
    """
    Basic ROCS scoring with given conformer file and query molecule.
    """

    ref_mol, conf_file, rocs_key = data
    rocs_opt = mp_global_param.rocs_opts[rocs_key]
    rocs = oeshape.OEROCS(rocs_opt)
    rocs.SetDatabase(oechem.oemolistream(conf_file))

    res = rocs.Overlay(ref_mol)
    out = [[_res.GetTanimotoCombo(), _res.GetOverlayConf().GetTitle()] for _res in res]

    return out


def rocs_write_out_overlay(
    input_conf_file,
    outfile_prefix,
    out_folder,
    ref_mol,
    rocs_key,
):
    """
    Run ROCS and write out overlay poses of top results.
    """

    rocs_opt = mp_global_param.rocs_opts[rocs_key]
    rocs = oeshape.OEROCS(rocs_opt)
    rocs.SetDatabase(oechem.oemolistream(input_conf_file))

    ostream = oechem.oemolostream(os.path.join(out_folder, f"{outfile_prefix}_rocs_overlay.sdf"))
    oechem.OEWriteMolecule(ostream, ref_mol)

    scores = []
    for res in rocs.Overlay(ref_mol):
        score = res.GetTanimotoCombo()
        title = res.GetOverlayConf().GetTitle()
        scores.append([score, title])
        overlay_conf = res.GetOverlayConf()
        oechem.OESetSDData(overlay_conf, "Tanimoto_combo", str(score))
        oeshape.OERemoveColorAtoms(overlay_conf)
        oechem.OEWriteMolecule(ostream, overlay_conf)

    ostream.close()
    return scores


def chunk_rocs_scores(
    ref_mol: oechem.OEMol,
    frag_idx: int,
    input_conf_file: str,
    temp_conf_dir: str,
    output_folder: str,
    n_chunks: int,
    outfile_prefix: str,
    conf_file_suffix: str,
    rocs_key: int,
    rocs_best_hits: int,
    query_cleanup: bool,
    warts_separator: str,
    use_self_cpu: bool = True,
) -> list:
    """
    Split synthon conf database, run ROCS on each conf chunk, and combine results.
    Deduplicate results based on id.

    Args:
        ref_mol: Reference mol (query mol).
        frag_idx: Index of the fragment. For naming the output score file.
        input_conf_file: Conformer file for scoring.
        temp_conf_dir: Temporary directory for storing chunked conformers.
        output_folder: Directory for storing the ROCS results.
        n_chunks: Number of chunks to split the conformer into.
        outfile_prefix: For naming the output score file.
        conf_file_suffix: Conformer file format. Usually "oez".
        rocs_key: Which ROCS option from `mp_global_param` to use.
        rocs_best_hits: Number of top ROCS results to return.
        query_cleanup: Whether to cleanup the query molecule.
        warts_separator: Separator for isomer/conformer ids.
        use_self_cpu: Whether to use the current processor to do computation, in addition
            to the jobs submitted to the compute cluster.
    """

    logging.info(f"Starting mp chunk ROCS. Reference molecule: {ref_mol.GetTitle()}")

    scores = []
    n = 0

    conf_files = [
        os.path.join(temp_conf_dir, f"{outfile_prefix}_conf_chunk_{n}.{conf_file_suffix}") for n in range(n_chunks)
    ]

    if not all(os.path.isfile(f) for f in conf_files):
        split_n_chunks(
            input_file=input_conf_file,
            chunk_n=n_chunks,
            chunk_size=None,
            output_folder=temp_conf_dir,
            file_prefix=f"{outfile_prefix}_conf",
            file_suffix=conf_file_suffix,
        )
        logging.info(f"Files {input_conf_file} split into {n_chunks} chunks.")
    else:
        logging.info(f"{input_conf_file} already split into {n_chunks} conf chunks.")

    ref_mol_file = os.path.join(output_folder, f"{uuid1()}.oeb")
    ostream = oechem.oemolostream(ref_mol_file)
    oechem.OEWriteMolecule(ostream, ref_mol)
    ostream.close()

    if use_self_cpu:
        start_chunk = 1
    else:
        start_chunk = 0
    submit_rocs(
        chunks=n_chunks,
        query_file=ref_mol_file,
        start_chunk=start_chunk,
        input_folder=temp_conf_dir,
        infile_prefix=f"{outfile_prefix}_conf",
        input_file_suffix=conf_file_suffix,
        output_folder=output_folder,
        outfile_prefix=f"{outfile_prefix}_{frag_idx}_rocs_score",
        best_hits=rocs_best_hits,
        rocs_key=rocs_key,
        color_ff_file=os.path.join(output_folder, f"custom_color_ff_{rocs_key}.txt"),
        query_cleanup=query_cleanup,
        wait_for_finish=False,  # wait outside this function.
        queue="short",
    )

    if use_self_cpu:
        scores_0 = sp_simple_rocs_scores(
            (ref_mol, os.path.join(temp_conf_dir, f"{outfile_prefix}_conf_chunk_0.{conf_file_suffix}"), rocs_key)
        )
        with open(os.path.join(output_folder, f"{outfile_prefix}_{frag_idx}_rocs_score_chunk_0.pkl"), "wb") as f:
            pickle.dump(scores_0, f)

    score_files = [
        os.path.join(output_folder, f"{outfile_prefix}_{frag_idx}_rocs_score_chunk_{i}.pkl") for i in range(n_chunks)
    ]
    wait(score_files, wait_time=5, final_delay=5)

    scores = merge_score_files(
        score_files,
        top_n=rocs_best_hits,
        warts_separator=warts_separator,
    )

    logging.info(f"Chunk ROCS finished for {input_conf_file} with {ref_mol.GetTitle()}.")

    # Cleanup
    for f in score_files:
        os.remove(f)
    os.remove(ref_mol_file)

    return scores


def mp_substitute_atoms(
    infile: Union[str, Path],
    outfile: Union[str, Path],
    sub_dict: dict,
    ncpu: int,
    mol_cleanup: bool = True,
):
    """
    Substitute atoms in input molecules, and write results to file.
    """

    ostream = oechem.oemolostream(outfile)
    istream = oechem.oemolistream(infile)
    if mp_global_param.ray_head is not None:
        pool = Ray_Pool(ray_address=mp_global_param.ray_head)
    else:
        pool = multiprocessing.Pool(ncpu)
    for mol in pool.imap(_substitute_atoms, zip(istream.GetOEMols(), repeat(sub_dict))):
        if mol_cleanup:
            mol = cleanup_mol(mol)
        oechem.OEWriteMolecule(ostream, mol)
    pool.terminate()

    istream.close()
    ostream.close()


def _substitute_atoms(input_data) -> oechem.OEMol:
    """
    Substitute atoms in a multi-conformer molecule.

    Sub-function for `mp_substitute_atoms` multiprocessing function.
    """

    mol, sub_dict = input_data

    return substitute_atoms(mol, sub_dict)


def _gen_library(input_args):
    (start_idx, chunk_size, reactants, smirks, enum_isomers, title_separator) = input_args
    reac1 = reactants[0]
    end_idx = min(len(reac1), start_idx + chunk_size)
    reactants[0] = reac1[start_idx:end_idx]

    mols = generate_library_products(
        reactants,
        smirks,
        sort_title=True,
        title_separator=title_separator,
    )

    if not enum_isomers:
        return list(mols)
    else:
        return list(
            generate_stereoisomers(
                list(mols),
                mp_global_param.flipper_option,
            )
        )


def mp_gen_library(
    reactants: Dict[int, List[Tuple[int, str]]],
    smirks: str,
    ncpu: int,
    enum_isomers: bool,
    title_separator: str,
):
    """
    Multiprocessing function to generate product libraries.
    """

    reac1 = reactants[0]
    chunk_size_1 = math.ceil(len(reac1) / ncpu)

    with multiprocessing.Pool(ncpu) as pool:
        for prods in pool.imap(
            _gen_library,
            zip(
                range(0, len(reac1), chunk_size_1),
                repeat(chunk_size_1),
                repeat(reactants),
                repeat(smirks),
                repeat(enum_isomers),
                repeat(title_separator),
            ),
        ):
            for prod in prods:
                yield prod


def get_distance(mol1, mol2, symbol, isotope) -> float:

    atom1 = list(find_atoms(mol1, symbol, isotope))[0]
    atom2 = list(find_atoms(mol2, symbol, isotope))[0]
    return oechem.OEGetDistance(mol1, atom1, mol2, atom2)


def validate_tasks(tasks):
    if tasks is not None:
        tasks = set(tasks)
        valid_tasks = set(
            [
                "ground_truth",
                "gen_synthon_conformers",
                "score_synthons",
                "select_synthons",
                "combine_products",
                "instantiate_products",
                "rescore_products",
            ]
        )
        invalid_tasks = [t for t in tasks if t not in valid_tasks]
        if invalid_tasks:
            raise ValueError(f"Invalid tasks: {invalid_tasks}")

        if "ground_truth" in tasks and len(tasks) > 1:
            raise ValueError("`ground_truth` task cannot be combined with other tasks.")


def order_fragments(
    synthon_smarts: Dict[int, str],
    smirks: str,
    synthon_connectors: List[str],
    fragments: List[oechem.OEMol],
    frag_connectors: List[str],
    frag_ring_connector_label: str,
    cross_score: bool,
) -> List[List[oechem.OEMol]]:
    """
    Create all valid query fragment molecule (ordered), and the corresponding synthon order.

    Args:
        synthon_smarts: *unordered* synthon SMARTS, as ordered in data file.
        fragments: *unordered* query fragments, from query fragmentation.
    """

    def check_S_F_matching(synthon_smis, frag_mols):
        frag_smis = [oechem.OEMolToSmiles(f) for f in frag_mols]
        for s_smi, f_smi in zip(synthon_smis, frag_smis):
            s_conn_set = set(get_conn_symbols(s_smi, synthon_connectors))
            f_conn_set = set(get_conn_symbols(f_smi, synthon_connectors))
            if s_conn_set != f_conn_set:
                return False
        return True

    # Order the synthons.
    synthon_ring_dict = label_synthon_ring_connector(
        synthon_smiles=synthon_smarts,
        connector_atoms=synthon_connectors,
        smirks=smirks,
    )

    synthon_smarts_list = list(synthon_smarts.values())
    synthon_order = enumerate_syn_frag_orders(
        smis=synthon_smarts_list,
        ring_connector_dict=synthon_ring_dict,
    )[0]
    # Any of the ordering will work. Return this order in the end.
    ordered_synthon_smarts = [synthon_smarts_list[i] for i in synthon_order]  # ordered synthons

    # Order the fragments.
    frag_ring_dict = get_frag_ring_connector_labels(frag_mols=fragments, connector_tag=frag_ring_connector_label)

    frag_smis = [oechem.OEMolToSmiles(f) for f in fragments]
    frag_orders = enumerate_syn_frag_orders(
        smis=frag_smis,
        ring_connector_dict=frag_ring_dict,
    )

    # Check for 3-ring synthon/frag (if each synthon/frag contains exactly 2 connectors)
    all_ring_synthon = (
        all(len(get_conn_symbols(smi, synthon_connectors)) == 2 for smi in synthon_smarts_list) and len(fragments) > 2
    )
    all_ring_frag = all(len(get_conn_symbols(smi, frag_connectors)) == 2 for smi in frag_smis) and len(fragments) > 2
    ordered_fragments = []
    if int(all_ring_synthon) - int(all_ring_frag) == 0:  # Either both True or both False
        for f_order in frag_orders:
            ordered_fragments.extend(
                enumerate_fragment_mol_orders(
                    smis=ordered_synthon_smarts,
                    frag_mols=[fragments[i] for i in f_order],
                    cross_score=cross_score,
                    frag_connector_symbols=set(frag_connectors),
                    synthon_connector_symbols=set(synthon_connectors),
                )
            )
    else:
        logging.warning("Synthon and frag sets are not both all-ring or both not all-ring. Matching skipped.")

    # Check each position connector atom match to synthon connectors (only for all-ring synthons/frags)
    if all_ring_synthon and all_ring_frag:
        ordered_fragments = [
            ordered_f for ordered_f in ordered_fragments if check_S_F_matching(ordered_synthon_smarts, ordered_f)
        ]

    return synthon_order, ordered_fragments


def mp_flipper(input_file, output_file, ncpu):

    istream = oechem.oemolistream(input_file)
    ostream = oechem.oemolostream(output_file)
    with multiprocessing.Pool(ncpu) as pool:
        for isomers in pool.imap(
            wrap_gen_stereoisomers,
            istream.GetOEMols(),
        ):
            for isomer in isomers:
                oechem.OEWriteMolecule(ostream, isomer)
    istream.close()
    ostream.close()


def wrap_gen_stereoisomers(mol):
    return list(generate_stereoisomers([mol], mp_global_param.flipper_option))


def run_std_omega_rocs(
    exp_dir,
    n_chunks,
    query_file,
    input_file_prefix,
    conf_file_format,
    num_conf: int,
    max_time: int,
    top_m: int,
    warts_separator: str,
):
    """
    Workflow for standard conformer search (OMEGA) and overlay with ROCS. Input files
    are chunks of enumerated isomers, and the output is a score pkl file.

    Args:
        exp_dir: Path of the experiment directory.
        n_chunks: Number of chunks of input molecules. Chunking allows convenient parallel
            processing.
        query_file: File containing the query molecule.
        input_file_prefix: Prefix of the input isomer files; full file name is
            `input_file_prefix`_chunk_<n>.<conf_file_format>.
        conf_file_format: Format of the conformer file format. 'oez' format is preferred.
        num_conf: Max number of conformers per molecule to be generated.
        max_time: Max time allowed for conformer search on each molecule.
        top_m: Top m number of ROCS results to write out.
        warts_separator: Separator symbol used after the molecule title to denote different
            conformers. e.g. 1234_1, 2234_5.
    """

    if all(os.path.isfile(os.path.join(exp_dir, f"conf_chunk_{n}.{conf_file_format}")) for n in range(n_chunks)):
        logging.info("Conf files already generated. OMEGA skipped.")
    else:
        logging.info("Start product conformer generation.")
        submit_omega(
            chunks=n_chunks,
            input_folder=exp_dir,
            input_file_prefix=input_file_prefix,
            output_folder=exp_dir,
            output_file_suffix=conf_file_format,
            num_conf=num_conf,
            max_time=max_time,
        )
        logging.info("Finished product conformer generation.")

    rocs_score_files = [os.path.join(exp_dir, f"rocs_score_chunk_{n}.pkl") for n in range(n_chunks)]
    if all(os.path.isfile(f) for f in rocs_score_files):
        logging.info("ROCS score files already exist. ROCS skipped.")
    else:
        logging.info("Start product ROCS scoring.")
        submit_rocs(
            chunks=n_chunks,
            query_file=query_file,
            input_folder=exp_dir,
            infile_prefix="conf",
            input_file_suffix=conf_file_format,
            output_folder=exp_dir,
            outfile_prefix="rocs_score",
            best_hits=top_m,
            rocs_key=0,
            query_cleanup=True,
        )
        logging.info("Finished product ROCS scoring.")

    if os.path.isfile(os.path.join(exp_dir, "rescoring_combined_rocs_res.pkl")):
        logging.info("Combined ROCS score exists. Score aggregation skipped.")
    else:
        logging.info("Start merging scores.")
        scores = merge_score_files(
            rocs_score_files,
            top_n=top_m,
            warts_separator=warts_separator,
        )
        with open(os.path.join(exp_dir, "rescoring_combined_rocs_res.pkl"), "wb") as f:
            pickle.dump(scores, f)
        logging.info("Finished merging scores.")


def submit_omega(
    chunks: int,
    input_folder: Union[Path, str],
    input_file_prefix: str,
    output_folder: Union[Path, str],
    output_file_suffix: str,
    num_conf: int,
    max_time: int,
):

    for i in range(chunks):
        input_file = os.path.join(input_folder, f"{input_file_prefix}_chunk_{i}.oeb.gz")
        output_file = os.path.join(output_folder, f"conf_chunk_{i}.{output_file_suffix}")

        subprocess.run(
            [
                "sbatch",
                "-c",
                "1",
                "--mem-per-cpu",
                "4G",
                f"{SCRIPT_DIR}/submit_omega.sh",
                input_file,
                output_file,
                str(num_conf),
                str(max_time),
            ]
        )

    wait([os.path.join(output_folder, f"conf_chunk_{i}.{output_file_suffix}.flag") for i in range(chunks)])


def submit_rocs(
    chunks: int,
    query_file: Union[Path, str],
    input_folder: Union[Path, str],
    infile_prefix: str,
    input_file_suffix: str,
    output_folder: Union[Path, str],
    outfile_prefix: str,
    best_hits: int,
    start_chunk: int = 0,
    rocs_key: int = 0,
    query_cleanup: bool = True,
    color_ff_file: str = None,
    wait_for_finish: bool = True,
    queue: str = "medium",
):
    """
    Submit ROCS jobs to compute cluster.
    """

    n = 0
    for i in range(start_chunk, chunks):
        input_file = f"{infile_prefix}_chunk_{i}.{input_file_suffix}"
        input_file = os.path.join(input_folder, input_file)
        file_size = os.path.getsize(input_file)

        if file_size < 1e8:
            mem = 8
        else:
            mem = 16 * math.ceil(file_size / 1e9)
            mem = min(32, mem)
        cmd = [
            "sbatch",
            "--qos",
            queue,
            "-c",
            "1",
            "--mem-per-cpu",
            f"{mem}G",
            f"{SCRIPT_DIR}/submit_rocs.sh",
            query_file,
            input_file,
            os.path.join(f"{output_folder}", f"{outfile_prefix}_chunk_{i}.pkl"),
            f"{best_hits}",
            f"{rocs_key}",
            f"{color_ff_file}",
        ]
        if query_cleanup:
            cmd.append("--query_cleanup")

        subprocess.run(cmd)
        n += 1

    logging.info(f"Submitted {n} jobs.")

    if wait_for_finish:
        wait([os.path.join(output_folder, f"{outfile_prefix}_chunk_{i}.pkl") for i in range(start_chunk, chunks)])
