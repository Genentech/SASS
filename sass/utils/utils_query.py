"""Miscellaneous util functions for running SASS."""

# Standard Library
import logging
import lzma
import math
import pickle
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Union

# Third Party Library
from openeye import oechem, oeshape

# Genentech Library
from sass.utils.utils_general import extract_base_id
from sass.utils.utils_mol import (
    FragNodeMatcher,
    count_non_H_atom,
    enumerate_fragment_connector_substitution,
    load_first_mol,
    map_mol_frag_graph,
    smiles_to_mol,
    special_substitute_atoms,
)


def split_n_chunks(
    input_file: Union[Path, str],
    chunk_n: int,
    chunk_size: int,
    output_folder: Union[Path, str],
    file_prefix: str,
    file_suffix: str,
):
    """Split file into chunks.

    Parameters
    ----------
    input_file : Union[Path, str]
        The path to the input file that needs to be split into chunks.
    chunk_n : int
        The number of chunks to create. If `chunk_size` is provided, this parameter is
        disregarded.
    chunk_size : int
        The size of each chunk. If `None`, the chunk size will be calculated based on
        `chunk_n`.
    output_folder : Union[Path, str]
        The path to the output folder where the chunks will be saved.
    file_prefix : str
        The prefix to be added to the filenames of the output chunks.
    file_suffix : str
        The suffix to be added to the filenames of the output chunks.

    Returns
    -------
    None

    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    db = oechem.OEMolDatabase(str(input_file))
    chunk = 0
    if chunk_size is None:
        n_mol = db.NumMols()
        chunk_size = math.ceil(n_mol / chunk_n)
    ostream = oechem.oemolostream()
    for i, mol in enumerate(db.GetOEMols()):
        if (i + 1) % chunk_size == 0:
            ostream.close()
            ostream = oechem.oemolostream(str(output_folder / f"{file_prefix}_chunk_{chunk}.{file_suffix}"))
            chunk += 1
        oechem.OEWriteMolecule(ostream, mol)
    ostream.close()

    logging.info(f"{input_file} split into {chunk_n} chunks.")


def merge_score_files(
    files: list[Union[Path, str]],
    top_n: int,
    warts_separator: str,
    keep_isomer_warts: bool,
    is_binary: bool = True,
    score_size_limit: int = 5e6,
) -> list:
    """
    Merge multiple score files into one. Scores are lists of [tuples (score, id, ...)].

    Parameters
    ----------
    files
        Score files to be merged.
    top_n
        Desired output size of the score file.
    warts_separator
        String separator between molecule id and isomer/conformer number in molecule titles.
    keep_isomer_warts
        Whether to keep isomer warts in the merged scores. Deduplication is still done
        on the base id (without isomer warts).
    is_binary
        Whether the files are binary or not. Defaults to True.
    score_size_limit
        Limit of the size of the combined scores, beyond which the list will deduplicate and trim to `top_n`.
        This is to prevent memory overflow and avoid sorting a huge list in the end.

    Returns
    -------
    list
        The merged and deduplicated scores.

    """
    scores = []
    open_flag = "rb" if is_binary else "r"
    for file in files:
        with Path(file).open(open_flag) as f:
            scores.extend(pickle.load(f))
        if len(scores) > score_size_limit:
            scores = deduplicate_scores(
                scores,
                sort_scores=True,
                limit=top_n,
                warts_separator=warts_separator,
                keep_isomer_warts=keep_isomer_warts,
            )

    return deduplicate_scores(
        scores,
        sort_scores=True,
        limit=top_n,
        warts_separator=warts_separator,
        keep_isomer_warts=keep_isomer_warts,
    )


def deduplicate_scores(
    scores: list[tuple[float, str, Any]],
    warts_separator: str,
    keep_isomer_warts: bool = False,
    sort_scores: bool = False,
    limit: Optional[int] = None,
) -> list[tuple[float, str, Any]]:
    """
    Deduplicate score list by mol id, return a sorted list.

    Parameters
    ----------
    scores
        Array of score data, e.g. [(score1, id1, ...), (score2, id2, ...), ...].
    warts_separator
        String separator between molecule id and isomer/conformer number in molecule titles.
    keep_isomer_warts
        Whether to keep isomer warts in the merged scores. Deduplication is still done
        on the base id (without isomer warts).
    sort_scores
        Whether to sort the array by the scores (1st element in the tuple). Most times the
        input array is already sorted by descending scores. However, use this
        option with caution, especially when score information is absent (i.e. scores
        are all None or ''), in which case sorting will cause unexpected behavior (i.e.
        sorting by other elements in the tuple).
    limit
        Number of top scores to output.

    Returns
    -------
    list[tuple[float, str, Any]]
        Array of [(score, id), ...] where all base ids are unique.
    """
    dedup_res = []
    seen = set()
    if sort_scores:
        scores.sort(reverse=True, key=lambda x: x[0])  # sort only based on 1st element

    n = 0
    for data in scores:
        sid = data[1]
        if limit is not None and n >= limit:
            break
        base_id = extract_base_id(mol_id=sid, keep_isomer_warts=False, warts_separator=warts_separator)
        if keep_isomer_warts is True:
            output_id = extract_base_id(sid, keep_isomer_warts=keep_isomer_warts, warts_separator=warts_separator)
        else:
            output_id = base_id
        if base_id not in seen:
            seen.add(base_id)
            new_entry = [data[0], output_id, *data[2:]]  # Retain additional information.
            if isinstance(data, tuple):
                new_entry = tuple(new_entry)
            dedup_res.append(new_entry)
            n += 1

    return dedup_res


def validate_tasks(tasks: list[str], query_file: str | None) -> None:
    """Validate the tasks and query file.

    Also determine whether to load the query and synthons based on the tasks.
    """
    load_query = False
    load_synthons = False
    if tasks is not None:
        tasks = set(tasks)
        valid_tasks = {  # tasks: whether to load synthons
            "full_library_enum": True,
            "full_library_conf_gen": False,
            "full_library_scoring": False,
            "gen_synthon_conformers": True,
            "score_synthons": True,
            "instantiate_products": False,
            "rescore_products": False,
            "get_product_poses": False,
            "get_single_fragment_poses": True,
            "score_single_fragment": True,
        }
        invalid_tasks = [t for t in tasks if t not in valid_tasks]
        if invalid_tasks:
            raise ValueError(f"Invalid tasks: {invalid_tasks}")
        load_synthons = any(valid_tasks[t] for t in tasks)

        scoring_tasks = {
            "full_library_scoring",
            "score_synthons",
            "rescore_products",
            "get_product_poses",
            "get_single_fragment_poses",
            "score_single_fragment",
        }
        if any(t in scoring_tasks for t in tasks):
            if query_file is None:
                raise ValueError(f"Must specify a query for tasks: {[t for t in tasks if t in scoring_tasks]}!")
            validate_query(query_file)
            load_query = True

        if any("single_fragment" in t for t in tasks) and any("single_fragment" not in t for t in tasks):
            raise ValueError("Single fragment tasks cannot be combined with other tasks.")

        if "instantiated_products" in tasks and "rescore_products" in tasks:
            raise ValueError("`Rescore products` step already contains product instantiation.")
    return load_query, load_synthons


def validate_query(file: Path | str) -> None:
    if not Path(file).is_file():
        raise ValueError(f"Query molecule file {file} does not exist!")

    # Check number of conformers. Not an issue for SDF file input.
    query = load_first_mol(file)
    n_confs = query.NumConfs()
    if n_confs > 1:
        raise ValueError(f"Query molecule should only have one conformer, but has {n_confs} conformers!")

    # Check if molecule can be set as ROCS reference.
    overlay = oeshape.OEOverlay()
    if not overlay.SetupRef(query):
        raise ValueError("Query molecule cannot be set as ROCS reference! Check if it is a 3D conformer.")


def order_and_substitute_fragments(
    synthon_smiles_eg: dict[int, str],
    # smirks: str,
    synthon_connectors: list[str],
    fragments: list[oechem.OEMol],
    frag_connectors: list[str],
    # frag_ring_connector_label: str,
    cross_score: bool,
) -> list[list[oechem.OEMol]]:
    """Generate all ordered fragment lists for a set of synthons.

    Given the synthon order, enumerate all ways to order the fragments, and all ways to
    map the fragment connector atoms to synthon connector atoms, and substitute those
    fragment connector atoms. The synthon index order is returned instead of the actual
    synthons because the synthons here are just representative of the entire synthon lists.
    The synthon index is used to look up the synthon conformer files.

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
    synthon_order, synthon_smiles = zip(*synthon_smiles_eg.items())
    syn_nodes = map_mol_frag_graph(synthon_smiles, set(synthon_connectors))
    frag_smiles = [oechem.OEMolToSmiles(f) for f in fragments]
    frag_nodes = map_mol_frag_graph(frag_smiles, set(frag_connectors))

    matches = FragNodeMatcher.match_all_graphs(syn_nodes, frag_nodes)
    ordered_fragments = []
    for match in matches:
        # Important! Sort the fragments based on the synthon order.
        # Get the integer index fo the frag-nodes based on the match dict.
        # Frag node indices are stored as f'{idx}_2'. See `match_all_graphs`.
        frag_order = [int(match[syn_node].index.split("_")[0]) for syn_node in syn_nodes]
        ordered_frags = [fragments[idx] for idx in frag_order]
        for res in enumerate_fragment_connector_substitution(syn_nodes, match, cross_score):
            if res:
                temp = [special_substitute_atoms(f, res) for f in ordered_frags]
                ordered_fragments.append(temp)

    return synthon_order, ordered_fragments


def allocate_synthons(
    synthon_counts: dict[int | str : int], min_num: int = 100, max_total: int = 1e6
) -> dict[int | str : int]:
    """Calculate the limit of each synthon score list to use for select synthons.

    This is to avoid overly large lists from 3-component reactions that can be on
    the order of 1e9, multiplied by hundreds of fragment sets.

    This takes synthons from each synthon list proportionally, while keeping a
    minimum number of synthons from each list.

    Parameters
    ----------
    synthon_counts
        The number of synthons in each reaction component.
    min_num
        The minimum number of synthons to take from each component.
    max_total
        The maximum total number of products to generate.

    Returns
    -------
        A dictionary mapping synthon index to the limit.
    """
    # sort the synthon count, process smallest first.
    syn_count = list(synthon_counts.items())
    syn_count.sort(key=lambda x: x[1])  # sort by count

    def _allocate_syn(idx: int, cur_prod: int, remaining_allowed: int) -> list:
        nonlocal res, syn_count, min_num
        if idx == len(syn_count) - 1:
            res.append((syn_count[idx][0], min(cur_prod, remaining_allowed)))
            return
        if cur_prod <= remaining_allowed:
            # technically, can also just take the entire remaining
            # res.append(_syn_count[idx])
            # _allocate_syn(idx + 1, int(cur_prod / _syn_count[idx]), int(remaining_allowed / _syn_count[idx]))
            res.extend(syn_count[idx:])
            return
        # calc reducing factor.
        reduce_factor = (cur_prod / remaining_allowed) ** (1 / (len(syn_count) - idx))
        first_actual = syn_count[idx][1]
        first_reduced = int(first_actual / reduce_factor)
        if first_reduced >= min_num:
            first_count = first_reduced
        else:  # if after scaling, the count is < min_num, use up to min_num of this synthon.
            first_count = min(first_actual, min_num)
        res.append((syn_count[idx][0], first_count))
        _allocate_syn(idx + 1, int(cur_prod / syn_count[idx][1]), int(remaining_allowed / first_count))

    cur_prod = math.prod([ele[1] for ele in syn_count])
    res = []
    _allocate_syn(0, cur_prod, max_total)
    return dict(res)


def simple_average(array: list[float]) -> float:
    # return mean(array) # This is slow!!
    return sum(array) / len(array)


def min_score(array: list) -> float:
    return min(array)


def get_weighted_average(n_heavy_atoms: list[int], scores: list[float]) -> float:
    return sum(n_ha * score for n_ha, score in zip(n_heavy_atoms, scores)) / sum(n_heavy_atoms)


def ha_weighted_avg(array: list, sids: list, synthon_data: dict) -> float:
    ha_counts = [synthon_data[str(sid)]["heavy_atom_count"] for sid in sids]
    return get_weighted_average(ha_counts, array)


def top_two_of_three(array: list) -> float:
    array.sort()
    return mean(array[1:])  # This is slow. Replace with sum/n.


def top_two_separately(array: list) -> float:
    a = [ele[0] for ele in array]
    b = [ele[1] for ele in array]
    return top_two_of_three(a) + top_two_of_three(b)


def aggregate_scores(agg_method: str, scores: list[float], **kwargs: Any) -> float:
    match agg_method:
        case "simple_average":
            return simple_average(scores)
        case "ha_weighted_avg":
            return ha_weighted_avg(scores, kwargs["sids"], kwargs["synthon_data"])
        case "min_score":
            return min_score(scores)
        case "top_two_of_three":
            return top_two_of_three(scores)
        case "top_two_separately":
            return top_two_separately(scores)
        case _:
            raise ValueError(f'"{agg_method}" is an unknown score aggregation method.')


def determine_rxn_id(rids: list[str | list[str] | set[str]]) -> str:
    """Determine the reaction id based on the input rids.

    20240816: For new REAL 2024 data, some synthons are mapped to multiple reactions, causing
    error when using a single rxn_id stored in synthon_data file. New synthon_data file
    stores all reaction ids for each synthon. This function finds the intersect of all
    reactions of all synthons and return an arbitrary reaction id from the intersection.
    For backward compatibility where synthon_data file only stores one reaction id, this
    function finds the most frequent reaction id among all synthons.
    """
    if isinstance(rids[0], (list, set)):
        common_rxn_id = set.intersection(*[set(r) for r in rids])
        if common_rxn_id:
            return common_rxn_id.pop()
        else:
            raise ValueError(f"No common reaction id found among all ids: {rids}.")
    else:
        return max(set(rids), key=rids.count)


def compress_pkl_files(folder: Path) -> None:
    """Compress all pkl files in a directory."""
    pkl_files = list(folder.rglob("*.pkl"))
    for file in pkl_files:
        with file.open("rb") as input_fh:
            data = input_fh.read()
        outfile = file.with_suffix(".pkl.xz")
        with lzma.open(outfile, "wb") as outpuf_fh:
            outpuf_fh.write(data)
        file.unlink()


def set_file_name(preferred: str, alternative: str | Path) -> Path:
    """Set a file name based on the preferred name and an alternative name.

    If the preferred name is `None`, use the alternative name.

    Parameters
    ----------
    preferred
        Preferred file name.
    alternative
        Alternative file name.
    """
    if preferred is not None:
        return Path(preferred)
    else:
        return Path(alternative)


def load_intermediate_files(file: str | Path) -> Any:
    """Load intermediate files.

    If the file is compressed, it will be decompressed before loading.
    """
    file = Path(file)
    if file.suffix == ".xz":
        with lzma.open(file, "rb") as f:
            return pickle.load(f)
    else:
        with file.open("rb") as f:
            return pickle.load(f)


def merge_synthon_conformer_chunks(file_dir: Path, conf_format: str) -> None:
    """Merge synthon conformer chunks into a single file."""
    conf_files = list(file_dir.glob(f"*.{conf_format}"))
    unique_synthons = {f.name.split(".")[0] for f in conf_files}
    for synthon in unique_synthons:
        synthon_files = list(file_dir.glob(f"{synthon}*.oez"))
        merged_synthon_file = file_dir / f"{synthon}.{conf_format}"
        ostream = oechem.oemolostream(str(merged_synthon_file))
        if len(synthon_files) == 1:
            synthon_files[0].rename(merged_synthon_file)
        else:
            for synthon_file in synthon_files:
                for mol in oechem.oemolistream(str(synthon_file)).GetOEMols():
                    oechem.OEWriteMolecule(ostream, mol)
                synthon_file.unlink()
            ostream.close()
        logging.info(f"{synthon} {len(synthon_files)} chunks have been merged.")


def map_synthon_ids(
    reactions: list[str],
    grouped_synthons: dict[str, dict[int, list[tuple[int, str]]]],
    synthon_connectors: list[str],
) -> dict:
    """Map synthon ids to SMILES, synthon index, heavy atom count, and reaction ids."""
    synthon_data = {}
    for rxn_id in reactions:
        for s_idx, synthons in grouped_synthons[rxn_id].items():
            for sid, smi in synthons:
                # heavy atom count
                mol = smiles_to_mol(smi)
                ha_count = count_non_H_atom(mol, set(synthon_connectors))
                # A synthon id may have multiple `rxn_id`, but only one `s_idx` or `SMILES`.
                if str(sid) in synthon_data:
                    for key, val in {"SMILES": smi, "s_idx": s_idx}.items():
                        if synthon_data[str(sid)][key] != val:
                            raise ValueError(
                                (
                                    f"Conflict in synthon data: {sid}, {key}, "
                                    f"prev: {synthon_data[str(sid)][key]}, new: {val}."
                                )
                            )
                    synthon_data[str(sid)]["rxn_id"].add(rxn_id)
                else:
                    synthon_data[str(sid)] = {}
                    for key, val in {"SMILES": smi, "s_idx": s_idx, "heavy_atom_count": ha_count}.items():
                        synthon_data[str(sid)][key] = val
                    synthon_data[str(sid)]["rxn_id"] = {rxn_id}
    return synthon_data
