"""Score synthons against query fragments and combine them into ranked pseudo-products.

The submission loop drives Phase A (`score_synthon_fragment`, one component against one query
fragment) and Phase B (`combine_forder_task`, one fragment order into pseudo-products). It runs
a small warm-up wave first so a score cutoff exists early, merges score files while scoring
runs, and feeds each merge's `top_m`-th score back into later tasks as a pruning cutoff.
"""

import logging
import math
import pickle
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

from dask.distributed import as_completed
from openeye import oechem

from sass.chem.fragment import fragment_memo_key, fragment_molecule
from sass.chem.synthon import order_and_substitute_fragments
from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.io import discard_superseded_dir, load_score_file
from sass.library.conformers import build_conformer_manifest
from sass.library.synthons import SynthonHandler
from sass.selection.dedup import deduplicate_scores
from sass.worker.merge_scores import merge_score_batch_task
from sass.worker.score_synthons import combine_forder_task, score_synthon_fragment

logger = logging.getLogger(__name__)


def dask_score_and_combine_synthon(
    client: DaskClient,
    reactions: list[str],
    shandler: SynthonHandler,
    query_fragments_dict: dict[int, list[list[oechem.OEMolBase]]],
    grouped_synthons: dict,
    synthon_conf_dir: Path,
    conf_format: str,
    align_connector_atom: bool,
    synthon_connectors: list[str],
    rocs_key_opt: int,
    rocs_key_score: int,
    opt_shape_func: str,
    opt_color_func: str | None,
    overlap_shape_func: str,
    overlap_color_func: str | None,
    top_m: int,
    dedup_limit: int,
    warts_separator: str,
    agg_method: str,
    synthon_comb_limit: int,
    frag_connectors: list[str],
    cross_score: bool,
    color_ff_dir: Path,
    pseudo_res_dir: Path,
    merge_fanin: int = 12,
    warmup_wave_size: int = 64,
    warmup_quorum: float = 0.5,
):
    """Score synthons against query fragments and combine synthons to generate pseudo scores.

    Pseudo scores are organized by number of reaction components to allow for
    merging results from different component reactions using a user-defined ratio.

    Parameters
    ----------
    reactions
        Reaction ids to be used for generating scoring tasks.
    shandler
        Synthon handler containing synthon and reaction information.
    query_fragments_dict
        Query fragments, organized by number of fragments.
    grouped_synthons
        Synthons organized by reactions.
    synthon_conf_dir
        Directory containing synthon conformers. Usually pre-generated.
    conf_format
        Format of the conformer files, e.g. oez, or oeb.
    align_connector_atom
        Whether to apply custom interactions on special connector atoms.
    synthon_connectors
        All special atoms denoting a connector atom on sythons.
    rocs_key_opt
        Key for ROCS option for overlay optimization.
    rocs_key_score
        Key for ROCS option for overlap scoring.
    **_shape/color_func:
        Shape and color functions for overlay optimization and overlap scoring.
    top_m
        Number of top pseudo products to output.
    dedup_limit
        Size above which a score list is deduplicated and trimmed to `top_m`, to bound memory.
    warts_separator
        String separator for naming isomers, conformers.
    agg_method
        Method for aggregating synthon scores to calculate the pseudo product scores.
    synthon_comb_limit
        Limit of number of combinations for each worker. Without a limit, all synthon
        combinations are considered, which is as costly as brute force.
    frag_connectors
        All special atoms denoting a connector atom on query fragments.
    cross_score
        See `sass.chem.synthon.order_and_substitute_fragments`.
    color_ff_dir
        Directory containing custom color interaction files.
    pseudo_res_dir
        Directory to write pseudo scores to.
    merge_fanin
        Number of score files merged into one by each incremental merge.
    warmup_wave_size
        Number of tasks to run before opening submission to the full cluster, so that a
        score cutoff exists before the bulk of the work starts. Set to 0 to disable.
    warmup_quorum
        Fraction of the wave that must finish before submission opens. Stragglers keep
        running alongside the rest.

    Returns
    -------
        One fully merged pseudo score file per component count, and the number of
        possible products per component count.
    """
    # One directory scan for the whole phase. The task generator slices it per task.
    conf_manifest = build_conformer_manifest(synthon_conf_dir, conf_format)
    logger.info(
        "Conformer manifest: %s files across %s (reaction, component) groups.",
        sum(len(v) for v in conf_manifest.values()),
        len(conf_manifest),
    )

    task_gen = generate_synthon_scoring_tasks(
        reactions,
        shandler,
        query_fragments_dict,
        grouped_synthons,
        synthon_connectors,
        frag_connectors,
        cross_score,
        conf_manifest,
    )

    phase_a_kwargs = {
        "align_connector_atom": align_connector_atom,
        "synthon_connectors": synthon_connectors,
        "rocs_key_opt": rocs_key_opt,
        "rocs_key_score": rocs_key_score,
        "opt_shape_func": opt_shape_func,
        "opt_color_func": opt_color_func,
        "overlap_shape_func": overlap_shape_func,
        "overlap_color_func": overlap_color_func,
        "warts_separator": warts_separator,
        "color_ff_dir": color_ff_dir,
    }
    phase_b_kwargs = {
        "top_m": top_m,
        "dedup_limit": dedup_limit,
        "warts_separator": warts_separator,
        "synthon_comb_limit": synthon_comb_limit,
        "agg_method": agg_method,
        "pseudo_res_dir": pseudo_res_dir,
    }
    #: `(rxn, s_idx, fragment)` -> the future holding that component's score list. Phase A
    #: is memoized on this key so each distinct triple is scored exactly once.
    phase_a_futures: dict[tuple, Any] = {}

    # Bounded submission, so each task can carry the newest `cutoff`. Task arguments are
    # fixed at submit time.
    max_in_flight = max(3 * client.target_max_worker, merge_fanin * 4)

    # Warm-up wave. Until some bucket has `top_m` rows there is no cutoff, and every task
    # writes a full-size file. A small first wave establishes cutoffs early. A cutoff from
    # any subset is a lower bound on the final threshold, so a poor sample costs pruning
    # efficiency, never correctness. The wave is a one-shot batch, not a concurrency cap,
    # and the gate opens on a quorum rather than on the whole wave.
    in_warmup = warmup_wave_size > 0
    wave_quorum = max(1, math.ceil(warmup_wave_size * warmup_quorum))
    n_wave_submitted = 0
    wave_merge_triggered = False
    buckets_seen: set[int] = set()

    pending: dict[int, list[Path]] = defaultdict(list)  # unmerged score files per component
    cutoffs: dict[int, float] = {}
    prod_count = defaultdict(int)
    unique_rxn = set()
    merge_counter = 0
    n_scored = 0
    client.disk.reset("pseudo_res")
    exhausted = False
    n_score_in_flight = 0
    n_merge_in_flight = 0

    ac = as_completed()

    def _submit_merges(n_component: int, force: bool) -> None:
        """Fold `pending` score files for one component count into fewer, larger files.

        `force` drains the queue down to a single file; used once no more score files
        can arrive for that component.
        """
        nonlocal merge_counter, n_merge_in_flight
        queue = pending[n_component]
        while len(queue) >= merge_fanin or (force and len(queue) > 1):
            batch = queue[:merge_fanin]
            del queue[:merge_fanin]
            out_file = pseudo_res_dir / f"merged_{n_component}comp_{merge_counter}.pkl"
            merge_counter += 1
            ac.add(
                client.dask.submit(
                    merge_score_batch_task,
                    batch,
                    out_file,
                    top_m,
                    warts_separator,
                    False,  # keep_isomer_warts: pseudo-product ids carry no warts.
                    tag=n_component,
                    pure=False,
                )
            )
            n_merge_in_flight += 1

    def _submit_one() -> bool:
        """Submit the next fragment order, reusing component scores. False if none left."""
        nonlocal exhausted, n_score_in_flight
        try:
            data = next(task_gen)
        except StopIteration:
            exhausted = True
            return False
        rxn_id, fset_idx, forder_idx, synthon_order, ordered_f, conf_files, frag_keys = data
        n_component = len(synthon_order)

        # Phase A, memoized by `(rxn, s_idx, fragment)`. Dask keeps each result alive only
        # while some fragment order still holds its future.
        component_futures = []
        for s_idx, frag, frag_key in zip(synthon_order, ordered_f, frag_keys, strict=True):
            key = (rxn_id, s_idx, frag_key)
            future = phase_a_futures.get(key)
            if future is None:
                future = client.dask.submit(
                    score_synthon_fragment,
                    (rxn_id, s_idx, frag, conf_files.get(s_idx, [])),
                    pure=False,
                    **phase_a_kwargs,
                )
                phase_a_futures[key] = future
            component_futures.append(future)

        # Component scores are passed as futures, so this runs as soon as its own
        # components are done.
        ac.add(
            client.dask.submit(
                combine_forder_task,
                (rxn_id, fset_idx, forder_idx, list(synthon_order)),
                *component_futures,
                cutoff=cutoffs.get(n_component),
                pure=False,
                **phase_b_kwargs,
            )
        )
        n_score_in_flight += 1
        return True

    def _top_up() -> None:
        """Submit the warm-up wave, or keep the queue full once the gate has opened."""
        nonlocal n_wave_submitted
        if in_warmup:
            # Submit the wave once, then stop.
            while not exhausted and n_wave_submitted < warmup_wave_size:
                if _submit_one():
                    n_wave_submitted += 1
            return
        while not exhausted and n_score_in_flight < max_in_flight:
            if not _submit_one():
                break

    _top_up()
    while ac.count():
        dask_res = next(ac).result()

        if dask_res and dask_res[0] == "merge":
            n_merge_in_flight -= 1
            _, n_component, out_file, _n_rows, delta_bytes, cutoff = dask_res
            pending[n_component].append(out_file)
            client.disk.add(delta_bytes)
            if cutoff is not None and cutoff > cutoffs.get(n_component, -math.inf):
                cutoffs[n_component] = cutoff
        else:
            n_score_in_flight -= 1
            if not dask_res:  # Guard against None returns.
                _top_up()
                continue
            _, rxn_id, fname, count, nbytes, local_cutoff = dask_res
            n_scored += 1
            n_component = shandler.get_number_of_components(rxn_id)
            buckets_seen.add(n_component)
            pending[n_component].append(fname)
            client.disk.add(nbytes)
            if rxn_id not in unique_rxn:
                unique_rxn.add(rxn_id)
                prod_count[n_component] += count  # Only count once per rxn_id.
            if local_cutoff is not None and local_cutoff > cutoffs.get(n_component, -math.inf):
                cutoffs[n_component] = local_cutoff
            if n_scored % client.target_max_worker == 0:
                client.disk.log_progress(n_scored, extra=f" cutoffs: {({k: round(v, 4) for k, v in cutoffs.items()})}.")

        # Once no more score files can arrive, drain every bucket to a single file.
        draining = exhausted and n_score_in_flight == 0
        if draining:
            for bucket in list(pending):
                _submit_merges(bucket, force=True)
        else:
            _submit_merges(n_component, force=False)

        wave_ready = n_scored >= wave_quorum or (exhausted and n_score_in_flight == 0)
        if in_warmup and wave_ready:
            # Enough of the wave has scored. Start a merge, since a merged bucket gives a
            # better cutoff than any single task, but do not wait on it.
            if not wave_merge_triggered:
                wave_merge_triggered = True
                for bucket in list(pending):
                    _submit_merges(bucket, force=True)
            # Every component count seen so far needs a cutoff before opening. The merge
            # clause covers runs too small for any bucket to reach `top_m`.
            have_cutoffs = bool(buckets_seen) and buckets_seen <= cutoffs.keys()
            if have_cutoffs or n_merge_in_flight == 0:
                in_warmup = False
                logger.info(
                    "Warm-up quorum reached: %s of %s wave tasks scored (%s still running); "
                    "cutoffs: %s. Opening submission to %s in flight.",
                    n_scored,
                    n_wave_submitted,
                    n_score_in_flight,
                    {k: round(v, 4) for k, v in cutoffs.items()},
                    max_in_flight,
                )
        _top_up()

    # Safety net in case the drain above left more than one file in a bucket.
    final_files = {}
    for n_component, queue in pending.items():
        if not queue:
            continue
        if len(queue) > 1:
            logger.warning("Bucket %s still had %s unmerged files; merging now.", n_component, len(queue))
            out_file = pseudo_res_dir / f"merged_{n_component}comp_final.pkl"
            merge_score_batch_task(queue, out_file, top_m, warts_separator, False)
            final_files[n_component] = out_file
        else:
            final_files[n_component] = queue[0]

    logger.info("Scored and combined a total of %s rxn_fset_forder res.", n_scored)
    client.disk.log_peak()
    logger.info("Final merged pseudo_res file per component: %s.", final_files)
    logger.info(
        "Total number of possible products for each component: %s.",
        dict(prod_count),
    )
    return final_files, prod_count


def combine_pseudo_res(
    pseudo_res: dict[int, Path],
    top_m: int,
    warts_separator: str,
    use_proportion: bool,
    product_count: dict[int, int],
) -> list:
    """Mix the per-component pseudo product scores into one final list.

    The per-component lists are already merged by `dask_score_and_combine_synthon`. This
    step only applies the component mixing ratio.

    Parameters
    ----------
    pseudo_res
        One merged pseudo res file per number of components.
    top_m
        Number of top pseudo products to output.
    warts_separator
        String separator for naming isomers, conformers.
    use_proportion
        Whether to use the proportion of total products by each number of reaction
        components to select the final product list. E.g. if total products of 2 vs
        3 component reaction is 30:70, pick the top 0.3 x `top_m` from all 2 component
        reaction products nad 0.7 x `top_m` from all 3 component reaction products.
    product_count
        Possible number of products grouped by number of components.

    Returns
    -------
        Combined pseudo products, sorted by descending scores.
    """
    res = {n_comp: load_score_file(file) for n_comp, file in pseudo_res.items()}

    # Mixing different n-component results by ratio.
    if not use_proportion:
        mixing_ratio = dict.fromkeys(res, 1)
    else:
        # Determine the ratio from product counts
        total_count = sum(product_count.values())
        mixing_ratio = {key: product_count[key] / total_count for key in product_count}
        logger.info("Ratio for mixing various component reaction pseudo results: %s.", mixing_ratio)

    combined_res = []
    for key, _res in res.items():
        combined_res.extend(_res[: int(top_m * mixing_ratio[key])])

    out = deduplicate_scores(combined_res, warts_separator=warts_separator, sort_scores=True, limit=top_m)

    return out


def generate_synthon_scoring_tasks(
    reactions: list[str],
    shandler: SynthonHandler,
    query_fragments_dict: dict[int, list],
    grouped_synthons: dict[Any, Any],
    synthon_connectors: list[str],
    frag_connectors: list[str],
    cross_score: bool,
    conf_manifest: dict[tuple[str, int], list[Path]] | None = None,
):
    conf_manifest = {} if conf_manifest is None else conf_manifest
    seen_components = set()
    streams_by_component: dict[int, list] = defaultdict(list)
    for rxn_id in reactions:
        n_component = shandler.get_number_of_components(rxn_id)
        seen_components.add(n_component)
        if n_component not in query_fragments_dict:
            logger.info(f"{n_component}-fragment sets were not generated from query. Skipping reaction {rxn_id}.")
            continue
        if not query_fragments_dict[n_component]:
            continue
        streams_by_component[n_component].append(
            _generate_reaction_scoring_tasks(
                rxn_id=rxn_id,
                n_component=n_component,
                query_fragments_dict=query_fragments_dict,
                grouped_synthons=grouped_synthons,
                synthon_connectors=synthon_connectors,
                frag_connectors=frag_connectors,
                cross_score=cross_score,
                rxn_conf_files={s_idx: files for (rxn, s_idx), files in conf_manifest.items() if rxn == rxn_id},
            )
        )

    streams = interleave_by_component(streams_by_component)

    # Round-robin one task per reaction per pass, over a reaction order that alternates
    # component counts. Any prefix of the task stream then spans many reactions and every
    # component count.
    n = 0
    while streams:
        alive = []
        for stream in streams:
            task = next(stream, None)
            if task is None:
                continue
            yield task
            n += 1
            alive.append(stream)
        streams = alive

    logger.info(f"Generated a total of {n} synthon scoring tasks.")
    if seen_components != set(query_fragments_dict.keys()):
        logger.warning("!" * 32)
        logger.warning(
            f"Query was fragmented into {list(query_fragments_dict.keys())} fragments, "
            f"but synthon conformers are from {seen_components}-component reactions."
        )
        logger.warning("!" * 32)


def _generate_reaction_scoring_tasks(
    rxn_id: str,
    n_component: int,
    query_fragments_dict: dict[int, list],
    grouped_synthons: dict[Any, Any],
    synthon_connectors: list[str],
    frag_connectors: list[str],
    cross_score: bool,
    rxn_conf_files: dict[int, list[Path]],
) -> Iterable[tuple]:
    """Yield every (fragment set, fragment order) scoring task for a single reaction.

    Each task carries only the conformer files for the components it scores.
    """
    # Take the 1st synthon SMILES from each reactant set. Do not use the SMARTS
    # from the reaction file, since OETK cannot parse "~" in SMARTS.
    synthon_smiles_eg = {}
    for key, syns in grouped_synthons[rxn_id].items():
        _, smi = syns[0]
        synthon_smiles_eg[key] = smi

    rxn_synthons_lens = [len(val) for val in grouped_synthons[rxn_id].values()]
    frag_sets = query_fragments_dict[n_component]
    logger.info(f"Start scoring {rxn_id} synthons ({rxn_synthons_lens}); frag_set_count: {len(frag_sets)}")

    for fset_idx, frags in enumerate(frag_sets):
        (
            synthon_order,
            ordered_frags,
        ) = order_and_substitute_fragments(  # synthon_order is 0-indexed!
            synthon_smiles_eg=synthon_smiles_eg,  # This contains info on how many component-reaction.
            synthon_connectors=synthon_connectors,
            fragments=frags,
            frag_connectors=frag_connectors,
            cross_score=cross_score,
        )
        for forder_idx, ordered_f in enumerate(ordered_frags):
            logger.debug(f"Start ROCS with reaction: {rxn_id}, frag set: {fset_idx}, forder: {forder_idx}.")
            yield (
                rxn_id,
                fset_idx,
                forder_idx,
                synthon_order,
                ordered_f,
                {s_idx: rxn_conf_files.get(s_idx, []) for s_idx in synthon_order},
                # Identity per fragment, used to memoize Phase A (see `fragment_memo_key`).
                [fragment_memo_key(f) for f in ordered_f],
            )


def interleave_by_component(streams_by_component: dict[int, list]) -> list:
    """Order per-reaction task streams so consecutive entries alternate component counts.

    Enamine reaction files list every 2-component reaction before every 3-component one.
    Alternating component counts makes any prefix cover every bucket, so each bucket gets a
    cutoff early.

    Parameters
    ----------
    streams_by_component
        Per-reaction task generators, grouped by number of reaction components.

    Returns
    -------
        A flat list of streams, alternating between component counts while each still has
        reactions left, then continuing with whatever remains.
    """
    groups = [streams_by_component[key] for key in sorted(streams_by_component)]
    return [stream for row in zip_longest(*groups) for stream in row if stream is not None]


def run(ctx: RunContext, client: DaskClient) -> None:
    """Fragment the query, score every synthon against it, and combine into pseudo-products."""
    ctx.logger.info(f"Starting synthon scoring using conformers from {ctx.synthon_conf_dir}.")

    query_fragments_dict = {
        n: fragment_molecule(
            mol=oechem.OEMol(ctx.query_mol),
            n_fragments=n,
            heavy_atom_limit=ctx.config["heavy_atom_limit"],
            cleave_acyclic_bonds=True,
            cleave_cyclic_bonds=ctx.config["cleave_cyclic_bonds"],
            # Non-zero isotope marks a connector atom for `check_fragment_size`.
            connector_isotope=1,
            connector_atoms=ctx.frag_connectors,
        )
        for n in ctx.config["n_fragments"]
    }

    ctx.pseudo_res_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    pseudo_res_file_dict, prod_count = dask_score_and_combine_synthon(
        client,
        reactions=ctx.reactions,
        shandler=ctx.shandler,
        query_fragments_dict=query_fragments_dict,
        synthon_conf_dir=ctx.synthon_conf_dir,
        grouped_synthons=ctx.grouped_synthons,
        conf_format=ctx.conf_format,
        align_connector_atom=ctx.config["align_on_dummy_atom"],
        synthon_connectors=ctx.synthon_connectors,
        rocs_key_opt=ctx.rocs_key_opt,
        rocs_key_score=ctx.rocs_key_score,
        opt_shape_func=ctx.opt_shape_func,
        opt_color_func=ctx.opt_color_func,
        overlap_shape_func=ctx.overlap_shape_func,
        overlap_color_func=ctx.overlap_color_func,
        top_m=ctx.top_m,
        dedup_limit=ctx.dedup_limit,
        warts_separator=ctx.warts_id_sep,
        agg_method=ctx.config["synthon_score_aggregation_method"],
        synthon_comb_limit=ctx.synthon_comb_limit,
        frag_connectors=ctx.frag_connectors,
        cross_score=ctx.config["cross_scoring"],
        color_ff_dir=ctx.color_ff_dir,
        pseudo_res_dir=ctx.pseudo_res_dir,
        merge_fanin=ctx.config["pseudo_res_merge_fanin"],
        warmup_wave_size=ctx.config["cutoff_warmup_wave_size"],
        warmup_quorum=ctx.config["cutoff_warmup_quorum"],
    )

    comb_pseudo_products = combine_pseudo_res(
        pseudo_res=pseudo_res_file_dict,
        top_m=ctx.top_m,
        warts_separator=ctx.warts_id_sep,
        use_proportion=ctx.config["mix_pseudo_res_by_proportion"],
        product_count=prod_count,
    )

    with ctx.combined_pseudo_res_file_w.open("wb") as f:
        pickle.dump(comb_pseudo_products, f)
    ctx.logger.info(f"Finished combining pseudo products. Result saved to {ctx.combined_pseudo_res_file_w}.")
    discard_superseded_dir(
        ctx.pseudo_res_dir,
        ctx.combined_pseudo_res_file_w,
        enabled=ctx.config["clean_up_intermediate_files"],
    )
