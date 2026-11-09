"""Score synthons against a single query fragment. Experimental."""

import logging
import pickle
from pathlib import Path

from openeye import oechem

from sass import EXTEND_ATOM_ISOTOPE, EXTEND_ATOM_SYMBOL
from sass.chem.mol import load_first_mol
from sass.chem.synthon import extend_dummy_bond
from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.worker.merge_scores import merge_score_batch_task
from sass.worker.score_single_fragment import score_single_fragment

logger = logging.getLogger(__name__)


def dask_score_single_fragment(
    client: DaskClient,
    conf_files: list[Path],
    query_fragment: oechem.OEMolBase,
    synthon_connectors: set[str],
    frag_connectors: set[str],
    color_ff_dir: Path,
    warts_separator: str,
    score_dir: Path,
    keep_top: int,
    merge_fanin: int,
) -> list[Path]:
    """Score a single fragment against a set of synthons, merging the score files as they arrive.

    Trimming a partial merge to `keep_top` is exact, since the top k of a union equals the top
    k of the union of each part's top k.

    Parameters
    ----------
    conf_files
        Synthon conformer files to score against.
    query_fragment
        Query fragment to score against synthons.
    synthon_connectors
        Special atoms denoting a connector atom on synthons.
    frag_connectors
        Special atoms denoting a connector atom on query fragment.
    color_ff_dir
        Directory containing custom color force field files.
    warts_separator
        Separator for warts in product ids.
    score_dir
        Directory to write out scores to.
    keep_top
        Number of top scores each merge keeps. See `_retention`.
    merge_fanin
        Number of score files folded into one per merge task.

    Returns
    -------
        Score files left to combine: the merged ones, plus any tail too short to merge.
    """
    client.disk.reset("single_frag_scores")
    pending: list[Path] = []
    merged: list[Path] = []
    merge_futures = []
    merge_counter = 0
    n = 0

    def _fold(batch: list[Path]) -> None:
        nonlocal merge_counter
        out_file = score_dir / f"merged_single_frag_{merge_counter}.pkl"
        merge_counter += 1
        merge_futures.append(
            client.dask.submit(
                merge_score_batch_task,
                batch,
                out_file,
                keep_top,
                warts_separator,
                True,  # keep_isomer_warts: pose extraction needs the wart to find the conformer.
                pure=False,
            )
        )

    def _reap() -> None:
        """Collect finished merges without blocking on the unfinished ones."""
        still_running = []
        for future in merge_futures:
            if not future.done():
                still_running.append(future)
                continue
            _kind, _tag, out_file, _n_rows, n_bytes, _cutoff = future.result()
            client.disk.add(n_bytes)
            merged.append(Path(out_file))
        merge_futures[:] = still_running

    for res in client.stream_map(
        score_single_fragment,
        iter(conf_files),
        query_fragment=query_fragment,
        synthon_connectors=synthon_connectors,
        frag_connectors=frag_connectors,
        color_ff_dir=color_ff_dir,
        warts_separator=warts_separator,
        score_dir=score_dir,
    ):
        if res is None:  # synthon had no single alignable connector
            continue
        score_file, n_bytes = res
        n += 1
        client.disk.add(n_bytes)
        pending.append(score_file)
        if len(pending) >= merge_fanin:
            _fold(pending)
            pending = []
            _reap()
        if n % client.target_max_worker == 0:
            client.disk.log_progress(n)

    while merge_futures:  # drain whatever was still running when the last score landed
        merge_futures[0].result()
        _reap()

    client.disk.log_peak()
    return merged + pending


def _prepared_query(ctx: RunContext) -> oechem.OEMol:
    """Reload the query fragment and, if configured, extend its connector bond.

    Single-fragment scoring aligns along the connection vector, which needs a second dummy
    atom. The single-fragment synthon conformers must use the same `extend_bond_distance`.
    """
    query_mol = load_first_mol(ctx.query_mol_file, mol_cleanup=False)
    ctx.logger.info(f"Query fragment (re)loaded from {ctx.query_mol_file}.")
    if ctx.config["align_on_bond"]:
        query_mol = extend_dummy_bond(
            mol=query_mol,
            dummy_atom_symbol=ctx.synthon_connectors[0],  # must match the stored conformers
            dummy_isotope=0,
            new_dummy_symbol=EXTEND_ATOM_SYMBOL,
            new_isotope=EXTEND_ATOM_ISOTOPE,
            distance=ctx.config["extend_bond_distance"],
        )
    return query_mol


def _conformer_files(ctx: RunContext) -> list[Path]:
    """Single-fragment conformer files belonging to a reaction this run covers.

    Names are `{rxn}_synthon_{i}_conf_single_frag_{k}.{fmt}`. No reaction id contains
    `_synthon_`, so splitting on it recovers the reaction.
    """
    wanted = set(ctx.reactions)
    return sorted(
        f
        for f in ctx.single_frag_synthon_conf_dir.glob(f"*.{ctx.config['conf_file_format']}")
        if f.name.split("_synthon_")[0] in wanted
    )


def max_tasks(ctx: RunContext) -> int | None:
    """Return the number of dask tasks this task submits, one per conformer file.

    Return `None` if this run also generates the conformer files, because they do not exist yet.
    Used by `preflight.resolve_num_workers`.
    """
    if "generate_single_fragment_conformers" in ctx.tasks:
        return None
    return len(_conformer_files(ctx))


def _retention(ctx: RunContext) -> int:
    """How many scored synthons to carry forward. Not `top_m`.

    Product pseudo-scores are reordered by `rescore_products`, so that pipeline keeps a long
    list. Single-fragment scores are final, and the pose writer reads only `num_write_pose` rows.
    """
    return ctx.config["num_write_pose"]


def run(ctx: RunContext, client: DaskClient) -> None:
    """Score synthons against a single query fragment."""
    query_mol = _prepared_query(ctx)
    ctx.pseudo_res_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    keep_top = _retention(ctx)

    conf_files = _conformer_files(ctx)
    ctx.logger.info(f"Scoring {len(conf_files)} single-frag synthon files, keeping the top {keep_top:,}.")

    synthon_score_files = dask_score_single_fragment(
        client,
        conf_files=conf_files,
        query_fragment=query_mol,
        synthon_connectors=ctx.synthon_connectors,
        frag_connectors=ctx.frag_connectors,
        color_ff_dir=ctx.color_ff_dir,
        warts_separator=ctx.warts_id_sep,
        score_dir=ctx.pseudo_res_dir,
        keep_top=keep_top,
        merge_fanin=ctx.config["pseudo_res_merge_fanin"],
    )
    ctx.logger.info(
        f"Finished scoring single fragment. {len(synthon_score_files)} score files in {ctx.pseudo_res_dir}."
    )

    comb_synthon_res = client.combine_files(
        file_list=synthon_score_files,
        output_dir=ctx.pseudo_res_dir,
        top_m=keep_top,
        warts_separator=ctx.warts_id_sep,
        file_name="single_fragment",
        keep_isomer_warts=True,  # required here to extract overlay poses
    )
    with ctx.single_frag_pseudo_res_file_w.open("wb") as f:
        pickle.dump(comb_synthon_res, f)
    ctx.logger.info(
        f"Finished combining single fragment scores. Combined result saved to {ctx.single_frag_pseudo_res_file_w}."
    )
