"""Re-instantiate the selected pseudo-products, build conformers, and score with ROCS.

One fused task per chunk without writing conformers to disk.
"""

import logging
import pickle
import random
from pathlib import Path
from typing import Any

from openeye import oechem

from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.io import discard_superseded_dir, load_intermediate_files
from sass.library.synthons import SynthonHandler
from sass.tasks.instantiate_products import generate_instantiation_tasks, rescore_chunk_size
from sass.worker.rescore_products import (
    instantiate_confgen_score,
)

logger = logging.getLogger(__name__)


def dask_instantiate_confgen_score(
    client: DaskClient,
    product_list: list,
    instantiation_chunk_size: int,
    shandler: SynthonHandler,
    synthon_data: dict,
    warts_separator: str,
    title_separator: str,
    output_dir: Path,
    **kwargs: Any,  # ROCS, OMEGA and pose parameters for `instantiate_confgen_score`
) -> tuple[list[Path], list[Path]]:
    """Instantiate, generate conformers for and score products in a single pass.

    Conformers are never written, so the phase writes only score files, plus pose files when
    a `pose_set_file` is supplied.

    Returns
    -------
        Score files, and pose files (empty unless a `pose_set_file` was supplied).
    """
    task_gen = generate_instantiation_tasks(
        product_list=product_list,
        chunk_size=instantiation_chunk_size,
        shandler=shandler,
        synthon_data=synthon_data,
        warts_separator=warts_separator,
        title_separator=title_separator,
    )

    score_files, pose_files = [], []
    n = 0
    for res in client.stream_map(
        instantiate_confgen_score,
        task_gen,
        output_dir=output_dir,
        warts_separator=warts_separator,
        title_separator=title_separator,
        **kwargs,
    ):
        if not res:  # Guard against None returns.
            continue
        score_file, pose_file, nbytes = res
        client.disk.add(nbytes)
        n += 1
        if n % client.target_max_worker == 0:
            client.disk.log_progress(n)
        if score_file is not None:
            score_files.append(score_file)
        if pose_file is not None and pose_file.is_file():
            pose_files.append(pose_file)
    return score_files, pose_files


def run(ctx: RunContext, client: DaskClient) -> None:
    """Re-instantiate, build conformers for, and ROCS the selected pseudo-products."""
    ctx.logger.info("Start instantiating, generating conformers for, and rescoring pseudo products.")
    ctx.rescore_res_dir_w.mkdir(mode=0o755, parents=True, exist_ok=True)
    client.disk.reset("rescore_results")

    products = load_intermediate_files(ctx.combined_pseudo_res_file_r)
    random.shuffle(products)
    ctx.logger.info(f"Combined pseudo products loaded from {ctx.combined_pseudo_res_file_r}.")

    # One fused task covers instantiation, OMEGA and ROCS for its chunk, so a single chunk
    # size serves all three.
    rescored_score_files, _ = dask_instantiate_confgen_score(
        client,
        product_list=products,
        instantiation_chunk_size=rescore_chunk_size(ctx, client, len(products)),
        shandler=ctx.shandler,
        synthon_data=ctx.synthon_data,
        title_separator=ctx.product_id_sep,
        warts_separator=ctx.warts_id_sep,
        output_dir=ctx.rescore_res_dir_w,
        ref_mol=oechem.OEMol(ctx.query_mol),
        opt_shape_func=ctx.opt_shape_func,
        opt_color_func=ctx.opt_color_func,
        overlap_shape_func=ctx.overlap_shape_func,
        overlap_color_func=ctx.overlap_color_func,
        color_ff_dir=ctx.color_ff_dir,
        max_stereocenter=ctx.max_stereocenter,
        top_n=ctx.config["num_final_list"],
        omega_max_conf=ctx.config["product_omega_max_conf"],
        omega_max_time=ctx.config["product_omega_max_time"],
        omega_e_window=ctx.config["product_omega_energy"],
    )
    ctx.logger.info(f"Finished rescoring pseudo products. Scores saved to {ctx.rescore_res_dir_w}.")

    rescored_scores = client.combine_files(
        file_list=rescored_score_files,
        output_dir=ctx.rescore_res_dir_w,
        top_m=ctx.config["num_final_list"],
        warts_separator=ctx.warts_id_sep,
        file_name="rescore",
        keep_isomer_warts=True,  # required here to extract overlay poses
    )
    with ctx.rescore_file_w.open("wb") as f:
        pickle.dump(rescored_scores, f)
    ctx.logger.info(f"Finished combining rescored scores. Combined result saved to {ctx.rescore_file_w}.")
    discard_superseded_dir(
        ctx.rescore_res_dir_w,
        ctx.rescore_file_w,
        enabled=ctx.config["clean_up_intermediate_files"],
    )
    client.disk.log_peak()
