"""ROCS every fully enumerated product against the query. Validation path."""

import logging
import pickle
from pathlib import Path
from typing import Any

from openeye import oechem

from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.worker.rescore_products import (
    overlay_molecules_wrapper,
)

logger = logging.getLogger(__name__)


def dask_overlay_molecules(
    client: DaskClient,
    input_dir: Path,
    output_dir: Path,
    delete_conf_file: bool,
    conformer_suffix: str = "conf.oez",
    **kwargs: Any,  # Parameters for overlay
) -> list:
    task_gen = input_dir.glob(f"*{conformer_suffix}")

    score_files = []
    n = 0
    for res in client.stream_map(
        overlay_molecules_wrapper,
        task_gen,
        delete_conf_file=delete_conf_file,
        output_dir=output_dir,
        **kwargs,
    ):
        if not res:  # Guard against None returns.
            continue
        score_file, delta = res
        client.disk.add(delta)
        if score_file:
            n += 1
            if n % client.target_max_worker == 0:
                client.disk.log_progress(n)
            score_files.append(score_file)
    return score_files


def run(ctx: RunContext, client: DaskClient) -> None:
    """ROCS every fully enumerated product against the query, and combine the scores."""
    ctx.logger.info("Starting ROCS scoring for full library.")
    ctx.full_enumeration_score_dir_w.mkdir(mode=0o755, parents=True, exist_ok=True)
    client.disk.reset("full_enumeration")
    score_files = dask_overlay_molecules(
        client,
        input_dir=ctx.full_enumeration_conf_dir_r,
        output_dir=ctx.full_enumeration_score_dir_w,
        delete_conf_file=False,
        ref_mol=oechem.OEMol(ctx.query_mol),
        opt_shape_func=ctx.opt_shape_func,
        opt_color_func=ctx.opt_color_func,
        overlap_shape_func=ctx.overlap_shape_func,
        overlap_color_func=ctx.overlap_color_func,
        color_ff_dir=ctx.color_ff_dir,
        warts_separator=ctx.warts_id_sep,
        top_n=5000,
    )
    ctx.logger.info(f"Finished ROCS scoring for full library. Chunks saved to {ctx.full_enumeration_score_dir_w}.")

    combined_scores = client.combine_files(
        file_list=score_files,
        output_dir=ctx.full_enumeration_score_dir_w,
        top_m=5000,
        warts_separator=ctx.warts_id_sep,
        file_name="ground_truth_score",
        keep_isomer_warts=True,  # True is optional here. Set to False for cleaner output.
    )
    combined_scores_file = ctx.exp_dir / "ground_truth_combined_rocs_res.pkl"
    with combined_scores_file.open("wb") as f:
        pickle.dump(combined_scores, f)
    ctx.logger.info(f"Finished combining ROCS scores. Combined result saved to {combined_scores_file}.")
