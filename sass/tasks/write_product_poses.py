"""Regenerate and write overlay poses for the top rescored products."""

import logging
import pickle
from pathlib import Path

from openeye import oechem

from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.ids import product_id_to_sids, restore_original_synthon_ids
from sass.io import discard_superseded_dir, load_intermediate_files
from sass.tasks.rescore_products import dask_instantiate_confgen_score
from sass.tasks.write_single_fragment_poses import _concat_unique_poses

logger = logging.getLogger(__name__)


def _require_rxn_id_column(scores: list, file: Path) -> None:
    """Fail clearly if a rescore file has no reaction-id column.

    Pose writing re-instantiates the top products, which needs to know the reaction each
    product came from.
    """
    if scores and len(scores[0]) < 3:
        raise ValueError(
            f"{file} has no reaction id column. Pose writing re-instantiates the top "
            "products, which needs the reaction each product came from. Re-run "
            "`rescore_products` to regenerate this file."
        )


def run(ctx: RunContext, client: DaskClient) -> None:
    """Regenerate conformers for the top rescored products and write their overlay poses."""
    num_pose = ctx.config["num_write_pose"]
    scores = load_intermediate_files(ctx.rescore_file_r)[:num_pose]
    _require_rxn_id_column(scores, ctx.rescore_file_r)

    # `rescore_products` may not have run in the same invocation.
    ctx.rescore_res_dir_w.mkdir(mode=0o755, parents=True, exist_ok=True)

    top_isomer_set_f = ctx.exp_dir / "top_isomer_set.pkl"
    with top_isomer_set_f.open("wb") as f:
        pickle.dump({ele[1] for ele in scores}, f)

    # Rebuild conformers for just the top products, since conformers are not stored.
    ctx.logger.info(f"Regenerating conformers for the top {len(scores)} isomers.")
    top_products = [
        (score, product_id_to_sids(pid, warts_separator=ctx.warts_id_sep, title_separator=ctx.product_id_sep), rxn_id)
        for score, pid, rxn_id in scores
    ]
    _, overlay_files = dask_instantiate_confgen_score(
        client,
        product_list=top_products,
        instantiation_chunk_size=ctx.config["dask_rescore_max_chunk_size"],
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
        top_n=num_pose,
        omega_max_conf=ctx.config["product_omega_max_conf"],
        omega_max_time=ctx.config["product_omega_max_time"],
        omega_e_window=ctx.config["product_omega_energy"],
        pose_set_file=top_isomer_set_f,
        sep_shape_color_score=ctx.config["sep_shape_color_score"],
    )

    # Restore the original library ids in the final output only. Everything upstream stays
    # numeric.
    retitle = None
    if ctx.config["restore_synthon_ids_in_poses"] and ctx.config["synthon_id_index"] is not None:
        id_index = load_intermediate_files(ctx.config["synthon_id_index"])
        ctx.logger.info("Pose titles will use the original synthon ids from the id index.")

        def retitle(title: str) -> str:
            return restore_original_synthon_ids(title, id_index, ctx.product_id_sep, ctx.warts_id_sep)

    overlay_pose_f = ctx.exp_dir / f"rescore_top_{num_pose}_mol_poses.sdf"
    n_unique = _concat_unique_poses(overlay_files, overlay_pose_f, ctx.query_mol, retitle=retitle)
    for f in overlay_files:
        f.unlink()
    top_isomer_set_f.unlink()

    ctx.logger.info(f"Top {n_unique} unique rescored molecule overlay poses saved to {overlay_pose_f}.")
    # The overlay files above have all been unlinked, so what is left is an empty scratch dir.
    discard_superseded_dir(
        ctx.rescore_res_dir_w,
        overlay_pose_f,
        enabled=ctx.config["clean_up_intermediate_files"],
    )
    client.disk.log_peak()
