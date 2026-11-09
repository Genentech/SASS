"""Write overlay poses for the top single-fragment hits. Experimental."""

import logging
import pickle
from pathlib import Path
from typing import Callable

from openeye import oechem

from sass.chem.mol import read_mols
from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.ids import restore_original_synthon_ids
from sass.io import load_intermediate_files
from sass.tasks.score_single_fragments import _conformer_files, _prepared_query
from sass.tasks.score_single_fragments import max_tasks as _scoring_max_tasks
from sass.worker.rescore_products import (
    extract_mols_and_score,
)

logger = logging.getLogger(__name__)


def extract_conformers_and_score(
    client: DaskClient,
    conf_files: list[Path],
    product_set_file: Path,
    ref_mol: oechem.OEMol | oechem.OEGraphMol,
    opt_shape_func: str,
    opt_color_func: str | None,
    overlap_shape_func: str,
    overlap_color_func: str | None,
    color_ff_dir: Path,
    rocs_key: int,
    out_dir: Path,
    sep_shape_color_score: bool,
    remove_input_file: bool,
) -> list[Path]:
    """Extract the selected synthons' conformers from conformer files, and score them.

    After single-fragment scoring, the top synthons are selected for writing out poses.
    Their conformers are read back from the stored single-fragment conformer files rather
    than regenerated.

    Parameters
    ----------
    conf_files
        Conformer files to search for the top products.
    product_set_file
        File containing the set of product ids to extract from all conformer files.
    ref_mol
        Reference molecule for overlay scoring.
    **_shape/color_func:
        Shape and color functions for overlay optimization and overlap scoring.
    color_ff_dir
        Directory containing custom color interaction files.
    rocs_key
        Key for ROCS option for overlay optimization.
    out_dir
        Directory to write out overlay poses.
    sep_shape_color_score
        Whether to write out separate shape and color scores in the output.
    remove_input_file
        Whether to remove the input conformer file after scoring.

    Returns
    -------
        A list of best overlaid conformer file paths.
    """
    out = []
    n = 0
    for out_file, delta in client.stream_map(
        extract_mols_and_score,
        iter(conf_files),
        product_set_file=product_set_file,
        ref_mol=ref_mol,
        opt_shape_func=opt_shape_func,
        opt_color_func=opt_color_func,
        overlap_shape_func=overlap_shape_func,
        overlap_color_func=overlap_color_func,
        color_ff_dir=color_ff_dir,
        rocs_key=rocs_key,
        out_dir=out_dir,
        sep_shape_color_score=sep_shape_color_score,
        remove_input_file=remove_input_file,
    ):
        client.disk.add(delta)
        if out_file:
            n += 1
            if n % client.target_max_worker == 0:
                client.disk.log_progress(n)
            out.append(out_file)
    return out


def max_tasks(ctx: RunContext) -> int | None:
    """Return the number of dask tasks this task submits. It reads the same files as scoring."""
    return _scoring_max_tasks(ctx)


def run(ctx: RunContext, client: DaskClient) -> None:
    """Write overlay poses for the top single-fragment hits."""
    query_mol = _prepared_query(ctx)
    num_pose = ctx.config["num_write_pose"]

    scores = load_intermediate_files(ctx.single_frag_pseudo_res_file_r)[:num_pose]
    top_synthon_set_f = ctx.exp_dir / "top_single_frag_synthon_set.pkl"
    with top_synthon_set_f.open("wb") as f:
        pickle.dump({ele[1] for ele in scores}, f)

    overlay_files = extract_conformers_and_score(
        client,
        # Search only the files that scoring read. The top hits can only come from those files.
        conf_files=_conformer_files(ctx),
        product_set_file=top_synthon_set_f,
        ref_mol=query_mol,
        opt_shape_func=ctx.opt_shape_func,
        opt_color_func=ctx.opt_color_func,
        overlap_shape_func=ctx.overlap_shape_func,
        overlap_color_func=ctx.overlap_color_func,
        color_ff_dir=ctx.color_ff_dir,
        rocs_key=1,
        out_dir=ctx.pseudo_res_dir,
        sep_shape_color_score=ctx.config["sep_shape_color_score"],
        remove_input_file=False,
    )

    # Restore the original library ids in the final SDF only. Single-fragment titles are
    # `<int id>_<wart>[<connector>]`, which `restore_original_synthon_ids` handles.
    retitle = None
    if ctx.config["restore_synthon_ids_in_poses"] and ctx.config["synthon_id_index"] is not None:
        id_index = load_intermediate_files(ctx.config["synthon_id_index"])
        ctx.logger.info("Pose titles will use the original synthon ids from the id index.")

        def retitle(title: str) -> str:
            return restore_original_synthon_ids(title, id_index, ctx.product_id_sep, ctx.warts_id_sep)

    overlay_pose_f = ctx.exp_dir / f"top_{num_pose}_single_frag_synthon_poses.sdf"
    n_unique = _concat_unique_poses(overlay_files, overlay_pose_f, query_mol, retitle=retitle)
    for f in overlay_files:
        f.unlink()
    ctx.logger.info(f"Top {n_unique} unique single frag synthon overlay poses saved to {overlay_pose_f}.")


def _concat_unique_poses(
    overlay_files: list[Path],
    out_file: Path,
    query_mol: oechem.OEMolBase,
    retitle: Callable[[str], str] | None = None,
) -> int:
    """Concatenate per-chunk pose files into one SDF, dropping duplicate structures.

    Poses are read as `OEGraphMol`, since an `OEMol` conformer's own title would override
    `SetTitle` at write time.
    """
    ostream = oechem.oemolostream(str(out_file))
    oechem.OEWriteMolecule(ostream, query_mol)
    n_unique, seen_smi = 0, set()
    for f in overlay_files:
        with read_mols(f, graph=True) as mols:
            for mol in mols:
                smi = oechem.OEMolToSmiles(mol)
                if smi in seen_smi:
                    continue
                seen_smi.add(smi)
                if retitle is not None:
                    mol.SetTitle(retitle(mol.GetTitle()))
                oechem.OEWriteMolecule(ostream, mol)
                n_unique += 1
    ostream.close()
    return n_unique
