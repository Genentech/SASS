"""Instantiating a product, building its conformers and scoring it, in one task.

Conformers are not written to disk, to reduce temporary disk usage.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Iterable

from openeye import oechem, oeomega

from sass.chem.confgen import build_conformers, omega_opt_builder
from sass.chem.libgen import (
    flipper_opt_builder,
    generate_stereoisomers,
    instantiate_singleton_product,
)
from sass.chem.mol import read_mols
from sass.chem.overlay import sp_simple_oeoverlay
from sass.ids import extract_base_id
from sass.io import dump_score_file, file_size, load_score_file
from sass.selection.dedup import deduplicate_scores

logger = logging.getLogger(__name__)


def instantiate_confgen_score(
    data: Any,
    output_dir: Path,
    ref_mol: oechem.OEMolBase,
    opt_shape_func: str,
    opt_color_func: str | None,
    overlap_shape_func: str,
    overlap_color_func: str | None,
    color_ff_dir: Path,
    warts_separator: str,
    title_separator: str,
    max_stereocenter: int,
    top_n: int,
    omega_max_conf: int,
    omega_max_time: int,
    omega_e_window: int,
    pose_set_file: Path | None = None,
    sep_shape_color_score: bool = False,
) -> tuple:
    """Instantiate, generate conformers for, and ROCS-score one chunk of products.

    Conformers are never written to disk. The few needed for poses are regenerated later.
    Molecules are streamed one at a time, so peak memory does not depend on chunk size.

    Parameters
    ----------
    data
        `(pid_data, idx)`, where `pid_data` is a list of `(reactants, smirks, rxn_id)`.
    output_dir
        Directory for the score file (and pose file, in pose mode).
    ref_mol
        Query molecule to score against.
    **_shape/color_func, color_ff_dir
        ROCS scoring configuration.
    warts_separator, title_separator
        Isomer-wart separator, and separator joining synthon ids into a product id.
    max_stereocenter
        Cap on stereocenters enumerated by FLIPPER.
    top_n
        Number of top scores to keep from this chunk.
    omega_max_conf, omega_max_time, omega_e_window
        OMEGA configuration.
    pose_set_file
        Pose mode. A pickled set of isomer titles. Only those isomers are built, and their
        best overlays are written to an `.oez` beside the score file.
    sep_shape_color_score
        Whether to record separate shape and color scores on the written poses.

    Returns
    -------
        `(score_file, pose_file_or_None, n_bytes)`.
    """
    pid_data, idx = data
    score_file = output_dir / f"chunk_{idx}.score.pkl"
    pose_file = output_dir / f"chunk_{idx}_poses.oez" if pose_set_file is not None else None
    pose_set = load_score_file(pose_set_file) if pose_set_file is not None else None

    rxn_by_product: dict[str, str] = {}

    def _isomer_stream() -> Iterable[oechem.OEMol]:
        """Yield the stereoisomers to build conformers for; nothing is held or written."""
        flipper_opt = flipper_opt_builder(max_stereocenter=max_stereocenter)
        for reactants, smirks, rxn_id in pid_data:
            mol = instantiate_singleton_product(reactants, smirks, sort_title=True, title_separator=title_separator)
            rxn_by_product[mol.GetTitle()] = rxn_id
            for isomer in generate_stereoisomers(mol, flipper_opt):
                if pose_set is not None and isomer.GetTitle() not in pose_set:
                    continue  # Pose mode: only the specific isomers that were selected.
                yield isomer

    def _mol_stream() -> Iterable[oechem.OEMol]:
        """Instantiate, enumerate isomers and build conformers, one molecule at a time."""
        omega = oeomega.OEOmega(
            omega_opt_builder(
                max_conf=omega_max_conf,
                max_search_time=omega_max_time,
                energy_window=omega_e_window,
            )
        )
        yield from build_conformers(_isomer_stream(), omega)

    scores = sp_simple_oeoverlay(
        ref_mol=ref_mol,
        conf_file=None,
        fit_mols=_mol_stream(),
        overlay_opts=None,
        rocs_key_opt=0,
        rocs_key_score=0,
        opt_shape_func=opt_shape_func,
        opt_color_func=opt_color_func,
        overlap_shape_func=overlap_shape_func,
        overlap_color_func=overlap_color_func,
        color_ff_dir=color_ff_dir,
        write_out_overlay=pose_set is not None,
        out_file=pose_file,
        keep_color=False,
        sep_shape_color_score=sep_shape_color_score,
    )
    scores = deduplicate_scores(
        scores, sort_scores=True, keep_isomer_warts=True, warts_separator=warts_separator, limit=top_n
    )

    # Stamp each row with its reaction, which pose writing needs to re-instantiate products.
    # The generator is fully consumed by now, so `rxn_by_product` is complete.
    rows = [
        (
            score,
            title,
            rxn_by_product[extract_base_id(title, warts_separator=warts_separator, keep_isomer_warts=False)],
        )
        for score, title in scores
    ]
    if pose_set is None:
        dump_score_file(score_file, rows)
    else:
        # Pose mode only needs the poses, so no score file is written.
        score_file = None
    logger.info("Instantiated, built conformers for and scored %s products (chunk %s).", len(pid_data), idx)

    return score_file, pose_file, file_size(score_file) + file_size(pose_file)


def extract_mols_and_score(
    data: list[Path],  # list[conf_file]
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
) -> Path:
    conf_file = data
    with product_set_file.open("rb") as f:
        product_set = pickle.load(f)  # Internal intermediate file, never external input.
    with read_mols(conf_file) as mols:
        matched_mols = [oechem.OEMol(mol) for mol in mols if mol.GetTitle() in product_set]
    freed = 0
    if remove_input_file:
        freed = file_size(conf_file)
        conf_file.unlink()

    if not matched_mols:
        return None, -freed

    # Score and write out overlay poses.
    out_file = out_dir / f"{conf_file.stem}_extracted.oez"
    _ = sp_simple_oeoverlay(
        ref_mol=oechem.OEMol(ref_mol),
        conf_file=None,
        overlay_opts=None,
        fit_mols=matched_mols,
        rocs_key_opt=rocs_key,
        rocs_key_score=rocs_key,
        opt_shape_func=opt_shape_func,
        opt_color_func=opt_color_func,
        overlap_shape_func=overlap_shape_func,
        overlap_color_func=overlap_color_func,
        color_ff_dir=color_ff_dir,
        num_hits=None,
        write_out_overlay=True,
        out_file=out_file,
        keep_color=False,
        sep_shape_color_score=sep_shape_color_score,
    )

    return out_file, file_size(out_file) - freed


def overlay_molecules_wrapper(
    data: Any,
    delete_conf_file: bool,
    output_dir: Path,
    ref_mol: oechem.OEMolBase,
    opt_shape_func: str,
    opt_color_func: str | None,
    overlap_shape_func: str,
    overlap_color_func: str | None,
    color_ff_dir: Path,
    warts_separator: str,
    top_n: int,
):
    conformer_file = data
    base_name = conformer_file.name.split(".")[0]
    score_file = output_dir / f"{base_name}.score.pkl"

    if score_file.exists():  # For occasional re-runs.
        logger.info(f"Score file {score_file} already exists, skipping.")
        return score_file, 0

    score = sp_simple_oeoverlay(
        ref_mol=ref_mol,
        conf_file=conformer_file,
        overlay_opts=None,
        rocs_key_opt=0,
        rocs_key_score=0,
        opt_shape_func=opt_shape_func,
        opt_color_func=opt_color_func,
        overlap_shape_func=overlap_shape_func,
        overlap_color_func=overlap_color_func,
        color_ff_dir=color_ff_dir,
    )

    score = deduplicate_scores(
        score, sort_scores=True, keep_isomer_warts=True, warts_separator=warts_separator, limit=top_n
    )

    with score_file.open("wb") as f:
        pickle.dump(score, f)
    logger.info(f"ROCS finished for {base_name}. {len(score)} molecules.")

    delta = file_size(score_file)
    if delete_conf_file:
        delta -= file_size(conformer_file)
        conformer_file.unlink()

    return score_file, delta
