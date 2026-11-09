"""Scoring one reaction component against one query fragment, and combining the results.

Phase A (`score_synthon_fragment`) is the unit of ROCS work. Phase B (`combine_forder_task`)
turns a fragment order's per-component score lists into pseudo-products.
"""

import logging
from pathlib import Path
from typing import Any

from openeye import oechem

from sass.chem.mol import get_conn_symbols
from sass.chem.overlay import sp_simple_oeoverlay
from sass.ids import synthon_id_to_int
from sass.io import dump_score_file
from sass.library.conformers import stream_conformers
from sass.selection.combine import combine_forder_scores
from sass.selection.dedup import deduplicate_scores, score_cutoff

logger = logging.getLogger(__name__)


def score_synthon_fragment(
    data: tuple[Any],
    align_connector_atom: bool,
    synthon_connectors: list[str],
    rocs_key_opt: int,
    rocs_key_score: int,
    opt_shape_func: str,
    opt_color_func: str | None,
    overlap_shape_func: str,
    overlap_color_func: str | None,
    warts_separator: str,
    color_ff_dir: Path,
) -> list:
    """Phase A: score one reaction component against one query fragment.

    The same `(rxn_id, s_idx, fragment)` triple recurs across fragment orders and fragment
    sets, since a fragment from one cut reappears in every finer partition that makes the same
    cut. The caller memoizes by triple.

    Returns the score list for the component, deduplicated to the best isomer per synthon and
    sorted by descending score, as `combine_forder_scores` requires.
    """
    rxn_id, s_idx, frag, conf_files = data

    if not conf_files:
        # oechem does not raise on a missing file, only warns.
        logger.error("No conformer files for %s synthon %s.", rxn_id, s_idx)
        return []

    if align_connector_atom is False:
        _rocs_key_opt = _rocs_key_score = 0
    else:
        num_connector = len(get_conn_symbols(oechem.OEMolToSmiles(frag), synthon_connectors))
        _rocs_key_opt = num_connector if rocs_key_opt is None else rocs_key_opt
        _rocs_key_score = num_connector if rocs_key_score is None else rocs_key_score

    # Chain every chunk into one overlay call, so the ROCS setup is paid once per component.
    score = sp_simple_oeoverlay(
        ref_mol=frag,
        conf_file=None if len(conf_files) > 1 else conf_files[0],
        fit_mols=stream_conformers(conf_files) if len(conf_files) > 1 else None,
        overlay_opts=None,
        rocs_key_opt=_rocs_key_opt,
        rocs_key_score=_rocs_key_score,
        opt_shape_func=opt_shape_func,
        opt_color_func=opt_color_func,
        overlap_shape_func=overlap_shape_func,
        overlap_color_func=overlap_color_func,
        color_ff_dir=color_ff_dir,
    )
    score = deduplicate_scores(score, sort_scores=True, warts_separator=warts_separator)
    # Conformer titles are strings. Cast the ids to integers once per surviving synthon.
    return [(value, synthon_id_to_int(sid, conf_files[0])) for value, sid in score]


def combine_forder_task(
    data: tuple[Any],
    *component_scores: list,
    top_m: int,
    dedup_limit: int,
    warts_separator: str,
    synthon_comb_limit: int,
    agg_method: str,
    pseudo_res_dir: Path,
    cutoff: float | None = None,
) -> tuple:
    """Phase B: combine one fragment order's per-component scores into pseudo-products.

    `component_scores` are the Phase A results, one per entry of `synthon_order`, passed as
    dask futures. The task runs as soon as its own components are scored.

    ``cutoff`` is the best known lower bound on the final selection threshold for this
    reaction's component count (see `combine_forder_scores`).
    """
    rxn_id, fset_idx, forder_idx, synthon_order = data

    scores = dict(zip(synthon_order, component_scores, strict=True))
    scores, count = combine_forder_scores(
        scores,
        top_m,
        dedup_limit,
        warts_separator,
        synthon_comb_limit,
        agg_method,
        rxn_id,
        cutoff=cutoff,
    )

    fname = pseudo_res_dir / f"{rxn_id}_{fset_idx}_{forder_idx}_pseudo_res.pkl"
    dump_score_file(fname, scores)

    # A task that filled its own `top_m` quota also yields a valid cutoff. It holds `top_m`
    # distinct ids at or above its lowest score, so the global `top_m`-th score cannot be lower.
    local_cutoff = score_cutoff(scores, top_m)

    return ("score", rxn_id, fname, count, fname.stat().st_size, local_cutoff)
