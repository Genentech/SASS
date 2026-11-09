"""Single-fragment scoring: synthon conformers scored against one query fragment with a single connector."""

import logging
import pickle
from pathlib import Path

from openeye import oechem

from sass.chem.mol import get_conn_symbols, load_first_mol
from sass.chem.overlay import sp_simple_oeoverlay
from sass.selection.dedup import deduplicate_scores

logger = logging.getLogger(__name__)


def score_single_fragment(
    data: list[Path],
    query_fragment: oechem.OEMolBase,
    synthon_connectors: set[str],
    frag_connectors: set[str],
    color_ff_dir: Path,
    warts_separator: str,
    score_dir: Path,
) -> tuple[Path, int] | None:
    conf_file = data
    outfile = score_dir / f"{conf_file.name}.pkl"
    sample_conf = load_first_mol(conf_file, mol_cleanup=False)
    synthon_conns = get_conn_symbols(oechem.OEMolToSmiles(sample_conf), synthon_connectors)
    frag_conns = get_conn_symbols(oechem.OEMolToSmiles(query_fragment), frag_connectors + synthon_connectors)
    if len(synthon_conns) == 1 and len(frag_conns) == 1:  # currently limited to 1 connector
        score = sp_simple_oeoverlay(
            ref_mol=query_fragment,
            conf_file=conf_file,
            overlay_opts=None,
            rocs_key_opt=1,
            rocs_key_score=1,
            # Fixed to TanimotoCombo. The `*_shape_func` / `*_color_func` config keys do not apply to this mode.
            opt_shape_func="GetTanimoto",
            opt_color_func="GetColorTanimoto",
            overlap_shape_func="GetTanimoto",
            overlap_color_func="GetColorTanimoto",
            color_ff_dir=color_ff_dir,
        )
        score = deduplicate_scores(score, sort_scores=True, warts_separator=warts_separator, keep_isomer_warts=True)

        with outfile.open("wb") as f:
            pickle.dump(score, f)
        return outfile, outfile.stat().st_size

    # With more than one connector on either side there is no single bond vector to align on,
    # so the synthon is skipped.
    return None
