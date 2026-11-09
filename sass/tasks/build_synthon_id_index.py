"""Build a library's string to integer synthon id index.

This is the first step of preparing a library, before `group_synthon_data` and
`generate_synthon_conformers`. Both read the index, so it is built during context setup.
"""

import logging
import pickle
from pathlib import Path

from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.library.synthons import build_synthon_id_index

logger = logging.getLogger(__name__)


def build_id_index(ctx: RunContext) -> list[str]:
    """Build the id index and save it to `synthon_id_index`."""
    out_file = ctx.config["synthon_id_index"]
    if not out_file:
        raise ValueError(
            "`build_synthon_id_index` needs `synthon_id_index` set to the path to write. It is "
            "the one library artifact this task produces."
        )

    index = build_synthon_id_index(ctx.config["synthon_file"])

    out_path = Path(out_file)
    if not out_path.is_absolute():
        out_path = ctx.exp_dir / out_path
    out_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    ctx.logger.info("Synthon id index (%s ids) saved to %s.", f"{len(index):,}", out_path)

    # Read it back to verify. A corrupt index silently maps ids to the wrong synthons.
    with out_path.open("rb") as f:
        if pickle.load(f) != index:
            raise ValueError(f"{out_path} did not round-trip; refusing to continue.")
    return index


def run(ctx: RunContext, client: DaskClient) -> None:
    """No-op: the index is built during context setup, since `SynthonHandler` needs it."""
