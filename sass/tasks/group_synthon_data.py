"""Group the raw synthon file by reaction and component, and build the synthon data table.

Runs on the head node with no cluster, as a library-preparation step.
"""

import logging
import pickle

from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.library.synthons import map_synthon_ids

logger = logging.getLogger(__name__)


def build_grouped_synthons(ctx: RunContext) -> dict:
    """Group the raw synthon file by reaction and component, and save it."""
    grouped_synthons = ctx.shandler.group_synthon_by_rxn_id(
        n_components=ctx.config["n_fragments"],
        rxn_ids=ctx.reactions,
        limit=ctx.config["synthon_limit"],
        random_seed=ctx.config["random_seed"],
        cleanup=False,
    )
    with ctx.grouped_synthon_w.open("wb") as f:
        pickle.dump(grouped_synthons, f)
    ctx.logger.info(f"Grouped synthons saved to {ctx.grouped_synthon_w}.")
    return grouped_synthons


def build_synthon_data(ctx: RunContext) -> dict:
    """Build the `{synthon_id: {SMILES, heavy_atom_count, rxn_id: {s_idx}}}` table, and save it."""
    ctx.logger.info("Start building synthon data dict.")
    synthon_data = map_synthon_ids(
        reactions=ctx.reactions,
        grouped_synthons=ctx.grouped_synthons,
        synthon_connectors=ctx.synthon_connectors,
    )
    with ctx.synthon_data_w.open("wb") as f:
        pickle.dump(synthon_data, f)
    ctx.logger.info(f"Synthon data dict saved to {ctx.synthon_data_w}!")
    return synthon_data


def run(ctx: RunContext, client: DaskClient) -> None:
    """No-op: both artifacts are built during context setup, since later tasks need them."""
