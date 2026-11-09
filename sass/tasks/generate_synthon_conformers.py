"""OMEGA conformer generation for every synthon in the library."""

import logging
import math
from pathlib import Path
from typing import Any

from sass.chem.mol import get_conn_symbols
from sass.chem.synthon import label_synthon_ring_connector
from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.library.conformers import conformer_chunk_stems, merge_conformer_chunks_for_stem
from sass.library.synthons import SynthonHandler
from sass.worker.generate_conformers import (
    synthon_confgen_wrapper,
)

logger = logging.getLogger(__name__)


def dask_generate_synthon_conformers(
    client: DaskClient,
    reactions: list[str],
    grouped_synthons: dict,
    synthon_handler: SynthonHandler,
    ofile_format: str,
    connector_atoms: list[str],
    chunk_size: int,
    output_dir: Path,
    **kwargs: Any,
):
    """Generate synthon conformers and write to file.

    Parameters
    ----------
    reactions
        Reaction ids to be included for synthon conformer generation.
    grouped_synthons
        Synthons organized by reactions.
    synthon_handler
        Synthon handler containing synthon and reaction information.
    ofile_format
        Format of the conformer files, e.g. oez.
    connector_atoms
        Special atoms denoting a connector atom on synthons.
    chunk_size
        Max size of synthons to be processed by one worker.
    output_dir
        Directory to write conformer files to.
    kwargs
        OMEGA parameters.
    """
    task_gen = generate_synthon_confgen_tasks(
        reactions=reactions,
        grouped_synthons=grouped_synthons,
        synthon_handler=synthon_handler,
        connector_atoms=connector_atoms,
        chunk_size=chunk_size,
    )

    for n, _ in enumerate(
        client.stream_map(
            synthon_confgen_wrapper,
            task_gen,
            conf_format=ofile_format,
            output_dir=output_dir,
            **kwargs,
        )
    ):
        if (n + 1) % client.target_max_worker == 0:
            logger.info(f"Finished {n + 1} synthon conformer generation tasks. Workers: {client.worker_count()}.")


def generate_synthon_confgen_tasks(
    reactions: list[str],
    synthon_handler: SynthonHandler,
    grouped_synthons: dict,
    connector_atoms: list[str],
    chunk_size: int,
):
    """Generate tasks for synthon conformer generation.

    Parameters
    ----------
    reactions
        List of reaction ids to include.
    synthon_handler
        Synthon handler containing synthon and reaction information.
    grouped_synthons
        Synthons grouped by reaction id.
    connector_atoms
        Special atoms denoting a connector atom on synthons.
    chunk_size
        Chunk size of a synthon component list for each dask worker to process.
    """
    n = 0
    for rxn_id in reactions:
        rxn_synthons = grouped_synthons[rxn_id]
        eg_synthon_smiles = {}
        for key, syns in rxn_synthons.items():
            _, smi = syns[0]
            eg_synthon_smiles[key] = smi

        smirks = synthon_handler.get_reaction_smirks_by_id(rxn_id)
        synthon_ring_connector_dict = label_synthon_ring_connector(
            synthon_smiles=eg_synthon_smiles,
            connector_atoms=connector_atoms,
            smirks=smirks,
        )
        # Ensure that the connector_dict contains the same number of unique connector atoms as the entire reaction.
        expected_connectors = set()
        for smi in eg_synthon_smiles.values():
            for conn in get_conn_symbols(smi, connector_atoms):
                expected_connectors.add(conn)
        if expected_connectors != set(synthon_ring_connector_dict.keys()):
            raise ValueError(
                f"Reaction {rxn_id} expected connectors: {expected_connectors} != found connectors: "
                f"{set(synthon_ring_connector_dict.keys())}. \n"
                f"Failing this step usually indicates a problem with atom mapping in the reaction SMIRKS."
            )
        logger.info(f"Reaction {rxn_id} ring connector atoms: {synthon_ring_connector_dict}")

        for s_idx, synthons in rxn_synthons.items():
            other_synthons = {key: val for key, val in eg_synthon_smiles.items() if key != s_idx}
            num_chunk = math.ceil(len(synthons) / chunk_size)
            actual_chunk_size = math.ceil(len(synthons) / num_chunk)
            for chunk_idx, i in enumerate(range(0, len(synthons), actual_chunk_size)):
                yield (
                    rxn_id,
                    smirks,
                    s_idx,
                    synthons[i : i + actual_chunk_size],
                    other_synthons,
                    synthon_ring_connector_dict,
                    chunk_idx,
                )
                n += 1
            logger.info(f"Generated {num_chunk} tasks for {rxn_id} synthon {s_idx}.")

    logger.info(f"Generated a total of {n} synthon confgen tasks.")


def run(ctx: RunContext, client: DaskClient) -> None:
    """Build conformers for every synthon in the library, then merge the chunks."""
    ctx.synthon_conf_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    dask_generate_synthon_conformers(
        client,
        reactions=ctx.reactions,
        grouped_synthons=ctx.grouped_synthons,
        synthon_handler=ctx.shandler,
        ofile_format=ctx.conf_format,
        connector_atoms=ctx.synthon_connectors,
        chunk_size=ctx.config["synthon_confgen_chunk_size"],
        output_dir=ctx.synthon_conf_dir,
        omega_max_conf=ctx.config["synthon_omega_max_conf"],
        omega_max_time=ctx.config["synthon_omega_max_time"],
        omega_e_window=ctx.config["synthon_omega_energy"],
    )
    ctx.logger.info(f"Finished generating synthon conformers. Chunks saved to {ctx.synthon_conf_dir}.")

    # Merge chunks on the cluster, one task per (reaction, component) stem.
    stems = conformer_chunk_stems(ctx.synthon_conf_dir, ctx.conf_format)
    ctx.logger.info("Merging chunks for %s (reaction, component) files.", len(stems))
    merged = 0
    for stem, n_chunks in client.stream_map(
        merge_conformer_chunks_for_stem,
        iter(stems),
        file_dir=ctx.synthon_conf_dir,
        conf_format=ctx.conf_format,
    ):
        merged += 1
        ctx.logger.debug("%s: %s chunk(s) merged.", stem, n_chunks)
    ctx.logger.info("Merged %s of %s conformer files.", merged, len(stems))
