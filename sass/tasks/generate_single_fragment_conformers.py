"""Rewrite synthon conformers keeping one connector atom each, for single-fragment scoring.

Experimental. A synthon with two connectors becomes two conformer files.
"""

import logging
from pathlib import Path

from sass import EXTEND_ATOM_ISOTOPE, EXTEND_ATOM_SYMBOL
from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.worker.generate_conformers import (
    generate_single_frag_synthon_conformers,
)

logger = logging.getLogger(__name__)


def _input_files(input_dir: Path) -> list[Path]:
    """Return the regular conformer files to convert. Each file is one dask task."""
    return sorted(input_dir.glob("*oez"))


def dask_generate_single_frag_synthon_conformers(
    client: DaskClient,
    input_dir: Path,
    output_dir: Path,
    synthon_connectors: list[str],
    target_connector_atom: str,
    extend_bond: bool,
    extend_atom_symbol: str,
    extend_atom_isotope: int,
    extend_bond_length: float,
    chunk_size: int,
):
    """Generate conformers suitable for single-fragment scoring from regular synthon conformers.

    Each output molecule keeps a single connector atom, so a synthon with several connectors
    yields several outputs. The connection bond can be extended with a dummy atom to allow
    aligning on the bond during scoring.

    Parameters
    ----------
    input_dir
        Directory containing regular synthon conformer files.
    output_dir
        Directory to write single-fragment synthon conformers to.
    synthon_connectors
        List of connector atom symbols in the regular synthons.
    target_connector_atom
        The connector atom symbol to be used for single-fragment synthon conformers.
    extend_bond
        Whether to extend the connection bond with an additional dummy atom.
    extend_atom_symbol
        Symbol of the dummy atom to extend the connection bond with.
    extend_atom_isotope
        Isotope of the dummy atom to extend the connection bond with.
    extend_bond_length
        Length of the extended bond.
    chunk_size
        Max chunk size of generated conformers.
    """
    for n, _ in enumerate(
        client.stream_map(
            generate_single_frag_synthon_conformers,
            iter(_input_files(input_dir)),
            output_dir=output_dir,
            synthon_connectors=synthon_connectors,
            target_connector_atom=target_connector_atom,
            extend_bond=extend_bond,
            extend_atom_symbol=extend_atom_symbol,
            extend_atom_isotope=extend_atom_isotope,
            extend_bond_length=extend_bond_length,
            chunk_size=chunk_size,
        )
    ):
        if (n + 1) % client.target_max_worker == 0:
            logger.info(
                "Finished %s single-fragment synthon conformer generation tasks. Workers: %s.",
                n + 1,
                client.worker_count(),
            )


def max_tasks(ctx: RunContext) -> int | None:
    """Return the number of dask tasks this task submits, one per regular conformer file.

    Return `None` if this run also generates the regular conformer files, because they do not
    exist yet. Used by `preflight.resolve_num_workers`.
    """
    if "generate_synthon_conformers" in ctx.tasks:
        return None
    return len(_input_files(ctx.synthon_conf_dir))


def run(ctx: RunContext, client: DaskClient) -> None:
    """Rewrite the synthon conformers keeping one connector atom each."""
    ctx.logger.info("Starting generation of conformers for single fragment scoring.")
    ctx.single_frag_synthon_conf_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    dask_generate_single_frag_synthon_conformers(
        client,
        input_dir=ctx.synthon_conf_dir,
        output_dir=ctx.single_frag_synthon_conf_dir,
        synthon_connectors=ctx.synthon_connectors,
        target_connector_atom=ctx.synthon_connectors[0],
        extend_bond=ctx.config["align_on_bond"],
        extend_atom_symbol=EXTEND_ATOM_SYMBOL,
        extend_atom_isotope=EXTEND_ATOM_ISOTOPE,
        extend_bond_length=ctx.config["extend_bond_distance"],
        chunk_size=300,
    )
    ctx.logger.info(
        "Finished generating conformers for single fragment scoring. "
        f"Synthon conformers saved to {ctx.single_frag_synthon_conf_dir}."
    )
