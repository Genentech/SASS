"""Enumerate every product of the library. This is the ground-truth path, for validation only."""

import logging
import math
import random
from pathlib import Path

from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.library.synthons import SynthonHandler
from sass.worker.instantiate_products import write_library_products_wrapper

logger = logging.getLogger(__name__)


def dask_instantiate_library_products(
    client: DaskClient,
    synthon_handler: SynthonHandler,
    grouped_synthons: dict,
    reaction_ids: list[str],
    output_dir: Path,
    product_chunk_size: int,  # at least 1000.
    out_file_suffix: str,
    sort_title: bool,
    title_separator: str,
    max_stereocenter: int,
) -> None:
    """Instantiate library products and write to file."""
    task_gen = generate_library_instantiation_tasks(
        grouped_synthons=grouped_synthons,
        synthon_handler=synthon_handler,
        reaction_ids=reaction_ids,
        chunk_size=product_chunk_size,
    )

    n = 0
    total_mol, total_iso = 0, 0
    for n_mol, n_iso in client.stream_map(
        write_library_products_wrapper,
        task_gen,
        output_dir=output_dir,
        out_file_suffix=out_file_suffix,
        sort_title=sort_title,
        title_separator=title_separator,
        max_stereocenter=max_stereocenter,
        chunk_size=product_chunk_size * 3,  # Assuming ~3 isomers per molecule.
    ):
        n += 1  # noqa: SIM113
        if n % client.target_max_worker == 0:
            logger.info(f"Finished {n} library instantiation tasks.")
        total_mol += n_mol
        total_iso += n_iso
    logger.info("Total of %s molecules and %s isomers written.", total_mol, total_iso)


def generate_library_instantiation_tasks(
    grouped_synthons: dict[str, dict],
    synthon_handler: SynthonHandler,
    reaction_ids: list[str],
    chunk_size: int,
):
    """Generate tasks for instantiating library products.

    Tasks are parallelized on the "1st dimension" of the synthon lists. i.e. for a
    reaction with n x m x k synthons, the n synthons are split such that each chunk
    contains ~`chunk_size` products.

    Parameters
    ----------
    synthon_handler
        SynthonHandler object.
    reaction_ids
        List of reaction ids.
    chunk_size
        Desired number of products to instantiate per chunk. Actual number will vary.
    """
    n = 0
    random.shuffle(reaction_ids)
    for rxn_id in reaction_ids:
        smirks = synthon_handler.get_reaction_smirks_by_id(rxn_id)
        reactants = grouped_synthons[rxn_id]
        if not all(reactants.values()):
            # A component filtered to empty would make `num_chunks` 0.
            logger.warning("Skipping %s: at least one component has no synthons.", rxn_id)
            continue
        num_products = math.prod([len(val) for val in reactants.values()])
        num_chunks = min(len(reactants[0]), math.ceil(num_products / chunk_size))
        this_chunk_size = math.ceil(len(reactants[0]) / num_chunks)
        other_reactant_sizes = [len(val) for key, val in reactants.items() if key != 0]
        other_size_text = "".join([f" x {val}" for val in other_reactant_sizes])
        logger.info(f"Splitting {rxn_id} into ~{num_chunks} chunks of size {this_chunk_size}{other_size_text}.")

        others = {key: val for key, val in reactants.items() if key != 0}
        for i in range(0, len(reactants[0]), this_chunk_size):
            # A fresh dict per chunk, so buffered tasks do not alias one object.
            yield ({**others, 0: reactants[0][i : i + this_chunk_size]}, smirks, n)
            n += 1
    logger.info(f"Instantiating a total of {n} chunks.")


def run(ctx: RunContext, client: DaskClient) -> None:
    """Enumerate every product of the library, for ground-truth validation."""
    ctx.logger.info("Starting full library enumeration.")
    ctx.full_enumeration_product_dir_w.mkdir(mode=0o755, parents=True, exist_ok=True)
    dask_instantiate_library_products(
        client,
        synthon_handler=ctx.shandler,
        grouped_synthons=ctx.grouped_synthons,
        reaction_ids=ctx.reactions,
        output_dir=ctx.full_enumeration_product_dir_w,
        product_chunk_size=ctx.config["product_chunk_size"],
        out_file_suffix="product.oeb.gz",
        sort_title=True,
        title_separator=ctx.product_id_sep,
        max_stereocenter=ctx.max_stereocenter,
    )
    ctx.logger.info(f"Finished full product enumeration. Chunks saved to {ctx.full_enumeration_product_dir_w}.")
