"""Enumerate the selected pseudo-products to disk, without rescoring them."""

import logging
import math
import random
from pathlib import Path

from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.ids import product_id_to_sids
from sass.io import load_intermediate_files
from sass.library.synthons import SynthonHandler
from sass.worker.instantiate_products import write_singleton_products_wrapper

logger = logging.getLogger(__name__)


def dask_instantiate_singleton_products(
    client: DaskClient,
    product_list: list,
    instantiation_chunk_size: int,
    file_chunk_size: int,
    shandler: SynthonHandler,
    synthon_data: dict,
    warts_separator: str,
    title_separator: str,
    output_dir: Path,
    product_file_suffix: str,
    max_stereocenter: int,
) -> None:
    """Instantiate selected products based on synthon-scores.

    Parameters
    ----------
    product_list
        List of product ids to instantiate.
    instantiation_chunk_size
        Chunk size for each dask worker to process.
    file_chunk_size
        Target chunk size for actual file writing.
    shandler
        Synthon handler containing synthon and reaction information.
    synthon_data
        Grouped synthons organized by reactions.
    warts_separator
        String separator for naming isomers, conformers.
    title_separator
        String separator for joining synthon ids to form product ids.
    output_dir
        Directory to write product files to.
    product_file_suffix
        Suffix for product files.
    max_stereocenter
        Maximum number of stereocenters to consider when enumerating isomers.
    """
    task_gen = generate_instantiation_tasks(
        product_list=product_list,
        chunk_size=instantiation_chunk_size,
        shandler=shandler,
        synthon_data=synthon_data,
        warts_separator=warts_separator,
        title_separator=title_separator,
    )

    for n, delta in enumerate(
        client.stream_map(
            write_singleton_products_wrapper,
            task_gen,
            output_dir=output_dir,
            out_file_suffix=product_file_suffix,
            title_separator=title_separator,
            file_chunk_size=file_chunk_size,
            max_stereocenter=max_stereocenter,
        )
    ):
        client.disk.add(delta or 0)
        if (n + 1) % client.target_max_worker == 0:
            client.disk.log_progress(n + 1)


def generate_instantiation_tasks(
    product_list: list,
    chunk_size: int,
    shandler: SynthonHandler,
    synthon_data: dict,
    warts_separator: str,
    title_separator: str,
):
    logger.info(f"Splitting pseudo-product list into {chunk_size}/chunk for instantiation (+ confgen/ROCS scoring).")
    chunk = []
    n = 0
    for data in product_list:
        pid, rxn_id = data[1], data[2]
        sids = product_id_to_sids(pid, warts_separator=warts_separator, title_separator=title_separator)
        reactants = []
        for sid in sids:
            smi = synthon_data[sid]["SMILES"]
            sorder = synthon_data[sid][rxn_id]["s_idx"]
            reactants.append((sid, smi, sorder))
        this_smirks = shandler.get_reaction_smirks_by_id(rxn_id)
        # `rxn_id` rides along so the rescore task can stamp it on its rows for pose writing.
        chunk.append((reactants, this_smirks, rxn_id))
        if len(chunk) >= chunk_size:
            if n % 1000 == 0:
                logger.info(f"Start instantiating chunk {n}")
            yield (chunk, n)
            chunk = []
            n += 1
    if chunk:
        logger.info(f"Start instantiating chunk {n}")
        yield (chunk, n)
        chunk = []
        n += 1
    logger.info("Instantiating a total of %s chunks.", n)


def rescore_chunk_size(ctx: RunContext, client: DaskClient, n_products: int) -> int:
    """Chunk size for the rescore-family tasks, scaled to the cluster.

    Small enough that every worker gets work, but floored so each chunk is worth the fixed
    ROCS setup.
    """
    min_chunk_size = max(ctx.config["dask_rescore_min_chunk_size"], math.ceil(n_products / client.target_max_worker))
    return min(ctx.config["dask_rescore_max_chunk_size"], min_chunk_size)


def run(ctx: RunContext, client: DaskClient) -> None:
    """Enumerate the selected pseudo-products to disk, without rescoring them."""
    ctx.logger.info("Start instantiating pseudo products.")
    ctx.rescore_res_dir_w.mkdir(mode=0o755, parents=True, exist_ok=True)
    client.disk.reset("instantiated_products")

    products = load_intermediate_files(ctx.combined_pseudo_res_file_r)
    random.shuffle(products)
    ctx.logger.info(f"Combined pseudo products loaded from {ctx.combined_pseudo_res_file_r}.")

    dask_instantiate_singleton_products(
        client,
        product_list=products,
        instantiation_chunk_size=1000,  # Manually set: too small a chunk size leads to a `dask` error.
        file_chunk_size=rescore_chunk_size(ctx, client, len(products)),
        shandler=ctx.shandler,
        synthon_data=ctx.synthon_data,
        title_separator=ctx.product_id_sep,
        warts_separator=ctx.warts_id_sep,
        output_dir=ctx.rescore_res_dir_w,
        product_file_suffix="product.oeb.gz",
        max_stereocenter=ctx.max_stereocenter,
    )
    ctx.logger.info(f"Finished instantiating products. Chunks saved to {ctx.rescore_res_dir_w}.")
    client.disk.log_peak()
