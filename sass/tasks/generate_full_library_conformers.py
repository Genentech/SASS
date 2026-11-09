"""Conformer generation over the fully enumerated library. Validation path."""

import logging
from pathlib import Path
from typing import Any

from sass.cluster.client import DaskClient
from sass.context import RunContext
from sass.worker.generate_conformers import (
    write_conformers_wrapper,
)

logger = logging.getLogger(__name__)


def dask_generate_conformers(
    client: DaskClient,
    input_dir: Path,
    output_dir: Path,
    product_file_suffix: str,
    delete_input_file: bool,
    **kwargs: Any,  # OMEGA parameters
) -> None:
    """Generate conformers for full products."""
    task_gen = input_dir.glob(f"*{product_file_suffix}")

    for n, delta in enumerate(
        client.stream_map(
            write_conformers_wrapper,
            task_gen,
            output_dir=output_dir,
            delete_input_file=delete_input_file,
            **kwargs,
        )
    ):
        client.disk.add(delta or 0)
        if (n + 1) % client.target_max_worker == 0:
            client.disk.log_progress(n + 1)


def run(ctx: RunContext, client: DaskClient) -> None:
    """Build conformers for the fully enumerated library."""
    ctx.logger.info("Starting conformer generation for full library.")
    ctx.full_enumeration_conf_dir_w.mkdir(mode=0o755, parents=True, exist_ok=True)
    dask_generate_conformers(
        client,
        input_dir=ctx.full_enumeration_product_dir_r,
        output_dir=ctx.full_enumeration_conf_dir_w,
        product_file_suffix="product.oeb.gz",
        delete_input_file=False,
        omega_max_conf=ctx.config["product_omega_max_conf"],
        omega_max_time=ctx.config["product_omega_max_time"],
        omega_e_window=ctx.config["product_omega_energy"],
    )
    ctx.logger.info(f"Finished full library conformer generation. Chunks saved to {ctx.full_enumeration_conf_dir_w}.")
