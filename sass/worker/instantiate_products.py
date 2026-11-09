"""Enumerating products from synthon combinations, run on a worker."""

import logging
from pathlib import Path
from typing import Any

from sass.chem.libgen import (
    write_library_products_chunks,
    write_singleton_products_chunks,
)
from sass.io import file_size

logger = logging.getLogger(__name__)


def write_library_products_wrapper(
    data: Any,
    output_dir: Path,
    out_file_suffix: str,
    sort_title: bool,
    title_separator: str,
    max_stereocenter: int,
    chunk_size: int,
):
    reactants, smirks, idx = data
    out_file = output_dir / f"chunk_{idx}.{out_file_suffix}"
    return write_library_products_chunks(
        out_file=out_file,
        max_stereocenter=max_stereocenter,
        chunk_size=chunk_size,
        reactants=reactants,
        smirks=smirks,
        sort_title=sort_title,
        title_separator=title_separator,
    )


def write_singleton_products_wrapper(
    data: Any,
    output_dir: Path,
    out_file_suffix: str,
    title_separator: str,
    file_chunk_size: int,
    max_stereocenter: int,
):
    """Instantiate molecules using data passed from dask head node."""
    pid_data, idx = data

    logger.info("Starting product instantiation.")
    out_files = write_singleton_products_chunks(
        pid_data,
        out_file_stem=f"chunk_{idx}",
        out_file_dir=output_dir,
        out_file_suffix=out_file_suffix,
        title_separator=title_separator,
        max_stereocenter=max_stereocenter,
        split_chunk_size=file_chunk_size,
    )
    return sum(file_size(f) for f in out_files)
