"""Locating and reading the library's synthon conformer files.

Conformers are either chunked, as in `reaction_11_synthon_0_conf.chunk3.oez`, or stored as one
file per reaction component.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from openeye import oechem

from sass.chem.mol import read_mols

logger = logging.getLogger(__name__)


def build_conformer_manifest(conf_dir: Path, conf_format: str) -> dict[tuple[str, int], list[Path]]:
    """Map `(rxn_id, s_idx)` to its conformer files, from a single directory scan.

    One scan is much cheaper than a glob per task when a directory holds many chunk files.

    Handles both layouts: `{rxn}_synthon_{s}_conf.{fmt}` (one merged file) and
    `{rxn}_synthon_{s}_conf.chunk{k}.{fmt}`.
    """
    manifest: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for entry in Path(conf_dir).iterdir():
        name = entry.name
        if not name.endswith(f".{conf_format}"):
            continue
        stem = name.split(".")[0]  # strip .chunkN and the extension
        marker = "_synthon_"
        if not stem.endswith("_conf") or marker not in stem:
            continue
        rxn_id, _, tail = stem.rpartition(marker)
        s_idx_text = tail.removesuffix("_conf")
        if not s_idx_text.isdigit():
            continue
        manifest[(rxn_id, int(s_idx_text))].append(entry)

    # Sort so chunk order, and therefore tie-breaking, is deterministic.
    for files in manifest.values():
        files.sort()
    return dict(manifest)


def resolve_conformer_files(
    conf_dir: Path,
    rxn_id: str,
    s_idx: int,
    conf_format: str,
    manifest: dict[tuple[str, int], list[Path]] | None = None,
) -> list[Path]:
    """Conformer files for one reaction component, from the manifest or by direct lookup."""
    if manifest is not None:
        return manifest.get((rxn_id, s_idx), [])

    single = Path(conf_dir) / f"{rxn_id}_synthon_{s_idx}_conf.{conf_format}"
    if single.is_file():
        return [single]
    return sorted(Path(conf_dir).glob(f"{rxn_id}_synthon_{s_idx}_conf.chunk*.{conf_format}"))


def stream_conformers(conf_files: list[Path]) -> Iterable[oechem.OEMol]:
    """Yield every molecule across `conf_files` as one continuous stream."""
    for conf_file in conf_files:
        with read_mols(conf_file) as mols:
            yield from mols


def conformer_chunk_stems(file_dir: Path, conf_format: str) -> list[str]:
    """The `(reaction, component)` stems with chunks waiting to be merged, in sorted order."""
    return sorted({f.name.split(".")[0] for f in file_dir.glob(f"*.{conf_format}")})


def merge_conformer_chunks_for_stem(stem: str, file_dir: Path, conf_format: str) -> tuple[str, int]:
    """Merge one `(reaction, component)`'s chunks into a single file. Returns `(stem, n_chunks)`.

    Chunks are merged in sorted order, the same order `resolve_conformer_files` reads them in,
    so the result is reproducible. One stem per call, so merging can be fanned out.
    """
    merged_file = file_dir / f"{stem}.{conf_format}"
    # Exclude the merge target, which matches the glob on a re-run.
    chunks = sorted(f for f in file_dir.glob(f"{stem}*.{conf_format}") if f != merged_file)
    if not chunks:
        return stem, 0

    if len(chunks) == 1:
        chunks[0].rename(merged_file)
    else:
        ostream = oechem.oemolostream(str(merged_file))
        for chunk in chunks:
            with read_mols(chunk) as mols:
                for mol in mols:
                    oechem.OEWriteMolecule(ostream, mol)
        ostream.close()
        for chunk in chunks:
            chunk.unlink()
    return stem, len(chunks)


def merge_synthon_conformer_chunks(file_dir: Path, conf_format: str) -> None:
    """Merge every stem's chunks, serially. `generate_synthon_conformers` fans this out instead."""
    for stem in conformer_chunk_stems(file_dir, conf_format):
        _, n_chunks = merge_conformer_chunks_for_stem(stem, file_dir, conf_format)
        if n_chunks:
            logger.info("%s: %s chunk(s) merged.", stem, n_chunks)
