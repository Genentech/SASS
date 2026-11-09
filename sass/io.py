"""Reading and writing the pickled intermediate files a run produces."""

import logging
import lzma
import pickle
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)

#: LZMA preset for result files. Faster than the default (6) at a modest cost in ratio.
_COMPRESS_PRESET = 1


def dump_score_file(file: Union[Path, str], rows: list) -> None:
    """Write a score list of `(float, tuple[int, ...], str)` rows as a plain pickle."""
    with Path(file).open("wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_score_file(file: Union[Path, str]) -> list:
    """Read a score list written by `dump_score_file`."""
    with Path(file).open("rb") as f:
        return pickle.load(f)  # Internal intermediate files only, never external input.


def discard_superseded_dir(directory: Path, superseded_by: Path, *, enabled: bool = True) -> None:
    """Delete a directory of per-chunk files once the combined artifact replaces it.

    Does nothing unless `superseded_by` exists and is non-empty, so a failed merge keeps its
    inputs.
    """
    if not enabled or not directory.is_dir():
        return
    if not (superseded_by.is_file() and superseded_by.stat().st_size > 0):
        logger.warning(
            "Keeping %s: %s is missing or empty, so the chunks it was merged into are still the "
            "only copy of those scores.",
            directory,
            superseded_by,
        )
        return

    freed = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    shutil.rmtree(directory)
    logger.info("Removed %s (%.2f MB), superseded by %s.", directory, freed / 1e6, superseded_by)


def compress_pkl_files(folder: Path, keep: Iterable[Path | str] = ()) -> None:
    """Compress every pkl file under `folder`, except the paths in `keep`.

    Each file is replaced by a `.pkl.xz`. Pass library artifacts in `keep`, since configs
    resolve them by exact path.
    """
    keep_resolved = {Path(path).resolve() for path in keep}
    for file in folder.rglob("*.pkl"):
        if file.resolve() in keep_resolved:
            continue
        with file.open("rb") as input_fh:
            data = input_fh.read()
        outfile = file.with_suffix(".pkl.xz")
        with lzma.open(outfile, "wb", preset=_COMPRESS_PRESET) as outpuf_fh:
            outpuf_fh.write(data)
        file.unlink()


def load_intermediate_files(file: str | Path) -> Any:
    """Load intermediate files.

    If the file is compressed, it will be decompressed before loading.
    """
    file = Path(file)
    if file.suffix == ".xz":
        with lzma.open(file, "rb") as f:
            return pickle.load(f)
    else:
        with file.open("rb") as f:
            return pickle.load(f)


def file_size(file: Path | str | None) -> int:
    """Size of `file` in bytes, or 0 if it is missing."""
    if file is None:
        return 0
    path = Path(file)
    return path.stat().st_size if path.is_file() else 0
