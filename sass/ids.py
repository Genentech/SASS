"""Encoding SASS ids into OpenEye molecule titles, and parsing them back out.

Titles take three forms::

    12345                         a synthon id, an integer assigned by `synthon_id_index`
    12345_1                       a synthon id plus a FLIPPER isomer wart
    12345&67890_1                 a product: its synthon ids joined, plus the wart

The separators come from `warts_id_sep` and `product_id_sep` and must differ. This module
imports nothing from `sass`, so workers and the head node can both use it.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _WART_RE(warts_separator: str) -> re.Pattern[str]:
    """`<base><sep><isomer number>` with an optional trailing connector tag, e.g. `7_1[Np]`."""
    return re.compile(rf"^(.*){re.escape(warts_separator)}\d+(?:\[[A-Za-z]+\])?$")


def extract_base_id(mol_id: str, warts_separator: str = "_", keep_isomer_warts: bool = False) -> str:
    """
    Extract base id from input id strings that contain isomer/conformer warts.

    Parameters
    ----------
    mol_id
        Input id string.
    warts_separator
        String separator used to identify id parts.
    keep_isomer_warts
        Whether to keep isomer warts in the id string.
    """
    if keep_isomer_warts is False:
        return remove_isomer_warts(mol_id, warts_separator)
    else:
        return mol_id


def remove_isomer_warts(mol_id: str, warts_separator: str = "_") -> str:
    """Remove an isomer wart from an id string, if it has one.

    A wart is a FLIPPER isomer *number* at the end of the id, optionally carrying the connector
    tag that single-fragment ids append. Only that shape is removed:

    ==============================  ==========================  =================================
    input                           output                      why
    ==============================  ==========================  =================================
    ``"12345"``                     ``"12345"``                 no separator, nothing to strip
    ``"12345_1"``                   ``"12345"``                 ``_1`` is the isomer wart
    ``"12345&67890_1"``             ``"12345&67890"``           a product; the wart is still last
    ``"12345_1_2"``                 ``"12345_1"``               only the *last* number comes off
    ``"7_1[Np]"``                   ``"7"``                     wart plus single-fragment tag
    ``"t_join_a_phenyl"``           ``"t_join_a_phenyl"``       ``phenyl`` is not a number
    ``"t_join_a_phenyl_1"``         ``"t_join_a_phenyl"``       ...but ``_1`` is
    ==============================  ==========================  =================================

    Ids that also carry conformer warts are not handled. Non-string ids, such as pseudo-product
    tuples of integer synthon ids, pass through unchanged.
    """
    if not isinstance(mol_id, str):
        return mol_id
    if warts_separator not in mol_id:
        return mol_id
    match = _WART_RE(warts_separator).match(mol_id)
    return match.group(1) if match else mol_id


def synthon_id_to_int(sid: str, source_file: Path | None = None) -> int:
    """Parse an integer synthon id from a conformer title.

    A non-numeric id means the conformers were built without a `synthon_id_index`.
    """
    try:
        return int(sid)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Synthon id {sid!r} is not an integer"
            f"{f' (from {source_file})' if source_file else ''}. Conformer libraries must be "
            "built with a `synthon_id_index`, so that every synthon id is an integer."
        ) from exc


def product_id_to_sids(pid: Any, warts_separator: str, title_separator: str) -> tuple[int, ...]:
    """Normalize a product id to its component synthon ids.

    Pseudo-product ids are tuples of integer synthon ids and pass through. Rescore rows carry
    a title string such as `"12345&67890_1"`, which is stripped of its wart and split.
    """
    if not isinstance(pid, str):
        return tuple(pid)
    base = extract_base_id(pid, warts_separator=warts_separator, keep_isomer_warts=False)
    return tuple(synthon_id_to_int(part) for part in base.split(title_separator))


def restore_original_synthon_ids(
    title: str,
    synthon_id_index: list[str],
    title_separator: str,
    warts_separator: str,
) -> str:
    """Translate a product title's integer synthon ids back to the original library ids.

    `"12345&67890_1"` -> `"synthonid000000000004A&synthonid000000000005A_1"`.

    Only the final pose file is translated. Score files stay numeric, because
    `write_product_poses` re-instantiates products from them.

    Titles that are not integer products, such as the query molecule, are returned unchanged.
    A trailing single-fragment connector tag such as `[U]` is preserved.
    """
    connector = ""
    if title.endswith("]") and "[" in title:
        title, _, tag = title.rpartition("[")
        connector = f"[{tag}"

    if warts_separator in title:
        base, wart = title.rsplit(warts_separator, 1)
    else:
        base, wart = title, ""
    parts = base.split(title_separator)
    if not all(part.isdigit() for part in parts):
        return title + connector
    try:
        restored = title_separator.join(synthon_id_index[int(part)] for part in parts)
    except IndexError:
        logger.warning("Synthon id in %s is outside the id index; leaving the title unchanged.", title)
        return title + connector
    return (f"{restored}{warts_separator}{wart}" if wart else restored) + connector
