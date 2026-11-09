"""Collapsing a score list to one row per molecule, and reading a selection threshold off it."""

from typing import Any, Optional

from sass.ids import extract_base_id


def score_cutoff(rows: list, top_m: int) -> float | None:
    """Score of the ``top_m``-th row, or ``None`` if there are fewer than that.

    Once a merged bucket holds `top_m` deduplicated rows, its lowest score is a lower bound on
    the final selection threshold, since merging only raises it. Scoring tasks can discard
    anything below it without changing the result.
    """
    return rows[top_m - 1][0] if len(rows) >= top_m else None


def deduplicate_scores(
    scores: list[tuple[float, str, Any]],
    warts_separator: str,
    keep_isomer_warts: bool = False,
    sort_scores: bool = False,
    limit: Optional[int] = None,
) -> list[tuple[float, str, Any]]:
    """
    Deduplicate score list by mol id, return a sorted list.

    Parameters
    ----------
    scores
        Array of score data, e.g. [(score1, id1, ...), (score2, id2, ...), ...].
    warts_separator
        String separator between molecule id and isomer/conformer number in molecule titles.
    keep_isomer_warts
        Whether to keep isomer warts in the merged scores. Deduplication is still done
        on the base id (without isomer warts).
    sort_scores
        Whether to sort the array by score (first element) first. The input is usually
        already sorted. Avoid when scores are missing (None or ''), since the sort then
        falls back to the other elements.
    limit
        Number of top scores to output.

    Returns
    -------
    list[tuple[float, str, Any]]
        Array of [(score, id), ...] where all base ids are unique.
    """
    dedup_res = []
    seen = set()
    if sort_scores:
        scores.sort(reverse=True, key=lambda x: x[0])  # sort only based on 1st element

    n = 0
    for data in scores:
        sid = data[1]
        if limit is not None and n >= limit:
            break
        base_id = extract_base_id(mol_id=sid, keep_isomer_warts=False, warts_separator=warts_separator)
        if keep_isomer_warts is True:
            output_id = extract_base_id(sid, keep_isomer_warts=keep_isomer_warts, warts_separator=warts_separator)
        else:
            output_id = base_id
        if base_id not in seen:
            seen.add(base_id)
            new_entry = [data[0], output_id, *data[2:]]  # Retain additional information.
            if isinstance(data, tuple):
                new_entry = tuple(new_entry)
            dedup_res.append(new_entry)
            n += 1

    return dedup_res
