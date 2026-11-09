"""Per-component caps on how many synthons a single fragment order may combine."""

import logging
import math

logger = logging.getLogger(__name__)


def allocate_synthons(
    synthon_counts: dict[int | str, int], min_num: int = 100, max_total: int = 1e6
) -> dict[int | str, int]:
    """Calculate the limit of each synthon score list to use for select synthons.

    This caps the combinations enumerated per fragment order, which for 3-component
    reactions can be very large. Synthons are taken from each list proportionally, with a
    minimum number from each list.

    Parameters
    ----------
    synthon_counts
        The number of synthons in each reaction component.
    min_num
        The minimum number of synthons to take from each component.
    max_total
        The maximum total number of products to generate.

    Returns
    -------
        A dictionary mapping synthon index to the limit.
    """
    # sort the synthon count, process smallest first.
    syn_count = list(synthon_counts.items())
    syn_count.sort(key=lambda x: x[1])  # sort by count

    def _allocate_syn(idx: int, cur_prod: int, remaining_allowed: int) -> list:
        nonlocal res, syn_count, min_num
        if idx == len(syn_count) - 1:
            # Never allocate 0. Going slightly over the cap beats an empty score list.
            res.append((syn_count[idx][0], max(1, min(cur_prod, remaining_allowed))))
            return
        if cur_prod <= remaining_allowed:
            res.extend(syn_count[idx:])
            return
        # calc reducing factor.
        reduce_factor = (cur_prod / remaining_allowed) ** (1 / (len(syn_count) - idx))
        first_actual = syn_count[idx][1]
        first_reduced = int(first_actual / reduce_factor)
        if first_reduced >= min_num:
            first_count = first_reduced
        else:  # if after scaling, the count is < min_num, use up to min_num of this synthon.
            first_count = min(first_actual, min_num)
        res.append((syn_count[idx][0], first_count))
        # Floor the budget at 1, since the next level divides by it.
        _allocate_syn(idx + 1, int(cur_prod / syn_count[idx][1]), max(1, int(remaining_allowed / first_count)))

    cur_prod = math.prod([ele[1] for ele in syn_count])
    res = []
    _allocate_syn(0, cur_prod, max_total)
    return dict(res)
