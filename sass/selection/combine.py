"""Enumerating synthon combinations into pseudo-products, with exact cutoff pruning."""

import logging
import math
from typing import Any

from sass.selection.aggregate import aggregate_scores, simple_average
from sass.selection.allocate import allocate_synthons
from sass.selection.dedup import deduplicate_scores

logger = logging.getLogger(__name__)


#: Measured memory of a row in the running result list, in bytes.
_COMBINE_BYTES_PER_ROW = 183


def estimate_peak_combine_gb(synthon_comb_limit: int | None, dedup_limit: float) -> float:
    """Peak memory of `combine_forder_scores` for a task that runs before a cutoff exists.

    The running result list grows to `min(synthon_comb_limit, dedup_limit)` rows before its
    first trim. `synthon_comb_limit=None` means no cap.
    """
    rows = dedup_limit if synthon_comb_limit is None else min(synthon_comb_limit, dedup_limit)
    return _COMBINE_BYTES_PER_ROW * rows / 1e9


#: Aggregation methods for which an admissible upper bound on the score still reachable
#: from a partial synthon combination is known. Only these can be pruned; anything else
#: falls back to exhaustive enumeration.
_PRUNE_BOUND_MODE = {
    "simple_average": "sum",
    "min_score": "min",
}

#: Slack on the pruning comparison, to absorb floating-point rounding. The bound and the real
#: score add the same values in a different order. Extra slack can only retain rows, never
#: drop ones that belong in the result.
_PRUNE_EPSILON = 1e-9


def combine_forder_scores(
    scores: dict[int, list[tuple[str, float, Any]]],
    top_m: int,
    dedup_limit: int,
    warts_separator: str,
    synthon_comb_limit: int,
    agg_method: str,
    rxn_id: str,
    cutoff: float | None = None,
) -> tuple[list, int]:
    """Combine per-component synthon scores into pseudo-product scores.

    Enumerates synthon combinations depth-first and aggregates each component's score into
    a single pseudo-product score.

    Parameters
    ----------
    scores
        Per-component score lists, ``{s_idx: [(score, sid), ...]}``, each sorted by
        descending score. Pruning relies on this ordering.
    top_m
        Number of top pseudo-products to keep.
    dedup_limit
        Size above which the running result list is deduplicated and trimmed.
    warts_separator
        Isomer-wart separator.
    synthon_comb_limit
        Cap on combinations considered, distributed across components by
        `allocate_synthons`.
    agg_method
        Score aggregation method. See `_PRUNE_BOUND_MODE`.
    rxn_id
        Reaction the combinations belong to.
    cutoff
        Score below which a pseudo-product cannot reach the final ``top_m``, taken from
        already-merged results for this reaction's component count. Combinations that
        cannot beat it are skipped. Pruning is exact: merging only raises the cutoff, and
        ties are kept, so the output matches full enumeration.

    Returns
    -------
        The top pseudo-products, and the total number of possible products for the reaction.
    """

    def backtrack(score_list: list, sid_list: list, partial: float) -> None:
        nonlocal res, threshold

        if len(res) > dedup_limit:
            res = deduplicate_scores(res, sort_scores=True, limit=top_m, warts_separator=warts_separator)
            if len(res) >= top_m:
                # The list is full, so its lowest score is now a valid local cutoff too.
                threshold = max(threshold, res[-1][0])

        if len(sid_list) == n_component:
            if agg_method == "simple_average":
                # Fast path for the common case.
                comb_score = simple_average(score_list)
            else:
                comb_score = aggregate_scores(agg_method, score_list, sids=sid_list, synthon_data=None)

            res.append((comb_score, tuple(sorted(sid_list)), rxn_id))
            return

        cur_s_idx = len(sid_list)  # use result list to track which s-score is on.
        cur_key = score_keys[cur_s_idx]
        cur_s_limit = s_limit_dict[cur_key]
        cur_scores = scores[cur_key]
        suffix = suffix_bound[cur_s_idx]

        for i in range(len(cur_scores)):
            # `cur_s_limit` is a count, not a last index.
            if i >= cur_s_limit:
                break

            _score, _sid = cur_scores[i]

            # Best score still reachable down this branch. `cur_scores` is sorted
            # descending, so the bound only decreases as `i` grows. Once it drops below the
            # threshold, the rest of the loop can stop.
            if bound_mode == "sum":
                if (partial + _score + suffix) / n_component < threshold - _PRUNE_EPSILON:
                    break
                next_partial = partial + _score
            elif bound_mode == "min":
                next_partial = _score if partial is None else min(partial, _score)
                if min(next_partial, suffix) < threshold - _PRUNE_EPSILON:
                    break
            else:
                next_partial = 0.0

            score_list.append(_score)
            sid_list.append(_sid)
            backtrack(score_list, sid_list, next_partial)
            score_list.pop()
            sid_list.pop()

    syn_counts = {key: len(val) for (key, val) in scores.items()}
    if synthon_comb_limit is None:
        s_limit_dict = syn_counts
    else:
        s_limit_dict = allocate_synthons(syn_counts, max_total=synthon_comb_limit)
        logger.info(f"Limit for each synthon score list: {s_limit_dict}.")

    n_component = len(scores)
    score_keys = list(scores.keys())

    # Use the total number of products instead of combinations evaluated.
    num_prod = math.prod(syn_counts.values())

    if any(not scores[key] for key in score_keys):
        # A component scored nothing (e.g. a missing conformer file); no combination exists.
        return [], num_prod

    # The threshold comes from `cutoff`, or from the local list filling up mid-enumeration.
    bound_mode = _PRUNE_BOUND_MODE.get(agg_method)
    threshold = -math.inf if cutoff is None else cutoff

    # suffix_bound[d] summarizes the best possible contribution of components after d:
    # their score sum for "sum" mode, the smallest of their maxima for "min" mode.
    suffix_bound: list[float] = [0.0] * n_component
    if bound_mode == "sum":
        running = 0.0
        for d in range(n_component - 1, -1, -1):
            suffix_bound[d] = running
            running += scores[score_keys[d]][0][0]
    elif bound_mode == "min":
        running = math.inf
        for d in range(n_component - 1, -1, -1):
            suffix_bound[d] = running
            running = min(running, scores[score_keys[d]][0][0])

    res = []
    backtrack([], [], 0.0 if bound_mode == "sum" else None)
    res.sort(reverse=True)

    res = deduplicate_scores(res, sort_scores=True, limit=top_m, warts_separator=warts_separator)

    return res, num_prod
