"""Ways of collapsing a synthon combination's per-component scores into one product score.

Cutoff pruning in `combine` needs a monotone upper bound on the score still reachable from a
partial combination. Only `simple_average` and `min_score` admit one.
"""

from typing import Any


def simple_average(array: list[float]) -> float:
    return sum(array) / len(array)


def min_score(array: list) -> float:
    return min(array)


def get_weighted_average(n_heavy_atoms: list[int], scores: list[float]) -> float:
    return sum(n_ha * score for n_ha, score in zip(n_heavy_atoms, scores, strict=False)) / sum(n_heavy_atoms)


def ha_weighted_avg(array: list, sids: list, synthon_data: dict) -> float:
    ha_counts = [synthon_data[sid]["heavy_atom_count"] for sid in sids]
    return get_weighted_average(ha_counts, array)


def top_two_of_three(array: list) -> float:
    """Mean of all except the lowest score.

    Must not sort in place, since `combine_forder_scores` passes its live depth-first stack.
    """
    ordered = sorted(array)
    return sum(ordered[1:]) / (len(ordered) - 1)


def aggregate_scores(agg_method: str, scores: list[float], **kwargs: Any) -> float:
    match agg_method:
        case "simple_average":
            return simple_average(scores)
        case "ha_weighted_avg":
            return ha_weighted_avg(scores, kwargs["sids"], kwargs["synthon_data"])
        case "min_score":
            return min_score(scores)
        case "top_two_of_three":
            return top_two_of_three(scores)
        case _:
            raise ValueError(f'"{agg_method}" is an unknown score aggregation method.')


#: Methods `combine_forder_scores` supports. `ha_weighted_avg` needs per-synthon heavy-atom
#: counts that the worker is not given.
SUPPORTED_AGG_METHODS = frozenset({"simple_average", "min_score", "top_two_of_three"})
