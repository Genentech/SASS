"""Tests that cutoff-based pruning does not change `combine_forder_scores` output.

Results are compared against a brute-force reference over randomized inputs.
"""

import itertools
import random

import pytest

from sass.selection.aggregate import min_score, simple_average
from sass.selection.allocate import allocate_synthons
from sass.selection.combine import combine_forder_scores
from sass.selection.dedup import deduplicate_scores

SASS_SEP = "&"
WARTS_SEP = "_"
RXN = "reaction_1"


def brute_force(scores, top_m, agg_method, synthon_comb_limit):
    """Enumerate every allowed combination with no pruning."""
    counts = {k: len(v) for k, v in scores.items()}
    limits = counts if synthon_comb_limit is None else allocate_synthons(counts, max_total=synthon_comb_limit)
    agg = simple_average if agg_method == "simple_average" else min_score

    keys = list(scores)
    # `allocate_synthons` returns a count per component, not a last index.
    ranges = [range(min(len(scores[k]), limits[k])) for k in keys]
    res = [
        (
            agg([scores[k][i][0] for k, i in zip(keys, combo, strict=True)]),
            tuple(sorted(scores[k][i][1] for k, i in zip(keys, combo, strict=True))),
            RXN,
        )
        for combo in itertools.product(*ranges)
    ]
    res.sort(reverse=True)
    return deduplicate_scores(res, sort_scores=True, limit=top_m, warts_separator=WARTS_SEP)


def random_scores(rng, n_comp, pool_size, tie_prone):
    """Per-component score lists, sorted descending, with overlapping synthon ids.

    Overlapping ids let different combinations share a product id, which exercises deduplication.
    """
    pool = [f"s{i}" for i in range(pool_size)]
    values = [0.1, 0.2, 0.3, 0.5, 0.5, 0.7, 0.9] if tie_prone else None
    scores = {}
    for c in range(n_comp):
        k = rng.randint(1, min(len(pool), 10))
        lst = [[round(rng.choice(values) if tie_prone else rng.random(), 6), s] for s in rng.sample(pool, k)]
        lst.sort(key=lambda x: x[0], reverse=True)
        scores[c] = lst
    return scores


@pytest.mark.parametrize("agg_method", ["simple_average", "min_score"])
@pytest.mark.parametrize("tie_prone", [True, False], ids=["ties", "distinct"])
def test_no_cutoff_matches_brute_force(agg_method, tie_prone):
    rng = random.Random(11)
    for _ in range(120):
        n_comp = rng.choice([2, 2, 3, 4])
        scores = random_scores(rng, n_comp, rng.choice([6, 12, 30]), tie_prone)
        top_m = rng.choice([1, 3, 10, 10**6])
        limit = rng.choice([None, 50, 10**6])

        got, _ = combine_forder_scores(dict(scores), top_m, 5e6, WARTS_SEP, limit, agg_method, RXN, cutoff=None)

        assert got == brute_force(scores, top_m, agg_method, limit)


@pytest.mark.parametrize("agg_method", ["simple_average", "min_score"])
@pytest.mark.parametrize("tie_prone", [True, False], ids=["ties", "distinct"])
def test_cutoff_never_drops_a_qualifying_row(agg_method, tie_prone):
    """Every row at or above the cutoff survives pruning unchanged.

    Pruning may keep extra rows below the cutoff, but only within `_PRUNE_EPSILON` of it.
    """
    from sass.selection.combine import _PRUNE_EPSILON

    rng = random.Random(23)
    checked = 0
    for _ in range(200):
        n_comp = rng.choice([2, 3])
        scores = random_scores(rng, n_comp, 20, tie_prone)

        full, _ = combine_forder_scores(dict(scores), 10**6, 5e6, WARTS_SEP, None, agg_method, RXN, cutoff=None)
        if len(full) < 3:
            continue
        checked += 1
        cutoff = full[rng.randint(1, len(full) - 1)][0]

        pruned, _ = combine_forder_scores(dict(scores), 10**6, 5e6, WARTS_SEP, None, agg_method, RXN, cutoff=cutoff)

        assert [row for row in pruned if row[0] >= cutoff] == [row for row in full if row[0] >= cutoff]
        assert all(row[0] >= cutoff - _PRUNE_EPSILON for row in pruned)
    assert checked > 50, "test data degenerate -- not enough non-trivial cases"


def test_cutoff_above_everything_yields_nothing():
    scores = {0: [[0.5, "a"], [0.4, "b"]], 1: [[0.5, "c"], [0.3, "d"]]}
    pruned, _ = combine_forder_scores(dict(scores), 10**6, 5e6, WARTS_SEP, None, "simple_average", RXN, cutoff=1.0)
    assert pruned == []


def test_unprunable_agg_method_still_correct():
    """An aggregation method with no known bound must fall back to full enumeration."""
    scores = {0: [[0.9, "a"], [0.4, "b"]], 1: [[0.8, "c"], [0.2, "d"]], 2: [[0.7, "e"], [0.1, "f"]]}
    got, _ = combine_forder_scores(dict(scores), 10**6, 5e6, WARTS_SEP, None, "top_two_of_three", RXN, cutoff=0.5)
    assert len(got) == 8  # 2 x 2 x 2, nothing pruned away


def test_empty_component_yields_no_combinations():
    scores = {0: [[0.9, "a"]], 1: []}
    got, num_prod = combine_forder_scores(dict(scores), 10, 5e6, WARTS_SEP, None, "simple_average", RXN, cutoff=None)
    assert got == []
    assert num_prod == 0


def test_num_prod_counts_all_products_not_combinations_evaluated():
    scores = {0: [[0.9, f"a{i}"] for i in range(30)], 1: [[0.8, f"b{i}"] for i in range(20)]}
    _, num_prod = combine_forder_scores(dict(scores), 10, 5e6, WARTS_SEP, 100, "simple_average", RXN, cutoff=None)
    assert num_prod == 600


def test_product_ids_are_tuples_not_joined_strings():
    """Product ids are sorted tuples of synthon ids, not separator-joined strings."""
    scores = {0: [[0.9, 101], [0.4, 102]], 1: [[0.8, 201], [0.2, 202]]}
    got, _ = combine_forder_scores(dict(scores), 10**6, 5e6, WARTS_SEP, None, "simple_average", RXN, cutoff=None)

    assert all(isinstance(row[1], tuple) for row in got)
    assert all(isinstance(sid, int) for row in got for sid in row[1])
    assert got[0][1] == (101, 201), "ids must be sorted within the product id"
    assert {row[1] for row in got} == {(101, 201), (101, 202), (102, 201), (102, 202)}
