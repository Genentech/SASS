"""Selection defects that would give a wrong answer rather than an error."""

import itertools

import pytest

from sass.selection.aggregate import SUPPORTED_AGG_METHODS, top_two_of_three
from sass.selection.allocate import allocate_synthons
from sass.selection.combine import combine_forder_scores

#: Three components of two synthons each, with distinct id spaces per component so that
#: `tuple(sorted(sids))` cannot collapse two different combinations into one key.
SCORES = {0: [(0.9, 1), (0.8, 2)], 1: [(0.7, 11), (0.6, 12)], 2: [(0.5, 21), (0.4, 22)]}


def brute_force(scores: dict, aggregate) -> dict:
    """Score every combination independently."""
    return {
        tuple(sorted(sid for _, sid in combo)): pytest.approx(aggregate([score for score, _ in combo]))
        for combo in itertools.product(*scores.values())
    }


class TestTopTwoOfThreeDoesNotMutate:
    """`top_two_of_three` leaves its argument unchanged, so combination scores stay correct."""

    def test_the_argument_is_left_alone(self):
        scores = [0.4, 0.9, 0.6]
        top_two_of_three(scores)
        assert scores == [0.4, 0.9, 0.6], "aggregation must not reorder the caller's list"

    def test_it_still_averages_the_top_two(self):
        assert top_two_of_three([0.4, 0.9, 0.6]) == pytest.approx((0.9 + 0.6) / 2)

    def test_every_combination_matches_brute_force(self):
        """Every combination must be scored and ranked correctly."""
        got, _ = combine_forder_scores(
            SCORES,
            rxn_id="r",
            agg_method="top_two_of_three",
            top_m=100,
            dedup_limit=10**9,
            warts_separator="_",
            synthon_comb_limit=None,
        )
        expected = brute_force(SCORES, lambda vals: sum(sorted(vals)[1:]) / 2)
        assert {tuple(sorted(sids)): score for score, sids, _ in got} == expected

    def test_the_top_score_is_right(self):
        """`top_two_of_three` drops the lowest component, so (1,11,21) and (1,11,22) tie for the top score.

        Both must be returned with that score. Their order is not checked.
        """
        got, _ = combine_forder_scores(
            SCORES,
            rxn_id="r",
            agg_method="top_two_of_three",
            top_m=100,
            dedup_limit=10**9,
            warts_separator="_",
            synthon_comb_limit=None,
        )
        ranked = sorted(got, key=lambda row: -row[0])
        assert ranked[0][0] == pytest.approx((0.9 + 0.7) / 2)
        winners = {tuple(sorted(sids)) for score, sids, _ in got if score == pytest.approx(0.8)}
        assert winners == {(1, 11, 21), (1, 11, 22)}


class TestCombinationLimitIsACount:
    def test_the_configured_cap_is_not_exceeded(self):
        """`synthon_comb_limit` caps the number of combinations returned."""
        scores = {
            0: [(1.0 - i / 10000, i) for i in range(1000)],
            1: [(1.0 - i / 10000, 1_000_000 + i) for i in range(1000)],
        }
        got, _ = combine_forder_scores(
            scores,
            rxn_id="r",
            agg_method="simple_average",
            top_m=10**9,
            dedup_limit=10**12,
            warts_separator="_",
            synthon_comb_limit=10_000,
        )
        assert len(got) <= 10_000


class TestAllocationNeverReturnsZero:
    @pytest.mark.parametrize(
        ("counts", "max_total"),
        [({0: 10, 1: 10}, 9), ({0: 2, 1: 2, 2: 500_000}, 3), ({0: 100, 1: 100, 2: 100}, 2)],
    )
    def test_no_component_is_allocated_nothing(self, counts, max_total):
        """Every component is allocated at least one synthon, even under a tight `max_total`."""
        allocated = allocate_synthons(counts, max_total=max_total)
        assert all(limit >= 1 for limit in allocated.values()), allocated


class TestSupportedAggregationMethods:
    def test_the_supported_set_is_what_combine_can_actually_run(self):
        for method in SUPPORTED_AGG_METHODS:
            combine_forder_scores(
                dict(SCORES),
                rxn_id="r",
                agg_method=method,
                top_m=10,
                dedup_limit=10**9,
                warts_separator="_",
                synthon_comb_limit=None,
            )

    def test_ha_weighted_avg_is_excluded(self):
        """It needs a per-synthon heavy-atom count the worker is not given."""
        assert "ha_weighted_avg" not in SUPPORTED_AGG_METHODS
