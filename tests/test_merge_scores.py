"""Tests for `merge_score_batch_task`, the incremental fold-away of score files.

Also covers `check_memory_budget`, which warns before a run when a worker's peak memory may exceed
its limit.
"""

import random

import pytest

from sass.io import dump_score_file, load_score_file
from sass.selection.dedup import deduplicate_scores
from sass.worker.merge_scores import merge_score_batch_task

WARTS_SEP = "_"


def write_batch(tmp_path, n_files, rows_per_file, seed=0, n_ids=5000):
    """Write `n_files` score files of random rows, and return them with the union of rows."""
    rng = random.Random(seed)
    files, all_rows = [], []
    for i in range(n_files):
        rows = [
            (rng.random(), f"{rng.randrange(n_ids)}{WARTS_SEP}{rng.randrange(1, 4)}", f"reaction_{rng.randrange(3)}")
            for _ in range(rows_per_file)
        ]
        f = tmp_path / f"part_{i}.pkl"
        dump_score_file(f, rows)
        files.append(f)
        all_rows.extend(rows)
    return files, all_rows


@pytest.mark.parametrize("top_m", [10, 200, 5000, 10**6])
@pytest.mark.parametrize("keep_isomer_warts", [False, True])
def test_trimming_while_reading_equals_reading_everything_first(tmp_path, top_m, keep_isomer_warts):
    """Trimming to `top_m` between files must give exactly the all-at-once answer.

    `top_m=10` forces a trim after almost every file, and `top_m=10**6` never trims.
    """
    files, all_rows = write_batch(tmp_path, n_files=12, rows_per_file=900)
    expected = deduplicate_scores(
        list(all_rows),
        sort_scores=True,
        limit=top_m,
        warts_separator=WARTS_SEP,
        keep_isomer_warts=keep_isomer_warts,
    )

    out = tmp_path / "merged.pkl"
    merge_score_batch_task(files, out, top_m, WARTS_SEP, keep_isomer_warts)

    assert load_score_file(out) == expected


def test_peak_rows_held_is_bounded_by_top_m_not_by_fan_in(tmp_path, monkeypatch):
    """The merge trims as it reads, so the rows it holds are bounded by `top_m`, not by fan-in."""
    top_m, fan_in, rows_per_file = 100, 12, 400
    files, _ = write_batch(tmp_path, n_files=fan_in, rows_per_file=rows_per_file)

    # The list is largest right before each `deduplicate_scores` call, so its length there is the peak.
    from sass.worker import merge_scores

    peak = 0
    real_dedup = merge_scores.deduplicate_scores

    def watched_dedup(rows, **kwargs):
        nonlocal peak
        peak = max(peak, len(rows))
        return real_dedup(rows, **kwargs)

    monkeypatch.setattr(merge_scores, "deduplicate_scores", watched_dedup)
    merge_score_batch_task(files, tmp_path / "merged.pkl", top_m, WARTS_SEP, False)

    unbounded = fan_in * rows_per_file
    assert peak <= 2 * top_m + rows_per_file, f"held {peak} rows; bound is {2 * top_m + rows_per_file}"
    assert peak < unbounded, f"no better than concatenating everything ({unbounded} rows)"


def test_inputs_are_deleted_and_the_byte_delta_is_reported(tmp_path):
    """Inputs are deleted, and the return value reports the net change in bytes on disk."""
    files, _ = write_batch(tmp_path, n_files=4, rows_per_file=100)
    freed = sum(f.stat().st_size for f in files)
    out = tmp_path / "merged.pkl"

    kind, tag, out_file, n_rows, n_bytes, cutoff = merge_score_batch_task(files, out, 50, WARTS_SEP, False, tag=3)

    assert (kind, tag, out_file) == ("merge", 3, out)
    assert not any(f.exists() for f in files), "inputs must be unlinked"
    assert n_rows == 50
    assert n_bytes == out.stat().st_size - freed
    assert cutoff == load_score_file(out)[49][0]


def preflight_messages(top_m, memory_gb, caplog, synthon_comb_limit=None, tasks=("score_synthons",)):
    """Run `check_memory_budget` and return everything it logged."""
    import logging as _logging

    from sass.preflight import check_memory_budget

    logger = _logging.getLogger("sass.test.preflight")
    dedup_limit = max(5e6, 10 * top_m)
    limit = max(synthon_comb_limit or top_m, top_m)
    with caplog.at_level(_logging.INFO, logger=logger.name):
        check_memory_budget(list(tasks), top_m, limit, dedup_limit, memory_gb, logger)
    return [r.getMessage() for r in caplog.records]


def preflight_warnings(top_m, memory_gb, caplog, synthon_comb_limit=None, tasks=("score_synthons",)):
    """Only the warnings, for the tests that care whether something was flagged."""
    import logging as _logging

    preflight_messages(top_m, memory_gb, caplog, synthon_comb_limit, tasks)
    return [r.getMessage() for r in caplog.records if r.levelno >= _logging.WARNING]


class TestTopMPreflightCheck:
    """A `top_m` that would exceed worker memory or the advised rescore runtime is warned about."""

    warnings_for = staticmethod(preflight_warnings)

    def test_the_default_is_silent(self, caplog):
        assert self.warnings_for(1_000_000, 4, caplog) == []

    def test_warns_when_the_merge_will_not_fit(self, caplog):
        """top_m=4e6 on a 4 GB worker must warn."""
        warnings = self.warnings_for(4_000_000, 4, caplog)
        assert any("KilledWorker" in w for w in warnings), warnings
        assert any("dask_worker_memory" in w for w in warnings), warnings

    def test_a_bigger_worker_silences_the_memory_warning(self, caplog):
        """A larger `dask_worker_memory` silences the memory warning."""
        assert not any("KilledWorker" in w for w in self.warnings_for(4_000_000, 16, caplog))

    def test_warns_about_rescore_runtime_past_the_advised_maximum(self, caplog):
        """The rescore runtime warning does not depend on worker memory."""
        warnings = self.warnings_for(4_000_000, 64, caplog)
        assert any("rescore_products" in w for w in warnings), warnings

    def test_raising_synthon_comb_limit_alone_cannot_blow_up_memory(self, caplog):
        """`dedup_limit` clamps the combination peak at `10 * top_m` rows, whatever the limit."""
        assert self.warnings_for(1_000_000, 4, caplog, synthon_comb_limit=50_000_000) == []

    def test_a_high_synthon_comb_limit_is_flagged_at_a_high_top_m(self, caplog):
        """At a high `top_m`, a raised limit warns about the combination step but not the merge."""
        warnings = self.warnings_for(2_000_000, 4, caplog, synthon_comb_limit=50_000_000)
        assert any("combination" in w for w in warnings), warnings
        assert not any("score merge" in w for w in warnings), "the 2.4 GB merge still fits"

    def test_the_default_synthon_comb_limit_is_safe_even_at_the_advised_max_top_m(self, caplog):
        """With the floor alone (`synthon_comb_limit == top_m`) the combination step is tiny."""
        assert self.warnings_for(2_000_000, 4, caplog) == []


class TestTheEstimateIsScopedToTheRun:
    """The memory estimate covers only the peaks the run's tasks can reach.

    `generate_synthon_conformers` has no score merge and no combination step, so it logs nothing.
    """

    def test_a_run_with_neither_peak_says_nothing(self, caplog):
        assert preflight_messages(1_000_000, 4, caplog, tasks=("generate_synthon_conformers",)) == []

    def test_index_building_says_nothing_either(self, caplog):
        assert preflight_messages(1_000_000, 4, caplog, tasks=("build_synthon_id_index", "group_synthon_data")) == []

    def test_score_synthons_reports_both_peaks(self, caplog):
        summary = preflight_messages(1_000_000, 4, caplog, tasks=("score_synthons",))[-1]
        assert "score merge" in summary
        assert "combination" in summary

    def test_a_merge_only_task_does_not_report_the_combination_peak(self, caplog):
        """`combine_forder_scores` runs only in `score_synthons`, while the merge runs in several tasks."""
        summary = preflight_messages(1_000_000, 4, caplog, tasks=("score_single_fragments",))[-1]
        assert "score merge" in summary
        assert "combination" not in summary, "single-fragment scoring never runs the combination step"

    def test_rescore_products_reports_the_merge(self, caplog):
        summary = preflight_messages(1_000_000, 4, caplog, tasks=("rescore_products",))[-1]
        assert "score merge" in summary

    def test_a_mixed_run_reports_the_union(self, caplog):
        summary = preflight_messages(1_000_000, 4, caplog, tasks=("generate_synthon_conformers", "score_synthons"))[-1]
        assert "score merge" in summary
        assert "combination" in summary

    def test_the_unreachable_peak_cannot_warn_either(self, caplog):
        """A peak the run cannot reach does not produce a warning either."""
        messages = preflight_messages(2_000_000, 4, caplog, synthon_comb_limit=50_000_000, tasks=("rescore_products",))
        warnings = [m for m in messages if "KilledWorker" in m]
        assert not warnings, "the combination peak is unreachable without `score_synthons`"

    def test_every_task_named_is_a_real_task(self):
        from sass.preflight import _MEMORY_PEAKS_BY_TASK, VALID_TASKS

        assert set(_MEMORY_PEAKS_BY_TASK) <= set(VALID_TASKS)
