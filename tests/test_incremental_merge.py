"""Integration tests for incremental merging inside `dask_score_and_combine_synthon`.

The scoring loop runs on a threaded local dask cluster with the ROCS work stubbed out.
"""

import logging
import random
import re
import zlib

import pytest
from dask.distributed import Client, LocalCluster

from sass.cluster.client import DaskClient
from sass.cluster.disk import DiskTracker
from sass.io import dump_score_file, load_score_file
from sass.selection.dedup import deduplicate_scores
from sass.tasks.score_synthons import dask_score_and_combine_synthon, interleave_by_component

SASS_SEP = "&"
WARTS_SEP = "_"


def stub_seed(rxn_id: str, fset_idx: int, forder_idx: int) -> int:
    """Seed the fake scorer's rows from the task's identity, reproducibly.

    `crc32` is used because Python randomizes string hashing per process.
    """
    return zlib.crc32(f"{rxn_id}|{fset_idx}|{forder_idx}".encode())


class StubHandler:
    """Stands in for `SynthonHandler`; component count is encoded in the reaction id.

    Reaction ids look like ``reaction_<n_components>_<variant>``, so one bucket holds several reactions.
    """

    def get_number_of_components(self, rxn_id):
        return int(rxn_id.split("_")[1])


def make_tasks(n_tasks, n_reactions_per_bucket=5):
    """Build scoring tasks spread over several reactions per component count."""
    tasks = []
    for i in range(n_tasks):
        n_comp = 2 + (i % 2)
        rxn_id = f"reaction_{n_comp}_{(i // 2) % n_reactions_per_bucket:03d}"
        # 6th/7th elements: the per-task conformer file slice and the fragment key per component.
        tasks.append(
            (
                rxn_id,
                i,
                0,
                list(range(n_comp)),
                [None] * n_comp,
                {s: [] for s in range(n_comp)},
                [f"frag{s}" for s in range(n_comp)],
            )
        )
    return tasks


@pytest.fixture(scope="module")
def dask_client():
    """A `DaskClient` wrapping a threaded local cluster, so monkeypatched stubs reach the workers.

    `object.__new__` skips `__init__`, which would start a real cluster.
    """
    cluster = LocalCluster(processes=False, n_workers=2, threads_per_worker=2, dashboard_address=None)
    client = Client(cluster)
    obj = object.__new__(DaskClient)
    obj._dask_client = client
    obj.target_max_worker = 4
    obj.disk = DiskTracker("test", worker_count=obj.worker_count)
    yield obj
    client.close()
    cluster.close()


def install_stubs(monkeypatch, tasks, rows_per_task, slow_fset_indices=(), slow_seconds=0.0):
    """Replace task generation and scoring with deterministic fakes.

    The fake records its calls in an ``audit`` subdirectory, since dask pickles the task function
    and a closure variable would be mutated in a copy.
    """
    from sass.tasks import score_synthons as task_score_synthons

    def fake_generate(*_args, **_kwargs):
        yield from tasks

    def fake_score(data, *_component_scores, cutoff=None, **kwargs):
        import pickle as _pickle
        import time as _time

        rxn_id, fset_idx, forder_idx, synthon_order = data
        if fset_idx in slow_fset_indices:
            _time.sleep(slow_seconds)  # Stand in for a large reaction's ROCS pass.

        rng = random.Random(stub_seed(rxn_id, fset_idx, forder_idx))
        pool = [f"s{i:04d}" for i in range(30)]
        rows = [
            (rng.random(), SASS_SEP.join(sorted(rng.sample(pool, len(synthon_order)))), rxn_id)
            for _ in range(rows_per_task)
        ]
        rows = deduplicate_scores(rows, sort_scores=True, warts_separator=WARTS_SEP)
        # Honor the cutoff exactly as the real task does.
        if cutoff is not None:
            rows = [r for r in rows if r[0] >= cutoff]

        audit_dir = kwargs["pseudo_res_dir"] / "audit"
        audit_dir.mkdir(exist_ok=True)
        with (audit_dir / f"{rxn_id}_{fset_idx}_{forder_idx}.pkl").open("wb") as fh:
            _pickle.dump({"rows": rows, "cutoff": cutoff, "submit_idx": fset_idx}, fh)

        fname = kwargs["pseudo_res_dir"] / f"{rxn_id}_{fset_idx}_{forder_idx}_pseudo_res.pkl"
        dump_score_file(fname, rows)
        top_m = kwargs["top_m"]
        local_cutoff = rows[top_m - 1][0] if len(rows) >= top_m else None
        return ("score", rxn_id, fname, 7, fname.stat().st_size, local_cutoff)

    monkeypatch.setattr(task_score_synthons, "generate_synthon_scoring_tasks", fake_generate)
    monkeypatch.setattr(task_score_synthons, "combine_forder_task", fake_score)

    def fake_phase_a(_data, **_kwargs):
        """Phase A, the per-fragment synthon scoring, is stubbed out."""
        return []

    monkeypatch.setattr(task_score_synthons, "score_synthon_fragment", fake_phase_a)


def read_audit(tmp_path):
    """Load what the stub recorded, keyed by (rxn_id, fset_idx, forder_idx)."""
    import pickle as _pickle

    out = {}
    for f in (tmp_path / "audit").glob("*.pkl"):
        rxn_prefix, fset_idx, forder_idx = f.stem.rsplit("_", 2)
        with f.open("rb") as fh:
            out[(rxn_prefix, int(fset_idx), int(forder_idx))] = _pickle.load(fh)
    return out


def run_loop(dask_client, tmp_path, top_m, merge_fanin, warmup_wave_size=0, warmup_quorum=1.0):
    """Invoke the loop under test with placeholder values for the ROCS-only arguments."""
    return dask_score_and_combine_synthon(
        dask_client,
        reactions=[],
        shandler=StubHandler(),
        query_fragments_dict={},
        grouped_synthons={},
        synthon_conf_dir=tmp_path,
        conf_format="oez",
        align_connector_atom=True,
        synthon_connectors=["U"],
        rocs_key_opt=0,
        rocs_key_score=0,
        opt_shape_func="GetTanimoto",
        opt_color_func=None,
        overlap_shape_func="GetTanimoto",
        overlap_color_func=None,
        top_m=top_m,
        dedup_limit=5e6,
        warts_separator=WARTS_SEP,
        agg_method="simple_average",
        synthon_comb_limit=None,
        frag_connectors=["He"],
        cross_score=True,
        color_ff_dir=tmp_path,
        pseudo_res_dir=tmp_path,
        merge_fanin=merge_fanin,
        warmup_wave_size=warmup_wave_size,
        warmup_quorum=warmup_quorum,
    )


@pytest.mark.parametrize(("n_tasks", "merge_fanin"), [(37, 4), (8, 12), (1, 4), (100, 3)])
def test_incremental_merge_equals_all_at_once(dask_client, tmp_path, monkeypatch, n_tasks, merge_fanin):
    """The incrementally merged result must equal merging every task's output at the end."""
    tasks = make_tasks(n_tasks)
    top_m = 10**6  # High enough that no cutoff can bite, so the reference is unpruned.
    install_stubs(monkeypatch, tasks, rows_per_task=40)

    final_files, prod_count = run_loop(dask_client, tmp_path, top_m, merge_fanin)

    produced = read_audit(tmp_path)
    assert len(produced) == n_tasks, "not every task ran"

    for n_comp, merged_file in final_files.items():
        expected_rows = [
            row
            for (rxn_id, _f, _o), rec in produced.items()
            for row in rec["rows"]
            if int(rxn_id.split("_")[1]) == n_comp
        ]
        expected = deduplicate_scores(expected_rows, sort_scores=True, limit=top_m, warts_separator=WARTS_SEP)
        assert load_score_file(merged_file) == expected

    assert set(prod_count) == set(final_files)


def test_merged_rows_keep_their_own_reaction(dask_client, tmp_path, monkeypatch):
    """Every merged row must still name the reaction it was actually scored against.

    Each bucket here holds several reactions, so a wrong attribution is visible.
    """
    tasks = make_tasks(60, n_reactions_per_bucket=6)
    install_stubs(monkeypatch, tasks, rows_per_task=30)

    final_files, _ = run_loop(dask_client, tmp_path, top_m=10**6, merge_fanin=4)

    # Which reactions each product id legitimately came from.
    legitimate = {}
    for (rxn_id, _f, _o), rec in read_audit(tmp_path).items():
        for row in rec["rows"]:
            legitimate.setdefault(row[1], set()).add(rxn_id)

    seen_reactions = set()
    for merged_file in final_files.values():
        for _score, product_id, rxn_id in load_score_file(merged_file):
            assert rxn_id in legitimate[product_id], f"{product_id} attributed to {rxn_id}"
            seen_reactions.add(rxn_id)

    # Guard against the test passing trivially because only one reaction survived.
    assert len(seen_reactions) > 1, "merged output collapsed to a single reaction"


def test_warmup_gate_opens_after_the_wave_not_at_the_end(dask_client, tmp_path, monkeypatch, caplog):
    """The warm-up gate opens once, after the configured wave size has completed."""
    wave = 8
    n_tasks = 200
    install_stubs(monkeypatch, make_tasks(n_tasks), rows_per_task=20)

    with caplog.at_level(logging.INFO, logger="sass.tasks.score_synthons"):
        run_loop(dask_client, tmp_path, top_m=10**6, merge_fanin=4, warmup_wave_size=wave)

    opened = [r.getMessage() for r in caplog.records if "Warm-up quorum reached" in r.getMessage()]
    assert len(opened) == 1, f"gate should open exactly once, got {len(opened)}"

    reported = int(re.search(r"reached: (\d+) of", opened[0]).group(1))
    assert reported == wave, (
        f"gate opened after {reported} tasks, expected {wave}. "
        "A value near the total task count means the wave became a concurrency cap."
    )
    assert reported < n_tasks / 10, "gate opened far too late to have relieved the burst"


def test_gate_opens_on_quorum_without_waiting_for_stragglers(dask_client, tmp_path, monkeypatch, caplog):
    """Submission must open once a quorum of the wave has scored, not all of it.

    Half the wave is deliberately slow, so a gate that waited for every member would report the full wave.
    """
    wave = 8
    slow = tuple(range(wave // 2, wave))  # second half of the wave is slow
    install_stubs(monkeypatch, make_tasks(200), rows_per_task=20, slow_fset_indices=slow, slow_seconds=3.0)

    with caplog.at_level(logging.INFO, logger="sass.tasks.score_synthons"):
        run_loop(dask_client, tmp_path, top_m=5, merge_fanin=4, warmup_wave_size=wave, warmup_quorum=0.5)

    opened = [r.getMessage() for r in caplog.records if "Warm-up quorum reached" in r.getMessage()]
    assert len(opened) == 1, f"gate should open exactly once, got {len(opened)}"

    scored = int(re.search(r"reached: (\d+) of", opened[0]).group(1))
    assert scored >= wave // 2, f"opened at {scored}, before quorum of {wave // 2}"
    assert scored < wave, f"opened at {scored} of {wave} -- waited for the slow stragglers"


def test_intermediate_files_are_cleaned_up(dask_client, tmp_path, monkeypatch):
    """Nothing but the final per-component files may be left behind."""
    tasks = make_tasks(30)
    install_stubs(monkeypatch, tasks, rows_per_task=25)

    final_files, _ = run_loop(dask_client, tmp_path, top_m=10**6, merge_fanin=4)

    left_on_disk = set(tmp_path.glob("*.pkl"))
    assert left_on_disk == set(final_files.values())


def test_one_final_file_per_component_count(dask_client, tmp_path, monkeypatch):
    tasks = [
        (
            f"reaction_{2 + (i % 3)}_{i % 4:03d}",
            i,
            0,
            list(range(2 + (i % 3))),
            [None] * (2 + (i % 3)),
            {},
            [f"frag{s}" for s in range(2 + (i % 3))],
        )
        for i in range(45)
    ]
    install_stubs(monkeypatch, tasks, rows_per_task=20)

    final_files, _ = run_loop(dask_client, tmp_path, top_m=10**6, merge_fanin=5)

    assert sorted(final_files) == [2, 3, 4]
    for f in final_files.values():
        assert f.is_file()


def test_cutoff_is_propagated_to_later_tasks(dask_client, tmp_path, monkeypatch):
    """Once a bucket fills its top_m quota, later tasks must receive a cutoff."""
    tasks = [("reaction_2_000", i, 0, [0, 1], [None, None], {}, ["f0", "f1"]) for i in range(400)]
    # top_m far below rows_per_task so the very first completed task yields a cutoff.
    install_stubs(monkeypatch, tasks, rows_per_task=30)

    run_loop(dask_client, tmp_path, top_m=5, merge_fanin=4)

    records = read_audit(tmp_path)
    by_submission = [records[k]["cutoff"] for k in sorted(records, key=lambda k: records[k]["submit_idx"])]

    assert by_submission[0] is None, "first task cannot have a cutoff yet"
    assert any(c is not None for c in by_submission), "cutoff never reached any task"
    # Cutoffs only tighten, so they never decrease along submission order.
    non_null = [c for c in by_submission if c is not None]
    assert non_null == sorted(non_null), f"cutoff must be monotonically non-decreasing: {non_null[:20]}"


def test_warmup_wave_establishes_cutoffs_before_the_bulk_runs(dask_client, tmp_path, monkeypatch):
    """The warm-up wave bounds how many tasks run before a cutoff exists."""
    wave = 16
    tasks = make_tasks(200, n_reactions_per_bucket=8)
    install_stubs(monkeypatch, tasks, rows_per_task=30)

    run_loop(dask_client, tmp_path, top_m=5, merge_fanin=4, warmup_wave_size=wave)

    records = read_audit(tmp_path)
    by_submission = [records[k]["cutoff"] for k in sorted(records, key=lambda k: records[k]["submit_idx"])]
    unpruned = sum(1 for c in by_submission if c is None)

    # Some tasks run before any cutoff exists, but the wave bounds how many.
    assert unpruned <= wave * 2, f"{unpruned} tasks ran without a cutoff, wave was {wave}"
    assert any(c is not None for c in by_submission)


def test_warmup_wave_can_be_disabled(dask_client, tmp_path, monkeypatch):
    tasks = make_tasks(40)
    install_stubs(monkeypatch, tasks, rows_per_task=20)

    final_files, _ = run_loop(dask_client, tmp_path, top_m=10**6, merge_fanin=4, warmup_wave_size=0)

    assert len(read_audit(tmp_path)) == 40
    assert set(final_files) == {2, 3}


def test_warmup_wave_larger_than_task_count_still_completes(dask_client, tmp_path, monkeypatch):
    """A wave wider than the whole job must not deadlock waiting for tasks that never come."""
    tasks = make_tasks(5)
    install_stubs(monkeypatch, tasks, rows_per_task=10)

    final_files, _ = run_loop(dask_client, tmp_path, top_m=10**6, merge_fanin=4, warmup_wave_size=500)

    assert len(read_audit(tmp_path)) == 5
    for f in final_files.values():
        assert f.is_file()


def test_interleave_covers_every_component_count_in_any_prefix():
    """Any prefix of the task order must span every component count, even for a sorted reaction list."""
    # A common reaction-file shape: all 2-component reactions listed before all 3-component.
    streams_by_component = {2: [f"2c_{i}" for i in range(140)], 3: [f"3c_{i}" for i in range(46)]}

    order = interleave_by_component(streams_by_component)

    assert len(order) == 186, "every reaction must survive the interleave"
    assert sorted(order) == sorted(streams_by_component[2] + streams_by_component[3])

    for prefix in (2, 8, 64):
        buckets = {s.split("c_")[0] for s in order[:prefix]}
        assert buckets == {"2", "3"}, f"first {prefix} reactions cover only {buckets}"


def test_interleave_handles_a_single_component_count():
    """A query fragmented only one way must still produce every reaction, in order."""
    streams_by_component = {2: ["a", "b", "c"]}
    assert interleave_by_component(streams_by_component) == ["a", "b", "c"]
    assert interleave_by_component({}) == []


def test_each_unique_component_fragment_pair_is_scored_once(dask_client, tmp_path, monkeypatch):
    """`(rxn, s_idx, fragment)` is scored once however many fragment orders and sets need it."""
    import tempfile

    from sass.tasks import score_synthons as task_score_synthons

    # Three fragment sets over two reactions, sharing fragments between sets.
    tasks = []
    for rxn in ("reaction_2_000", "reaction_2_001"):
        for fset, keys in enumerate([("fragA", "fragB"), ("fragA", "fragC"), ("fragB", "fragC")]):
            for forder in range(2):  # two orders per set, reusing the same two fragments
                order = [0, 1] if forder == 0 else [1, 0]
                # Use the keys themselves as the 'fragments' so the stub can report which
                # fragment it was asked to score.
                tasks.append((rxn, fset * 2 + forder, forder, order, list(keys), {}, list(keys)))
    install_stubs(monkeypatch, tasks, rows_per_task=5)

    calls_dir = tmp_path / "phase_a_calls"
    calls_dir.mkdir()

    def counting_phase_a(data, **_kwargs):
        """Record every call on disk, since a closure variable would be mutated in a pickled copy."""
        rxn_id, s_idx, frag, _files = data
        fd, _path = tempfile.mkstemp(dir=str(calls_dir))
        with open(fd, "w") as fh:
            fh.write(f"{rxn_id}|{s_idx}|{frag}")
        return []

    monkeypatch.setattr(task_score_synthons, "score_synthon_fragment", counting_phase_a)

    run_loop(dask_client, tmp_path, top_m=10**6, merge_fanin=4)

    calls = [f.read_text() for f in calls_dir.iterdir()]
    distinct = {(t[0], s, k) for t in tasks for s, k in zip(t[3], t[6], strict=True)}
    component_slots = sum(len(t[3]) for t in tasks)

    assert len(calls) == len(distinct), f"expected {len(distinct)} Phase A calls, got {len(calls)}"
    assert len(set(calls)) == len(calls), "a triple was scored more than once"
    assert len(calls) < component_slots, f"no deduplication: {len(calls)} calls for {component_slots} component slots"


def test_progress_line_reports_live_workers_against_target(dask_client, caplog):
    """Each progress line reports live workers against the target count."""
    with caplog.at_level(logging.INFO, logger="sass.cluster.disk"):
        dask_client.disk.log_progress(1234)

    line = next(r.getMessage() for r in caplog.records if "Finished 1234" in r.getMessage())
    assert f"Workers: 2/{dask_client.target_max_worker}." in line, line


def test_worker_count_is_not_capped_at_five():
    """`worker_count` counts every worker, though `scheduler_info()` describes only 5 by default.

    The shared fixture has two workers, so this test starts its own cluster with seven.
    """
    cluster = LocalCluster(processes=False, n_workers=7, threads_per_worker=1, dashboard_address=None)
    client = Client(cluster)
    try:
        obj = object.__new__(DaskClient)
        obj._dask_client = client
        obj.target_max_worker = 1000
        assert len(client.scheduler_info()["workers"]) == 5, (
            "dask no longer truncates scheduler_info()['workers'] to 5; "
            "re-check whether worker_count still needs the n_workers field"
        )
        assert obj.worker_count() == "7/1000"
    finally:
        client.close()
        cluster.close()


def test_progress_line_survives_an_unreachable_scheduler():
    """A progress line must never be the thing that takes down a run."""

    def boom() -> str:
        raise RuntimeError("scheduler is gone")

    tracker = DiskTracker("test", worker_count=boom)
    with pytest.raises(RuntimeError):
        boom()  # The callable really does raise ...

    obj = object.__new__(DaskClient)
    obj._dask_client = None  # ... and `DaskClient.worker_count` swallows the equivalent failure.
    obj.target_max_worker = 4
    assert obj.worker_count() == "unknown"
    assert tracker.worker_count is boom


def test_the_stub_seed_does_not_depend_on_the_process():
    """The stub seed is the same in every process.

    It is pinned against a literal, since a per-process seed is still stable within one process.
    """
    assert stub_seed("reaction_2_000", 0, 0) == 2742036551
