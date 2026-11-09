"""Request fewer SLURM workers when a run has fewer dask tasks than `num_workers`.

A task module can report its dask task count with `max_tasks(ctx)`, and
`preflight.resolve_num_workers` requests no more workers than the largest count.
"""

import inspect
import logging
import types
from importlib import import_module

from sass import main
from sass.main import TASK_MODULES
from sass.preflight import resolve_num_workers, useful_workers

logger = logging.getLogger("sass.test.workercap")


def make_ctx(tmp_path, tasks, n_single_frag=0, n_regular=0, reactions=("reaction_1",), num_workers=1000):
    single = tmp_path / "single_frag"
    regular = tmp_path / "regular"
    single.mkdir(exist_ok=True)
    regular.mkdir(exist_ok=True)
    for i in range(n_single_frag):
        (single / f"reaction_1_synthon_0_conf_single_frag_{i}.oez").touch()
    for i in range(n_regular):
        (regular / f"reaction_{i}_synthon_0_conf.oez").touch()
    return types.SimpleNamespace(
        tasks=list(tasks),
        reactions=list(reactions),
        single_frag_synthon_conf_dir=single,
        synthon_conf_dir=regular,
        exp_dir=tmp_path,
        logger=logger,
        config={
            "conf_file_format": "oez",
            "num_workers": num_workers,
            "dask_cluster_type": "slurm",
            "qos": None,
            "slurm_partition": None,
            "dask_worker_memory": 4,
        },
    )


class TestUsefulWorkers:
    def test_a_single_fragment_run_reports_its_width(self, tmp_path):
        ctx = make_ctx(tmp_path, ["score_single_fragments", "write_single_fragment_poses"], n_single_frag=30)
        assert useful_workers(ctx) == {"score_single_fragments": 30, "write_single_fragment_poses": 30}

    def test_one_task_that_cannot_say_leaves_the_run_uncapped(self, tmp_path):
        """`score_synthons` only knows its task count after fragmenting the query."""
        ctx = make_ctx(tmp_path, ["score_synthons", "rescore_products"], n_single_frag=10)
        assert useful_workers(ctx) is None

    def test_building_the_directory_in_the_same_run_makes_the_width_unknown(self, tmp_path):
        """The conformer files do not exist yet, so they cannot be counted."""
        ctx = make_ctx(
            tmp_path,
            ["generate_single_fragment_conformers", "score_single_fragments"],
            n_regular=18,
            n_single_frag=0,
        )
        assert useful_workers(ctx) is None

    def test_generating_single_fragment_conformers_counts_the_regular_files(self, tmp_path):
        ctx = make_ctx(tmp_path, ["generate_single_fragment_conformers"], n_regular=18)
        assert useful_workers(ctx) == {"generate_single_fragment_conformers": 18}

    def test_regenerating_the_regular_conformers_first_makes_it_unknown(self, tmp_path):
        from sass.tasks.generate_single_fragment_conformers import max_tasks

        ctx = make_ctx(tmp_path, ["generate_synthon_conformers", "generate_single_fragment_conformers"])
        assert max_tasks(ctx) is None

    def test_only_this_runs_reactions_are_counted(self, tmp_path):
        """Only files from this run's reactions are counted, as with `limit_reactions`."""
        ctx = make_ctx(tmp_path, ["score_single_fragments"], n_single_frag=50, reactions=["reaction_2"])
        assert useful_workers(ctx) == {"score_single_fragments": 0}


class TestResolveNumWorkers:
    def test_a_narrow_run_requests_only_what_it_can_use(self, tmp_path, caplog):
        ctx = make_ctx(tmp_path, ["score_single_fragments", "write_single_fragment_poses"], n_single_frag=30)
        with caplog.at_level(logging.INFO, logger=logger.name):
            assert resolve_num_workers(ctx) == 30
        assert "instead of num_workers=1000" in caplog.text, "log why the request differs from the config"

    def test_a_wide_run_keeps_num_workers(self, tmp_path):
        """More tasks than workers does not change the request."""
        ctx = make_ctx(tmp_path, ["score_single_fragments"], n_single_frag=12, num_workers=10)
        assert resolve_num_workers(ctx) == 10

    def test_exactly_as_many_tasks_as_workers_is_left_alone(self, tmp_path, caplog):
        ctx = make_ctx(tmp_path, ["score_single_fragments"], n_single_frag=10, num_workers=10)
        with caplog.at_level(logging.INFO, logger=logger.name):
            assert resolve_num_workers(ctx) == 10
        assert "instead of" not in caplog.text, "no log line when the request is unchanged"

    def test_an_unknown_width_keeps_num_workers(self, tmp_path):
        ctx = make_ctx(tmp_path, ["score_synthons", "rescore_products"])
        assert resolve_num_workers(ctx) == 1000

    def test_never_zero_workers(self, tmp_path):
        """Request at least one worker, even when there are no tasks."""
        ctx = make_ctx(tmp_path, ["score_single_fragments"], n_single_frag=50, reactions=["reaction_2"])
        assert resolve_num_workers(ctx) == 1

    def test_a_local_cluster_is_left_alone(self, tmp_path):
        """Local clusters use the node's own cores, so they are not changed."""
        ctx = make_ctx(tmp_path, ["score_single_fragments"], n_single_frag=3)
        ctx.config["dask_cluster_type"] = "local"
        assert resolve_num_workers(ctx) == 1000


def test_make_client_requests_the_resolved_count(tmp_path, monkeypatch):
    """`_make_client` passes the result of `resolve_num_workers` to `DaskClient`."""
    seen = {}

    def fake_client(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(main, "DaskClient", fake_client)
    monkeypatch.setattr(main, "resolve_cluster_name", lambda *_a: "my_cluster")
    main._make_client(make_ctx(tmp_path, ["score_single_fragments"], n_single_frag=30))
    assert seen["min_workers"] == seen["max_workers"] == 30


class TestPoseWritingReadsWhatScoringRead:
    def test_it_maps_over_exactly_the_files_it_is_given(self, tmp_path):
        """It searches only the files it is given."""
        from sass.tasks import write_single_fragment_poses as task

        given = [tmp_path / f"reaction_1_synthon_0_conf_single_frag_{i}.oez" for i in range(3)]
        mapped = []

        class Client:
            target_max_worker = 1000
            disk = types.SimpleNamespace(add=lambda _d: None, log_progress=lambda _n: None)

            def stream_map(self, func, data, **_kwargs):  # noqa: ARG002
                for f in data:
                    mapped.append(f)
                    yield None, 0

        task.extract_conformers_and_score(
            Client(),
            conf_files=given,
            product_set_file=tmp_path / "set.pkl",
            ref_mol=None,
            opt_shape_func="GetTanimoto",
            opt_color_func=None,
            overlap_shape_func="GetTanimoto",
            overlap_color_func=None,
            color_ff_dir=tmp_path,
            rocs_key=1,
            out_dir=tmp_path,
            sep_shape_color_score=False,
            remove_input_file=False,
        )
        assert mapped == given

    def test_its_width_is_scorings_width(self, tmp_path):
        from sass.tasks import score_single_fragments, write_single_fragment_poses

        ctx = make_ctx(tmp_path, ["write_single_fragment_poses"], n_single_frag=7)
        assert write_single_fragment_poses.max_tasks(ctx) == score_single_fragments.max_tasks(ctx) == 7


def test_every_max_tasks_hook_takes_only_the_context():
    """Every `max_tasks` takes only `ctx`, because `resolve_num_workers` calls it that way."""
    hooks = 0
    for name in TASK_MODULES:
        hook = getattr(import_module(f"sass.tasks.{name}"), "max_tasks", None)
        if hook is None:
            continue
        hooks += 1
        assert list(inspect.signature(hook).parameters) == ["ctx"], name
    assert hooks == 3, "each single-fragment task defines max_tasks"
