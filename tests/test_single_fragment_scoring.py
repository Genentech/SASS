"""Selecting conformer files, and folding score files while the scoring still runs.

The folded result keeps as many rows as poses will be written, not `top_m`.
"""

import pickle
import types

import pytest

from sass.cluster.disk import DiskTracker
from sass.io import dump_score_file, load_score_file
from sass.selection.dedup import deduplicate_scores
from sass.tasks.score_single_fragments import _conformer_files, _retention, dask_score_single_fragment
from sass.worker.merge_scores import merge_score_batch_task

WARTS_SEP = "_"


class InProcessClient:
    """Runs every task synchronously, so the fold logic is testable without a cluster."""

    def __init__(self):
        self.disk = DiskTracker("test")
        self.target_max_worker = 4
        self.dask = self

    def stream_map(self, func, data_iterator, **kwargs):
        for data in data_iterator:
            yield func(data, **kwargs)

    def submit(self, func, *args, pure=None, **kwargs):  # noqa: ARG002
        value = func(*args, **kwargs)
        return types.SimpleNamespace(result=lambda: value, done=lambda: True)


@pytest.fixture()
def scored(monkeypatch):
    """Stub the ROCS worker with one that writes a score file and remembers what was in it."""
    from sass.tasks import score_single_fragments as task

    written = {}

    def fake_score(conf_file, *, score_dir, **_kwargs):
        # Mirrors the real worker: `None` when the synthon has no single alignable connector.
        if "skip" in conf_file.name:
            return None
        n = int(conf_file.stem.split("_")[-1])
        rows = [[1.0 - (n * 10 + i) / 1000, f"{n * 10 + i}_1[U]"] for i in range(10)]
        out = score_dir / f"{conf_file.name}.pkl"
        dump_score_file(out, rows)
        written[out] = rows
        return out, out.stat().st_size

    monkeypatch.setattr(task, "score_single_fragment", fake_score)
    return written


def run_scoring(tmp_path, n_files, merge_fanin, top_m=10**6, extra=()):
    """`top_m` is passed to the task as `keep_top`."""
    conf_dir = tmp_path / "confs"
    conf_dir.mkdir()
    files = [conf_dir / f"reaction_1_synthon_0_conf_single_frag_{i}.oez" for i in range(n_files)]
    files += [conf_dir / name for name in extra]
    for f in files:
        f.touch()

    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    client = InProcessClient()
    out = dask_score_single_fragment(
        client,
        conf_files=files,
        query_fragment=None,
        synthon_connectors=["U"],
        frag_connectors=["He"],
        color_ff_dir=tmp_path,
        warts_separator=WARTS_SEP,
        score_dir=score_dir,
        keep_top=top_m,
        merge_fanin=merge_fanin,
    )
    return client, out


class TestFoldingWhileScoring:
    @pytest.mark.parametrize(("n_files", "merge_fanin"), [(24, 12), (25, 12), (1, 12), (12, 12), (100, 7)])
    def test_the_folded_rows_are_the_same_rows(self, tmp_path, scored, n_files, merge_fanin):
        """Folding neither loses nor duplicates rows."""
        _client, out = run_scoring(tmp_path, n_files, merge_fanin)

        got = []
        for f in out:
            got.extend(load_score_file(f))
        expected = deduplicate_scores(
            [row for rows in scored.values() for row in rows],
            sort_scores=True,
            warts_separator=WARTS_SEP,
            keep_isomer_warts=True,
        )
        assert sorted(map(tuple, got)) == sorted(map(tuple, expected))

    @pytest.mark.usefixtures("scored")
    def test_a_tail_shorter_than_the_fanin_is_carried_not_dropped(self, tmp_path):
        """25 files at fan-in 12 leave one unfolded file, which is still returned."""
        _client, out = run_scoring(tmp_path, n_files=25, merge_fanin=12)
        rows = [row for f in out for row in load_score_file(f)]
        assert len(rows) == 250, f"expected every file's 10 rows, got {len(rows)}"

    @pytest.mark.usefixtures("scored")
    def test_it_returns_far_fewer_files_than_it_scored(self, tmp_path):
        """Folding returns one file per `merge_fanin` inputs."""
        _client, out = run_scoring(tmp_path, n_files=120, merge_fanin=12)
        assert len(out) == 10, f"120 files at fan-in 12 should fold to 10, got {len(out)}"

    def test_folded_inputs_are_gone(self, tmp_path, scored):
        """The merge deletes the score files it consumed."""
        _client, _out = run_scoring(tmp_path, n_files=24, merge_fanin=12)
        assert not [f for f in scored if f.exists()], "per-chunk score files should be consumed"

    @pytest.mark.usefixtures("scored")
    def test_a_skipped_synthon_does_not_become_a_score_file(self, tmp_path):
        """The worker returns `None` when there is no single connector to align on."""
        _client, out = run_scoring(tmp_path, n_files=12, merge_fanin=12, extra=["skip_synthon_0_conf.oez"])
        rows = [row for f in out for row in load_score_file(f)]
        assert len(rows) == 120, "only the 12 real files should contribute"

    @pytest.mark.usefixtures("scored")
    def test_the_disk_peak_is_recorded(self, tmp_path):
        """Every scoring task reports its disk peak."""
        client, _out = run_scoring(tmp_path, n_files=24, merge_fanin=12)
        assert client.disk.label == "single_frag_scores"
        assert client.disk.peak > 0, "bytes written must be accounted for"

    def test_folding_lowers_the_peak_below_holding_every_file(self, tmp_path, scored):
        """The merge frees its inputs as it goes, so the high-water mark is not the total."""
        client, _out = run_scoring(tmp_path, n_files=120, merge_fanin=12)
        unfolded = sum(len(pickle.dumps(rows)) for rows in scored.values())
        assert client.disk.peak < unfolded, f"peak {client.disk.peak} should beat holding all {unfolded} bytes"


class TestRetentionIsNotTopM:
    """Single-fragment scoring retains `num_write_pose` rows, not `top_m`.

    Its scores are final, and pose writing reads only the first `num_write_pose` rows.
    """

    @staticmethod
    def ctx_with(num_write_pose, top_m=2_000_000):
        return types.SimpleNamespace(top_m=top_m, config={"num_write_pose": num_write_pose})

    def test_the_retention_is_the_pose_count(self):
        """The combined score file's only consumer reads exactly `num_write_pose` rows."""
        assert _retention(self.ctx_with(10_000)) == 10_000

    def test_top_m_does_not_enter_into_it(self):
        """Retention, not `top_m`, sets the merge width."""
        assert _retention(self.ctx_with(10_000, top_m=2_000_000)) == _retention(self.ctx_with(10_000, top_m=50))

    def test_asking_for_fewer_poses_retains_fewer(self):
        assert _retention(self.ctx_with(500)) == 500

    def test_asking_for_more_poses_retains_more(self):
        assert _retention(self.ctx_with(50_000)) == 50_000

    def test_the_retained_rows_are_the_genuine_top_of_all_of_them(self, tmp_path, scored):
        """Trimming early must give the same answer as trimming once at the end."""
        keep = 15
        _client, out = run_scoring(tmp_path, n_files=30, merge_fanin=4, top_m=keep)
        got = sorted((row[0] for f in out for row in load_score_file(f)), reverse=True)

        every_score = sorted((row[0] for rows in scored.values() for row in rows), reverse=True)
        assert got[:keep] == every_score[:keep], "the top `keep` scores must survive the folding"

    def test_nothing_below_the_cut_survives_the_final_combine(self, tmp_path, scored):
        """After the final combine, exactly the top `keep` scores remain."""
        keep = 15
        _client, out = run_scoring(tmp_path, n_files=30, merge_fanin=4, top_m=keep)
        merged = tmp_path / "final.pkl"
        merge_score_batch_task(out, merged, keep, WARTS_SEP, True)

        rows = load_score_file(merged)
        every_score = sorted((row[0] for rows_ in scored.values() for row in rows_), reverse=True)
        assert len(rows) == keep
        assert [r[0] for r in rows] == every_score[:keep]


class TestConformerFileSelection:
    @staticmethod
    def ctx_for(tmp_path, reactions, names):
        conf_dir = tmp_path / "confs"
        conf_dir.mkdir(exist_ok=True)
        for name in names:
            (conf_dir / name).touch()
        return types.SimpleNamespace(
            single_frag_synthon_conf_dir=conf_dir,
            reactions=reactions,
            config={"conf_file_format": "oez"},
        )

    def test_only_the_reactions_this_run_covers_are_selected(self, tmp_path):
        """Only files for the reactions in `ctx.reactions` are selected."""
        ctx = self.ctx_for(
            tmp_path,
            reactions=["reaction_11"],
            names=[
                "reaction_11_synthon_0_conf_single_frag_0.oez",
                "reaction_11_synthon_1_conf_single_frag_0.oez",
                "reaction_12_synthon_0_conf_single_frag_0.oez",
            ],
        )
        assert [f.name for f in _conformer_files(ctx)] == [
            "reaction_11_synthon_0_conf_single_frag_0.oez",
            "reaction_11_synthon_1_conf_single_frag_0.oez",
        ]

    def test_a_reaction_id_is_matched_whole(self, tmp_path):
        """`reaction_1` does not match the files of `reaction_11`."""
        ctx = self.ctx_for(
            tmp_path,
            reactions=["reaction_1"],
            names=["reaction_1_synthon_0_conf_single_frag_0.oez", "reaction_11_synthon_0_conf_single_frag_0.oez"],
        )
        assert [f.name for f in _conformer_files(ctx)] == ["reaction_1_synthon_0_conf_single_frag_0.oez"]

    def test_reaction_ids_with_letters_and_underscores_work(self, tmp_path):
        """Ids with underscores and a letter suffix, such as `reaction_3b`, are matched whole."""
        ctx = self.ctx_for(
            tmp_path,
            reactions=["reaction_3b"],
            names=["reaction_3b_synthon_2_conf_single_frag_33.oez"],
        )
        assert len(_conformer_files(ctx)) == 1

    def test_the_quarantine_directory_is_not_scored(self, tmp_path):
        """`generate_single_fragment_conformers` writes failures to `failed/`."""
        ctx = self.ctx_for(tmp_path, reactions=["reaction_11"], names=["reaction_11_synthon_0_conf_single_frag_0.oez"])
        (ctx.single_frag_synthon_conf_dir / "failed").mkdir()
        (ctx.single_frag_synthon_conf_dir / "failed" / "reaction_11_synthon_0_conf_fail.oez").touch()
        assert len(_conformer_files(ctx)) == 1

    def test_the_configured_conformer_format_is_honored(self, tmp_path):
        ctx = self.ctx_for(
            tmp_path,
            reactions=["reaction_11"],
            names=["reaction_11_synthon_0_conf_single_frag_0.oez", "reaction_11_synthon_0_conf_single_frag_1.oeb"],
        )
        assert [f.suffix for f in _conformer_files(ctx)] == [".oez"]

    def test_the_order_is_stable(self, tmp_path):
        """Files are returned in sorted order, whatever the directory order."""
        ctx = self.ctx_for(
            tmp_path,
            reactions=["reaction_11"],
            names=[f"reaction_11_synthon_0_conf_single_frag_{i}.oez" for i in (10, 2, 33, 1)],
        )
        assert [f.name for f in _conformer_files(ctx)] == sorted(f.name for f in _conformer_files(ctx))
