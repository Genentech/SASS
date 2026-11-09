"""Post-run disk hygiene: what gets compressed, and what gets deleted outright."""

import lzma
import pickle

import pytest

from sass.io import compress_pkl_files, discard_superseded_dir, load_intermediate_files


@pytest.fixture()
def run_dir(tmp_path):
    """A run directory holding one result file and one library artifact, plus a nested chunk."""
    (tmp_path / "pseudo_res").mkdir()
    for name in ("combined_pseudo_res.pkl", "pseudo_res/chunk_0.pkl", "synthon_data.pkl"):
        with (tmp_path / name).open("wb") as f:
            pickle.dump({"name": name}, f)
    return tmp_path


class TestCompression:
    def test_result_files_are_compressed_recursively(self, run_dir):
        compress_pkl_files(run_dir)
        assert not (run_dir / "combined_pseudo_res.pkl").exists()
        assert (run_dir / "combined_pseudo_res.pkl.xz").is_file()
        assert (run_dir / "pseudo_res" / "chunk_0.pkl.xz").is_file(), "rglob should reach nested chunks"

    def test_the_compressed_file_still_loads(self, run_dir):
        compress_pkl_files(run_dir)
        assert load_intermediate_files(run_dir / "combined_pseudo_res.pkl.xz") == {"name": "combined_pseudo_res.pkl"}

    def test_it_is_really_compressed(self, run_dir):
        compress_pkl_files(run_dir)
        with lzma.open(run_dir / "combined_pseudo_res.pkl.xz", "rb") as f:
            assert pickle.load(f)["name"] == "combined_pseudo_res.pkl"


class TestLibraryArtifactsAreKept:
    """Paths passed as `keep` are not compressed, since library artifacts are resolved by exact path."""

    def test_a_kept_path_is_left_alone(self, run_dir):
        compress_pkl_files(run_dir, keep=[run_dir / "synthon_data.pkl"])
        assert (run_dir / "synthon_data.pkl").is_file()
        assert not (run_dir / "synthon_data.pkl.xz").exists()

    def test_keeping_one_does_not_spare_the_others(self, run_dir):
        compress_pkl_files(run_dir, keep=[run_dir / "synthon_data.pkl"])
        assert (run_dir / "combined_pseudo_res.pkl.xz").is_file()

    def test_keep_accepts_strings_and_unrelated_paths(self, run_dir, tmp_path):
        """`keep` is built from config values, which are strings and may point outside the run."""
        compress_pkl_files(run_dir, keep=[str(run_dir / "synthon_data.pkl"), str(tmp_path / "elsewhere.pkl")])
        assert (run_dir / "synthon_data.pkl").is_file()

    def test_keep_matches_through_a_non_normalized_path(self, run_dir):
        """Config paths are user-written; `./x/../synthon_data.pkl` must still match."""
        compress_pkl_files(run_dir, keep=[run_dir / "pseudo_res" / ".." / "synthon_data.pkl"])
        assert (run_dir / "synthon_data.pkl").is_file()


class TestDiscardSupersededDir:
    """`pseudo_res/` and `rescore_results/` duplicate the combined file written beside them."""

    @pytest.fixture()
    def chunks(self, tmp_path):
        d = tmp_path / "pseudo_res"
        (d / "nested").mkdir(parents=True)
        (d / "merged_2comp.pkl").write_bytes(b"x" * 100)
        (d / "nested" / "merged_3comp.pkl").write_bytes(b"y" * 100)
        return d

    def test_the_directory_goes_once_the_combined_file_exists(self, chunks, tmp_path):
        combined = tmp_path / "combined_pseudo_res.pkl"
        combined.write_bytes(b"z" * 10)
        discard_superseded_dir(chunks, combined)
        assert not chunks.exists()
        assert combined.is_file(), "the file that supersedes it must survive"

    def test_a_missing_combined_file_keeps_the_chunks(self, chunks, tmp_path):
        """They are the only copy of those scores; deleting them would lose the run."""
        discard_superseded_dir(chunks, tmp_path / "never_written.pkl")
        assert chunks.is_dir()

    def test_an_empty_combined_file_keeps_the_chunks(self, chunks, tmp_path):
        """A merge that produced nothing is a failure, not a reason to delete its inputs."""
        empty = tmp_path / "combined_pseudo_res.pkl"
        empty.touch()
        discard_superseded_dir(chunks, empty)
        assert chunks.is_dir()

    def test_disabling_keeps_the_chunks(self, chunks, tmp_path):
        combined = tmp_path / "combined_pseudo_res.pkl"
        combined.write_bytes(b"z" * 10)
        discard_superseded_dir(chunks, combined, enabled=False)
        assert chunks.is_dir()

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        combined = tmp_path / "combined_pseudo_res.pkl"
        combined.write_bytes(b"z" * 10)
        discard_superseded_dir(tmp_path / "never_created", combined)
