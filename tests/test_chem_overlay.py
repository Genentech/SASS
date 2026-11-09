"""Tests for sass.chem.overlay module."""

import json
import sys

import pytest
import pytest_check as check
from openeye import oechem

from sass import TEST_DIR
from sass.chem.mol import load_first_mol
from sass.chem.overlay import overlay_opt_builder, sp_simple_oeoverlay

DATA_DIR = TEST_DIR / "data"


@pytest.fixture(scope="module")
def expected_rocs_scores():
    with (DATA_DIR / "expected_rocs_scores.json").open("r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def overlay_opts():
    return overlay_opt_builder()


@pytest.fixture(scope="module")
def fit_mol():
    """Use OETK2023-generated conformers for reproducibility."""
    return load_first_mol(str(DATA_DIR / "test_query_4_conformer_oetk_2023.1.0.oeb"))


@pytest.mark.parametrize("opt_shape_func", ["GetTanimoto", "GetRefTversky"])
@pytest.mark.parametrize("opt_color_func", ["GetColorTanimoto", None, "GetRefColorTversky"])
@pytest.mark.parametrize("overlap_shape_func", ["GetTanimoto", "GetRefTversky"])
@pytest.mark.parametrize("overlap_color_func", ["GetColorTanimoto", None, "GetRefColorTversky"])
def test_various_shape_color_func(  # noqa: PLR0913
    opt_shape_func: str,
    opt_color_func: str,
    overlap_shape_func: str,
    overlap_color_func: str,
    expected_rocs_scores,
    overlay_opts,
    fit_mol,
):
    """Function outputs correct scores with various shape and color functions."""
    out = sp_simple_oeoverlay(
        ref_mol=load_first_mol(str(DATA_DIR / "test_query_1.sdf")),
        conf_file=None,
        overlay_opts=overlay_opts,
        fit_mols=[fit_mol],
        rocs_key_opt=0,
        rocs_key_score=0,
        opt_shape_func=opt_shape_func,
        opt_color_func=opt_color_func,
        overlap_shape_func=overlap_shape_func,
        overlap_color_func=overlap_color_func,
    )[0][0]

    check.almost_equal(
        out,
        expected_rocs_scores["test_various_shape_color_func"][str(opt_shape_func)][str(opt_color_func)][
            str(overlap_shape_func)
        ][str(overlap_color_func)],
        abs=1e-5,
    )


@pytest.mark.parametrize("opt_key", [0, 1, 2, 3])
@pytest.mark.parametrize("score_key", [0, 1, 2, 3])
def test_sp_simple_oeoverlay(
    opt_key: int,
    score_key: int,
    expected_rocs_scores,
    overlay_opts,
):
    """Function outputs correct scores with various optimization and score options."""
    out = sp_simple_oeoverlay(
        ref_mol=load_first_mol(str(DATA_DIR / "rocs_test_conf1.sdf")),
        conf_file=str(DATA_DIR / "rocs_test_conf2.sdf"),
        overlay_opts=overlay_opts,
        rocs_key_opt=opt_key,
        rocs_key_score=score_key,
        opt_shape_func="GetTanimoto",
        opt_color_func="GetColorTanimoto",
        overlap_shape_func="GetTanimoto",
        overlap_color_func="GetColorTanimoto",
    )[0][0]

    check.almost_equal(
        out,
        expected_rocs_scores["test_sp_simple_oeoverlay"][str(opt_key)][str(score_key)],
        abs=1e-5,
    )


@pytest.mark.parametrize("overlap_color_func", ["GetColorTanimoto", None])
def test_scores_are_titled_by_molecule_not_by_conformer(tmp_path, overlay_opts, overlap_color_func):
    """Score rows carry the molecule's title, not its conformers' titles.

    `overlap_color_func=None` takes the re-scoring branch, and "GetColorTanimoto" takes the direct one.
    """
    fit = load_first_mol(str(DATA_DIR / "rocs_test_conf2.sdf"))
    fit.SetTitle("12345_1")
    for conf in fit.GetConfs():
        conf.SetTitle("a-string-id_1")
    conf_file = tmp_path / "stale_titles.oeb"
    ostream = oechem.oemolostream(str(conf_file))
    oechem.OEWriteMolecule(ostream, fit)
    ostream.close()

    out = sp_simple_oeoverlay(
        ref_mol=load_first_mol(str(DATA_DIR / "rocs_test_conf1.sdf")),
        conf_file=str(conf_file),
        overlay_opts=overlay_opts,
        rocs_key_opt=1,
        rocs_key_score=1,
        opt_shape_func="GetTanimoto",
        opt_color_func="GetColorTanimoto",
        overlap_shape_func="GetTanimoto",
        overlap_color_func=overlap_color_func,
    )

    assert [title for _score, title in out] == ["12345_1"]


def test_written_pose_carries_the_same_title_as_its_score_row(tmp_path, overlay_opts):
    """The pose SDF is filtered by the titles in the score rows, so the two must agree."""
    fit = load_first_mol(str(DATA_DIR / "rocs_test_conf2.sdf"))
    fit.SetTitle("12345_1")
    for conf in fit.GetConfs():
        conf.SetTitle("a-string-id_1")
    conf_file = tmp_path / "stale_titles.oeb"
    ostream = oechem.oemolostream(str(conf_file))
    oechem.OEWriteMolecule(ostream, fit)
    ostream.close()
    pose_file = tmp_path / "poses.sdf"

    out = sp_simple_oeoverlay(
        ref_mol=load_first_mol(str(DATA_DIR / "rocs_test_conf1.sdf")),
        conf_file=str(conf_file),
        overlay_opts=overlay_opts,
        rocs_key_opt=1,
        rocs_key_score=1,
        opt_shape_func="GetTanimoto",
        opt_color_func="GetColorTanimoto",
        overlap_shape_func="GetTanimoto",
        overlap_color_func="GetColorTanimoto",
        write_out_overlay=True,
        out_file=pose_file,
    )

    written = [mol.GetTitle() for mol in oechem.oemolistream(str(pose_file)).GetOEGraphMols()]
    assert written == [title for _score, title in out] == ["12345_1"]


if __name__ == "__main__":
    sys.exit(pytest.main())
