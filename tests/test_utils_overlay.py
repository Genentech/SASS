"""Tests for sass.utils.utils_overlay module."""

import json
import sys

import pytest
import pytest_check as check
from openeye import oeomega

from sass import TEST_DIR
from sass.utils.utils_confgen import build_conformer
from sass.utils.utils_mol import load_first_mol
from sass.utils.utils_overlay import overlay_opt_builder, sp_simple_oeoverlay

DATA_DIR = TEST_DIR / "data"


@pytest.fixture(scope="module")
def expected_rocs_scores():
    with (DATA_DIR / "expected_rocs_scores.json").open("r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def overlay_opts():
    return overlay_opt_builder()


@pytest.mark.parametrize("opt_shape_func", ["GetTanimoto", "GetRefTversky"])
@pytest.mark.parametrize("opt_color_func", ["GetColorTanimoto", None, "GetRefColorTversky"])
@pytest.mark.parametrize("overlap_shape_func", ["GetTanimoto", "GetRefTversky"])
@pytest.mark.parametrize("overlap_color_func", ["GetColorTanimoto", None, "GetRefColorTversky"])
# @pytest.mark.parametrize("opt_shape_func", ["GetTanimoto"])
# @pytest.mark.parametrize("opt_color_func", ["GetColorTanimoto", None])
# @pytest.mark.parametrize("overlap_shape_func", ["GetRefTversky"])
# @pytest.mark.parametrize("overlap_color_func", [None])
def test_various_shape_color_func(
    opt_shape_func: str,
    opt_color_func: str,
    overlap_shape_func: str,
    overlap_color_func: str,
    expected_rocs_scores,
    overlay_opts,
):
    """Function outputs correct scores with various shape and color functions."""
    fit_mol = load_first_mol(str(DATA_DIR / "test_query_4.sdf"))
    omega = oeomega.OEOmega()
    omega.GetOptions().SetMaxConfs(25)
    build_conformer(fit_mol, omega)

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

    check.equal(
        out,
        expected_rocs_scores["test_various_shape_color_func"][str(opt_shape_func)][str(opt_color_func)][
            str(overlap_shape_func)
        ][str(overlap_color_func)],
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

    check.equal(out, expected_rocs_scores["test_sp_simple_oeoverlay"][str(opt_key)][str(score_key)])


if __name__ == "__main__":
    sys.exit(pytest.main())
