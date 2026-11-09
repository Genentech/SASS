"""End-to-end runs of every task group against the synthetic library.

The library is small enough to enumerate in full, so the synthon search is checked against
exhaustive ground truth. The query is itself a library product, so the right answer is known.
"""

from pathlib import Path

import pytest
from openeye import oechem

from sass.analysis.results import ResultAnalysis
from sass.io import load_intermediate_files

from .conftest import (
    KNOWN_FRAGMENT_SMILES,
    KNOWN_PRODUCT_SMILES,
    KNOWN_PRODUCT_SYNTHONS,
    N_PRODUCTS,
    config_against_library,
    run_sass,
    write_3d_query,
    write_query_fragment,
)

#: Each run takes longer than the project-wide unit-test timeout.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

#: Synthons in `tests/data/test_library/synthons.txt`.
N_SYNTHONS = 20

#: Coordinates below this are indistinguishable from an unset atom position.
ORIGIN_TOL = 1e-6

#: A perfect TanimotoCombo. The query conformer comes from the same `product_omega_*` ensemble.
SELF_MATCH = 2.0


def _load(path: Path):
    """Load a run artifact, whether or not `compress_pkl_files` renamed it to `.pkl.xz`."""
    if not path.exists():
        path = path.with_suffix(".pkl.xz")
    return load_intermediate_files(path)


def _count_mols(conf_file: Path) -> int:
    """Molecules (not conformers) in an OpenEye multi-conformer file."""
    return sum(1 for _ in oechem.oemolistream(str(conf_file)).GetOEMols())


def _decode(raw_id: str, id_index: list[str]) -> tuple[str, ...]:
    """`'1&4&7_1'` -> the original synthon ids, dropping the isomer wart."""
    base = raw_id.split("_")[0]
    return tuple(id_index[int(part)] for part in base.split("&"))


# --------------------------------------------------------------------------------------
# Library build
# --------------------------------------------------------------------------------------


def test_library_build(built_library):
    """The three build tasks produce a loadable library from the two committed tables."""
    for key in ("synthon_id_index", "synthon_data_file", "grouped_synthon_file", "synthon_conf_dir"):
        assert built_library[key].exists(), f"{key} missing"

    id_index = _load(built_library["synthon_id_index"])
    assert len(id_index) == N_SYNTHONS
    assert len(set(id_index)) == N_SYNTHONS, "ids must be unique -- position in this list IS the id"
    assert id_index == sorted(id_index), "index is built sorted; lookups rely on a stable order"

    # One conformer file per (reaction, component), holding one molecule per synthon.
    conf_files = sorted(built_library["synthon_conf_dir"].glob("*.oez"))
    assert {f.name: _count_mols(f) for f in conf_files} == {
        "t_bisamide_synthon_0_conf.oez": 3,  # diamines
        "t_bisamide_synthon_1_conf.oez": 3,  # first acyl
        "t_bisamide_synthon_2_conf.oez": 3,  # second cap
        "t_join_synthon_0_conf.oez": 6,  # acyl
        "t_join_synthon_1_conf.oez": 5,  # amine
    }


def test_every_stored_conformer_has_real_coordinates(built_library):
    """No molecule may be written with every atom sitting on the origin."""
    for conf_file in sorted(built_library["synthon_conf_dir"].glob("*.oez")):
        for mol in oechem.oemolistream(str(conf_file)).GetOEMols():
            assert mol.NumConfs() >= 1, f"{mol.GetTitle()} in {conf_file.name} has no conformer"
            coords = oechem.OEFloatArray(mol.NumAtoms() * 3)
            mol.GetCoords(coords)
            assert any(
                abs(value) > ORIGIN_TOL for value in coords
            ), f"{mol.GetTitle()} in {conf_file.name} has all atoms at the origin"


# --------------------------------------------------------------------------------------
# Synthon search
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def query_run(out_root, built_library) -> Path:
    """Score a query that is itself a library product, all the way to written poses."""
    exp_dir = out_root / "query"
    exp_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    config = config_against_library(
        built_library, exp_dir, ["score_synthons", "rescore_products", "write_product_poses"]
    )
    # Config first: the query conformer must come from the same OMEGA settings this run uses.
    config["query_molecule"] = str(
        write_3d_query(KNOWN_PRODUCT_SMILES, "known_bisamide_product", exp_dir / "query.sdf", config)
    )
    return run_sass(config, exp_dir)


def test_query_recovers_known_product(query_run, built_library):
    """The query IS a product of the library, so the search must rank it first, at 2.0."""
    id_index = _load(built_library["synthon_id_index"])
    scores = _load(query_run / "rescore_combined_rocs_res.pkl")
    assert scores, "no rescored products"

    top_score, top_id = scores[0][0], scores[0][1]
    assert _decode(top_id, id_index) == KNOWN_PRODUCT_SYNTHONS
    assert top_score == pytest.approx(
        SELF_MATCH, abs=1e-6
    ), "the query is its own product; expect a perfect self-overlay"
    assert scores == sorted(scores, key=lambda row: row[0], reverse=True), "results must be score-ordered"


def test_query_poses_are_written(query_run):
    """`write_product_poses` writes a readable SDF, query first, then hits in score order."""
    poses = list(query_run.glob("rescore_top_*_mol_poses.sdf"))
    assert len(poses) == 1, f"expected one pose file, found {poses}"

    titles = [mol.GetTitle() for mol in oechem.oemolistream(str(poses[0])).GetOEGraphMols()]
    assert titles[0] == "known_bisamide_product", "the reference query is written first"
    assert len(titles) > 1, "no hit poses written"
    # Titles are restored to original synthon ids, not left as the internal integers.
    assert all(not title.split("&")[0].isdigit() for title in titles[1:])
    assert "&".join(KNOWN_PRODUCT_SYNTHONS) in titles[1]


# --------------------------------------------------------------------------------------
# Full enumeration (ground truth)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ground_truth_run(out_root, built_library) -> Path:
    """Enumerate, generate conformers for, and score every product in the library."""
    exp_dir = out_root / "ground_truth"
    exp_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    config = config_against_library(
        built_library,
        exp_dir,
        ["enumerate_full_library", "generate_full_library_conformers", "score_full_library"],
    )
    config["query_molecule"] = str(
        write_3d_query(KNOWN_PRODUCT_SMILES, "known_bisamide_product", exp_dir / "query.sdf", config)
    )
    return run_sass(config, exp_dir)


def test_ground_truth_enumeration(ground_truth_run, built_library):
    """Exhaustive enumeration yields exactly the combinatorial product count."""
    id_index = _load(built_library["synthon_id_index"])
    gt = _load(ground_truth_run / "ground_truth_combined_rocs_res.pkl")

    # t_join: 6 x 5 = 30. t_bisamide: 3 x 3 x 3 = 27.
    assert len(gt) == N_PRODUCTS

    top_score, top_id = gt[0][0], gt[0][1]
    assert _decode(top_id, id_index) == KNOWN_PRODUCT_SYNTHONS
    assert top_score == pytest.approx(SELF_MATCH, abs=1e-6)


def test_recall_against_ground_truth(query_run, ground_truth_run):
    """The synthon search must recover the exhaustive top hits it is an approximation of."""
    gt = _load(ground_truth_run / "ground_truth_combined_rocs_res.pkl")
    exp = _load(query_run / "rescore_combined_rocs_res.pkl")

    analysis = ResultAnalysis(true_scores=gt, exp_scores=exp)
    for top_n in (1, 5, 10, 20):
        assert analysis.calc_single_recall(top_n) == 1.0, f"recall@{top_n} below 1.0"


# --------------------------------------------------------------------------------------
# Single-fragment scoring
# --------------------------------------------------------------------------------------


def test_single_fragment_scoring(out_root, built_library):
    """A query fragment must rank the synthon it was carved from first.

    The fragment is cleaved off the known product, so `t_bisamide_c_isonicotinoyl` is the right answer.
    """
    exp_dir = out_root / "single_fragment"
    exp_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    query = write_query_fragment(
        KNOWN_PRODUCT_SMILES, KNOWN_FRAGMENT_SMILES, "query_fragment_isonicotinoyl", exp_dir / "query.sdf"
    )
    config = config_against_library(
        built_library,
        exp_dir,
        ["generate_single_fragment_conformers", "score_single_fragments", "write_single_fragment_poses"],
    )
    config["query_molecule"] = str(query)
    config["single_frag_synthon_conf_dir"] = str(exp_dir / "single_frag_conformers")
    run_sass(config, exp_dir)

    id_index = _load(built_library["synthon_id_index"])
    scores = _load(exp_dir / "combined_single_fragment_res.pkl")
    assert scores, "single-fragment scoring produced nothing"

    # Ids here carry a trailing connector tag, e.g. `7_1[Np]`; `_decode` drops the wart and
    # the tag along with it.
    assert _decode(scores[0][1], id_index) == ("t_bisamide_c_isonicotinoyl",)

    # The count in the name comes from `num_write_pose`, so match on the pattern.
    poses = list(exp_dir.glob("top_*_single_frag_synthon_poses.sdf"))
    assert len(poses) == 1, f"expected one pose file, found {poses}"
    titles = [mol.GetTitle() for mol in oechem.oemolistream(str(poses[0])).GetOEGraphMols()]
    assert titles[0] == "query_fragment_isonicotinoyl"
    # Restored ids keep the connector tag, and the id in front of it must still be translated.
    assert any(title.endswith(("[U]", "[Np]")) for title in titles[1:])


# --------------------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------------------


def test_two_dimensional_query_is_rejected(out_root, built_library):
    """A query without coordinates cannot be a ROCS reference, and must fail before any work."""
    exp_dir = out_root / "flat_query"
    exp_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    flat = exp_dir / "flat.smi"
    flat.write_text(f"{KNOWN_PRODUCT_SMILES} flat_query\n")

    config = config_against_library(built_library, exp_dir, ["score_synthons"])
    config["query_molecule"] = str(flat)
    # `run_sass` re-raises as AssertionError with the run log attached.
    with pytest.raises(AssertionError, match="not a 3D conformer"):
        run_sass(config, exp_dir)
