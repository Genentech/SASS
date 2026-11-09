"""Tests for the fused instantiate -> confgen -> ROCS rescore task.

`instantiate_confgen_score` instantiates, builds conformers and scores in one task, and writes
only scores. The tests use a plain amide-coupling SMIRKS, so they need no library data.
"""

import pickle

import pytest
from openeye import oechem

from sass.chem.overlay import overlay_opt_builder
from sass.ids import extract_base_id
from sass.worker.rescore_products import instantiate_confgen_score

SMIRKS = "[C:1](=[O:2])[OH].[N:3][H]>>[C:1](=[O:2])[N:3]"
ACIDS = ["CC(C)(C)CCC(=O)O", "OC(=O)C1CCCCC1", "CCCCC(=O)O"]
AMINE = "NCCc1ccccc1"
WARTS_SEP = "_"
SASS_SEP = "&"


@pytest.fixture(scope="module")
def color_ff_dir(tmp_path_factory):
    """Color force field files, as a run writes them for the workers."""
    d = tmp_path_factory.mktemp("color_ff")
    for key, opt in overlay_opt_builder().items():
        opt.GetColorOptions().GetColorForceField().Write(oechem.oeofstream(str(d / f"custom_color_ff_{key}.txt")))
    return d


@pytest.fixture(scope="module")
def query_mol():
    mol = oechem.OEMol()
    oechem.OEReadMolecule(oechem.oemolistream("tests/data/test_query_4.sdf"), mol)
    return mol


def make_chunk(n, rxn_id="rxn_a"):
    """A chunk shaped exactly like `generate_instantiation_tasks` output."""
    return [([(f"acid{i}", ACIDS[i % len(ACIDS)], 0), (f"amine{i}", AMINE, 1)], SMIRKS, rxn_id) for i in range(n)]


def run_task(chunk, out_dir, query_mol, color_ff_dir, idx=0, **overrides):
    kwargs = {
        "output_dir": out_dir,
        "ref_mol": query_mol,
        "opt_shape_func": "GetTanimoto",
        "opt_color_func": "GetColorTanimoto",
        "overlap_shape_func": "GetTanimoto",
        "overlap_color_func": "GetColorTanimoto",
        "color_ff_dir": color_ff_dir,
        "warts_separator": WARTS_SEP,
        "title_separator": SASS_SEP,
        "max_stereocenter": 3,
        "top_n": 1000,
        "omega_max_conf": 10,
        "omega_max_time": 20,
        "omega_e_window": 10,
    }
    kwargs.update(overrides)
    return instantiate_confgen_score((chunk, idx), **kwargs)


def test_scores_products_and_writes_no_conformers(tmp_path, query_mol, color_ff_dir):
    """Scores are written, and conformer and product files are not."""
    score_file, pose_file, nbytes = run_task(make_chunk(4), tmp_path, query_mol, color_ff_dir)

    rows = pickle.load(score_file.open("rb"))
    assert len(rows) == 4, "one row per product after base-id dedup"
    assert pose_file is None
    assert nbytes == score_file.stat().st_size

    leftovers = list(tmp_path.glob("*.oez")) + list(tmp_path.glob("*.oeb.gz"))
    assert leftovers == [], f"conformers must not be written: {leftovers}"


def test_every_row_carries_its_reaction_id(tmp_path, query_mol, color_ff_dir):
    """Each score row carries the reaction id of its own product.

    The chunk mixes two reactions, so stamping one reaction on every row would fail.
    """
    chunk = make_chunk(3, rxn_id="rxn_a") + make_chunk(3, rxn_id="rxn_b")
    # Distinguish the two halves by synthon id so attribution can be checked.
    chunk = [
        ([(f"{sid}_{tag}".replace("_", "x"), smi, pos) for sid, smi, pos in reactants], smirks, rxn)
        for tag, (reactants, smirks, rxn) in enumerate(chunk)
    ]
    score_file, _, _ = run_task(chunk, tmp_path, query_mol, color_ff_dir)

    rows = pickle.load(score_file.open("rb"))
    assert all(len(r) == 3 for r in rows), "rows must be (score, title, rxn_id)"

    expected = {}
    for reactants, _smirks, rxn in chunk:
        expected[SASS_SEP.join(sorted(sid for sid, _s, _p in reactants))] = rxn
    for _score, title, rxn_id in rows:
        base = extract_base_id(title, warts_separator=WARTS_SEP, keep_isomer_warts=False)
        assert rxn_id == expected[base], f"{base} attributed to {rxn_id}"
    assert {r[2] for r in rows} == {"rxn_a", "rxn_b"}, "both reactions must survive"


def test_pose_mode_writes_only_the_requested_isomers(tmp_path, query_mol, color_ff_dir):
    """`write_product_poses` regenerates only the selected isomers, not every one."""
    chunk = make_chunk(4)
    score_file, _, _ = run_task(chunk, tmp_path, query_mol, color_ff_dir)
    rows = pickle.load(score_file.open("rb"))

    wanted = {r[1] for r in rows[:2]}
    pose_set_file = tmp_path / "wanted.pkl"
    with pose_set_file.open("wb") as f:
        pickle.dump(wanted, f)

    _, pose_file, _ = run_task(
        chunk, tmp_path, query_mol, color_ff_dir, idx=1, pose_set_file=pose_set_file, sep_shape_color_score=True
    )

    assert pose_file.is_file()
    written = [mol.GetTitle() for mol in oechem.oemolistream(str(pose_file)).GetOEMols()]
    assert set(written) == wanted
    assert len(written) == len(wanted), "no duplicates"


def test_omega_return_code_is_read_as_a_code_not_a_bool(tmp_path, query_mol, color_ff_dir):
    """`OEOmega.Build` returns 0 on success, so successful builds must still produce scores."""
    score_file, _, _ = run_task(make_chunk(3), tmp_path, query_mol, color_ff_dir)

    rows = pickle.load(score_file.open("rb"))
    assert rows, "empty scores means successful OMEGA builds were thrown away"
    assert all(isinstance(r[0], float) for r in rows)


def test_pose_mode_writes_no_score_file(tmp_path, query_mol, color_ff_dir):
    """Pose mode writes poses and no score file."""
    chunk = make_chunk(3)
    score_file, _, _ = run_task(chunk, tmp_path, query_mol, color_ff_dir)
    wanted = {r[1] for r in pickle.load(score_file.open("rb"))[:2]}
    pose_set_file = tmp_path / "wanted.pkl"
    with pose_set_file.open("wb") as f:
        pickle.dump(wanted, f)
    score_file.unlink()

    returned_score_file, pose_file, _ = run_task(
        chunk, tmp_path, query_mol, color_ff_dir, idx=1, pose_set_file=pose_set_file
    )

    assert returned_score_file is None, "pose mode must not report a score file"
    assert pose_file.is_file()
    assert list(tmp_path.glob("*.score.pkl")) == [], "pose mode must not write a score file"


def test_pose_titles_survive_retitling(tmp_path, query_mol, color_ff_dir):
    """Retitled poses keep their new titles when written and read back.

    A pose whose conformer kept the old title would be written with that title instead.
    """
    chunk = make_chunk(2)
    score_file, _, _ = run_task(chunk, tmp_path, query_mol, color_ff_dir)
    wanted = {r[1] for r in pickle.load(score_file.open("rb"))}
    pose_set_file = tmp_path / "wanted.pkl"
    with pose_set_file.open("wb") as f:
        pickle.dump(wanted, f)

    _, pose_file, _ = run_task(chunk, tmp_path, query_mol, color_ff_dir, idx=1, pose_set_file=pose_set_file)

    out = tmp_path / "renamed.sdf"
    ostream = oechem.oemolostream(str(out))
    for mol in oechem.oemolistream(str(pose_file)).GetOEGraphMols():
        mol.SetTitle(f"RENAMED_{mol.GetTitle()}")
        oechem.OEWriteMolecule(ostream, mol)
    ostream.close()

    written = [m.GetTitle() for m in oechem.oemolistream(str(out)).GetOEMols()]
    assert written, "no poses written"
    assert all(t.startswith("RENAMED_") for t in written), f"retitling was discarded: {written}"
