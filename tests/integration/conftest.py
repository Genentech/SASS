"""Fixtures for the end-to-end tests: build the synthetic library once, then query it.

Only the two source tables under `tests/data/test_library/` are committed, and the rest of the
library is built by the real tasks. Outputs land in `tests/_output/integration/` rather than a
`tmp_path`, so a failed run can be inspected.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from openeye import oechem, oeomega

from sass import SRC_DIR, TEST_DIR
from sass.chem.confgen import omega_opt_builder
from sass.chem.fragment import fragment_molecule

#: Committed source tables, the only persisted part of the library.
LIBRARY_SRC = TEST_DIR / "data" / "test_library"
SYNTHON_FILE = LIBRARY_SRC / "synthons.txt"
REACTION_FILE = LIBRARY_SRC / "reactions.txt"

#: Kept out of `tmp_path` on purpose; see the module docstring.
OUT_ROOT = TEST_DIR / "_output" / "integration"

#: The full `t_join` (6 x 5) + `t_bisamide` (3 x 3 x 3) product space.
N_PRODUCTS = 57

#: Defaults from `config_template.yml`, mirrored so fragmentation here matches `score_synthons`.
FRAG_CONNECTORS = ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"]
HEAVY_ATOM_LIMIT = 5

#: A molecule that IS a library product: t_bisamide_a_piperazine + _b_pentanoyl + _c_isonicotinoyl.
KNOWN_PRODUCT_SMILES = "CCCCC(=O)N1CCN(CC1)C(=O)c1ccncc1"

#: The isonicotinoyl cap of `KNOWN_PRODUCT_SMILES`, i.e. `t_bisamide_c_isonicotinoyl`.
KNOWN_FRAGMENT_SMILES = "[1He]C(=O)c1ccncc1"
KNOWN_PRODUCT_SYNTHONS = (
    "t_bisamide_a_piperazine",
    "t_bisamide_b_pentanoyl",
    "t_bisamide_c_isonicotinoyl",
)


def base_config(exp_dir: Path, tasks: list[str]) -> dict:
    """Default config, pointed at `exp_dir` and running `tasks` on a 1-worker local cluster."""
    with (SRC_DIR / "config_template.yml").open("r") as f:
        config = yaml.safe_load(f)
    config["exp_dir"] = str(exp_dir)
    config["tasks"] = tasks
    # Unset, so no key resolves to a library named in the site's template.
    config["synthon_library"] = None
    config["synthon_file"] = str(SYNTHON_FILE)
    config["reaction_file"] = str(REACTION_FILE)
    config["dask_cluster_type"] = "local"
    config["num_workers"] = 1
    # `compress_pkl_files` stays at its default, so the library build must survive compression.
    config["clean_up_dask_logs"] = True
    # Small library: keep conformer counts low so a full build stays in the tens of seconds.
    config["synthon_omega_max_conf"] = 10
    config["top_m"] = 200
    return config


def sass_executable() -> str:
    """The `sass` console script belonging to the interpreter running the tests.

    It is looked up next to `sys.executable` first, since the venv's `bin` may not be on `PATH`.
    """
    candidate = Path(sys.executable).parent / "sass"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("sass")
    if found is None:
        raise RuntimeError("`sass` console script not found; is the package installed?")
    return found


def run_sass(config: dict, exp_dir: Path) -> Path:
    """Write `config` into `exp_dir` and run it through the real `sass run` CLI.

    On failure, raises with the captured output and the tail of the run log attached.
    """
    exp_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    config_file = exp_dir / "config.yml"
    with config_file.open("w") as f:
        yaml.dump(config, f, sort_keys=False)
    # `cwd` matters: a SASS run chdirs into `exp_dir`, and dask drops scratch in the cwd.
    result = subprocess.run(
        [sass_executable(), "run", "--config", str(config_file)],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logs = sorted(exp_dir.glob("sass_*.log"))
        tail = logs[-1].read_text().splitlines()[-40:] if logs else ["<no run log was written>"]
        raise AssertionError(
            f"`sass run` failed with exit {result.returncode} for {config_file}\n"
            f"--- stdout ---\n{_excerpt(result.stdout)}\n"
            f"--- stderr ---\n{_excerpt(result.stderr)}\n"
            f"--- {logs[-1].name if logs else 'run log'} (last 40 lines) ---\n" + "\n".join(tail)
        )
    return exp_dir


def _excerpt(text: str, head: int = 3000, tail: int = 2000) -> str:
    """Both ends of a captured stream.

    The child's error is printed before the CLI's own traceback, so the head matters too.
    """
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n... [{len(text) - head - tail} characters elided] ...\n{text[-tail:]}"


def write_3d_query(smiles: str, title: str, out_file: Path, config: dict) -> Path:
    """Write a single-conformer 3D SDF, drawn from the ensemble the pipeline will regenerate.

    OMEGA runs with this config's `product_omega_*` settings, which makes an exact self-match possible.
    """
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    omega = oeomega.OEOmega(
        omega_opt_builder(
            max_conf=config["product_omega_max_conf"],
            max_search_time=config["product_omega_max_time"],
            energy_window=config["product_omega_energy"],
        )
    )
    if not omega(mol):
        raise RuntimeError(f"OMEGA failed on {smiles}")

    # One conformer only, since `validate_query` rejects a multi-conformer query.
    query = oechem.OEGraphMol(next(iter(mol.GetConfs())))
    query.SetTitle(title)
    ofs = oechem.oemolostream(str(out_file))
    oechem.OEWriteMolecule(ofs, query)
    ofs.close()
    return out_file


def write_query_fragment(parent_smiles: str, fragment_smiles: str, title: str, out_file: Path) -> Path:
    """Cleave a 3D parent into two fragments and write the one matching `fragment_smiles`.

    The fragment is derived with the same `fragment_molecule` arguments `score_synthons` uses, and
    keeps the parent's coordinates. OMEGA cannot build it from its own SMILES, since it cannot
    parameterize the connector atoms.
    """
    parent = oechem.OEMol()
    oechem.OESmilesToMol(parent, parent_smiles)
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetStrictStereo(False)
    if not omega(parent):
        raise RuntimeError(f"OMEGA failed on {parent_smiles}")

    fragment_sets = fragment_molecule(
        mol=parent,
        n_fragments=2,
        heavy_atom_limit=HEAVY_ATOM_LIMIT,
        cleave_acyclic_bonds=True,
        cleave_cyclic_bonds=False,
        connector_isotope=1,  # matches `score_synthons`; marks connectors for `check_fragment_size`
        connector_atoms=FRAG_CONNECTORS,
    )
    wanted = oechem.OEMol()
    oechem.OESmilesToMol(wanted, fragment_smiles)
    canonical = oechem.OEMolToSmiles(wanted)
    for fragment_set in fragment_sets:
        for fragment in fragment_set:
            if oechem.OEMolToSmiles(fragment) == canonical:
                fragment.SetTitle(title)
                ofs = oechem.oemolostream(str(out_file))
                oechem.OEWriteMolecule(ofs, fragment)
                ofs.close()
                return out_file
    found = sorted({oechem.OEMolToSmiles(f) for fs in fragment_sets for f in fs})
    raise ValueError(f"{fragment_smiles} is not a 2-way fragment of {parent_smiles}; got {found}")


@pytest.fixture(scope="session")
def out_root() -> Path:
    """A clean, inspectable output root for the whole session."""
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(mode=0o755, parents=True, exist_ok=True)
    return OUT_ROOT


@pytest.fixture(scope="session")
def built_library(out_root: Path) -> dict:
    """Build the library from the two committed tables with the real tasks, and return the artifact paths."""
    exp_dir = out_root / "library"
    config = base_config(exp_dir, ["build_synthon_id_index", "group_synthon_data", "generate_synthon_conformers"])
    config["synthon_id_index"] = "synthon_id_index.pkl"
    config["synthon_conf_dir"] = str(exp_dir / "synthon_conformers")
    run_sass(config, exp_dir)
    return {
        "exp_dir": exp_dir,
        "synthon_id_index": exp_dir / "synthon_id_index.pkl",
        "synthon_data_file": exp_dir / "synthon_data.pkl",
        "grouped_synthon_file": exp_dir / "grouped_synthons.pkl",
        "synthon_conf_dir": exp_dir / "synthon_conformers",
    }


def config_against_library(built_library: dict, exp_dir: Path, tasks: list[str]) -> dict:
    """A `base_config` that reads the already-built library instead of rebuilding it."""
    config = base_config(exp_dir, tasks)
    for key in ("synthon_id_index", "synthon_data_file", "grouped_synthon_file", "synthon_conf_dir"):
        config[key] = str(built_library[key])
    return config
