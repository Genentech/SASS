# Shape-Aware Synthon Search (SASS)

<p align="left">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/charliermarsh/ruff"><img alt="lint: ruff" src="https://img.shields.io/badge/linting-ruff-d24b4b.svg"></a>
<a href="https://mypy-lang.org/"><img alt="lint: mypy" src="https://img.shields.io/badge/linting-mypy-d24b4b.svg"></a>
</p>

Last updated: 2026-11-03

This repo contains the code for running combinatorial, synthon-based ROCS queries, as described in [Shape-Aware Synthon Search (SASS) for virtual screening of synthon-based chemical spaces](https://pubs.acs.org/doi/full/10.1021/acs.jcim.3c01865). The current version contains major improvements over what is described in the paper, including searching reactions with 3 or more components, and much more streamlined parallelization.

<center>
    <img src="data/SASS.jpg?raw=true" width="500">
</center>

SASS is a synthon-based virtual screening method that carries out shape similarity searches in the synthon space instead of the enumerated product space. Queries are fragmented, query fragments are scored against synthons, and top synthon combinations are selected. A tiny fraction of the full library is then instantiated and rescored, thereby avoiding full enumeration/scoring and significantly accelerating large-scale, shape-based virtual screening.

**This is a reference implementation.** It does not run off the shelf: you configure it for your own SLURM cluster and your own synthon library first (see [Configure for your site](#configure-for-your-site)).

## Setup

### Install

SASS uses the OpenEye toolkits, which are installed from OpenEye's package index.

```bash
git clone https://github.com/Genentech/SASS.git
cd SASS

# uv
uv venv
uv sync --group dev
source .venv/bin/activate

# or pip, with the OpenEye index from config/pip.conf
python3.11 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://pypi.anaconda.org/OpenEye/simple/ -e .
```

An editable install is recommended, because configuring SASS means editing two files in the package.

### OpenEye license

SASS requires a valid OpenEye license. Set its path in your environment:

```bash
export OE_LICENSE=/path/to/oe_license.txt
```

## Configure for your site

### 1. Your cluster

SASS runs its workers as SLURM jobs through `dask-jobqueue`. Cluster presets live in `DASK_SLURM_CONFIG` in [`sass/cluster/slurm.py`](sass/cluster/slurm.py). Replace the `my_cluster` example with your own:

```python
DASK_SLURM_CONFIG = {
    "my_cluster": {
        "partition": "cpu",
        "queue_time_map": {
            "1d": {"walltime": "1-0", "lifetime": "1435m", "lifetime-stagger": "4m"},
        },
        "default_qos": "1d",
        "job_extra_directives": ["--account=my_project"],
    },
}
```

- `queue_time_map` holds named walltime presets for worker jobs. The config key `qos` picks one, and `default_qos` is used otherwise. `lifetime` should be a few minutes shorter than `walltime`, so dask retires a worker cleanly before SLURM kills it.
- `job_extra_directives` are extra `#SBATCH` lines for worker jobs, such as a QoS or an account.
- `cluster_name` in [`sass/config_template.yml`](sass/config_template.yml) defaults to `my_cluster`. If you rename the preset, change it there too. A site with several clusters can let SASS detect which one it is on; see [guide/internals.md](guide/internals.md#locating-the-library-and-the-cluster).

To run without SLURM, on the current node only, set `dask_cluster_type: local`.

### 2. Your synthon library

SASS searches a synthon library: a synthon file and a reaction file in the format shown in [`data/example_library`](data/example_library), plus artifacts that SASS builds from them once. Build them with one run:

```yaml
tasks: ['build_synthon_id_index', 'group_synthon_data', 'generate_synthon_conformers']
```

See [guide/tasks.md](guide/tasks.md) for these tasks and [guide/configuration.md](guide/configuration.md#the-synthon-library) for the library keys. Then register the library in [`sass/libraries.yml`](sass/libraries.yml), so a run config only needs its name (`synthon_library: my_library`). The shipped file is a template with a commented example entry.

For a one-off library, you can instead leave `synthon_library` unset and set the library keys (`synthon_conf_dir`, `synthon_data_file`, `grouped_synthon_file`, `synthon_file`, `reaction_file`, `synthon_id_index`, `single_frag_synthon_conf_dir`) in the run config. Do not mix the two, to avoid mismatched library files.

### 3. Check

```bash
sass list-libraries
```

lists every registered library, what it resolves to on this machine, and whether its files are present.

## Usage

### Quick start

Once your cluster and library are configured, a query molecule and a library name are all you have to supply:

```bash
cd /path/to/an/empty/results/directory
sbatch --mem=16G --time=5:00:00 sass run -q query.sdf --library my_library
```

Output lands in the directory you launched from. Add `--set KEY=VALUE` (repeatable) to change any config key without editing a config file:

```bash
sass run -q query.sdf --library my_library --set top_m=500000 --set num_workers=200
```

For repeated runs or advanced options, use a config file. Start from a fresh copy of [`sass/config_template.yml`](sass/config_template.yml):

```bash
sass run --config /path/to/config.yml
```

Precedence is **CLI > config file > template default**. Everything a run actually used is written to `sass_<timestamp>.yml` beside the results, so a run can be reproduced from its own output.

### Details

* **Run SASS on a fast scratch filesystem.** The intermediate files take up about 10 GB of disk space, and are removed after a run finishes.
* A config missing a key (likely copied from an old run) is refused at startup, listing every missing key with its default.
* With default settings on reasonably modern CPUs, a query against a commercial library of about 100 billion products takes 0.5-1 hr on 1000 CPUs, depending on the size of the query molecule.
* After a run completes, an SDF file containing the query molecule and the best overlay poses of the top products (`rescore_top_<n>_mol_poses.sdf`) is written. The first molecule is the query, and the rest are **not** sorted by ROCS score.
* **Save the query 3D conformer in SDF V2000 format.** The OpenEye toolkits may have problems reading SDF V3000 files.

## Key terms

- **synthons**: Building blocks that can be directly connected to form products. Synthons are different from reactants in that synthons have labeled connector atoms, and have been modified from the reactants (e.g. leaving groups removed).
- **connector atom**: Dummy atoms (e.g. U, Np, Pu, Am) used to designate connecting vectors where two compatible synthons join.
- **chemical space**: All molecules generated from exhaustively instantiating all combinations of compatible synthons.
- **query**: The input molecule for which shape-similar molecules in the chemical space are to be searched.
- **query fragment**: Fragments of a query molecule, generated by cleaving bonds of the query.

## Tasks

A run executes the `tasks:` listed in the config. For a regular query that is:

```yaml
tasks: ['score_synthons', 'rescore_products', 'write_product_poses']
```

<center>
    <img src="data/SASS_workflow.png?raw=true" width="700">
</center>

There are thirteen tasks and three run modes (regular query, single-fragment query, full-library enumeration). See **[guide/tasks.md](guide/tasks.md)** for what each one does and which to combine.

## Documentation

| | |
|---|---|
| **[guide/tasks.md](guide/tasks.md)** | the thirteen tasks, and the three run modes |
| **[guide/configuration.md](guide/configuration.md)** | every config key, grouped |
| **[guide/tuning.md](guide/tuning.md)** | what to *set* things to: `top_m`, recall, OMEGA quality, worker memory |
| **[guide/internals.md](guide/internals.md)** | package layout, parallelization strategy, what is exact and what is approximate |
| **[guide/testing.md](guide/testing.md)** | the synthetic test library, and how to add an end-to-end test |
| **[CHANGELOG.md](CHANGELOG.md)** | changes between public releases |

**`top_m` is the most expensive parameter in SASS.** Do not exceed 2,000,000 without reading [guide/tuning.md](guide/tuning.md). It drives rescore runtime, worker memory, and search depth at once. SASS estimates the memory peaks at startup and warns if they will not fit.

## Advanced usage

* **Single-fragment query mode**: instead of matching a whole molecule, score the synthon library against *one* fragment with a defined attachment point. This is useful for growing or replacing a substituent while the rest of the molecule stays fixed.

  * **The query must carry a `[U]` connector atom** marking where the fragment attaches. Typically you take a whole 3D molecule and break one acyclic bond at the position you want to vary, keeping the piece you want to search and capping the cut with `[U]`. The connector must be a real atom with coordinates, since the alignment uses the bond vector, not just the shape.
  * OMEGA cannot generate conformers for a molecule containing `[U]`, so build the 3D structure *first* and substitute the connector afterwards.
  * Run the single-fragment tasks; they cannot be mixed with the whole-molecule tasks in one run:

    ```
    sass run -q fragment.sdf --set 'tasks=[score_single_fragments, write_single_fragment_poses]'
    ```

  * This needs `single_frag_synthon_conf_dir`, a separate set of conformers aligned on the connector bond. `sass list-libraries` reports which libraries contain it. To build one, run `generate_single_fragment_conformers` against that library's regular synthon conformers.
  * `align_on_bond` and `extend_bond_distance` must match the values the conformers were generated with. The stored conformers carry a second dummy atom placed at that distance, and the query is given one at run time to match. Record the values in the library's `single_frag_geometry` entry; `sass list-libraries` prints them. A mismatch is refused at startup, measured from a sampled conformer file rather than taken from the registry, so it holds for unregistered libraries too. Otherwise a mismatch would be silent: the overlay still runs and still ranks, against a bond vector that does not line up.

* **Restarting from intermediate steps**: SASS writes out and stores the following intermediate files. You can specify the paths to those files/folders to restart from an intermediate step.
  * Combined pseudo result file: contains `top_m` synthon combinations to be rescored with ROCS.
  * Combined rescore result file: contains ROCS scores of `top_m` products, and can be used to generate overlay poses.

* **Using custom color force field files**: You can generate your own color FF file and provide its path under the `base_cff_file` key in the config file. This file overrides the default Mills-Dean force field used by ROCS. The base color FF is extended with appropriate connector atom color features.
  * You can also re-use an entire set of color FF files by providing the path to the directory under the `color_ff_dir` key.

* **Custom scoring in ROCS**: ROCS uses the TanimotoCombo score by default. You can specify the shape and color score functions to use for overlay optimization and scoring by modifying the `[optimization/overlap]_[shape/color]_func` keys in the config file (single-fragment scoring always uses TanimotoCombo). The values must be valid public methods of the [`oeshape.OEOverlapResults`](https://docs.eyesopen.com/toolkits/python/shapetk/OEShapeClasses/OEOverlapResults.html) class.

## Testing

```bash
pytest -m "not integration"   # unit tests
pytest -m integration         # the whole pipeline on a small synthetic library, on a local dask cluster
```

Both need `OE_LICENSE`. See [guide/testing.md](guide/testing.md).

## Contributing

Issues and pull requests are welcome.
