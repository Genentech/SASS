# Changelog

User-facing changes between public releases.

## 2.2.8

A major update from 1.1.0. Configs and imports written for 1.x need updating.

### Breaking

- **New package layout.** `sass/query_main.py` and `sass.utils.*` are replaced by `sass.chem`,
  `sass.library`, `sass.cluster`, `sass.selection`, `sass.tasks`, `sass.worker`, and
  `sass.analysis`. For example, `ResultAnalysis` is now `sass.analysis.results.ResultAnalysis`.
- **Config format changed.** Start from a fresh `sass/config_template.yml`. A config that is
  missing a key is refused at startup, with every missing key and its default listed.
- **Task names changed** to match the modules that implement them. Old names raise an error.

  | 1.x | 2.x |
  |---|---|
  | `get_product_poses` | `write_product_poses` |
  | `gen_synthon_conformers` | `generate_synthon_conformers` |
  | `full_library_enum` | `enumerate_full_library` |
  | `full_library_conf_gen` | `generate_full_library_conformers` |
  | `full_library_scoring` | `score_full_library` |
  | (`tasks: null`) | `group_synthon_data` |

- **Synthon libraries use integer synthon ids.** Build a library's id index with the new
  `build_synthon_id_index` task. Original ids are restored in the final pose file.
- **Cluster settings moved** to `DASK_SLURM_CONFIG` in `sass/cluster/slurm.py`, a template to
  fill in. Account, QoS, and other `#SBATCH` lines go in `job_extra_directives`.
- **`openeye-toolkits` is pinned exactly** (2026.1.0), because ROCS scores shift slightly between
  toolkit releases.

### Added

- Single-fragment query mode: `score_single_fragments`, `write_single_fragment_poses`, and
  `generate_single_fragment_conformers`.
- A library registry, `sass/libraries.yml`, so a run names a library instead of its files, and
  `sass list-libraries` to show what each name resolves to.
- `sass run --query`, `--library`, and `--set KEY=VALUE`. A config file is optional.
- By default, the results are stored where `sass run` command is called.
- Startup checks that fail in seconds instead of hours: config keys, library files, worker
  memory estimates, single-fragment geometry, and color force fields.
- Fewer SLURM workers are requested when a run has fewer tasks than `num_workers`, and worker
  walltime is fitted to the head job's remaining time.
- An end-to-end test suite on a small synthetic library (`pytest -m integration`).

### Changed

- Peak disk use for a typical query run is about 100 times lower, around 10 GB.
- Faster scoring: fewer ROCS calls, exact cutoff pruning when combining synthon scores, and
  score files merged while scoring runs.
- A failed run exits non-zero and prints the reason on the terminal. Dask worker logs are kept
  when a run fails.
