# Configuration reference

Every key in `sass/config_template.yml`, grouped by what it affects. This is the *what*; for
*what to set it to*, see [tuning.md](tuning.md).

Copy `sass/config_template.yml` to your experiment directory, edit, and run `sass run --config <your.yml>`.
Unless `exp_dir` says otherwise, **output lands in the directory you launched from**.

## Setting values on the command line

A config file is optional. Without `--config`, the packaged template's defaults are used, so
once your cluster and library are configured, a search needs only a query and a library:

```bash
sass run -q query.sdf --library my_library               # template defaults
sass run -c my.yml -q query.sdf                          # config file, query overridden
sass run -c my.yml --set top_m=500000 --set num_workers=200
```

| | |
|---|---|
| `-q`, `--query` | `query_molecule`. Resolved against the launch directory, so a relative path is safe. |
| `--library` | `synthon_library`. The template leaves it unset. |
| `--set KEY=VALUE` | Any key in this reference. Repeatable. |

**Precedence is CLI > config file > template default**, and `--query`/`--library` win over a
`--set` naming the same key, so the answer never depends on argument order.

`--set` values are parsed as YAML, so they arrive as the type the config file would give:
`--set num_workers=200` is an int, `--set compress_pkl_files=false` is a bool, `--set
exp_dir=null` is `None`, `--set 'n_fragments=[2, 3]'` is a list.

A `--set` key that is not in this reference is **refused**, with a suggestion if it looks like a
near miss. That differs from an unrecognized key in a config *file*, which only warns: a flag
was typed just now and meant, so silently ignoring it would run the whole job wrong.

Whatever the run resolves to -- defaults, file and overrides combined -- is written to
`exp_dir/sass_<timestamp>.yml`, so the record shows what ran rather than what was typed.

**Every key below must be present.** SASS diffs your config against the template at startup and
refuses to run if any key is missing, listing them all with their defaults so you can paste them
in. Keys are read at different points in a run. Without this check, a missing key would fail
only when the task that reads it runs, possibly hours in. Start from a fresh copy
of the template when upgrading rather than adding keys to an old config one error at a time. A
key the template does *not* define is only warned about, since it is almost always a typo.

---

## Query and output

| key | default | meaning |
|---|---|---|
| `query_molecule` | `null` | Path to the query, `.sdf` or `.oeb`. Must be a single conformer. |
| `clean_up_query` | `false` | Neutralize and strip salts before fragmenting. |
| `tasks` | see below | Which pipeline steps to run. See [tasks.md](tasks.md). |
| `top_m` | `2000000` | Products carried from synthon scoring into rescoring. **The most expensive parameter in SASS — read [tuning.md](tuning.md) before changing it.** |
| `num_final_list` | `10000` | Top rescored product ids written to `rescore_combined_rocs_res.pkl`. |
| `num_write_pose` | `10000` | Top rescored products written as overlay poses. Fewer after duplicate removal. |
| `restore_synthon_ids_in_poses` | `true` | Translate integer synthon ids back to original library ids in the final pose SDF. Affects that file only — everything upstream stays numeric, because `write_product_poses` re-instantiates from it. Needs `synthon_id_index`. |

---

## The synthon library

A library is a set of artifacts. Name it and they resolve:

```yaml
synthon_library: my_library
```

That name is looked up in `sass/libraries.yml`, which ships with the package as a template for
you to fill in. Two things follow:

- **A library version bump is an edit to the registry**, not a message telling every user to edit
  a path. An alias such as `my_library` is a moving pointer to the recommended release;
  `my_library_202601` pins a specific release.
- **Paths are relative to a data root**, so moving the data means editing one line.

Runs log the *concrete* entry, never the alias, because an alias is a moving target:

```
INFO  Resolved synthon_library='my_library' -> 'my_library_202601' under /path/to/SASS_data (my_cluster).
INFO    My synthon library, 2026-01.
```

### The library keys

| key | what it is |
|---|---|
| `synthon_conf_dir` | directory of synthon conformer files |
| `synthon_data_file` | `{synthon_id: {SMILES, heavy_atom_count, reactions}}` |
| `grouped_synthon_file` | synthons grouped by reaction and component |
| `synthon_file` | the raw synthon list |
| `reaction_file` | reaction SMARTS |
| `synthon_id_index` | `list[str]` mapping integer ids back to original library ids. Optional, but a library without one cannot be scored. |
| `single_frag_synthon_conf_dir` | conformers rewritten to carry one connector each, for single-fragment scoring. Optional — only the single-fragment tasks read it. |

Set any of them to override the registry, e.g. to use an older library it does not list. An
override is logged as a warning, because a half-overridden library is exactly the mismatch
`synthon_library` exists to prevent: resolving integer synthon ids through the **wrong** index
yields plausible-looking wrong synthons rather than an error.

**Every key that is set is checked to exist at startup**, whichever way it was resolved, and
every missing one is reported at once. Two keys are optional, and *optional describes the
library, not the run*: a library may ship without `synthon_id_index` or
without `single_frag_synthon_conf_dir` (nothing but the single-fragment tasks reads it), but a
run that reads one is refused rather than allowed to score nothing.

```
ValueError: Synthon library is incomplete -- these are missing:
  synthon_file: /nope/synthons.txt
Check `synthon_library` (currently 'my_library') and any explicit overrides.
```

One `stat` per artifact, before a cluster is created — rather than a worker failing to open a
conformer file an hour into the run.

### A library the registry does not list

Omit `synthon_library` and set the keys yourself. They are still checked at startup.

`synthon_library` is a **name, not a path**. SASS does not infer filenames from a directory,
because no naming convention (`synthons.txt`, `reaction_SMARTS.txt`, …) fits every library.
Passing a path fails with a message that points at the two supported routes.

### Sites with several clusters

An entry's optional `hosts:` lists the clusters it is available on; omitting it means "expected
everywhere". Using a library on a host it does not list warns if every artifact is there, and
fails with an explanation if something is missing.

### Adding a library to the registry

Edit `sass/libraries.yml`. Each entry names every path explicitly, relative to a data root.

```yaml
data_roots:
  - path: /path/to/SASS_data

aliases:
  my_library: my_library_202601

libraries:
  my_library_202601:
    description: My synthon library, 2026-01.
    synthon_conf_dir: my_library_202601/synthon_conformers
    synthon_id_index: my_library_202601/synthon_id_index.pkl
    ...
    synthon_file: my_library_202601/synthons.txt
    single_frag_synthon_conf_dir: my_library_202601/single_frag_synthon_conformers
    single_frag_geometry:
      align_on_bond: true
      extend_bond_distance: 3
```

| key | default | meaning |
|---|---|---|
| `limit_reactions` | `null` | Restrict to a list of reaction ids. `null` uses all. |
| `synthon_limit` | `null` | Cap synthons per component. No effect when using a pre-built `grouped_synthon_file`. For reproducibility, generate your own fixed subset instead. |
| `random_seed` | `63` | Seed for `synthon_limit` sampling. |
| `synthon_connectors` | `['U','Np','Pu','Am']` | Connector atoms in the synthon files. |

## Paths

`exp_dir` (default: the directory you launched from) is where everything is written. The rest are
*read* paths, for resuming from a previous run's output.

| key | resume from |
|---|---|
| `combined_pseudo_res_file` | after `score_synthons` |
| `combined_rescore_res_file` | after `rescore_products` |
| `full_enumeration_product_dir` | after `enumerate_full_library` |
| `full_enumeration_conf_dir` | after `generate_full_library_conformers` |
| `single_frag_synthon_conf_dir` | after `generate_single_fragment_conformers` |
| `single_frag_pseudo_res_file` | after `score_single_fragments` |

`logging_dir` defaults to `exp_dir`.

> `rescore_products` writes only scores, not conformers, so a run cannot resume from
> conformers. Resume from `combined_pseudo_res_file` or `combined_rescore_res_file`.

---

## Query fragmentation

| key | default | meaning |
|---|---|---|
| `n_fragments` | `[2, 3]` | Fragment counts to generate. Also selects which reactions apply — a 3-fragment set can only match a 3-component reaction. |
| `heavy_atom_limit` | `5` | Fragments below this many heavy atoms are discarded. |
| `frag_connectors` | `['He','Ne','Ar','Kr','Xe','Rn','Og']` | Rare-gas placeholders marking cut bonds. Distinct from `synthon_connectors` so the two can be told apart. |
| `cleave_cyclic_bonds` | `true` | Allow cuts inside rings. |

---

## Product enumeration

| key | default | meaning |
|---|---|---|
| `enum_max_stereocenter` | `3` | Maximum stereocenters enumerated per product. |
| `warts_id_sep` | `_` | Separates a base id from its isomer wart (`12345_1`). |
| `product_id_sep` | `&` | Joins synthon ids into a product id (`12345&67890`). Must differ from `warts_id_sep`. |
| `product_chunk_size` | `1000` | Chunk size for full-library enumeration. Actual isomer count per chunk is ~3x this. Below ~100 sometimes triggers dask errors. |

---

## Conformer generation

Two independent sets. Synthon conformers are built **once per library**; product conformers are
built **on every query**, which is why only the latter is performance-critical.

| key | default | meaning |
|---|---|---|
| `synthon_omega_max_conf` | `50` | Conformers per synthon. |
| `synthon_omega_max_time` | `120` | Seconds per synthon. |
| `synthon_omega_energy` | `10` | kcal/mol window above the minimum. |
| `synthon_confgen_chunk_size` | `500` | Synthons per conformer-generation task. |
| `product_omega_max_conf` | `200` | Conformers per product isomer. **The accuracy/throughput dial for rescoring — see [tuning.md](tuning.md).** |
| `product_omega_max_time` | `120` | Seconds per product isomer. |
| `product_omega_energy` | `10` | kcal/mol window above the minimum. |
| `conf_file_format` | `oez` | Conformer file format. |

OMEGA uses the `OEOmegaSampling_Classic` preset, which enables Thompson sampling. The explicit
energy window is the deliberate divergence from that preset, which otherwise uses a varying one.

---

## ROCS scoring

### How the color force fields are built

SASS does not use one color force field; it **builds a set** and writes them to
`exp_dir/color_ffs/` as `custom_color_ff_<n>.txt`, one per connector count. A worker picks by
the number of connector atoms on the synthon it is scoring, which is what keeps color scores
comparable between a synthon with one connector and one with three — `scale_weight_per_atom`
divides the connector weight by that count. A four-entry `synthon_connectors` therefore means
five files, keys `0` through `4`, where `0` carries no connector features at all.

Each file is layered, in this order:

1. **a base** — Mills-Dean, or `base_cff_file` if set. Skipped entirely by `no_default_color`,
   which also discards `base_cff_file`.
2. **`additional_color_patterns`** — extra features, e.g. to overweight a functional group.
3. **connector atom features** from `custom_color_ff`, if `align_on_dummy_atom`.
4. **bond-extension features** on the `33Si` dummy atom, if `align_on_bond`.

So `base_cff_file` replaces only step 1; everything else is still applied on top of it.
`custom_color_ff` needs one entry per `synthon_connectors` entry, because the file is chosen by
connector count — a shorter list leaves the highest keys unwritten, and overlaying a synthon with that many
connectors then has no file to load. Refused at startup.

### Reusing a directory instead

`color_ff_dir` is **not** another input to that build — it replaces the build entirely. The
directory is used exactly as found and nothing is added to it, so `base_cff_file` and the
keys in the table below pertaining to color ff build have no effect. It is meant for reusing a previous run's
`color_ffs/`.

A run that sets `color_ff_dir` alongside any of the keys it disables is warned, listing them. A
directory missing the file for some connector count is refused before a cluster is created,
rather than failing on a worker an hour in.

| key | default | meaning |
|---|---|---|
| `base_cff_file` | `null` | Custom base color force field, replacing Mills-Dean. Connector-atom features are added on top. Ignored under `no_default_color`, or when `color_ff_dir` is set. |
| `color_ff_dir` | `null` | Reuse a directory of color FF files verbatim. Overrides the whole color ff build process above. |
| `align_on_dummy_atom` | `true` | Add custom color features on connector atoms, so overlays align on the connection vectors. |
| `custom_color_ff` | `['[U]','[Np]','[Pu]','[Am]']` | SMARTS for the connector features. |
| `custom_color_weight` | `-10` | Weight of those features. Large and negative to dominate the overlay. |
| `custom_color_radius` | `10` | Radius of those features. |
| `color_interaction_type` | `gaussian` | `gaussian` or `discrete`. |
| `scale_weight_per_atom` | `true` | Normalize connector weight across synthons with several connectors. |
| `additional_color_patterns` | `null` | Extra custom color features, e.g. to overweight a functional group. See the README. |

Four score functions, which must be public methods of
[`oeshape.OEOverlapResults`](https://docs.eyesopen.com/toolkits/python/shapetk/OEShapeClasses/OEOverlapResults.html):

| key | default |
|---|---|
| `optimization_shape_func` | `GetTanimoto` |
| `optimization_color_func` | `GetColorTanimoto` |
| `overlap_shape_func` | `GetTanimoto` |
| `overlap_color_func` | `GetColorTanimoto` |

When the optimization and overlap settings are identical, the optimized score is used directly.
When they differ, the best conformer is transformed and re-scored — slower, but it lets you
optimize on one metric and rank on another.

| key | default | meaning |
|---|---|---|
| `no_default_color` | `false` | Score shape plus only connector/custom color features. Effectively a shape-only overlay aligned on connectors. |
| `cross_scoring` | `true` | Score ring synthons against linear fragments and vice versa. |
| `rocs_key_for_overlay_optimization` | `null` | Which color FF to optimize with. `null` selects by the fragment's connector count. |
| `rocs_key_for_inplace_scoring` | `null` | Which color FF to re-score with. |
| `sep_shape_color_score` | `true` | Write separate shape and color scores as SD data on the poses. |

Deprecated, ignore: `ring_fragment_size_match`, `ring_fragment_size_match_strict`,
`weight_factor`.

---

## Bond alignment (single-fragment mode only)

| key | default | meaning |
|---|---|---|
| `align_on_bond` | `false` | Place a second dummy atom along the connection vector and align on it. |
| `extend_bond_distance` | `3` | Distance of that atom, in Å. **Must match the value used to build the single-fragment conformers.** |

---

## Synthon selection

| key | default | meaning |
|---|---|---|
| `synthon_combination_limit` | `2000000` | Combinations enumerated per fragment order. A **recall** knob, not an output-size one, and raised to `top_m` if set lower. **See [tuning.md](tuning.md) — the cost depends on the aggregation method.** |
| `synthon_score_aggregation_method` | `simple_average` | How component scores become a product score. One of `simple_average`, `min_score`, `top_two_of_three`; anything else is rejected at startup. Also decides whether cutoff pruning is possible: only `simple_average` and `min_score` admit a bound, so `top_two_of_three` is substantially slower. |
| `mix_pseudo_res_by_proportion` | `true` | Take `top_m x ratio` from each component-count bucket, by product count, so 3-component reactions don't swamp the list by sheer number. |
| `pseudo_res_merge_fanin` | `12` | Score files folded into one per incremental merge. Merging runs concurrently with scoring, so peak disk scales with the worker count rather than the task count. |
| `cutoff_warmup_wave_size` | `64` | Tasks run and merged before submission opens to the whole cluster, so a score cutoff exists before the bulk of the run starts writing. Peak disk is otherwise set by the initial burst of unpruned tasks. `0` disables. |
| `cutoff_warmup_quorum` | `0.5` | Fraction of the wave that must finish before opening up. Stragglers keep running alongside; task cost varies widely, so waiting for the whole wave idles the cluster far longer than waiting for half. |

`synthon_combination_limit` and `mix_pseudo_res_by_proportion` are the only two lossy steps in
the selection path. Everything downstream of them — cutoff pruning, deduplication, the merge —
is exact.

---

## Rescoring

| key | default | meaning |
|---|---|---|
| `dask_rescore_max_chunk_size` | `100` | Upper bound on products per rescore task. |
| `dask_rescore_min_chunk_size` | `10` | Lower bound. Below this, fixed ROCS setup dominates. |

The actual chunk size is chosen between these from the worker count, so every worker gets work
without any chunk being too small to be worth starting.

---

## Cluster

Cluster presets live in `DASK_SLURM_CONFIG` in `sass/cluster/slurm.py`. Each names a partition,
walltime presets (`queue_time_map`), and extra `#SBATCH` lines for worker jobs
(`job_extra_directives`), such as a QoS or an account. The shipped file is a template; fill in
your own cluster before running with `dask_cluster_type: slurm`.

| key | default | meaning |
|---|---|---|
| `dask_cluster_type` | `slurm` | `slurm` or `local`. `local` uses the current node, which is what the end-to-end tests run on. |
| `cluster_name` | `my_cluster` | A key of `DASK_SLURM_CONFIG` in `sass/cluster/slurm.py`. Set `null` to detect it instead, on a site with several clusters (see [internals.md](internals.md#locating-the-library-and-the-cluster)). |
| `num_workers` | `1000` | Workers requested. |
| `dask_worker_memory` | `4` | GB per worker. Raise only alongside `top_m` past ~2.5e6 — see [tuning.md](tuning.md). A larger request is harder for SLURM to schedule. |
| `qos` | `null` | Walltime preset: a key of the cluster's `queue_time_map`. Overrides the cluster's `default_qos`. To send a SLURM QoS, add `--qos=...` to the cluster's `job_extra_directives`. |
| `slurm_partition` | `null` | Overrides the cluster default. |
| `clean_up_intermediate_files` | `true` | Delete `pseudo_res/` and `rescore_results/` once the combined file that supersedes them is written. They are the *inputs* to a merge whose output sits beside them, so keeping both is storing the same scores twice, and removing them significantly reduces the final footprint. Never deleted if the combined file is missing or empty, so a failed merge leaves its inputs to resume from. Set `false` to keep the per-chunk scores for debugging. |
| `clean_up_dask_logs` | `true` | Delete per-worker dask logs after a **successful** run. They are kept automatically when a run fails: a `KilledWorker` says only that four workers died, and the per-worker logs are the only place the reason is written. |
| `compress_pkl_files` | `true` | LZMA-compress result pickles at the end of the run. Library artifacts are excluded: they are resolved by exact path by every later run, so renaming one to `.pkl.xz` would break the next startup check. |
