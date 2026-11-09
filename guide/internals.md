# Internals

How the package is laid out and why, and how each phase is parallelized. For running SASS see
the [README](../README.md); for config keys, [configuration.md](configuration.md).

## Package layout

`sass/` is layered, and the layering is enforced by `tests/test_layering.py` — an AST pass over
every module — rather than by convention.

```
sass/
  ids.py           the identifier scheme: synthon ids, isomer warts, product ids
  io.py            reading/writing the pickled intermediates
  config.py        loading a run config (that is all it does)
  preflight.py     everything checked before a cluster exists: tasks, query, memory, cluster
  context.py       RunContext: the resolved config, paths and loaded library
  main.py          argument parsing and task dispatch
  cli.py           typer entry point
  logging_setup.py logging and the OpenEye error handler

  chem/            OpenEye. mol, fragment, synthon, overlay, confgen, libgen
  library/         the synthon library on disk. registry, synthons, conformers
  selection/       combine, aggregate, allocate, dedup
  cluster/         dask plumbing. client, disk, plugins, slurm
  worker/          what executes on a dask worker
  tasks/           one module per `tasks:` entry
  analysis/        result analysis and plotting
```

### Two rules that are load-bearing

**`worker/` may not import dask, matplotlib, seaborn, sklearn, `cluster/` or `tasks/`.** A
worker imports these modules once per process and a run starts ~1000 processes, so anything
reachable from `worker/` is paid ~1000 times. Plotting and analysis libraries must stay off
this path. `tests/test_layering.py` enforces it, and `worker/` imports nothing outside the
standard library except `openeye`.

**`selection/` may not import OpenEye, dask, or touch the filesystem.** It holds the code where
a bug changes *results* rather than crashing — the pruning bound, deduplication, the
per-component caps — so it stays testable in milliseconds with nothing to mock.

```
ids, io, selection/  ->  (nothing else in sass)
chem/                ->  ids
library/             ->  ids, chem/
worker/              ->  ids, io, selection/, chem/, library/
cluster/             ->  (dask only)
tasks/               ->  everything
main.py              ->  tasks/, config, cluster/
```

### "task" means two different things

Unavoidably, given the config key. `sass/tasks/` holds **pipeline phases** — the things you
list under `tasks:`. `sass/worker/` holds **dask tasks** — the functions that execute remotely.
A single phase submits many dask tasks.

## Locating the library and the cluster

`sass/libraries.yml` holds the data roots and the library entries. The first data root that
exists is used, and each entry names its artifacts explicitly, relative to it.

`cluster_name` picks a preset in `DASK_SLURM_CONFIG` (`sass/cluster/slurm.py`). A site that runs
on several clusters can leave it unset and let SASS detect the cluster: from the `host` of the
mounted data root, or failing that from SLURM's `ClusterName` via `SLURM_CLUSTER_NAMES`.

## What is exact and what is not

Only two steps in the selection path are lossy:

- **`allocate_synthons`** takes a prefix of each component's score list, because SASS never
  combines all `s1 x s2 x s3`, which is billions per reaction. This is what makes each
  reaction's contribution an approximate top `top_m`. Governed by `synthon_combination_limit`;
  see [tuning.md](tuning.md).
- **Proportional mixing** in `combine_pseudo_res` takes `top_m x ratio` from each
  component-count bucket so 3-component reactions don't swamp the list by sheer count. Disable
  with `mix_pseudo_res_by_proportion: false`.

Everything downstream is exact: the cutoff pruning (the cutoff is a lower bound that only
rises, so pruning below it cannot change the result), deduplication, the incremental merge, and
the phase A / phase B scoring split.

## Parallelization strategy, task by task

Given the data volume and independence of most computation, parallelization in computing is highly desirable. Currently, parallelization is achieved using `dask` for all steps, and `dask` is configured to run on `SLURM`-based compute clusters, or on local machines.

- **library product enumeration**: Because exhaustive product enumeration from sets of compatible synthons uses the OpenEye `OELibraryGen` object, which itself is likely already quite optimized, it is inefficient to loop through the synthons one-by-one. Instead, for an $n \times m \times k \times ...$ sized library, only the 1st "dimension" is split, i.e. each dask worker only enumerates $n[i: i+c] \times m \times k \times ...$ products.
  - Tautomer selection and isomer enumeration (FLIPPER) is done on each product.
  - The split on the 1st dimension for each reaction is dynamically determined to fit the target product chunk size. e.g. for a two-component reaction of $50 \times 100$ and a target chunk size of $1000$, the 1st dimension is split 5-fold with 10 synthons each (i.e. a total of 5 tasks of $10 \times 100 = 1000$)
  - In cases where the actual chunk size exceeds the target chunk size even when the 1st "dimension" is split to the smallest possible chunk (e.g. for a three-component reaction of $100 \times 100 \times 100$, even $1 \times 100 \times 100 = 10000$ is larger than the target chunk size of $1000$), the oversized chunk is still instantiated on 1 worker, but split to the target chunk size when writing to file. Even though this does not improve the efficiency of library product instantiation, having even chunk sizes greatly improves the parallelization efficiency for library product conformer generation and scoring.
  - It is recommended to set the target product chunk size to be at least 1000. Too small chunk size leads to `dask` error. The actual number of isomers per chunk will be ~3000 due to multiple isomers per product.

- **product conformer generation**: Each dask worker processes one product chunk (see above).

- **product scoring (ROCS)**: Each dask worker processes one conformer chunk (see above).

- **synthon conformer generation**: Synthon counts vary by orders of magnitude between reactions, and there are relatively few reactions, so parallelization on reaction level leads to very different runtime for each dask worker. To alleviate this, synthon lists are split to a target size and fed to each dask worker. Conformer chunks are written to files. The chunks of each (reaction, component) file are then merged on the workers, one file per task, in a fixed chunk order.

- **synthon scoring**: Scoring is parallelized on the `fragment_order` level. For a large commercial library with a typical query molecule, the number of total `fragment_order` is on the order of 1e4, which is suitable with ~1000 dask workers. Each task of synthon vs query fragment scoring is not further split.

- **singleton product instantiation**: Unlike **library product enumeration**, instantiating the SASS-selected pseudo-products does not require enumeration of all combinations of compatible synthons, but only 1 product from each of the aggregated synthon-id. `OELibraryGen` objects were used, but only with 1 synthon at each reactant position (and 1 expected product per function call).
  - It is recommended to set the chunk size to be processed by each dask worker to be no smaller than 1000. Too small chunk size can lead to `dask` error. The instantiated products are further split into the desired chunk size (usually 10-100/chunk) for downstream rescoring.

- **write_product_poses**: The top products from rescoring are re-instantiated. Each dask worker regenerates conformers for a chunk of them, overlays them on the query, and writes out the pose files. The head node then deduplicates and aggregates the poses into a single file.

- **generate_single_fragment_conformers**: synthon conformer files from `generate_synthon_conformers` step are processed (substituting connector atoms, extending connector bonds) and chunked to ~200-300 isomers per file for higher parallelization efficiency for downstream tasks.

- **score_single_fragments/write_single_fragment_poses**: Each worker operates on one conformer chunk file.

## Scoring phase mechanics

`score_synthons` is the most intricate phase. Three mechanisms interleave in one submission
loop, and they exist for measured reasons:

- **Warm-up wave.** A small batch of reactions spanning both component-count buckets runs
  first, so a score cutoff exists before the bulk of the run starts writing files. Without it
  peak disk is set by the initial burst of unpruned tasks. Sized by `cutoff_warmup_wave_size`,
  opened at `cutoff_warmup_quorum` so stragglers don't idle the cluster.
- **Incremental merging.** Score files are folded together while scoring is still running, so
  peak disk tracks the worker count rather than the task count. Fan-in from
  `pseudo_res_merge_fanin`.
- **Cutoff propagation.** Each merge's `top_m`-th score feeds back into later tasks, which skip
  combinations that cannot reach the final list. Exact, and worth many fewer combinations
  enumerated at realistic cutoffs.

Scoring itself is split in two: **phase A** scores one `(reaction, component, fragment)` triple,
**phase B** combines a fragment order's per-component lists into pseudo-products. Phase A
results are memoized by triple, so a fragment shared between fragment sets is scored once —
fewer ROCS calls overall, more so on 3-component reactions.
