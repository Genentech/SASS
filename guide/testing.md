# Testing

```bash
uv run pytest tests/ --ignore=tests/integration   # unit tests, seconds
uv run pytest tests/integration                   # end-to-end, ~2 min on one cpu
```

## The synthetic test library

`tests/integration/` runs the whole pipeline against a library that lives in the repo, so it
needs no cluster, no account, and no access to commercial synthon data.

Two tab-separated tables under `tests/data/test_library/` are all that is committed:

| file | contents |
|---|---|
| `reactions.txt` | `t_join`, a 2-component amide coupling; `t_bisamide`, a diamine acylated on both ends |
| `synthons.txt` | 20 synthons — 11 for `t_join`, 9 for `t_bisamide` |

Everything else — the id index, grouped synthons, synthon conformers — is built by the fixture
running the real tasks. There is no checked-in build output to go stale.

The 20 synthons are chosen, not sampled. They cover ring synthons, two-connector synthons,
stereocenters, and the same structure appearing under two reactions (`[U]C(=O)c1ccccc1` is both
`t_join_a_phenyl` and `t_bisamide_b_phenyl`), which is what exercises pose deduplication.
`t_bisamide` uses the component-grouping SMIRKS form that Enamine-format reaction files use:

```
([U]-[*:2].[Np]-[*:3]).[U]-[*:1].[Np]-[*:4]>>([*:1]-[*:2].[*:3]-[*:4])
```

## Why it is small on purpose

Six times five, plus three cubed, is **57 products** — the entire library. That is what makes
these tests assertions rather than smoke checks:

- The full enumeration path (`enumerate_full_library` → `generate_full_library_conformers` →
  `score_full_library`) can actually be run, giving real ground truth to compare against. It
  cannot be run on a production library, so this is the only place that path is tested.
- The query is a molecule the library can build
  (`t_bisamide_a_piperazine` + `_b_pentanoyl` + `_c_isonicotinoyl`), so the right answer is known
  before the run starts: that combination, at a TanimotoCombo of exactly 2.0.

So the search is checked against exhaustive ground truth with `ResultAnalysis`, and recall must
be 1.0 at n = 1, 5, 10 and 20. A change that degrades selection shows up as a number, not as a
crash.

Note the flip side: 57 products is far too small to say anything about *selection quality* on a
real library. It will catch a break, not a regression in recall on a production library.

## Adding a test

Fixtures are in `tests/integration/conftest.py`. `built_library` is session-scoped and builds
once; use `config_against_library` to point a new run at it.

Output goes to `tests/_output/integration/` rather than a `tmp_path`, so a failed run leaves
logs and artifacts you can open. The directory is gitignored and wiped at the start of each
session.

Things that will trip you up:

- **OMEGA cannot parameterize connector atoms.** `[U]`, `[Np]` and the `frag_connectors` rare
  gases all fail with "missing parameters", so you cannot build a synthon or a query fragment
  from its own SMILES. The library path works around this by building a *minimum product* and
  stripping it back (`sample_general_mixed_synthon_conf`); the test path avoids it entirely by
  cleaving a fragment off an already-3D parent with `fragment_molecule`, which is what
  `write_query_fragment` does and what `score_synthons` does to a real query.
- **Set `synthon_library: None`.** If your `config_template.yml` defaults it to a production
  library, any library key you leave unset resolves to that library. A run that *builds* an unset
  artifact is refused outright, but a run that *reads* one silently reads that file from the
  production library, mixing production data into the test.
- **Single-fragment scoring needs a query carrying a `frag_connectors` atom.** Given a whole
  molecule it completes cleanly and scores nothing.

## What is not covered here

No test here exercises SLURM scale-out. The integration suite runs on a one-worker local dask
cluster; it tests the pipeline, not the cluster.
