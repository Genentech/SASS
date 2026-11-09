# Tuning SASS

How to choose values, and what each choice costs. For *what a key does*, see
[configuration.md](configuration.md); this document is about what to set it to.

Numbers below come from runs on a large synthon library and from synthetic benchmarks, which
are labeled as such. Treat them as orders of magnitude: synthetic numbers describe the
mechanism rather than your query.

---

## The three knobs that matter

| knob | governs | cost of raising |
|---|---|---|
| `top_m` | how many products are rescored | rescore runtime (dominant past 2e6), worker memory, and search depth via the floor below |
| `synthon_combination_limit` | how deep the synthon search goes | depends entirely on the aggregation method — see below |
| `cutoff_warmup_wave_size` | how quickly a score cutoff appears | roughly one task duration of idle cluster |

Everything else is a correctness or output-format choice, not a performance dial.

---

## `top_m` — do not exceed 2,000,000

`top_m` is the single most expensive parameter in SASS, and it drives three things at once.

### 1. Rescore runtime — the binding limit

`rescore_products` runs OMEGA and ROCS on **every** selected product, and it is already the
largest phase of a run. Past ~2e6 the run is essentially just rescoring, and the synthon-scoring
work that makes SASS worth using stops being the point.

This, not memory, is the reason for the advised maximum.

### 2. Worker memory

The score merge holds `2 x top_m` rows plus one input file — so `3 x top_m` worst case — at
~300 B per row. Measured peak resident memory, fan-in 12, in a clean process:

| `top_m` | peak | in a 4 GB worker |
|---|---|---|
| 1e6 | ~1.2 GB | comfortable |
| 2e6 | ~2.4 GB | fine |
| 2.5e6 | ~3.0 GB | practical ceiling — dask *pauses* a worker at 80% (3.2 GB) and kills at 95% |
| 4e6 | ~4.7 GB | over |

Linear at **~1.18 GB per 1e6 of `top_m`**. Beyond ~2.5e6 you must raise `dask_worker_memory`
as well — but a larger memory request is harder for SLURM to schedule, so the run may start
slower or with fewer workers.

Note that **lowering `pseudo_res_merge_fanin` does not help.** The merge trims as it reads, so
its peak does not depend on the fan-in.

### 3. Search depth

`synthon_combination_limit` is silently raised to at least `top_m`. That floor is a
**correctness** requirement, not a convenience: in the limiting case of a single fragment order,
the global top `top_m` must all come from that one task, so capping its enumeration lower would
make it impossible to produce `top_m` products at all. Raising `top_m` therefore also raises the
per-component caps, and the enumeration work with them.

### The startup check

SASS estimates both memory peaks before the cluster is created and prints them. You do not have
to do this arithmetic:

```
INFO     Peak worker memory estimate: 2.4 GB score merge, 0.4 GB combination (cold),
         against 4 GB workers.

WARNING  top_m=4,000,000 exceeds the advised maximum of 2,000,000. Past this the run is
         dominated by `rescore_products`, which runs OMEGA and ROCS on every selected product.
WARNING  The score merge (`merge_score_batch_task`, driven by top_m) will peak near 4.7 GB,
         against a 4 GB worker (usable ~2.4 GB after dask's own headroom). This run is likely
         to fail with KilledWorker. Raise `dask_worker_memory` to >= 8, or lower `top_m`.
```

The two warnings are independent: a 64 GB worker still warns about runtime, because memory was
never the reason 4e6 products is a bad idea.

---

## `synthon_combination_limit` — a recall knob, not an output-size knob

It is independent of `top_m`, which caps how many rows are *written*. This one caps how many are
*looked at*. `allocate_synthons` divides the budget across components, so a 3-component reaction
at 1e6 only ever considers the top ~100 synthons of **each**:

| limit | caps for a 3 x 5,000 reaction | enumerated |
|---|---|---|
| 1e6 | 100 / 100 / 100 | 1,000,000 |
| 2e6 | 125 / 126 / 126 | 1,984,500 |
| 1e7 | 215 / 215 / 216 | 9,984,600 |
| 1e8 | 464 / 464 / 464 | 99,897,344 |

A product built from two excellent synthons and one ranked 5,000th is unreachable, however well
it would have scored. **This is what makes each reaction's contribution an approximate top
`top_m` rather than an exact one** — SASS never combines all `s1 x s2 x s3`, which is billions
per reaction.

Measured on a 3 x 150 problem small enough to enumerate exhaustively, with `top_m` held at 500
throughout — so output size never changed, only search depth:

| limit | per-component caps | recall of the true top 500 |
|---|---|---|
| 1,000 | 10 / 10 / 10 | **47.8%** |
| 10,000 | 21 / 21 / 22 | 60.6% |
| 100,000 | 46 / 46 / 47 | 92.8% |
| 1,000,000 | 100 / 100 / 100 | **100.0%** |

### What raising it costs — depends on the aggregation method

**With `simple_average` or `min_score`,** cutoff pruning applies. The enumeration stops at
the first synthon that cannot beat the running cutoff, and *that stopping point does not depend
on this limit* — which is why a tight cutoff makes the limit irrelevant. Until the cutoff
tightens, though, the limit is the binding exit and the extra depth is paid in full:

| cutoff | rows out @2e6 | 2e6 | 1e7 | ratio |
|---|---|---|---|---|
| none | 100,000 | 4.09 s | 12.07 s | **3.0x** |
| loose | 100,000 | 4.21 s | 12.29 s | 2.9x |
| moderate | 100,000 | 1.90 s | 4.00 s | 2.1x |
| tight | 29,254 | 0.04 s | 0.08 s | 1.7x |
| tighter | 984 | ~0 | ~0 | 1.0x |

So: **~3x until the cutoff tightens past the depth the limit allows, then free.** How long that
takes is query-dependent.

**With `top_two_of_three` there is no pruning bound**
(see `_PRUNE_BOUND_MODE` in `sass/selection/combine.py`). The cutoff never helps, and the limit
is a direct multiplier on every task for the whole run:

| method | bound | @2e6 | @1e7 | |
|---|---|---|---|---|
| `simple_average` | `sum` | 0.05 s | 0.05 s | 1.0x |
| `top_two_of_three` | none | 21.74 s | 99.85 s | **4.6x** |

Same tight cutoff in both rows. Note the absolute figures: 21.7 s vs 0.05 s at the *same* limit,
so these methods are ~400x more expensive per task before the limit is touched at all.
**With them, leave `synthon_combination_limit` at the `top_m` floor.**

### Memory

The combination step grows its result list to `min(synthon_comb_limit, 10 x top_m)` rows at
~183 B each, and only tasks running before a cutoff exists pay it — but that means the *warm-up
wave*, which runs on the same workers:

| `top_m` | `synthon_comb_limit` | cold combination peak | score merge peak |
|---|---|---|---|
| 1e6 | 1e6 (floor) | 0.2 GB | 1.2 GB |
| 2e6 | 2e6 (floor) | 0.4 GB | 2.4 GB |
| 1e6 | 5e7 | 1.8 GB (clamped) | 1.2 GB |
| **2e6** | **5e7** | **3.7 GB** ← binding | 2.4 GB |

The `10 x top_m` clamp is a useful safety property: at the default `top_m` you cannot exhaust a
worker by raising this for recall, however far you raise it.

### Recommendation

Keep `synthon_combination_limit == top_m` unless you have a specific recall problem. At the
floor it is still doing its original job — bounding enumeration, and doing it hard. What you
give up is only the headroom above `top_m`, which is a recall-for-CPU trade rather than an
unknown.

---

## Conformer generation quality: OMEGA settings

`rescore_products` is the largest phase, and most of its time is OMEGA conformer generation.
The defaults (`OEOmegaSampling_Classic`, `product_omega_max_conf: 200`,
`product_omega_max_time: 120`) favor speed.

There is a trade-off between speed and ROCS scores. More conformers give a better chance of
finding the one that overlays best on the query, so scores are higher, but the phase takes
longer. The point of SASS is to rank-order molecules, and the exact absolute score is not the
primary goal, so the defaults suit most searches. To favor scores, raise
`product_omega_max_conf` and `product_omega_max_time` (for example to 500 and 300). To compare
against an earlier run, match its settings exactly, because a change in `max_conf` alone
reshuffles which products land in the top N.

`synthon_omega_max_conf` (50) is a different parameter on a different phase: synthon conformers
are generated once per library, not per query, so there is no runtime pressure to lower it.

---

## Cluster sizing

- `num_workers: 1000` suits a full-size library. Scoring usually generates many more tasks than
  the per-user CPU cap, so the cap, not the work, sets the useful worker count.
- `dask_worker_memory: 4` (GB) suits `top_m` up to ~2.5e6. See the tables above before raising
  it; a larger request is harder for SLURM to schedule.
- Every progress line reports the live worker count against the target
  (`Workers: 812/1000`), so a slow phase can be told apart from a cluster that never scaled up.
  If that number stays well below the target, the run is queue-limited, not compute-limited.

## Cluster startup and shutdown: what actually costs time

Both cluster startup and shutdown time scale with the **number of worker jobs**,
not with the amount of work, because both
are one Slurm RPC per job through a shared controller. Submission throughput can drop by an
order of magnitude when the controller has many pending jobs, and SASS cannot control that.

### Startup does not wait for the full cluster

A run begins as soon as 2 workers are connected (as a proxy for general cluster health);
the remaining are submitted in the background while scoring is already under way.

Example log:
```
Cluster startup: 14s to first workers (14s total); submitting 1000 worker jobs continues
                 in the background.
Started slurm cluster; 115 workers joined so far, scaling to 1000.
```

Starting early costs nothing measurable. Workers keep joining while scoring runs, and it
finishes in about the same time as it would after waiting for the full cluster.

If not even two workers arrive within **10 minutes** the run fails, pointing at `squeue` for the pending reason.
A cold node can take minutes, but *no worker at
all* for 10 minutes means a wrong partition, an unauthorized account, a QOS limit or
a sick controller — none of which resolve by waiting, and all of which would otherwise sit on
the head allocation until its walltime.

### Workers end with the head job

Inside a Slurm job, SASS reads the head's `SLURM_JOB_END_TIME`. When the head has less time
left than the queue's worker walltime, the workers ask for only what the head has left, and
their dask `--lifetime` shrinks to match. The log says so:

```
Worker walltime 1-0 shortened to 1:59:00 (lifetime 114m) to end with the head job.
```

A worker outliving its head has nothing to do. Asking for the default (longer) walltime anyway makes
worker jobs harder to schedule.

### Shutdown does not wait for the workers

`close()` drops the scheduler and returns in seconds. It does not cancel the worker jobs:
cancelling N jobs costs N RPCs exactly like submitting them, so `scancel` is no faster than
dask's own teardown. Both take minutes for a large cluster.

Dropping the scheduler is O(1) instead, and every worker notices at once. Worker reconnection
is unconditionally disabled in `distributed`, so a running worker closes itself immediately,
and most running workers release their CPUs within seconds. A worker still *starting* has
no scheduler to find and gives up after its death timeout, which is why a few stragglers take a
couple of minutes.

Two consequences worth knowing:

- **The run's log ends before the cluster is empty.** That is deliberate. Checking `squeue`
  just after a run finishes will show workers still retiring.
- **Back-to-back runs do not wait for each other.** Worker jobs request one CPU each and are
  allocated individually as CPUs free, so the next run starts immediately rather than waiting
  for a block.

### If startup is slow

Check whether it is submission or the queue:

```bash
sacct -X --name sass-worker-<head_pid> -o Submit,Start --parsable2
```

Spread-out `Submit` times mean the controller is slow to accept jobs — huge number of pending jobs,
usually, and nothing to do but wait or ask for fewer workers. A gap between
`Submit` and `Start` means the nodes themselves are not available, which is capacity or cold
start.

The two timings in the startup logging line should be nearly equal. If the total is much larger than
the time to first workers, something is blocking before the cluster is even asked to scale.

Lowering `num_workers` is worth considering on a busy
cluster or when the per-user CPU cap leaves little headroom for anything else you run.
