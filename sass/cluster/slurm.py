"""Per-cluster SLURM defaults: partition, QoS, worker walltime/lifetime pairs, and extra directives.

This is a template. Fill in the `my_cluster` example in `DASK_SLURM_CONFIG` with your cluster's
settings. `cluster_name` in the run config picks the entry. A site with several clusters can map
SLURM's `ClusterName` in `SLURM_CLUSTER_NAMES`, so SASS detects which one it is on.

Keep `lifetime` a few minutes short of `walltime`, so dask retires a worker before SLURM kills
it. `lifetime-stagger` spreads out when workers expire.
"""

import logging
import subprocess

logger = logging.getLogger(__name__)

#: SLURM's own `ClusterName` -> the key in `DASK_SLURM_CONFIG`. Used only as a fallback, after
#: the data roots in `libraries.yml`. Find the name with `scontrol show config | grep ClusterName`.
SLURM_CLUSTER_NAMES: dict[str, str] = {
    # "my_slurm_cluster_name": "my_cluster",
}


def detect_host_from_slurm() -> str | None:
    """Ask SLURM which cluster this is, or `None` if it cannot say."""
    try:
        result = subprocess.run(
            ["scontrol", "show", "config"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Could not run scontrol: %s", exc)
        return None

    for line in result.stdout.splitlines():
        if line.startswith("ClusterName"):
            reported = line.split("=", 1)[1].strip()
            host = SLURM_CLUSTER_NAMES.get(reported)
            if host is None:
                logger.warning(
                    "SLURM reports ClusterName=%r, which SASS does not recognize. Add it to "
                    "`SLURM_CLUSTER_NAMES`, or set `cluster_name` in the config. Known: %s.",
                    reported,
                    ", ".join(sorted(SLURM_CLUSTER_NAMES)),
                )
            return host
    return None


#: One entry per cluster. `cluster_name` in the run config, or detection, picks one.
#:
#: - `partition`: SLURM partition for worker jobs. `slurm_partition` in the config overrides it.
#: - `queue_time_map`: named walltime presets. The config's `qos` (or `default_qos`) picks one.
#: - `job_extra_directives`: extra `#SBATCH` lines for worker jobs, e.g. a QoS or an account.
DASK_SLURM_CONFIG = {
    "my_cluster": {
        "partition": "TODO",
        "queue_time_map": {
            "1d": {"walltime": "1-0", "lifetime": "1435m", "lifetime-stagger": "4m"},
        },
        "default_qos": "1d",
        "job_extra_directives": [],  # e.g. ["--qos=normal", "--account=my_project"]
    },
}
