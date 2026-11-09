"""Worker walltime is fitted to the head job, so workers never ask for time the head cannot use."""

import pytest
from dask.utils import parse_timedelta
from dask_jobqueue.slurm import SLURMJob

from sass.cluster.client import _fit_worker_time_to_head, _slurm_time_to_minutes

#: A one-day worker preset, as in a `queue_time_map` entry of `DASK_SLURM_CONFIG`.
PRESET = {"walltime": "1-0", "lifetime": "1435m", "lifetime-stagger": "4m"}
NOW = 1_800_000_000.0


@pytest.mark.parametrize(
    ("value", "minutes"),
    [("90", 90), ("90:30", 90), ("2:00:00", 120), ("1-0", 1440), ("1-12", 2160), ("1-12:30", 2190)],
)
def test_parses_every_sbatch_time_form(value, minutes):
    assert _slurm_time_to_minutes(value) == minutes


def preset_fit(head_minutes_left):
    return _fit_worker_time_to_head(
        PRESET["walltime"],
        PRESET["lifetime"],
        PRESET["lifetime-stagger"],
        NOW + head_minutes_left * 60,
        NOW,
    )


def test_a_two_hour_head_job_gets_workers_that_fit_under_a_two_hour_cap():
    walltime, lifetime = preset_fit(119)
    assert _slurm_time_to_minutes(walltime) <= 120
    assert walltime == "1:59:00"
    # Retires before Slurm kills it, even at the top of its random stagger.
    stagger = parse_timedelta(PRESET["lifetime-stagger"])
    assert parse_timedelta(lifetime) + stagger < _slurm_time_to_minutes(walltime) * 60


def test_a_long_head_job_keeps_the_queue_profile():
    assert preset_fit(3 * 1440) == (PRESET["walltime"], PRESET["lifetime"])


def test_outside_a_slurm_job_the_profile_is_unchanged():
    assert _fit_worker_time_to_head("1-0", "1435m", "4m", None, NOW) == ("1-0", "1435m")


def test_a_nearly_finished_head_job_still_yields_a_usable_worker_time():
    walltime, lifetime = preset_fit(1)
    assert _slurm_time_to_minutes(walltime) >= 6
    assert parse_timedelta(lifetime) > 0


def test_the_fitted_time_reaches_the_worker_job_script(tmp_path):
    walltime, lifetime = preset_fit(119)
    script = SLURMJob(
        scheduler="tcp://127.0.0.1:8786",
        name="w",
        cores=1,
        processes=1,
        memory="4GB",
        walltime=walltime,
        log_directory=str(tmp_path),
        worker_extra_args=[f"--lifetime={lifetime}", f"--lifetime-stagger={PRESET['lifetime-stagger']}"],
    ).job_script()
    assert "#SBATCH -t 1:59:00" in script
    assert f"--lifetime={lifetime}" in script
