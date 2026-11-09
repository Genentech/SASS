"""Functions that execute on a dask worker.

These are plain functions that take and return data, and also run without a cluster.
Nothing here may import dask, `sass.cluster` or `sass.tasks`. Keep imports minimal, since
every worker process pays for them.
"""
