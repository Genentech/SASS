"""One module per `tasks:` entry in the config.

Each module enumerates the work for one pipeline phase, submits it to the cluster, and
reads its inputs and writes its outputs. Module names match the config values. The
functions that run on a dask worker live in `sass.worker`.
"""
