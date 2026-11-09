"""Checks that run before a cluster is created, so a bad config fails in seconds.

Covers task names, the query, library artifacts, worker memory, single-fragment geometry and
color force fields.
"""

import argparse
import logging
import math
from importlib import import_module
from pathlib import Path

import yaml
from openeye import oechem, oeshape

from sass import EXTEND_ATOM_ISOTOPE, EXTEND_ATOM_SYMBOL, SRC_DIR
from sass.chem.mol import load_first_mol, read_mols
from sass.chem.overlay import COLOR_FF_FILE
from sass.cluster.slurm import DASK_SLURM_CONFIG, detect_host_from_slurm
from sass.config import load_config
from sass.context import RunContext
from sass.library.registry import detect_host
from sass.selection.aggregate import SUPPORTED_AGG_METHODS
from sass.selection.combine import estimate_peak_combine_gb
from sass.worker.merge_scores import estimate_peak_merge_gb

logger = logging.getLogger(__name__)


#: Library artifact -> the tasks that build it. A run that builds an artifact does not require it.
_BUILDS = {
    "synthon_id_index": ("build_synthon_id_index",),
    "grouped_synthon_file": ("group_synthon_data",),
    "synthon_data_file": ("group_synthon_data",),
    "synthon_conf_dir": ("generate_synthon_conformers",),
    "single_frag_synthon_conf_dir": ("generate_single_fragment_conformers",),
}


def artifacts_this_run_builds(tasks: list[str], args: argparse.Namespace) -> set[str]:
    """Which library artifacts this run produces rather than reads."""
    produced = {key for key, builders in _BUILDS.items() if any(t in tasks for t in builders)}
    if args.group_synthons:
        produced.add("grouped_synthon_file")
    if args.map_id:
        produced.add("synthon_data_file")
    return produced


#: Config `tasks:` value -> whether that task needs the synthon/reaction tables loaded.
VALID_TASKS = {
    "build_synthon_id_index": False,  # reads the raw synthon file directly, not the tables
    "group_synthon_data": True,
    "enumerate_full_library": True,
    "generate_full_library_conformers": False,
    "score_full_library": False,
    "generate_synthon_conformers": True,
    "generate_single_fragment_conformers": False,
    "score_synthons": True,
    "instantiate_products": True,
    "rescore_products": True,  # In case `rescore_products` also includes instantiation
    "write_product_poses": True,  # Re-instantiates the top products, so needs synthon data.
    "write_single_fragment_poses": True,
    "score_single_fragments": True,
}

#: The query molecule, as it appears in `TASK_REQUIREMENTS`.
QUERY_INPUT = "query"

#: Task -> the inputs it needs that a run may not have. `validate_tasks` checks the query, and
#: `resolve_library_paths` checks the artifacts. Artifacts listed here must also be in
#: `registry.OPTIONAL_KEYS` and `_BUILDS`.
TASK_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "score_full_library": (QUERY_INPUT,),
    "score_synthons": (QUERY_INPUT,),
    "rescore_products": (QUERY_INPUT,),
    "write_product_poses": (QUERY_INPUT,),
    "score_single_fragments": (QUERY_INPUT, "single_frag_synthon_conf_dir"),
    "write_single_fragment_poses": (QUERY_INPUT, "single_frag_synthon_conf_dir"),
}

#: Tasks that overlay something against the query, and therefore require one.
SCORING_TASKS = frozenset(task for task, needs in TASK_REQUIREMENTS.items() if QUERY_INPUT in needs)


def optional_artifacts_this_run_reads(tasks: list[str]) -> dict[str, str]:
    """Library artifacts the requested tasks read, mapped to why the run requires them.

    An optional library artifact is required by any run that reads it.
    """
    readers: dict[str, list[str]] = {}
    for task, needs in TASK_REQUIREMENTS.items():
        if task not in tasks:
            continue
        for need in needs:
            # A run that builds the artifact must not be asked to already have it.
            if need != QUERY_INPUT and not set(_BUILDS[need]) & set(tasks):
                readers.setdefault(need, []).append(task)
    return {
        artifact: f"{', '.join(tasks_)}; build it with the `{'`, `'.join(_BUILDS[artifact])}` task"
        for artifact, tasks_ in readers.items()
    }


def valid_task_names() -> list[str]:
    """Every task name a config may name."""
    return list(VALID_TASKS)


#: Advised maximum `top_m`. Above it, `rescore_products` dominates the run. Not enforced.
_TOP_M_ADVISED_MAX = 2_000_000

#: Fraction of a worker's memory the score merge may use before a warning. dask pauses a
#: worker at 80% and kills it at 95%, and the worker also holds other score lists.
_MERGE_MEMORY_HEADROOM = 0.6


def useful_workers(ctx: RunContext) -> dict[str, int] | None:
    """Return the number of dask tasks each requested task will submit.

    A task module defines `max_tasks(ctx)` if it can count its tasks before the cluster starts.
    If any requested task cannot, return `None`.
    """
    widths = {}
    for name in ctx.tasks:
        count = getattr(import_module(f"sass.tasks.{name}"), "max_tasks", None)
        width = count(ctx) if count else None
        if width is None:
            return None
        widths[name] = width
    return widths or None


def resolve_num_workers(ctx: RunContext) -> int:
    """Return the number of SLURM workers to request.

    Request fewer workers than `num_workers` when the run has fewer dask tasks than that. The
    cluster has a fixed size, so extra workers stay idle until the run ends. Tasks run one after
    another, so the task with the most dask tasks decides the number. The result is never more
    than `num_workers` and never less than 1. Local clusters are not changed.
    """
    requested = ctx.config["num_workers"]
    if ctx.config["dask_cluster_type"] != "slurm":
        return requested
    widths = useful_workers(ctx)
    if not widths or max(widths.values()) >= requested:
        return requested

    # Request at least one worker, even when there are no tasks.
    n_workers = max(1, *widths.values())
    ctx.logger.info(
        "Requesting %d SLURM workers instead of num_workers=%d, because this run has no more dask "
        "tasks than that (%s).",
        n_workers,
        requested,
        ", ".join(f"{name}: {width}" for name, width in widths.items()),
    )
    return n_workers


def resolve_cluster_name(config: dict, logger: logging.Logger) -> str:
    """Which SLURM cluster's partition/QoS defaults to use.

    An explicit `cluster_name` wins. Otherwise the cluster is detected from the environment.
    """
    name = config.get("cluster_name") or _detect_cluster_name(logger)
    if name not in DASK_SLURM_CONFIG:
        raise ValueError(f"Invalid cluster name: {name}. Valid options: {sorted(DASK_SLURM_CONFIG)}.")
    return name


def _detect_cluster_name(logger: logging.Logger) -> str:
    """Work out which cluster this is, from the environment."""
    # Try the mounted data root first.
    detected = detect_host()
    if detected in DASK_SLURM_CONFIG:
        logger.info("Detected cluster from the mounted data root: %s.", detected)
        return detected

    # Then ask SLURM, for nodes where no known data root is mounted.
    detected = detect_host_from_slurm()
    if detected in DASK_SLURM_CONFIG:
        logger.info("Detected cluster from SLURM ClusterName: %s.", detected)
        return detected

    raise ValueError(
        "Could not detect which cluster this is -- no known SASS data root is mounted and SLURM "
        f"did not report a recognized ClusterName -- and `cluster_name` is not set. Set it to "
        f"one of: {sorted(DASK_SLURM_CONFIG)}."
    )


#: Task -> the memory peaks it can reach. Tasks not listed get no estimate.
_MEMORY_PEAKS_BY_TASK = {
    "score_synthons": ("merge", "combine"),
    "score_single_fragments": ("merge",),
    "score_full_library": ("merge",),
    "rescore_products": ("merge",),
}


def check_memory_budget(
    tasks: list[str],
    top_m: int,
    synthon_comb_limit: int | None,
    dedup_limit: float,
    worker_memory_gb: int,
    logger: logging.Logger,
) -> None:
    """Warn at startup if a worker is likely to run out of memory, with the estimates.

    Two peaks are checked, each only for the tasks that can reach it:

    * the **score merge** (`merge_score_batch_task`), driven by `top_m`;
    * the **combination step** (`combine_forder_scores`) before a score cutoff exists, driven
      by `synthon_combination_limit` capped at `dedup_limit`. Only the warm-up wave reaches it.
    """
    reachable = {peak for task in tasks for peak in _MEMORY_PEAKS_BY_TASK.get(task, ())}
    if not reachable:
        return

    if top_m > _TOP_M_ADVISED_MAX:
        logger.warning(
            "top_m=%s exceeds the advised maximum of %s. Past this the run is dominated by "
            "`rescore_products`, which runs OMEGA and ROCS on every selected product.",
            f"{top_m:,}",
            f"{_TOP_M_ADVISED_MAX:,}",
        )

    budget_gb = worker_memory_gb * _MERGE_MEMORY_HEADROOM
    candidates = {
        "merge": (
            "score merge (`merge_score_batch_task`, driven by top_m)",
            lambda: estimate_peak_merge_gb(top_m),
        ),
        "combine": (
            "combination before a cutoff exists (`combine_forder_scores`, driven by synthon_combination_limit)",
            lambda: estimate_peak_combine_gb(synthon_comb_limit, dedup_limit),
        ),
    }
    peaks = {label: estimate() for peak, (label, estimate) in candidates.items() if peak in reachable}

    for what, peak_gb in peaks.items():
        if peak_gb > budget_gb:
            logger.warning(
                "The %s will peak near %.1f GB, against a %s GB worker (usable ~%.1f GB after "
                "dask's own headroom). This run is likely to fail with KilledWorker. Raise "
                "`dask_worker_memory` to >= %d, or lower the parameter driving it.",
                what,
                peak_gb,
                worker_memory_gb,
                budget_gb,
                math.ceil(peak_gb / _MERGE_MEMORY_HEADROOM),
            )

    summary = ", ".join(f"{gb:.1f} GB {label.split(' (')[0]}" for label, gb in peaks.items())
    logger.info("Peak worker memory estimate: %s, against %s GB workers.", summary, worker_memory_gb)


#: Config keys with a dedicated CLI flag, so `--set` can point at the friendlier spelling.
_CLI_FLAG_FOR_KEY = {"query_molecule": "--query/-q", "synthon_library": "--library"}


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    """Fold `--query`, `--library` and `--set key=value` into an already-loaded config.

    An unrecognized `--set` key raises, unlike an unknown key in a config file, which only warns.
    """
    settings = getattr(args, "overrides", None) or []
    template = load_config(SRC_DIR / "config_template.yml") if settings else {}
    for setting in settings:
        key, sep, raw = setting.partition("=")
        key = key.strip()
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {setting!r}.")
        if key not in template:
            close = [k for k in template if k.startswith(key[:4])] if len(key) >= 4 else []
            hint = f" Did you mean: {', '.join(sorted(close))}?" if close else ""
            raise ValueError(f"--set {key}=... is not a config key.{hint}")
        if key in _CLI_FLAG_FOR_KEY:
            logger.info("`--set %s=` works, but %s is the documented spelling.", key, _CLI_FLAG_FOR_KEY[key])
        # Parse as YAML, so values get the same types as in a config file.
        config[key] = yaml.safe_load(raw)

    # Applied last, so the dedicated flags win over `--set`.
    if getattr(args, "query", None):
        config["query_molecule"] = args.query
    if getattr(args, "library", None):
        config["synthon_library"] = args.library


def validate_config_keys(config: dict) -> None:
    """Check the config's keys against `config_template.yml` before anything else runs.

    Missing keys raise, listing each with its default. Unknown keys only warn. This runs before
    the run log exists, so the warning goes to stderr.
    """
    template = load_config(SRC_DIR / "config_template.yml")

    unknown = sorted(set(config) - set(template))
    if unknown:
        logger.warning(
            "These config keys are not recognized and will be ignored -- check for a typo: %s.",
            ", ".join(unknown),
        )

    missing = [key for key in template if key not in config]
    if missing:
        # Dump each as a one-key mapping, since `safe_dump` on a bare scalar appends `...`.
        lines = "".join(
            "  " + yaml.safe_dump({key: template[key]}, default_flow_style=False) for key in missing
        ).rstrip()
        raise ValueError(
            f"This config is missing {len(missing)} key(s) that SASS reads. This config likely "
            f"predates the current SASS version; see CHANGELOG.md. Add them to continue:\n{lines}"
        )


def validate_aggregation_method(agg_method: str) -> None:
    """Reject a `synthon_score_aggregation_method` the worker cannot actually run.

    Some aggregation methods need data that `combine_forder_scores` does not have. Rejecting
    them here avoids a `TypeError` inside a dask task.
    """
    if agg_method not in SUPPORTED_AGG_METHODS:
        raise ValueError(
            f"synthon_score_aggregation_method: {agg_method!r} is not supported. "
            f"Choose one of: {', '.join(sorted(SUPPORTED_AGG_METHODS))}. "
            "Note that only `simple_average` and `min_score` allow cutoff pruning, so the "
            "others are substantially slower."
        )


#: Number of molecules to read from the sampled conformer file.
_GEOMETRY_PROBE_MOLS = 5

#: Tolerance on the measured connector-to-extension distance, in Angstrom.
_BOND_DISTANCE_TOL = 0.01


def probe_single_fragment_geometry(
    conf_dir: Path, conf_format: str, connector: str
) -> tuple[Path, float | None] | None:
    """Measure the bond-extension geometry of a single-fragment conformer directory.

    Returns the sampled file and the connector-to-extension distance. The distance is `None` if
    the conformers carry no extension atom (built with `align_on_bond: false`). Returns `None`
    if nothing could be read.
    """
    sample = next(iter(sorted(conf_dir.glob(f"*.{conf_format}"))), None)
    if sample is None:
        return None

    connector_num = oechem.OEGetAtomicNum(connector)
    distances = []
    try:
        with read_mols(sample, graph=True) as mols:
            for n, mol in enumerate(mols):
                if n >= _GEOMETRY_PROBE_MOLS:
                    break
                coords = mol.GetCoords()
                conn = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == connector_num]
                # Match the isotope too, since a synthon may itself contain silicon.
                ext = [
                    a.GetIdx()
                    for a in mol.GetAtoms()
                    if a.GetAtomicNum() == oechem.OEGetAtomicNum(EXTEND_ATOM_SYMBOL)
                    and a.GetIsotope() == EXTEND_ATOM_ISOTOPE
                ]
                if len(conn) != 1:
                    continue
                distances.append(math.dist(coords[conn[0]], coords[ext[0]]) if len(ext) == 1 else None)
    except (ValueError, OSError) as e:
        logger.warning("Could not read %s to check the single-fragment geometry: %s", sample, e)
        return None

    if not distances:
        return None
    return sample, distances[0]


def validate_single_fragment_geometry(config: dict) -> None:
    """Refuse a run whose `align_on_bond` settings disagree with the conformers it will score.

    Single-fragment scoring aligns along the connector bond, using a second dummy atom placed
    `extend_bond_distance` along it. The query is extended at run time from the config, so the
    config must match how the stored conformers were built.
    """
    conf_dir = Path(config["single_frag_synthon_conf_dir"])
    probed = probe_single_fragment_geometry(conf_dir, config["conf_file_format"], config["synthon_connectors"][0])
    if probed is None:
        logger.warning(
            "Could not sample %s, so `align_on_bond` and `extend_bond_distance` were not checked "
            "against the conformers this run will score.",
            conf_dir,
        )
        return

    sample, measured = probed
    align, configured = config["align_on_bond"], config["extend_bond_distance"]
    built_with = (
        f"`align_on_bond: false` (no {EXTEND_ATOM_ISOTOPE}{EXTEND_ATOM_SYMBOL} extension atom)"
        if measured is None
        else f"`align_on_bond: true` and `extend_bond_distance: {measured:g}`"
    )
    remedy = (
        f"Set `align_on_bond: {str(measured is not None).lower()}`"
        + ("" if measured is None else f" and `extend_bond_distance: {measured:g}`")
        + ", or point `single_frag_synthon_conf_dir` at conformers built the way this config "
        "describes. Both libraries and queries must agree: the query is extended at run time "
        "from these same settings."
    )
    mismatch = (
        (align and measured is None)
        or (not align and measured is not None)
        or (align and measured is not None and abs(measured - configured) > _BOND_DISTANCE_TOL)
    )
    if mismatch:
        raise ValueError(
            f"Single-fragment conformers in {conf_dir} were built with {built_with}, but this run "
            # Lowercased: the message is meant to be pasted into YAML, where `True` is a string.
            f"is configured for `align_on_bond: {str(align).lower()}`"
            + (f" and `extend_bond_distance: {configured:g}`" if align else "")
            + f". Measured from {sample.name}. Scoring would still produce numbers, computed "
            f"against a bond vector that does not line up. {remedy}"
        )

    logger.info("Single-fragment conformers in %s match this run's geometry: %s.", conf_dir, built_with)


#: Config keys that only affect the color force fields SASS generates. They have no effect when
#: `color_ff_dir` is set. `align_on_bond` is absent because it also changes the query.
_GENERATED_COLOR_FF_KEYS = (
    "base_cff_file",
    "align_on_dummy_atom",
    "custom_color_ff",
    "custom_color_weight",
    "custom_color_radius",
    "color_interaction_type",
    "scale_weight_per_atom",
    "no_default_color",
    "additional_color_patterns",
)


def validate_color_force_fields(config: dict, tasks: list[str], logger: logging.Logger) -> None:
    """Check the color force fields a run will overlay with, before a cluster exists.

    There is one file per connector count, `0` through `len(synthon_connectors)`, and a worker
    picks the file by the number of connectors on the molecule it scores. This checks that a
    given `color_ff_dir` has every file, and that `custom_color_ff` has one pattern per
    connector. Only runs that overlay something are checked.
    """
    if not any(t in SCORING_TASKS for t in tasks):
        return

    max_key = len(config["synthon_connectors"])
    expected = [COLOR_FF_FILE.format(key=key) for key in range(max_key + 1)]
    directory = config.get("color_ff_dir")

    if not directory:
        patterns = config["custom_color_ff"] or []
        if config["align_on_dummy_atom"] and len(patterns) < max_key:
            raise ValueError(
                f"`custom_color_ff` has {len(patterns)} pattern(s) but `synthon_connectors` has "
                f"{max_key}, so the generated color force fields would only cover connector counts "
                f"0-{len(patterns)}. A synthon with {len(patterns) + 1} connectors would fail on a "
                "worker with no such file. Give `custom_color_ff` one pattern per connector atom."
            )
        return

    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"`color_ff_dir` is not a directory: {path}. Unset it to have SASS generate one.")

    missing = [name for name in expected if not (path / name).is_file()]
    if missing:
        raise ValueError(
            f"`color_ff_dir` {path} is missing {len(missing)} of the {len(expected)} color force field "
            f"files this run can ask for:\n  " + "\n  ".join(missing) + "\n"
            f"There is one per connector count, 0 through {max_key} (`synthon_connectors`), and a "
            "worker picks by the number of connector atoms on the synthon it is scoring. Reuse a "
            "complete `color_ffs/` from an earlier run, or unset `color_ff_dir` to have SASS "
            "generate one."
        )

    # Warn, since nothing else reports that these keys are ignored.
    template = load_config(SRC_DIR / "config_template.yml")
    overridden = [key for key in _GENERATED_COLOR_FF_KEYS if config.get(key) != template.get(key)]
    if overridden:
        logger.warning(
            "`color_ff_dir` is set, so %s is used as found and these keys have no effect: %s. "
            "Unset `color_ff_dir` to have them applied.",
            path,
            ", ".join(overridden),
        )
    else:
        logger.info("Using the color force fields in %s as found; nothing is added to them.", path)


def validate_tasks(tasks: list[str], query_file: str | None) -> tuple[bool, bool, bool]:
    """Validate the tasks and query file.

    Also determine whether to load the query and synthons based on the tasks.
    """
    load_query = False
    load_synthons = False
    # These two tasks do their work in `build_context`, so they need no cluster.
    use_dask = not set(tasks or []) <= {"group_synthon_data", "build_synthon_id_index"}

    if tasks is not None:
        tasks = set(tasks)
        invalid_tasks = [t for t in tasks if t not in VALID_TASKS]
        if invalid_tasks:
            raise ValueError(f"Invalid tasks: {invalid_tasks}")
        load_synthons = any(VALID_TASKS[t] for t in tasks)

        if any(t in SCORING_TASKS for t in tasks):
            if query_file is None:
                raise ValueError(f"Must specify a query for tasks: {[t for t in tasks if t in SCORING_TASKS]}!")
            validate_query(query_file)
            load_query = True

        if any("single_fragment" in t for t in tasks) and any("single_fragment" not in t for t in tasks):
            raise ValueError("Single fragment tasks cannot be combined with other tasks.")

        if "instantiate_products" in tasks and "rescore_products" in tasks:
            raise ValueError("`Rescore products` step already contains product instantiation.")

    return load_query, load_synthons, use_dask


def validate_query(file: Path | str) -> None:
    if not Path(file).is_file():
        raise ValueError(f"Query molecule file {file} does not exist!")

    # Check number of conformers. Not an issue for SDF file input.
    query = load_first_mol(file)
    n_confs = query.NumConfs()
    if n_confs > 1:
        raise ValueError(f"Query molecule should only have one conformer, but has {n_confs} conformers!")

    # Check dimension first, so a flat query gets a clear error.
    oechem.OESetDimensionFromCoords(query)
    if query.GetDimension() != 3:
        raise ValueError(
            f"Query molecule in {file} is {query.GetDimension()}D, not a 3D conformer. "
            "Generate one with OMEGA (or any 3D structure) before searching."
        )

    # Check if molecule can be set as ROCS reference.
    overlay = oeshape.OEOverlay()
    if not overlay.SetupRef(query):
        raise ValueError("Query molecule cannot be set as ROCS reference! Check if it is a 3D conformer.")
