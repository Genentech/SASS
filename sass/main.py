"""Entry point: parse arguments, build the run context, dispatch the requested tasks.

Each task module in `sass.tasks` exposes `run(ctx, client)`. It may also expose `max_tasks(ctx)`,
which `preflight.resolve_num_workers` uses to size the cluster.
"""

import argparse
import logging
import os
import sys
import time
from importlib import import_module
from pathlib import Path

import yaml
from openeye import oechem

import sass
from sass import PSEUDO_RES_DIR, SRC_DIR, SYNTHON_CONF_DIR
from sass.chem.mol import load_first_mol
from sass.cluster.client import DaskClient
from sass.cluster.slurm import DASK_SLURM_CONFIG
from sass.config import load_config
from sass.context import RunContext
from sass.io import compress_pkl_files, load_intermediate_files
from sass.library.registry import resolve_library_paths
from sass.library.synthons import SynthonHandler
from sass.logging_setup import CustomOEErrorHandler, StreamToLogger, report_failure, set_logger
from sass.preflight import (
    apply_cli_overrides,
    artifacts_this_run_builds,
    check_memory_budget,
    optional_artifacts_this_run_reads,
    resolve_cluster_name,
    resolve_num_workers,
    validate_aggregation_method,
    validate_color_force_fields,
    validate_config_keys,
    validate_single_fragment_geometry,
    validate_tasks,
)
from sass.tasks.build_synthon_id_index import build_id_index
from sass.tasks.group_synthon_data import build_grouped_synthons, build_synthon_data

#: Task modules in `sass.tasks`, named as in the config `tasks:` list, in run order.
TASK_MODULES = [
    "build_synthon_id_index",
    "group_synthon_data",
    "enumerate_full_library",
    "generate_full_library_conformers",
    "score_full_library",
    "generate_synthon_conformers",
    "generate_single_fragment_conformers",
    "score_single_fragments",
    "score_synthons",
    "instantiate_products",
    "rescore_products",
    "write_single_fragment_poses",
    "write_product_poses",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Run config. Optional -- without it the packaged template's defaults are used.",
    )

    parser.add_argument("--query", "-q", type=str, default=None, help="Query molecule file (a 3D conformer).")

    parser.add_argument(
        "--library",
        type=str,
        default=None,
        help="Synthon library name from sass/libraries.yml. See `sass list-libraries`.",
    )

    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config key, e.g. --set top_m=500000. Repeatable.",
    )

    parser.add_argument("--map_id", "--map-id", action="store_true", help=("If set, maps synthon id to smiles."))

    parser.add_argument(
        "--group-synthons",
        "--group_synthons",
        action="store_true",
        help=("If set, groups synthons by reaction id and number of fragments."),
    )

    return parser.parse_args()


def build_context(args: argparse.Namespace) -> tuple[RunContext, CustomOEErrorHandler, bool]:
    """Resolve the config into a `RunContext`, and set up logging and the OE error handler.

    Returns the context, the error handler (whose warning totals are reported at the end), and
    whether the requested tasks need a cluster at all.
    """
    config = load_config(args.config or SRC_DIR / "config_template.yml")
    # Check before any key is read, so a missing key fails at startup.
    validate_config_keys(config)
    apply_cli_overrides(config, args)
    # Resolve against the launch directory, before the chdir into `exp_dir` below.
    if config["query_molecule"]:
        config["query_molecule"] = str(Path(config["query_molecule"]).resolve())
    tasks = config["tasks"] or []
    required_artifacts = optional_artifacts_this_run_reads(tasks)
    config = resolve_library_paths(
        config,
        produced=artifacts_this_run_builds(tasks, args),
        required=required_artifacts,
    )
    # After resolution, so this checks the directory the run will read.
    if "single_frag_synthon_conf_dir" in required_artifacts:
        validate_single_fragment_geometry(config)

    query_mol_file = config["query_molecule"]
    load_query, load_synthons, use_dask = validate_tasks(tasks, query_mol_file)

    # Resolve before the chdir below. Unset means the launch directory.
    exp_dir = Path(config["exp_dir"] or Path.cwd()).resolve()
    exp_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    os.chdir(exp_dir)

    def folder(val: str | None, sub_dir: str | None = None) -> Path:
        # Absolute, because some of these are handed to dask workers.
        if val is not None:
            return Path(val).resolve()
        return exp_dir if sub_dir is None else exp_dir / sub_dir

    full_exp_name = f"sass_{time.strftime('%Y%m%d_%H%M%S')}"
    logger_path = folder(config["logging_dir"]) / f"{full_exp_name}.log"
    logger = set_logger(logger_path)
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    logger.info("-" * 32)
    logger.info(f"sass version: {sass.__version__}")
    logger.info(f"Start {full_exp_name}, tasks: {tasks}")

    top_m = config["top_m"]
    dedup_limit = max(5e6, 10 * top_m)
    synthon_comb_limit = config["synthon_combination_limit"]
    if synthon_comb_limit is not None:
        # Needs to be at least `top_m` to guarantee correctness of top_m.
        synthon_comb_limit = max(synthon_comb_limit, top_m)
    check_memory_budget(tasks, top_m, synthon_comb_limit, dedup_limit, config["dask_worker_memory"], logger)
    validate_aggregation_method(config["synthon_score_aggregation_method"])
    validate_color_force_fields(config, tasks, logger)

    error_handler = CustomOEErrorHandler(str(logger_path), logger)
    oechem.OEThrow.SetHandlerImpl(error_handler, False)

    ctx = RunContext(
        config=config,
        args=args,
        logger=logger,
        exp_dir=exp_dir,
        tasks=tasks,
        query_mol_file=query_mol_file,
        top_m=top_m,
        dedup_limit=dedup_limit,
        reactions=config["limit_reactions"],
        max_stereocenter=config["enum_max_stereocenter"],
        conf_format=config["conf_file_format"],
        synthon_connectors=config["synthon_connectors"],
        frag_connectors=config["frag_connectors"],
        warts_id_sep=config["warts_id_sep"],
        product_id_sep=config["product_id_sep"],
        opt_shape_func=config["optimization_shape_func"],
        opt_color_func=config["optimization_color_func"],
        overlap_shape_func=config["overlap_shape_func"],
        overlap_color_func=config["overlap_color_func"],
        rocs_key_opt=config["rocs_key_for_overlay_optimization"],
        rocs_key_score=config["rocs_key_for_inplace_scoring"],
        synthon_comb_limit=synthon_comb_limit,
        pseudo_res_dir=exp_dir / PSEUDO_RES_DIR,
        synthon_conf_dir=folder(config["synthon_conf_dir"], SYNTHON_CONF_DIR),
        rescore_file_w=exp_dir / "rescore_combined_rocs_res.pkl",
        rescore_file_r=Path(config["combined_rescore_res_file"] or exp_dir / "rescore_combined_rocs_res.pkl"),
        combined_pseudo_res_file_w=exp_dir / "combined_pseudo_res.pkl",
        combined_pseudo_res_file_r=Path(config["combined_pseudo_res_file"] or exp_dir / "combined_pseudo_res.pkl"),
        synthon_data_w=exp_dir / "synthon_data.pkl",
        synthon_data_r=Path(config["synthon_data_file"] or exp_dir / "synthon_data.pkl"),
        grouped_synthon_w=exp_dir / "grouped_synthons.pkl",
        grouped_synthon_r=Path(config["grouped_synthon_file"] or exp_dir / "grouped_synthons.pkl"),
        rescore_res_dir_w=exp_dir / "rescore_results",
        full_enumeration_product_dir_w=exp_dir / "full_enumeration_products",
        full_enumeration_product_dir_r=Path(
            config["full_enumeration_product_dir"] or exp_dir / "full_enumeration_products"
        ),
        full_enumeration_conf_dir_w=exp_dir / "full_enumeration_conformers",
        full_enumeration_conf_dir_r=Path(
            config["full_enumeration_conf_dir"] or exp_dir / "full_enumeration_conformers"
        ),
        full_enumeration_score_dir_w=exp_dir / "full_enumeration_scores",
        single_frag_synthon_conf_dir=folder(config["single_frag_synthon_conf_dir"], "single_frag_synthon_conformers"),
        single_frag_pseudo_res_file_w=exp_dir / "combined_single_fragment_res.pkl",
        single_frag_pseudo_res_file_r=Path(
            config["single_frag_pseudo_res_file"] or exp_dir / "combined_single_fragment_res.pkl"
        ),
    )

    with (exp_dir / f"{full_exp_name}.yml").open("w") as f:
        yaml.dump(config, f, sort_keys=False)

    if load_synthons:
        _load_synthon_data(ctx)
    if load_query:
        ctx.query_mol = load_first_mol(query_mol_file, mol_cleanup=config["clean_up_query"])
        logger.info(f"Query molecule loaded from {query_mol_file}.")

    return ctx, error_handler, use_dask


def _load_synthon_data(ctx: RunContext) -> None:
    """Load or build `grouped_synthons` and the synthon data table onto `ctx`."""
    config, logger = ctx.config, ctx.logger
    if "build_synthon_id_index" in ctx.tasks:
        # Before `SynthonHandler`, which reads the index to map ids to integers.
        build_id_index(ctx)
    ctx.shandler = SynthonHandler(
        reaction_file=config["reaction_file"],
        synthon_file=config["synthon_file"],
        synthon_id_index=config["synthon_id_index"],
    )
    rebuild = ctx.args.group_synthons or "group_synthon_data" in ctx.tasks

    if rebuild:
        ctx.grouped_synthons = build_grouped_synthons(ctx)
    else:
        ctx.grouped_synthons = load_intermediate_files(ctx.grouped_synthon_r)
        logger.info(f"Grouped synthons loaded from {ctx.grouped_synthon_r}.")

    if ctx.reactions is None:  # Use all reactions
        ctx.reactions = list(ctx.grouped_synthons.keys())

    if ctx.args.map_id or "group_synthon_data" in ctx.tasks:
        ctx.synthon_data = build_synthon_data(ctx)
    else:
        ctx.synthon_data = load_intermediate_files(ctx.synthon_data_r)
        logger.info(f"id mapping skipped. Loaded synthon data from {ctx.synthon_data_r}.")


def _make_client(ctx: RunContext) -> DaskClient:
    config = ctx.config
    cluster_type = config["dask_cluster_type"]
    if cluster_type == "slurm":
        cluster_name = resolve_cluster_name(config, ctx.logger)
        cluster_config = DASK_SLURM_CONFIG[cluster_name]
    else:
        # A local cluster does not read the SLURM settings.
        cluster_config = {"default_qos": None, "queue_time_map": {}, "partition": None}

    n_workers = resolve_num_workers(ctx)

    return DaskClient(
        cluster_type=cluster_type,
        log_dir=ctx.exp_dir / "dask_logs",
        min_workers=n_workers,
        max_workers=n_workers,
        qos=config["qos"] or cluster_config["default_qos"],
        queue_time_map=cluster_config["queue_time_map"],
        partition=config["slurm_partition"] or cluster_config["partition"],
        job_extra_directives=cluster_config.get("job_extra_directives", []),
        memory=config["dask_worker_memory"],
    )


def main():
    args = parse_args()
    try:
        ctx, error_handler, use_dask = build_context(args)
        if not use_dask:
            return
        dask_client = _make_client(ctx)
    except Exception as exc:
        # Startup failures happen before the run-level handler below.
        logging.getLogger(__name__).exception("SASS failed to start.")
        report_failure(exc)
        raise SystemExit(1) from exc

    logger = ctx.logger

    try:  # Catch everything so the dask cluster is always shut down.
        for name in TASK_MODULES:
            if name in ctx.tasks:
                import_module(f"sass.tasks.{name}").run(ctx, dask_client)

        logger.info("Workflow finished")

        if ctx.config["compress_pkl_files"]:
            compress_pkl_files(ctx.exp_dir, keep=ctx.library_artifact_paths)

    except (Exception, KeyboardInterrupt, SystemExit):
        # `SystemExit` is caught because the OpenEye error handler calls `sys.exit(1)`, and the
        # cluster must still shut down. The run still exits non-zero below.
        logger.exception("An error occurred during the SASS run.")
        run_failed = True
    else:
        run_failed = False

    dask_client.close()
    logger.info("Dask client closed.")

    if run_failed:
        # Worker logs are the only record of why a worker died.
        logger.info("Run failed -- keeping dask worker logs in %s for diagnosis.", dask_client.dask_log_dir)
    elif ctx.config["clean_up_dask_logs"] is True:
        dask_client.clean_up_logs()
        logger.info("Deleted dask logs.")

    error_handler.log_warning_summary()

    if run_failed:
        # Exit non-zero so SLURM sees the failure.
        report_failure()
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
