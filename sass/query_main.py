"""Main script for running SASS queries."""

# Standard Library
import argparse
import logging
import math
import os
import pickle
import random
import sys
from pathlib import Path

# Third Party Library
import yaml
from openeye import oechem

# Genentech Library
import sass
from sass import COLOR_FF_DIR, PSEUDO_RES_DIR, SRC_DIR, SYNTHON_CONF_DIR
from sass.utils.utils_dask import DaskClient
from sass.utils.utils_data import SynthonHandler
from sass.utils.utils_general import CustomOEErrorHandler, StreamToLogger, extract_base_id, load_config, set_logger
from sass.utils.utils_mol import (
    fragment_molecule,
    load_first_mol,
)
from sass.utils.utils_overlay import overlay_opt_builder
from sass.utils.utils_query import (
    compress_pkl_files,
    load_intermediate_files,
    map_synthon_ids,
    merge_synthon_conformer_chunks,
    set_file_name,
    validate_tasks,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=SRC_DIR / "config_template.yml",
    )

    parser.add_argument("--map_id", "--map-id", action="store_true", help=("If set, maps synthon id to smiles."))

    parser.add_argument(
        "--group-synthons",
        "--group_synthons",
        action="store_true",
        help=("If set, groups synthons by reaction id and number of fragments."),
    )

    return parser.parse_args()


def main():

    def _set_folder_path(val: str, sub_dir: str | None = None) -> Path:
        nonlocal exp_dir
        if val is None:
            if sub_dir is None:
                return exp_dir
            else:
                return exp_dir / sub_dir
        else:
            return Path(val)

    args = parse_args()
    config = load_config(args.config)

    tasks = config["tasks"]
    if tasks is None:
        tasks = []
    query_mol_file = config["query_molecule"]
    load_query, load_synthons = validate_tasks(tasks, query_mol_file)

    if load_query:
        query_mol = load_first_mol(query_mol_file, mol_cleanup=config["clean_up_query"])
        logging.info(f"Query molecule loaded from {query_mol_file}.")

    exp_dir = config["exp_dir"]
    if not exp_dir:
        exp_dir = Path(args.config).parent.resolve()
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    os.chdir(exp_dir)

    top_m = config["top_m"]
    top_m_high = 5e6
    if top_m > top_m_high:
        logging.warning((f"A very large top_m of {top_m} is chosen. This will cause long runtime for rescoring!"))
    full_exp_name = f"sass_{top_m}"

    # Set up logging.
    logging_dir = _set_folder_path(config["logging_dir"])
    logger_path = logging_dir / f"{full_exp_name}.log"
    logger = set_logger(logger_path)
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    logging.info("-" * 32)
    logging.info(f"sass version: {sass.__version__}")
    logging.info(f"Start {full_exp_name}, tasks: {tasks}")

    # Configure OETK error handler
    error_handler = CustomOEErrorHandler(str(logger_path))
    oechem.OEThrow.SetHandlerImpl(error_handler, False)

    # Optionally load cff from files:
    if config["color_ff_dir"] is not None:
        color_ff_dir = Path(config["color_ff_dir"])
    else:
        # Write out copies of color FF.
        custom_overlay_opts = overlay_opt_builder(
            align_on_dummy_atom=config["align_on_dummy_atom"],
            color_weight=config["custom_color_weight"],
            color_radius=config["custom_color_radius"],
            interaction_type=config["color_interaction_type"],
            color_patterns=config["custom_color_ff"],
            additional_colors=config["additional_color_patterns"],
            scale_weight_per_atom=config["scale_weight_per_atom"],
            no_default_color=config["no_default_color"],
            base_cff_file=config["base_cff_file"],
        )
        color_ff_dir = exp_dir / COLOR_FF_DIR
        color_ff_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        for key, overlay_opt in custom_overlay_opts.items():
            custom_ff = overlay_opt.GetColorOptions().GetColorForceField()
            custom_ff.Write(oechem.oeofstream(str(color_ff_dir / f"custom_color_ff_{key}.txt")))

    # Other config parameters.
    reactions = config["limit_reactions"]
    max_stereocenter = config["enum_max_stereocenter"]
    conf_format = config["conf_file_format"]
    synthon_connectors = config["synthon_connectors"]
    frag_connectors = config["frag_connectors"]
    warts_id_sep = config["warts_id_sep"]
    product_id_sep = config["product_id_sep"]
    opt_shape_func = config["optimization_shape_func"]
    opt_color_func = config["optimization_color_func"]
    overlap_shape_func = config["overlap_shape_func"]
    overlap_color_func = config["overlap_color_func"]
    rocs_key_opt = config["rocs_key_for_overlay_optimization"]
    rocs_key_score = config["rocs_key_for_inplace_scoring"]
    synthon_comb_limit = config["synthon_combination_limit"]
    if synthon_comb_limit is not None:
        synthon_comb_limit = max(synthon_comb_limit, top_m)
        # Needs to be at least `top_m` to guarantee correctness of top_m.

    # Directory paths.
    pseudo_res_dir = Path(PSEUDO_RES_DIR)
    synthon_conf_dir = _set_folder_path(config["synthon_conf_dir"], SYNTHON_CONF_DIR)
    rescore_file_w = exp_dir / "rescore_combined_rocs_res.pkl"
    rescore_file_r = set_file_name(config["combined_rescore_res_file"], rescore_file_w)
    combined_pseudo_res_file_w = exp_dir / f"{full_exp_name}_combined_pseudo_res.pkl"
    combined_pseudo_res_file_r = set_file_name(config["combined_pseudo_res_file"], combined_pseudo_res_file_w)
    synthon_data_w = exp_dir / "synton_data.pkl"
    synthon_data_r = set_file_name(config["synthon_data_file"], synthon_data_w)
    grouped_synthon_w = exp_dir / "grouped_synthons.pkl"
    grouped_synthon_r = set_file_name(config["grouped_synthon_file"], grouped_synthon_w)
    rescore_conformer_dir_r = _set_folder_path(config["rescore_conformer_dir"], "rescore_results")
    rescore_res_dir_w = exp_dir / "rescore_results"
    full_enumeration_product_dir_w = exp_dir / "full_enumeration_products"
    full_enumeration_product_dir_r = set_file_name(
        config["full_enumeration_product_dir"], full_enumeration_product_dir_w
    )
    full_enumeration_conf_dir_w = exp_dir / "full_enumeration_conformers"
    full_enumeration_conf_dir_r = set_file_name(config["full_enumeration_conf_dir"], full_enumeration_conf_dir_w)
    full_enumeration_score_dir_w = exp_dir / "full_enumeration_scores"

    # Save a copy of the config file.
    with (exp_dir / f"{full_exp_name}.yml").open("w") as f:
        yaml.dump(config, f, sort_keys=False)

    # Load synthon/reaction data.
    if load_synthons:
        shandler = SynthonHandler(
            reaction_file=config["reaction_file"],
            synthon_file=config["synthon_file"],
        )
        if args.group_synthons or config["grouped_synthon_file"] is None:
            # Build the `grouped_synthons` on the fly.
            grouped_synthons = shandler.group_synthon_by_rxn_id(
                n_components=config["n_fragments"],
                rxn_ids=reactions,
                limit=config["synthon_limit"],
                random_seed=config["random_seed"],
                cleanup=False,
            )
            with grouped_synthon_w.open("wb") as f:
                pickle.dump(grouped_synthons, f)
            logging.info(f"Grouped synthons saved to {grouped_synthon_w}.")
        else:
            grouped_synthons = load_intermediate_files(grouped_synthon_r)
            logging.info(f"Grouped synthons loaded from {grouped_synthon_r}.")

        if reactions is None:  # Use all reactions
            reactions = list(grouped_synthons.keys())

        # Build synthon data dict with id as the key.
        # {id: {'SMILES': str, 's_idx': int, 'ha_count': int, 'rxn_id': str}
        if args.map_id or config["synthon_data_file"] is None:
            logging.info("Start building synthon data dict.")  # TODO: Abstract this to a function.
            synthon_data = map_synthon_ids(
                reactions=reactions,
                grouped_synthons=grouped_synthons,
                synthon_connectors=synthon_connectors,
            )
            with synthon_data_w.open("wb") as f:
                pickle.dump(synthon_data, f)
            logging.info(f"Synthon data dict saved to {synthon_data_w}!")

        else:
            synthon_data = load_intermediate_files(synthon_data_r)
            logging.info(f"id mapping skipped. Loaded synthon data from {synthon_data_r}.")

    # Main tasks.
    dask_client = DaskClient(
        min_workers=max(config["num_dask_workers"] // 10, 2),
        max_workers=config["num_dask_workers"],
        qos=config["qos"],
        cluster_type=config["dask_cluster_type"],
        log_dir=exp_dir / "dask_logs",
        queue_time_map=None,  # TODO: read from config.
    )
    logging.info(f"Dask cluster initialized with {dask_client.n_workers} workers.")

    # Ground truth calculations.
    if "full_library_enum" in tasks:
        logging.info("Starting full library enumeration.")
        full_enumeration_product_dir_w.mkdir(mode=0o755, parents=True, exist_ok=True)
        dask_client.dask_instantiate_library_products(
            synthon_handler=shandler,
            grouped_synthons=grouped_synthons,
            reaction_ids=reactions,
            output_dir=full_enumeration_product_dir_w,
            product_chunk_size=config["product_chunk_size"],
            out_file_suffix="product.oeb.gz",
            sort_title=True,
            title_separator=product_id_sep,
            max_stereocenter=max_stereocenter,
        )
        logging.info("Finished full product enumeration. Chunks saved to {full_enumeration_product_dir_w}.")

    if "full_library_conf_gen" in tasks:
        logging.info("Starting conformer generation for full library.")
        full_enumeration_conf_dir_w.mkdir(mode=0o755, parents=True, exist_ok=True)
        dask_client.dask_generate_conformers(
            input_dir=full_enumeration_product_dir_r,
            output_dir=full_enumeration_conf_dir_w,
            product_file_suffix="product.oeb.gz",
            delete_input_file=False,
            omega_max_conf=config["product_omega_max_conf"],
            omega_max_time=config["product_omega_max_time"],
            omega_e_window=config["product_omega_energy"],
        )
        logging.info("Finished full library conformer generation. Chunks saved to {full_enumeration_conf_dir_w}.")

    if "full_library_scoring" in tasks:
        logging.info("Starting ROCS scoring for full library.")
        full_enumeration_score_dir_w.mkdir(mode=0o755, parents=True, exist_ok=True)
        score_files = dask_client.dask_overlay_molecules(
            input_dir=full_enumeration_conf_dir_r,
            output_dir=full_enumeration_score_dir_w,
            delete_conf_file=False,
            ref_mol=oechem.OEMol(query_mol),
            opt_shape_func=opt_shape_func,
            opt_color_func=opt_color_func,
            overlap_shape_func=overlap_shape_func,
            overlap_color_func=overlap_color_func,
            color_ff_dir=color_ff_dir,
            warts_separator=warts_id_sep,
            top_n=5000,
        )
        combined_scores = dask_client.combine_files(
            file_list=score_files,
            output_dir=full_enumeration_score_dir_w,
            top_m=5000,
            warts_separator=warts_id_sep,
            dedup_limit=1e6,
            file_name="ground_truth_score",
            keep_isomer_warts=True,
        )
        combined_scores_file = exp_dir / "ground_truth_combined_rocs_res.pkl"
        with combined_scores_file.open("wb") as f:
            pickle.dump(combined_scores, f)
        logging.info(f"Finished ROCS scoring for full library. Result saved to {combined_scores_file}.")

    # Generate synthon conformers.
    if "gen_synthon_conformers" in tasks:
        synthon_conf_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        dask_client.dask_generate_synthon_conformers(
            reactions=reactions,
            grouped_synthons=grouped_synthons,
            synthon_handler=shandler,
            ofile_format=conf_format,
            connector_atoms=synthon_connectors,
            chunk_size=config["synthon_confgen_chunk_size"],
            output_dir=synthon_conf_dir,
            omega_max_conf=config["synthon_omega_max_conf"],
            omega_max_time=config["synthon_omega_max_time"],
            omega_e_window=config["synthon_omega_energy"],
        )
        logging.info(f"Finished generating synthon conformers. Chunks saved to {synthon_conf_dir}.")

        # Merge synthon conformer chunks.
        merge_synthon_conformer_chunks(
            file_dir=synthon_conf_dir,
            conf_format=conf_format,
        )

    # Score synthons, pick top combinations.
    if "score_synthons" in tasks:
        logging.info((f"Starting synthon scoring using conformers from {synthon_conf_dir}."))

        query_fragments_dict = {}
        for n in config["n_fragments"]:
            query_fragments_dict[n] = fragment_molecule(
                mol=oechem.OEMol(query_mol),
                n_fragments=n,
                heavy_atom_limit=config["heavy_atom_limit"],
                cleave_acyclic_bonds=True,
                cleave_cyclic_bonds=config["cleave_cyclic_bonds"],
                connector_isotope=1,  # Use non-0 isotope to signify a connector atom for `check_fragment_size`.
                connector_atoms=frag_connectors,
            )

        pseudo_res_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        pseudo_res_file_dict, prod_count = dask_client.dask_score_and_combine_synthon(
            reactions=reactions,
            shandler=shandler,
            query_fragments_dict=query_fragments_dict,
            synthon_conf_dir=synthon_conf_dir,
            grouped_synthons=grouped_synthons,
            conf_format=conf_format,
            align_connector_atom=config["align_on_dummy_atom"],
            synthon_connectors=synthon_connectors,
            rocs_key_opt=rocs_key_opt,
            rocs_key_score=rocs_key_score,
            opt_shape_func=opt_shape_func,
            opt_color_func=opt_color_func,
            overlap_shape_func=overlap_shape_func,
            overlap_color_func=overlap_color_func,
            top_m=top_m,
            dedup_limit=max(top_m_high, 10 * top_m),
            warts_separator=warts_id_sep,
            sass_separator=product_id_sep,
            agg_method=config["synthon_score_aggregation_method"],
            synthon_comb_limit=synthon_comb_limit,
            frag_connectors=frag_connectors,
            cross_score=config["cross_scoring"],
            color_ff_dir=color_ff_dir,
            pseudo_res_dir=pseudo_res_dir,
        )

        comb_pseudo_products = dask_client.combine_pseudo_res(
            pseudo_res=pseudo_res_file_dict,
            pseudo_res_dir=pseudo_res_dir,
            top_m=top_m,
            warts_separator=warts_id_sep,
            use_proportion=config["mix_pseudo_res_by_proportion"],
            product_count=prod_count,
            dedup_limit=max(top_m_high, 10 * top_m),
        )

        with combined_pseudo_res_file_w.open("wb") as f:
            pickle.dump(comb_pseudo_products, f)
        logging.info(f"Finished combining pseudo products. Result saved to {combined_pseudo_res_file_w}.")

    # Instantiate and rescore products.
    if "instantiate_products" in tasks or "rescore_products" in tasks:
        logging.info("Start instantiating pseudo products for rescoring.")
        rescore_res_dir_w.mkdir(mode=0o755, parents=True, exist_ok=True)
        products = load_intermediate_files(combined_pseudo_res_file_r)
        random.shuffle(products)
        logging.info(f"Combined pseudo products loaded from {combined_pseudo_res_file_r}.")

        # Dynamically set the chunk size based on number of dask workers.
        min_chunk_size = max(config["dask_rescore_min_chunk_size"], math.ceil(len(products) / dask_client.n_workers))
        chunk_size = min(config["dask_rescore_max_chunk_size"], min_chunk_size)

        dask_client.dask_instantiate_singleton_products(
            product_list=products,
            instantiation_chunk_size=1000,  # Manually set chunk size. Too small chunk size leads to `dask` error.
            file_chunk_size=chunk_size,
            shandler=shandler,
            synthon_data=synthon_data,
            title_separator=product_id_sep,
            warts_separator=warts_id_sep,
            output_dir=rescore_res_dir_w,
            product_file_suffix="product.oeb.gz",
            max_stereocenter=max_stereocenter,
        )
        logging.info(f"Finished instantiating products. Chunks saved to {rescore_res_dir_w}.")

    if "rescore_products" in tasks:
        logging.info("Start rescoring pseudo products.")
        rescore_res_dir_w.mkdir(mode=0o755, parents=True, exist_ok=True)

        dask_client.dask_generate_conformers(
            input_dir=rescore_res_dir_w,
            output_dir=rescore_res_dir_w,
            product_file_suffix="product.oeb.gz",
            delete_input_file=True,
            omega_max_conf=config["product_omega_max_conf"],
            omega_max_time=config["product_omega_max_time"],
            omega_e_window=config["product_omega_energy"],
        )

        rescored_score_files = dask_client.dask_overlay_molecules(
            input_dir=rescore_res_dir_w,
            output_dir=rescore_res_dir_w,
            delete_conf_file=False,
            ref_mol=oechem.OEMol(query_mol),
            opt_shape_func=opt_shape_func,
            opt_color_func=opt_color_func,
            overlap_shape_func=overlap_shape_func,
            overlap_color_func=overlap_color_func,
            color_ff_dir=color_ff_dir,
            warts_separator=warts_id_sep,
            top_n=config["num_final_list"],
        )

        rescored_scores = dask_client.combine_files(
            file_list=rescored_score_files,
            output_dir=rescore_res_dir_w,
            top_m=config["num_final_list"],
            warts_separator=warts_id_sep,
            dedup_limit=max(1e6, 10 * config["num_final_list"]),
            file_name="rescore",
            keep_isomer_warts=True,
        )
        with rescore_file_w.open("wb") as f:
            pickle.dump(rescored_scores, f)
        logging.info(f"Finished rescoring pseudo products. File saved to {rescore_file_w}.")

    if "get_product_poses" in tasks:
        num_pose = config["num_write_pose"]
        scores = load_intermediate_files(rescore_file_r)[:num_pose]
        top_isomer_set = {extract_base_id(ele[1], warts_id_sep, True) for ele in scores}
        top_isomer_set_f = exp_dir / "top_isomer_set.pkl"
        with top_isomer_set_f.open("wb") as f:
            pickle.dump(top_isomer_set, f)

        overlay_pose_f = exp_dir / f"rescore_top_{num_pose}_mol_poses.sdf"

        logging.info(f"Picking molecules from {rescore_conformer_dir_r}.")
        overlay_files = dask_client.extract_conformers_and_score(
            conf_file_dir=rescore_conformer_dir_r,
            f_name_pattern="*.conf.oez",
            product_set_file=top_isomer_set_f,
            ref_mol=query_mol,
            opt_shape_func=opt_shape_func,
            opt_color_func=opt_color_func,
            overlap_shape_func=overlap_shape_func,
            overlap_color_func=overlap_color_func,
            color_ff_dir=color_ff_dir,
            rocs_key=0,
            out_dir=rescore_res_dir_w,
            sep_shape_color_score=config["sep_shape_color_score"],
        )

        # Concat all the overlay files and deduplicate:
        ostream = oechem.oemolostream(str(overlay_pose_f))
        oechem.OEWriteMolecule(ostream, query_mol)
        n_unique = 0
        seen_smi = set()
        for f in overlay_files:
            for mol in oechem.oemolistream(str(f)).GetOEMols():
                smi = oechem.OEMolToSmiles(mol)
                if smi not in seen_smi:
                    seen_smi.add(smi)
                    oechem.OEWriteMolecule(ostream, mol)
                    n_unique += 1
        ostream.close()
        [f.unlink() for f in overlay_files]
        top_isomer_set_f.unlink()

        logging.info(f"Top {n_unique} unique rescored molecule overlay poses saved to {overlay_pose_f}.")

    logging.info("Workflow finished")

    dask_client.close()

    if config["clean_up_dask_logs"] is True:
        dask_client.clean_up_logs()

    # compress all pkl result files
    if config["compress_pkl_files"]:
        compress_pkl_files(exp_dir)


if __name__ == "__main__":
    sys.exit(main())
