"""
Main script for running SASS queries.
"""

# Standard Library
import argparse
from glob import glob
import logging
import os
import pickle
import sys

# Third Party Library
from openeye import oechem
import ray
import yaml

# Genentech Library
from constants import SCRIPT_DIR
import mp_global_param
from utils_data import Synthon_handler
from utils_general import CustomOEErrorHandler, StreamToLogger, load_config, set_logger
from utils_mol import count_non_connector_heavy_atoms, fragment_molecule, load_first_mol
from utils_query import (
    merge_score_files,
    mp_flipper,
    mp_frag_rocs_score,
    mp_frag_select_synthon,
    mp_gen_library,
    mp_write_final_products,
    run_std_omega_rocs,
    sample_synthon_conformers,
    validate_tasks,
)

os.environ["OE_LICENSE"] = ...


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-name",
        "--exp_name",
        default=None,
        action="store",
        type=str,
        help="Base exp name and directory name to store results.",
    )

    parser.add_argument(
        "--output-parent-dir",
        "--output_parent_dir",
        default=None,
        help=(
            "Parent directory to store the results in. Individual experiment",
            " results wil be store under <output_parent_dir>/<exp_name>_<suffix>/",
        ),
    )

    parser.add_argument(
        "--ncpu",
        default=len(os.sched_getaffinity(0)),
        required=False,
        type=int,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SCRIPT_DIR, "config_template.yml"),
        help="Config file to use.",
    )

    parser.add_argument(
        "--gen_synthon_conf",
        "--gen-synthon-conf",
        action="store_true",
        help=(
            "If set, run conformation generation (OMEGA) on the synthons. "
            "Otherwise assuming that the synthon conformers are pre-generated. "
        ),
    )

    parser.add_argument("--map_id", "--map-id", action="store_true", help=("If set, maps synthon id to smiles."))

    parser.add_argument(
        "--head_node",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--reactions",
        nargs="*",
        type=str,
        default=None,
    )

    parser.add_argument("--omega_flag", action="store_true", help="If set, generate a flag for OMEGA job completion.")

    parser.add_argument(
        "--do_not_save_config",
        action="store_true",
    )

    return parser.parse_args()


def main():

    args = parse_args()
    config = load_config(args.config)

    tasks = config["tasks"]
    validate_tasks(tasks)

    query_method = "sass"

    n_fragments = config["n_fragments"]

    rank_method = config["non_frag_synthon_rank_method"]  # TODO deprecate
    top_m = config["top_m"]
    reactions = args.reactions if args.reactions else config["limit_reactions"]

    output_parent_dir = args.output_parent_dir if args.output_parent_dir else config["output_parent_dir"]
    exp_name = args.exp_name if args.exp_name else config["exp_name"]
    exp_dir = os.path.join(output_parent_dir, exp_name)
    os.makedirs(exp_dir, mode=0o755, exist_ok=True)
    os.chdir(exp_dir)

    full_exp_name = f"{query_method}_{top_m}"
    score_prefix = f"{query_method}_{rank_method}_synthon_scores"
    pseudo_res_prefix = f"{query_method}_{rank_method}_{top_m}_pseudo_res"

    if args.reactions:
        full_exp_name += "_"  # For large scale, parallelization over reaction level.
        full_exp_name += "_".join(args.reactions)  # Should only have one rxn_id.

    logger_path = os.path.join(exp_dir, f"{full_exp_name}.log")
    logger = set_logger(logger_path)
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    logging.info("-" * 32)
    logging.info(f"Start {full_exp_name}")
    ncpu = args.ncpu
    logging.info(f"cpu count: {ncpu}")

    # Configure OETK error handler
    error_handler = CustomOEErrorHandler(logger_path)
    oechem.OEThrow.SetHandlerImpl(error_handler, False)

    # Convert the list data in yaml to tuple keys.
    _sub_dict = config["atom_substitution_dict"]
    sub_dict = {}
    for src_symbol, src_isotope, dst_symbol, dst_isotope in _sub_dict:
        sub_dict[(src_symbol, src_isotope)] = (dst_symbol, dst_isotope)

    dummy_patterns = config["custom_color_ff"]
    syn_limit = config["synthon_limit"]
    if syn_limit is None:
        syn_limit = 100000

    custom_rocs_opts = mp_global_param.init(
        synthon_omega_max_conf=config["synthon_omega_max_conf"],
        synthon_omega_max_time=config["synthon_omega_max_time"],
        synthon_omega_energy=config["synthon_omega_energy"],
        product_omega_max_conf=config["product_omega_max_conf"],
        product_omega_max_time=config["product_omega_max_time"],
        product_omega_energy=config["product_omega_energy"],
        align_on_dummy_atom=config["align_on_dummy_atom"],
        color_weight=config["custom_color_weight"],
        color_radius=config["custom_color_radius"],
        interaction_type=config["color_interaction_type"],
        color_patterns=dummy_patterns,
        rocs_num_hits=syn_limit * 3,  # this can also be used to limit the score list length.
        scale_weight_per_atom=config["scale_weight_per_atom"],
        ray_head_node=args.head_node,
    )

    # Remaining config parameters.
    query_mol_file = config["query_molecule"]
    heavy_atom_limit = config["heavy_atom_limit"]
    enum_isomers = config["enum_isomers"]
    top_s_frac = config["top_s_frac"]
    conf_format = config["conf_file_format"]
    cross_score = config["cross_scoring"]
    connector_tag = "is_ring_connector"
    synthon_connectors = config["synthon_connectors"]
    frag_connectors = config["frag_connectors"]
    warts_id_sep = config["warts_id_sep"]
    sass_id_sep = config["sass_id_sep"]

    # Other common variables.
    instantiated_product_prefix = "product_isomer"

    # Save a copy of the config file.
    if not args.do_not_save_config:
        config["exp_name"] = exp_name
        with open(os.path.join(exp_dir, f"{full_exp_name}.yml"), "w") as f:
            yaml.dump(config, f, sort_keys=False)

    # Save copies of color FF.
    for key, custom_rocs_opt in custom_rocs_opts.items():
        custom_ff = custom_rocs_opt.GetOverlayOptions().GetColorOptions().GetColorForceField()
        custom_ff.Write(oechem.oeofstream(f"custom_color_ff_{key}.txt"))

    # Load synthon/reaction data.
    shandler = Synthon_handler(
        reaction_file=config["reaction_file"],
        synthon_file=config["synthon_file"],
        sub_dict=sub_dict,
    )

    if config["grouped_synthon_dir"] is not None:
        synthons_f = os.path.join(config["grouped_synthon_dir"], "grouped_synthons.pkl")
        with open(synthons_f, "rb") as f:
            grouped_synthons = pickle.load(f)
            logging.info(f"Grouped synthons loaded from {synthons_f}.")
    else:
        # Build the `grouped_synthons` on the fly, and limit to `syn_limit`
        grouped_synthons = shandler.group_synthon_by_rxn_id(
            n_components=n_fragments,
            rxn_ids=reactions,
            sub_dummy_atom=False,
            limit=syn_limit,
            random_seed=config["random_seed"],
            cleanup=False,
        )

        with open(os.path.join(exp_dir, "grouped_synthons.pkl"), "wb") as f:
            pickle.dump(grouped_synthons, f)
        logging.info("Grouped synthons saved.")

    if reactions is None:  # Reaction list for all reactions
        reactions = list(grouped_synthons.keys())

    # Build an synthon-id to SMILES mapping.
    if args.map_id or config["synthon_id_map_dir"] is None:
        logging.info("Start id-SMILES mapping.")

        id_map = {}
        ha_map = {}  # Possibly combine with `id_map`.
        for rxn_id in reactions:
            for s_idx, synthons in grouped_synthons[rxn_id].items():
                for sid, smi in synthons:
                    id_map[int(sid)] = (smi, s_idx)

                    # heavy atom count
                    mol = oechem.OEGraphMol()
                    oechem.OESmilesToMol(mol, smi)
                    # ha_count = oechem.OECount(mol, oechem.OEIsHeavy())
                    ha_count = count_non_connector_heavy_atoms(mol, synthon_connectors)
                    ha_map[int(sid)] = ha_count

        with open(os.path.join(exp_dir, "synthon_id_map.pkl"), "wb") as f:
            pickle.dump(id_map, f)
        logging.info("Id-SMILES mapping saved!")

        with open(os.path.join(exp_dir, "heavy_atom_map.pkl"), "wb") as f:
            pickle.dump(ha_map, f)
        logging.info("Heavy atom map saved!")

    else:
        id_map_dir = config["synthon_id_map_dir"]
        id_map_f = os.path.join(id_map_dir, "synthon_id_map.pkl")
        ha_map_f = os.path.join(exp_dir, "heavy_atom_map.pkl")

        logging.info(f"id mapping skipped. Loading id mapping from {id_map_f}.")
        with open(id_map_f, "rb") as f:
            id_map = pickle.load(f)

        logging.info(f"Loading heavy atom map from {ha_map_f}")
        with open(ha_map_f, "rb") as f:
            ha_map = pickle.load(f)

    # Main tasks.
    if tasks is None:
        return
    # Sample conformers.
    if config["synthon_conf_dir"] is not None:
        synthon_conf_dir = config["synthon_conf_dir"]
        logging.info((f"Synthon conformer generation skipped. Will load conformers from {synthon_conf_dir}."))
    else:
        synthon_conf_dir = exp_dir

        if args.gen_synthon_conf or "gen_synthon_conformers" in tasks:
            for rxn_id in reactions:
                sample_synthon_conformers(
                    rxn_id=rxn_id,
                    rxn_synthons=grouped_synthons[rxn_id],
                    ncpu=ncpu,
                    sub_dict=sub_dict,
                    ofile_format=conf_format,
                    connector_atoms=synthon_connectors,
                    smirks=shandler.get_reaction_smirks_by_id(rxn_id),
                    exp_dir=exp_dir,
                )

                if args.omega_flag:
                    with open(os.path.join(exp_dir, f"{rxn_id}_omega.flag"), "w") as f:
                        pass

    # Score synthons
    if "score_synthons" in tasks:
        assert os.path.isfile(query_mol_file)
        query_mol = load_first_mol(query_mol_file, mol_cleanup=True)
        logging.info(f"Query molecule loaded from {query_mol_file}.")

        query_fragments_dict = {}
        for n in n_fragments:
            query_fragments_dict[n] = fragment_molecule(
                mol=query_mol,
                n_fragments=n,
                heavy_atom_limit=heavy_atom_limit,
                cleave_acyclic_bonds=True,
                cleave_cyclic_bonds=True,
                connector_isotope=1,  # use non-0 isotope to signify a connector atom for the`check_fragment_size` fn.
                connector_atoms=frag_connectors,
                connector_tag=connector_tag,
            )

        logging.info("Start scoring synthons against query-fragments.")

        for rxn_id in reactions:
            n_component = shandler.get_number_of_components(rxn_id)
            smirks = shandler.get_reaction_smirks_by_id(rxn_id)
            # Take the 1st synthon SMILES from each reactant set. Should not use the SMARTS
            # from the reaction file, since OETK cannot parse "~" in SMARTS.
            synthon_smarts = {}
            for key, syns in grouped_synthons[rxn_id].items():
                _, smi = syns[0]
                synthon_smarts[key] = smi
            res = {}
            conf_chunk_dir = os.path.join(exp_dir, "conf_chunks")
            os.makedirs(conf_chunk_dir, exist_ok=True)
            rxn_synthons_lens = {}
            for key, val in grouped_synthons[rxn_id].items():
                rxn_synthons_lens[key] = len(val)

            frag_sets = query_fragments_dict[n_component]
            if len(frag_sets) > 0:
                logging.info(f"Start ROCS with {rxn_id} synthons. frag_set_count: {len(frag_sets)}")

                res = mp_frag_rocs_score(
                    frag_sets,
                    smirks,
                    synthon_smarts,
                    synthon_connectors,
                    frag_connectors,
                    connector_tag,
                    cross_score,
                    synthon_conf_dir,
                    rxn_id,
                    conf_format,
                    rxn_synthons_lens,
                    conf_chunk_dir,
                    exp_dir,
                    warts_id_sep,
                    ncpu,
                )

                with open(os.path.join(exp_dir, f"{score_prefix}_{rxn_id}.pkl"), "wb") as f:
                    pickle.dump(res, f)

            else:
                logging.info(f"ROCS scoring of {rxn_id} synthons skipped. No query fragments.")

            with open(os.path.join(exp_dir, f"{score_prefix}_{rxn_id}_scoring.flag"), "w") as f:
                pass

        logging.info("Finished synthon scoring.")

    # Pick top synthons, enumerate products, score.
    if "ground_truth" in tasks:
        # For ground truth calculation, "select" all synthons and combine.

        logging.info(f"Start {full_exp_name} full product enumeration.")
        ground_truth_prefix = "ground_truth_mol"
        n = 0
        i = 0
        ostream = oechem.oemolostream(f"{ground_truth_prefix}_chunk_{i}.oeb.gz")
        # 230811 Don't deduplicate. Should not have duplicate products.
        for rxn_id in reactions:
            smirks = shandler.get_reaction_smirks_by_id(rxn_id)
            reactants = grouped_synthons[rxn_id]
            logging.info(
                (
                    f"Reaction id: {rxn_id}; n-component: {len(reactants)}; "
                    f"synthon numbers: {[(key, len(val)) for key, val in reactants.items()]}"
                )
            )

            for prod in mp_gen_library(
                reactants, smirks=smirks, ncpu=ncpu, enum_isomers=False, title_separator=sass_id_sep
            ):
                # 231113 disabled `enum_isomers` due to out of memory.
                n += 1
                if n % 1000000 == 0:
                    print(f"Molecules enumerated: {n}")
                if n % 2000 == 0:
                    ostream.close()
                    i += 1
                    ostream = oechem.oemolostream(f"{ground_truth_prefix}_chunk_{i}.oeb.gz")
                oechem.OEWriteMolecule(ostream, prod)
        ostream.close()
        logging.info(
            (
                f"Finished full product enumeration. Total {n} isomers. "
                "Use separate workflow for cleanup, Flipper, OMEGA, ROCS."
            )
        )

        num_chunks = len(glob(f"{exp_dir}/{ground_truth_prefix}*"))

        if enum_isomers:
            isomer_prefix = f"{ground_truth_prefix}_isomer"
            for n in range(num_chunks):
                mp_flipper(
                    input_file=os.path.join(exp_dir, f"{ground_truth_prefix}_chunk_{n}.oeb.gz"),
                    output_file=os.path.join(exp_dir, f"{isomer_prefix}_chunk_{n}.oeb.gz"),
                    ncpu=ncpu,
                )
        else:
            isomer_prefix = ground_truth_prefix

        run_std_omega_rocs(
            exp_dir=exp_dir,
            n_chunks=num_chunks,
            query_file=query_mol_file,
            input_file_prefix=isomer_prefix,
            conf_file_format="oez",
            num_conf=config["product_omega_max_conf"],
            max_time=config["product_omega_max_time"],
            top_m=5000,
            warts_separator=warts_id_sep,
        )

    else:
        if "select_synthons" in tasks:

            logging.info(f"Start {full_exp_name} synthon selection.")

            # Compute aggregated ROCS scores for synthon combinations, then rank by rocs scores.
            logging.info(f"Start selecting top {top_m} synthon combinations.")

            synthon_score_dir = config["synthon_score_dir"]
            if synthon_score_dir is None:
                synthon_score_dir = exp_dir

            for rxn_id in reactions:
                score_file = os.path.join(synthon_score_dir, f"{score_prefix}_{rxn_id}.pkl")
                assert os.path.isfile(score_file)

                products = mp_frag_select_synthon(
                    score_file=score_file,
                    rxn_id=rxn_id,
                    top_m=top_m,
                    top_s_frac=top_s_frac,
                    ncpu=ncpu,
                    warts_separator=warts_id_sep,
                    sass_separator=sass_id_sep,
                    ha_map=ha_map,
                )

                with open(os.path.join(exp_dir, f"{pseudo_res_prefix}_{rxn_id}.pkl"), "wb") as f:
                    pickle.dump(products, f)
                logging.info(f"Wrote out pseudo_res {rxn_id}. Length: {len(products)}")

        if "combine_products" in tasks:
            combined_pseudo_res = os.path.join(exp_dir, f"{full_exp_name}_combined_pseudo_res.pkl")
            logging.info("Start combining pseudo products from individual reactions.")

            pseudo_res_files = [os.path.join(exp_dir, f"{pseudo_res_prefix}_{rxn_id}.pkl") for rxn_id in reactions]
            assert all(os.path.isfile(f) for f in pseudo_res_files), "Missing pseudo result files."

            products = merge_score_files(
                pseudo_res_files,
                top_n=top_m,
                warts_separator=warts_id_sep,
            )
            with open(combined_pseudo_res, "wb") as f:
                pickle.dump(products, f)
            logging.info(f"Finished combining pseudo products. File saved to {combined_pseudo_res}.")

        if "instantiate_products" in tasks:
            p_file = os.path.join(exp_dir, f"{full_exp_name}_combined_pseudo_res.pkl")
            with open(p_file, "rb") as f:
                products = pickle.load(f)
            logging.info(f"Combined pseudo products loaded from {p_file}.")
            mp_write_final_products(  # This has mol cleanup built-in.
                products=products,
                output_folder=exp_dir,
                outfile_prefix=instantiated_product_prefix,
                id_map=id_map,
                shandler=shandler,
                ncpu=ncpu,
                chunk_size=config["instantiate_chunk_size"],
                title_separator=sass_id_sep,
            )

            with open(os.path.join(exp_dir, "instantiate_products.flag"), "w"):
                pass

        if "rescore_products" in tasks:
            logging.info("Start rescoring selected products.")
            num_chunks = len(glob(f"{exp_dir}/{instantiated_product_prefix}*"))
            run_std_omega_rocs(
                exp_dir=exp_dir,
                n_chunks=num_chunks,
                query_file=query_mol_file,
                input_file_prefix=instantiated_product_prefix,
                conf_file_format="oez",
                num_conf=config["product_omega_max_conf"],
                max_time=config["product_omega_max_time"],
                top_m=top_m,
                warts_separator=warts_id_sep,
            )

    logging.info("Workflow finished")


if __name__ == "__main__":
    main()
