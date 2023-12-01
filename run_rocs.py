"""
Stand-alone mp rocs (parallelization on each molecule level).
"""

# Standard Library
import argparse
import logging
import os
import pickle

# Third Party Library
from openeye import oechem, oeshape

# Genentech Library
import mp_global_param
from utils_mol import load_first_mol
from utils_query import sp_simple_rocs_scores


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--database",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output_file",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--best_hits",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--rocs_key",
        required=False,
        default=0,
        type=int,
    )

    parser.add_argument(
        "--color_ff_file",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "--query_cleanup",
        action="store_true",
    )

    return parser.parse_args()


def run_sp_rocs(
    query_mol: oechem.OEMol,
    dbase_file: str,
    output_file: str,
    rocs_key: int,
):

    scores = sp_simple_rocs_scores((query_mol, dbase_file, rocs_key))

    with open(output_file, "wb") as f:
        pickle.dump(scores, f)

    logging.info(f"ROCS finished. Score saved to {output_file}.")


def main():

    args = get_args()
    query_mol = load_first_mol(args.query, clear_title=False, mol_cleanup=args.query_cleanup)
    best_hits = args.best_hits
    output_file = args.output_file
    rocs_key = args.rocs_key
    color_ff_file = args.color_ff_file

    _ = mp_global_param.init(
        align_on_dummy_atom=False,
        rocs_num_hits=best_hits,
    )

    if rocs_key != 0:
        color_ff = oeshape.OEColorForceField()
        color_ff.Init(oechem.oeifstream(color_ff_file))
        color_opt = oeshape.OEColorOptions()
        color_opt.SetColorForceField(color_ff)
        mp_global_param.rocs_opts[rocs_key] = mp_global_param.rocs_opt_builder(color_opt, best_hits)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S"
    )
    logging.info(args.database)
    logging.info(args.query)

    run_sp_rocs(
        query_mol,
        args.database,
        output_file,
        rocs_key,
    )


if __name__ == "__main__":
    main()
