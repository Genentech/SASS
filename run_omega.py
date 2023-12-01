"""
Stand-alone mp omega (parallelization on each molecule level).

Use for full products only (not synthons), since this doesn't have the ring-synthon-
specific workflow (ring-completion, deletion).
"""

# Standard Library
import argparse
import logging
import os

# Third Party Library
from openeye import oechem

# Genentech Library
import mp_global_param
from utils_query import mp_get_conf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output_file",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--num_conf",
        type=int,
    )

    parser.add_argument(
        "--max_time",
        type=int,
    )

    return parser.parse_args()


def main():
    args = get_args()

    mp_global_param.init(
        product_omega_max_conf=int(args.num_conf),
        product_omega_max_time=int(args.max_time),
        align_on_dummy_atom=False,
    )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S"
    )
    logging.info(args.input_file)

    mp_get_conf(
        oechem.oemolistream(args.input_file).GetOEMols(),
        args.output_file,
        omega_opt_key="product",
        ncpu=len(os.sched_getaffinity(0)),
        sub_dict=None,
    )

    # Write out a flag to signify completion.
    with open(f"{args.output_file}.flag", "w") as f:
        pass


if __name__ == "__main__":
    main()
