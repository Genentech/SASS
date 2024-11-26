"""dask-based parallelization.

Scoring, combining, etc. tasks may be coupled in a single function for efficiency reasons.
"""

import logging
import math
import os
import pickle
import random
import shutil
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from dask.distributed import Client, LocalCluster, as_completed
from dask_jobqueue import SLURMCluster
from openeye import oechem

from sass.utils.utils_confgen import sample_general_mixed_synthon_conf, write_conformers
from sass.utils.utils_data import SynthonHandler
from sass.utils.utils_general import extract_base_id, wait
from sass.utils.utils_libgen import write_library_products_chunks, write_singleton_products_chunks
from sass.utils.utils_mol import get_conn_symbols, label_synthon_ring_connector
from sass.utils.utils_overlay import sp_simple_oeoverlay
from sass.utils.utils_query import (
    aggregate_scores,
    allocate_synthons,
    deduplicate_scores,
    determine_rxn_id,
    merge_score_files,
    order_and_substitute_fragments,
    simple_average,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S"
)


class DaskClient:
    """Run SASS tasks using dask for parallelization.

    For synthon scoring, parallelization is done on rxn_fset_forder level. Each worker
    scores all reaction component synthons in all forders. Each rxn_fset_forder score
    are combined and written to file, and the score files are merged in the end (also
    parallelized).
    For rescoring, each worker processes a chunk of the combined pseudo results. The
    process includes instantiation, conformer generation, and ROCS scoring. # TODO: write out file or not?
    For final overlay pose writing, each worker extracts the matching conformers from a
    conformer file generated during rescoring, and the head node combines all matches
    in memory (due to relatively small size of the final pose writing list).
    """

    def __init__(
        self,
        min_workers: int,
        max_workers: int,
        qos: str,
        cluster_type: str,
        log_dir: Path | str,
        memory: int = 8,
        scheduler_port: int | None = None,
        queue_time_map: dict | None = None,
        parition: str = "defq",
    ) -> None:
        """Initialize Dask client.

        Parameters
        ----------
        min_workers
            Minimum number of workers to dyanmically scale to.
        max_workers
            Maximum number of workers to dyanmically scale to.
        qos
            Quality of Service (QoS) to use for dask workers.
        cluster_type
            Type of cluster to use, either "slurm" or "local".
        log_dir
            Directory to write dask logs to.
        memory
            Memory (in GB) per dask worker.
        scheduler_port
            Port for dask scheduler. Will be randomly assigned if not provided.
        queue_time_map
            Map of queue names to walltime and lifetime.
        parition
            Queue/partition to use for dask workers.
        """
        self._dask_temporary_directory = (
            Path(os.getenv("TMPDIR", "/tmp")) / f"dask-worker-space-{os.getpid()}"  # noqa: S108
        )
        self.dask_log_dir = log_dir

        if scheduler_port is None:
            scheduler_port = random.randint(49152, 65535)  # noqa: S311

        if queue_time_map is None:
            queue_time_map = {
                "veryshort": {"walltime": "0:10:00", "lifetime": "9m", "lifetime-stagger": "1m"},
                "short": {"walltime": "2:00:00", "lifetime": "115m", "lifetime-stagger": "4m"},
                "medium": {"walltime": "1-0", "lifetime": "1435m", "lifetime-stagger": "4m"},
                "long": {"walltime": "3-0", "lifetime": "4315m", "lifetime-stagger": "4m"},
            }

        if cluster_type == "slurm":
            _cluster_args = {
                "scheduler_options": {"port": scheduler_port, "dashboard_address": None},
                "cores": 1,
                "processes": 1,
                "job_cpu": 1,
                "log_directory": str(self.dask_log_dir),
                "local_directory": self._dask_temporary_directory,
                "memory": f"{memory}GB",
                "walltime": queue_time_map[qos]["walltime"],
                "queue": parition,
                "worker_extra_args": [
                    f"--lifetime={queue_time_map[qos]['lifetime']}",
                    f"--lifetime-stagger={queue_time_map[qos]['lifetime-stagger']}",
                ],
                "job_extra_directives": [f"--qos={qos}"],
            }

            self._cluster = SLURMCluster(**_cluster_args, asynchronous=False)
            self._cluster.adapt(
                minimum=min_workers,
                maximum=max_workers,
                wait_count=10,
                target_duration="1s",
            )
            self._dask_client = Client(self._cluster, heartbeat_interval="10s")
            self._dask_client.wait_for_workers(min(2, min_workers), timeout=300)
            self.n_workers = max_workers  # TODO: investigate if can set min/max worker differently!
            time.sleep(2)
            num_workers = len(self._dask_client.scheduler_info()["workers"])
            logging.info(f"Started dask cluster with {num_workers} workers. Target max workers: {max_workers}")

        if cluster_type == "local":
            self._cluster = LocalCluster(
                processes=True,
                local_directory=self._dask_temporary_directory,
                scheduler_port=0,
                dashboard_address=None,
            )
            self._cluster.scale(len(os.sched_getaffinity(0)))
            self._dask_client = Client(self._cluster)
            self._dask_client.wait_for_workers(1, timeout=30)
            time.sleep(2)
            scheduler_info = self._dask_client.scheduler_info()
            self.n_workers = len(scheduler_info["workers"])

    def close(self):
        time.sleep(3)
        self._dask_client.close()
        time.sleep(5)
        self._dask_client = None
        if self._cluster is not None:
            self._cluster.close()
            self._cluster = None
        if Path(self._dask_temporary_directory).exists():
            shutil.rmtree(self._dask_temporary_directory)

    def clean_up_logs(self):
        shutil.rmtree(self.dask_log_dir)

    def stream_map(
        self,
        func: callable,
        data_iterator: Iterable[Any],
        **kwargs: Any,
    ):
        """Submit jobs to dask workers based a stream of data. Yields results as they come in.

        Parameters
        ----------
        func
            Function to be executed.
        data_iterator
            Iterable of data to be passed to the function.
        kwargs
            Additional keyword arguments to be passed to the function.

        Yields
        ------
            Results from the function.
        """
        futures = [self._dask_client.submit(func, data, **kwargs) for data in data_iterator]
        for future in as_completed(futures):
            yield future.result()

    def dask_generate_synthon_conformers(
        self,
        reactions: list[str],
        grouped_synthons: dict,
        synthon_handler: SynthonHandler,
        ofile_format: str,
        connector_atoms: list[str],
        chunk_size: int,
        output_dir: Path,
        **kwargs: Any,
    ):
        """Generate synthon conformers and write to file.

        Parameters
        ----------
        reactions
            Reaction ids to be included for synthon conformer generation.
        grouped_synthons
            Synthons organized by reactions.
        synthon_handler
            Synthon handler containing synthon and reaction information.
        ofile_format
            Format of the conformer files, e.g. oez.
        connector_atoms
            Special atoms denoting a connector atom on synthons.
        chunk_size
            Max size of synthons to be processed by one worker.
        output_dir
            Directory to write conformer files to.
        kwargs
            OMEGA parameters.
        """
        task_gen = generate_synthon_confgen_tasks(
            reactions=reactions,
            grouped_synthons=grouped_synthons,
            synthon_handler=synthon_handler,
            connector_atoms=connector_atoms,
            chunk_size=chunk_size,
        )

        for n, _ in enumerate(
            self.stream_map(
                synthon_confgen_wrapper,
                task_gen,
                conf_format=ofile_format,
                output_dir=output_dir,
                **kwargs,
            )
        ):
            if (n + 1) % self.n_workers == 0:
                logging.info(f"Finished {n + 1} synthon conformer generation tasks.")

    # Technically, these don't need to be class methods of `DaskClient`. They are here just for convenience.
    def dask_score_and_combine_synthon(
        self,
        reactions: list[str],
        shandler: SynthonHandler,
        query_fragments_dict: dict[int : list[list[oechem.OEMolBase]]],
        grouped_synthons: dict,
        synthon_conf_dir: Path,
        conf_format: str,
        align_connector_atom: bool,
        synthon_connectors: list[str],
        rocs_key_opt: int,
        rocs_key_score: int,
        opt_shape_func: str,
        opt_color_func: str | None,
        overlap_shape_func: str,
        overlap_color_func: str | None,
        top_m: int,
        dedup_limit: int,
        warts_separator: str,
        sass_separator: str,
        agg_method: str,
        synthon_comb_limit: int,
        frag_connectors: list[str],
        cross_score: bool,
        color_ff_dir: Path,
        pseudo_res_dir: Path,
    ):
        """Score synthons against query fragments and combine synthons to generate pseudo scores.

        Pseudo scores are organized by number of reaction components to allow for
        merging results from different component reactions using a user-defined ratio.

        Parameters
        ----------
        reactions
            Reaction ids to be used for generating scoring tasks.
        shandler
            Synthon handler containing synthon and reaction information.
        query_fragments_dict
            Query fragments, organized by number of fragments.
        grouped_synthons
            Synthons organized by reactions.
        synthon_conf_dir
            Directory containing synthon conformers. Usually pre-generated.
        conf_format
            Format of the conformer files, e.g. oez, or oeb.
        align_connector_atom
            Whether to apply custom interactions on special connector atoms.
        synthon_connectors
            All special atoms denoting a connector atom on sythons.
        rocs_key_opt
            Key for ROCS option for overlay optimization.
        rocs_key_score
            Key for ROCS option for overlap scoring.
        **_shape/color_func:
            Shape and color functions for overlay optimization and overlap scoring.
        top_m
            Number of top pseudo products to output.
        dedup_limit
            Limit above which a score list will deduplicate and retain the top_m items.
            This is to prevent memory overflow with very large score lists.
        warts_separator
            String separator for naming isomers, conformers.
        sass_separator
            String separator for joining synthon ids to form product ids.
        agg_method
            Method for aggregating synthon scores to calculate the pseudo product scores.
        synthon_comb_limit
            Limit of number of combinations for each worker. Without a limit, all synthon
            combinations are considered, which increases the time complexity of SASS back
            to the brute-force method.
        frag_connectors
            All special atoms denoting a connector atom on query fragments.
        cross_score
            See docstring of utils_query.order_and_substitute_fragments.
        color_ff_dir
            Directory containing custom color interaction files.
        pseudo_res_dir
            Directory to write pseudo scores to.

        Returns
        -------
            A dict of pseudo score files, organized by number of components.
        """
        task_gen = generate_synthon_scoring_tasks(
            reactions,
            shandler,
            query_fragments_dict,
            grouped_synthons,
            synthon_connectors,
            frag_connectors,
            cross_score,
        )

        all_res = defaultdict(list)
        prod_count = defaultdict(int)
        n = 0
        unique_rxn = set()
        for dask_res in self.stream_map(
            score_and_combine_synthon,
            task_gen,
            synthon_conf_dir=synthon_conf_dir,
            conf_format=conf_format,
            align_connector_atom=align_connector_atom,
            synthon_connectors=synthon_connectors,
            rocs_key_opt=rocs_key_opt,
            rocs_key_score=rocs_key_score,
            opt_shape_func=opt_shape_func,
            opt_color_func=opt_color_func,
            overlap_shape_func=overlap_shape_func,
            overlap_color_func=overlap_color_func,
            top_m=top_m,
            dedup_limit=dedup_limit,
            warts_separator=warts_separator,
            sass_separator=sass_separator,
            synthon_comb_limit=synthon_comb_limit,
            agg_method=agg_method,
            color_ff_dir=color_ff_dir,
            pseudo_res_dir=pseudo_res_dir,
        ):
            if dask_res:  # Need to guard against None returns. Exact reason need to look up on dask.
                n += 1
                if n % self.n_workers == 0:
                    logging.info(f"Finished scoring {n} tasks.")
                rxn_id, fname, count = dask_res
                n_component = shandler.get_number_of_components(rxn_id)
                all_res[n_component].append(fname)
                if rxn_id not in unique_rxn:
                    unique_rxn.add(rxn_id)
                    prod_count[n_component] += count  # Only count once per rxn_id.

        logging.info("TOTAL of %s rxn_fset_forder res.", n)
        time.sleep(5)
        len_pseudo = {key: len(val) for key, val in all_res.items()}
        logging.info("Number of pseudo_res file for each component: %s.", len_pseudo)
        logging.info(
            "Total number of possible products for each component: %s.",
            dict(prod_count),
        )
        return all_res, prod_count

    def combine_pseudo_res(
        self,
        pseudo_res: dict,
        pseudo_res_dir: Path,
        top_m: int,
        warts_separator: str,
        use_proportion: bool,
        product_count: dict[int, int],
        dedup_limit: int,
    ) -> list:
        """Combine pseudo product scores from different component reactions.

        Parameters
        ----------
        pseudo_res
            Dict of lists of pseudo res files organized by number of components.
        pseudo_res_dir
            Directory containing the pseudo res files, and to write the merged score
            file to.
        top_m
            Number of top pseudo products to output.
        warts_separator
            String separator for naming isomers, conformers.
        use_proportion
            Whether to use the proportion of total products by each number of reaction
            components to select the final product list. E.g. if total products of 2 vs
            3 component reaction is 30:70, pick the top 0.3 x `top_m` from all 2 component
            reaction products nad 0.7 x `top_m` from all 3 component reaction products.
        product_count
            Possible number of products grouped by number of components.
        dedup_limit
            Limit above which a score list will deduplicate and retain the top_m items.
            This is to prevent memory overflow with very large score lists.

        Returns
        -------
            Combined pseudo products, sorted by descending scores.
        """
        res = {}
        for n_comp, files in pseudo_res.items():
            res[n_comp] = self.combine_files(
                file_list=files,
                output_dir=pseudo_res_dir,
                top_m=top_m,
                warts_separator=warts_separator,
                dedup_limit=dedup_limit,
                file_name=f"pseudo_res_{n_comp}_comp",
                keep_isomer_warts=False,
            )

        # Mixing different n-component results by ratio.
        if not use_proportion:
            mixing_ratio = {key: 1 for key in res}
        else:
            # Determine the ratio from product counts
            total_count = sum(product_count.values())
            mixing_ratio = {key: product_count[key] / total_count for key in product_count}
        logging.info("Ratio for mixing various component reaction pseudo results: %s.", mixing_ratio)

        combined_res = []
        for key, _res in res.items():
            combined_res.extend(_res[: int(top_m * mixing_ratio[key])])

        out = deduplicate_scores(combined_res, warts_separator=warts_separator, sort_scores=True, limit=top_m)

        return out

    def combine_files(  # TODO: Investigate why this doesn't work with dask 2024.5.1 when min-worker != max.
        self,
        file_list: list[Path],
        output_dir: Path,
        top_m: int,
        warts_separator: str,
        dedup_limit: int,
        file_name: str,
        keep_isomer_warts: bool,
    ) -> list:
        """Combine a list of score files using multiple dask workers.

        Parameters
        ----------
        file_list
            List of files to be combined.
        output_dir
            Directory to write the results to.
        top_m
            Number of top scores to output.
        warts_separator
            String separator for naming isomers, conformers.
        dedup_limit
            Limit above which a score list will deduplicate and retain the `top_m` items.
        file_name
            Base name for the combined file.
        keep_isomer_warts
            Whether to keep isomer warts in the product ids.

        Returns
        -------
            Combined pseudo products, sorted by descending scores.
        """
        task_gen = generate_combine_file_tasks(
            initial_list=file_list,
            output_dir=output_dir,
            file_name=file_name,
        )

        for _ in self.stream_map(
            merge_score_files_wrapper,
            task_gen,
            top_n=top_m,
            warts_separator=warts_separator,
            keep_isomer_warts=keep_isomer_warts,
            is_binary=True,
            score_size_limit=dedup_limit,
        ):
            pass

        with (output_dir / f"combined_{file_name}.pkl").open("rb") as f:
            combined_res = pickle.load(f)

        time.sleep(5)

        return combined_res

    def dask_instantiate_library_products(
        self,
        synthon_handler: SynthonHandler,
        grouped_synthons: dict,
        reaction_ids: list[str],
        output_dir: Path,
        product_chunk_size: int,  # at least 1000.
        out_file_suffix: str,
        sort_title: bool,
        title_separator: str,
        max_stereocenter: int,
    ) -> None:
        """Instantiate library products and write to file."""
        task_gen = generate_library_instantiation_tasks(
            grouped_synthons=grouped_synthons,
            synthon_handler=synthon_handler,
            reaction_ids=reaction_ids,
            chunk_size=product_chunk_size,
        )

        n = 0
        total_mol, total_iso = 0, 0
        for n_mol, n_iso in self.stream_map(
            write_library_products_wrapper,
            task_gen,
            output_dir=output_dir,
            out_file_suffix=out_file_suffix,
            sort_title=sort_title,
            title_separator=title_separator,
            max_stereocenter=max_stereocenter,
            chunk_size=product_chunk_size * 3,  # Assuming ~3 isomers per molecule.
        ):
            n += 1  # noqa: SIM113
            if n % self.n_workers == 0:
                logging.info(f"Finished {n} library instantiation tasks.")
            total_mol += n_mol
            total_iso += n_iso
        logging.info("Total of %s molecules and %s isomers written.", total_mol, total_iso)

    def dask_instantiate_singleton_products(
        self,
        product_list: list,
        instantiation_chunk_size: int,
        file_chunk_size: int,
        shandler: SynthonHandler,
        synthon_data: dict,
        warts_separator: str,
        title_separator: str,
        output_dir: Path,
        product_file_suffix: str,
        max_stereocenter: int,
    ) -> None:
        """Instantiate selected products based on synthon-scores.

        Parameters
        ----------
        product_list
            List of product ids to instantiate.
        instantiation_chunk_size
            Chunk size for each dask worker to process.
        file_chunk_size
            Target chunk size for actual file writing.
        shandler
            Synthon handler containing synthon and reaction information.
        synthon_data
            Grouped synthons organized by reactions.
        warts_separator
            String separator for naming isomers, conformers.
        title_separator
            String separator for joining synthon ids to form product ids.
        output_dir
            Directory to write product files to.
        product_file_suffix
            Suffix for product files.
        max_stereocenter
            Maximum number of stereocenters to consider when enumerating isomers.
        """
        task_gen = generate_instantiation_tasks(
            product_list=product_list,
            chunk_size=instantiation_chunk_size,
            shandler=shandler,
            synthon_data=synthon_data,
            warts_separator=warts_separator,
            title_separator=title_separator,
        )

        for _ in self.stream_map(
            write_singleton_products_wrapper,
            task_gen,
            output_dir=output_dir,
            out_file_suffix=product_file_suffix,
            title_separator=title_separator,
            file_chunk_size=file_chunk_size,
            max_stereocenter=max_stereocenter,
        ):
            pass

    def dask_generate_conformers(
        self,
        input_dir: Path,
        output_dir: Path,
        product_file_suffix: str,
        delete_input_file: bool,
        **kwargs: Any,  # OMEGA parameters
    ) -> None:
        task_gen = input_dir.glob(f"*{product_file_suffix}")

        for n, _ in enumerate(
            self.stream_map(
                write_conformers_wrapper,
                task_gen,
                output_dir=output_dir,
                delete_input_file=delete_input_file,
                **kwargs,
            )
        ):
            if (n + 1) % self.n_workers == 0:
                logging.info(f"Finished {n + 1} conformer generation tasks.")

    def dask_overlay_molecules(
        self,
        input_dir: Path,
        output_dir: Path,
        delete_conf_file: bool,
        conformer_suffix: str = "conf.oez",
        **kwargs: Any,  # Parameters for overlay
    ) -> list:
        task_gen = input_dir.glob(f"*{conformer_suffix}")

        score_files = []
        n = 0
        for res in self.stream_map(
            overlay_molecules_wrapper,
            task_gen,
            delete_conf_file=delete_conf_file,
            output_dir=output_dir,
            **kwargs,
        ):
            if res:
                n += 1
                if n % self.n_workers == 0:
                    logging.info(f"Finished {n} overlay tasks.")
                score_files.append(res)  # `res` here is a file name.
        return score_files

    def extract_conformers_and_score(
        self,
        conf_file_dir: Path,
        f_name_pattern: str,
        product_set_file: Path,
        ref_mol: oechem.OEMol | oechem.OEGraphMol,
        opt_shape_func: str,
        opt_color_func: str | None,
        overlap_shape_func: str,
        overlap_color_func: str | None,
        color_ff_dir: Path,
        rocs_key: int,
        out_dir: Path,
        sep_shape_color_score: bool,
    ) -> list[Path]:
        """Extract desired conformers from conformer files, and score those conformers.

        After product rescoring, select the top products for writing out poses. Extract
        the conformers of those products for ROCS scoring/writing pose.
        This avoids re-instantiation and re-generating conformers of those selected top
        products.

        Parameters
        ----------
        conf_file_dir
            Directory containing conformer files.
        f_name_pattern
            File name pattern to conformer files to search by.
        product_set_file
            File containing the set of product ids to extract from all conformer files.
        ref_mol
            Reference molecule for overlay scoring.
        **_shape/color_func:
            Shape and color functions for overlay optimization and overlap scoring.
        color_ff_dir
            Directory containing custom color interaction files.
        rocs_key
            Key for ROCS option for overlay optimization.
        out_dir
            Directory to write out overlay poses.
        sep_shape_color_score
            Whether to write out separate shape and color scores in the output.

        Returns
        -------
            A list of best overlaid conformer file paths.
        """
        conf_files = conf_file_dir.glob(f_name_pattern)  # .glob return is a generator

        out = []
        n = 0
        for res in self.stream_map(
            extract_mols_and_score,
            conf_files,
            product_set_file=product_set_file,
            ref_mol=ref_mol,
            opt_shape_func=opt_shape_func,
            opt_color_func=opt_color_func,
            overlap_shape_func=overlap_shape_func,
            overlap_color_func=overlap_color_func,
            color_ff_dir=color_ff_dir,
            rocs_key=rocs_key,
            out_dir=out_dir,
            sep_shape_color_score=sep_shape_color_score,
        ):
            if res:
                n += 1
                if n % self.n_workers == 0:
                    logging.info("Finished %s tasks.", n)
                out.append(res)
        return out


# Functions for dask tasks
def extract_mols_and_score(
    data: list[Path],  # list[conf_file]
    product_set_file: Path,
    ref_mol: oechem.OEMol | oechem.OEGraphMol,
    opt_shape_func: str,
    opt_color_func: str | None,
    overlap_shape_func: str,
    overlap_color_func: str | None,
    color_ff_dir: Path,
    rocs_key: int,
    out_dir: Path,
    sep_shape_color_score: bool,
) -> Path:
    conf_file = data
    product_set = pickle.load(product_set_file.open("rb"))
    matched_mols = [
        oechem.OEMol(mol) for mol in oechem.oemolistream(str(conf_file)).GetOEMols() if mol.GetTitle() in product_set
    ]
    conf_file.unlink()

    if not matched_mols:
        return None

    # Score and write out overlay poses.
    out_file = out_dir / f"{conf_file.name}.sdf"
    _ = sp_simple_oeoverlay(
        ref_mol=oechem.OEMol(ref_mol),
        conf_file=None,
        overlay_opts=None,
        fit_mols=matched_mols,
        rocs_key_opt=rocs_key,
        rocs_key_score=rocs_key,
        opt_shape_func=opt_shape_func,
        opt_color_func=opt_color_func,
        overlap_shape_func=overlap_shape_func,
        overlap_color_func=overlap_color_func,
        color_ff_dir=color_ff_dir,
        num_hits=None,
        write_out_overlay=True,
        out_file=out_file,
        keep_color=False,
        sep_shape_color_score=sep_shape_color_score,
    )

    return out_file


def write_library_products_wrapper(
    data: Any,
    output_dir: Path,
    out_file_suffix: str,
    sort_title: bool,
    title_separator: str,
    max_stereocenter: int,
    chunk_size: int,
):
    reactants, smirks, idx = data
    out_file = output_dir / f"chunk_{idx}.{out_file_suffix}"
    return write_library_products_chunks(
        out_file=out_file,
        max_stereocenter=max_stereocenter,
        chunk_size=chunk_size,
        reactants=reactants,
        smirks=smirks,
        sort_title=sort_title,
        title_separator=title_separator,
    )


def write_singleton_products_wrapper(
    data: Any,
    output_dir: Path,
    out_file_suffix: str,
    title_separator: str,
    file_chunk_size: int,
    max_stereocenter: int,
):
    """Instantiate molecules using data passed from dask head node."""
    pid_data, idx = data

    logging.info("Starting product instantiation.")
    write_singleton_products_chunks(
        pid_data,
        out_file_stem=f"chunk_{idx}",
        out_file_dir=output_dir,
        out_file_suffix=out_file_suffix,
        title_separator=title_separator,
        max_stereocenter=max_stereocenter,
        split_chunk_size=file_chunk_size,
    )


def write_conformers_wrapper(
    data: Any,
    output_dir: Path,
    delete_input_file: bool,
    **kwargs: Any,  # OMEGA parameters
):
    logging.info("Starting conformer generation.")
    mol_file = data
    base_name = mol_file.name.split(".")[0]
    out_file = output_dir / f"{base_name}.conf.oez"
    write_conformers(
        in_file=mol_file,
        out_file=out_file,
        **kwargs,
    )
    if delete_input_file:
        mol_file.unlink()


def overlay_molecules_wrapper(
    data: Any,
    delete_conf_file: bool,
    output_dir: Path,
    ref_mol: oechem.OEMolBase,
    opt_shape_func: str,
    opt_color_func: str | None,
    overlap_shape_func: str,
    overlap_color_func: str | None,
    color_ff_dir: Path,
    warts_separator: str,
    top_n: int,
):
    conformer_file = data
    score = sp_simple_oeoverlay(
        ref_mol=ref_mol,
        conf_file=conformer_file,
        overlay_opts=None,
        rocs_key_opt=0,
        rocs_key_score=0,
        opt_shape_func=opt_shape_func,
        opt_color_func=opt_color_func,
        overlap_shape_func=overlap_shape_func,
        overlap_color_func=overlap_color_func,
        color_ff_dir=color_ff_dir,
    )

    score = deduplicate_scores(
        score, sort_scores=True, keep_isomer_warts=True, warts_separator=warts_separator, limit=top_n
    )

    base_name = conformer_file.name.split(".")[0]
    score_file = output_dir / f"{base_name}.score.pkl"
    with score_file.open("wb") as f:
        pickle.dump(score, f)
    logging.info("rocs finished.")

    if delete_conf_file:
        conformer_file.unlink()

    return score_file


def merge_score_files_wrapper(data: tuple, **kwargs: Any) -> tuple:
    files_to_merge, merged_name = data
    merged_f_flags = [Path(f"{f}.flag") for f in files_to_merge if f.name.startswith("merged")]
    # only need to wait for the flag of "merged" files
    wait(merged_f_flags, wait_time=1, final_delay=1)

    scores = merge_score_files(files_to_merge, **kwargs)
    with merged_name.open("wb") as f:
        pickle.dump(scores, f)

    # Write a fast flag to avoid reading from unfinished pkl files.
    with Path(f"{merged_name}.flag").open("w"):
        pass

    for f in files_to_merge:
        f.unlink()
    for f in merged_f_flags:  # this is a list, does not concat to `files_to_merge`
        f.unlink()
    return None


def score_and_combine_synthon(
    data: tuple[Any],
    synthon_conf_dir: Path,
    conf_format: str,
    align_connector_atom: bool,
    synthon_connectors: list[str],
    rocs_key_opt: int,
    rocs_key_score: int,
    opt_shape_func: str,
    opt_color_func: str | None,
    overlap_shape_func: str,
    overlap_color_func: str | None,
    top_m: int,
    dedup_limit: int,
    warts_separator: str,
    sass_separator: str,
    synthon_comb_limit: int,
    agg_method: str,
    color_ff_dir: Path,
    pseudo_res_dir: Path,
) -> tuple:

    (
        rxn_id,
        fset_idx,
        forder_idx,
        synthon_order,
        ordered_f,
    ) = data

    scores = score_forder_synthon(
        synthon_order,
        ordered_f,
        synthon_conf_dir,
        conf_format,
        rxn_id,
        align_connector_atom,
        synthon_connectors,
        rocs_key_opt,
        rocs_key_score,
        opt_shape_func=opt_shape_func,
        opt_color_func=opt_color_func,
        overlap_shape_func=overlap_shape_func,
        overlap_color_func=overlap_color_func,
        warts_separator=warts_separator,
        color_ff_dir=color_ff_dir,
    )
    scores, count = combine_forder_scores(
        scores,
        top_m,
        dedup_limit,
        warts_separator,
        sass_separator,
        synthon_comb_limit,
        agg_method,
    )

    fname = pseudo_res_dir / f"{rxn_id}_{fset_idx}_{forder_idx}_pseudo_res.pkl"
    with fname.open("wb") as f:
        pickle.dump(scores, f)

    return rxn_id, fname, count


def synthon_confgen_wrapper(
    data: tuple,
    output_dir: Path,
    conf_format: str,
    **kwargs: Any,  # OMEGA parameters
):
    (rxn_id, smirks, s_idx, synthons, other_synthons, synthon_ring_connector_dict, chunk_idx) = data
    out_file = output_dir / f"{rxn_id}_synthon_{s_idx}_conf.chunk{chunk_idx}.{conf_format}"
    sample_general_mixed_synthon_conf(
        smirks=smirks,
        synthon_ring_connector_dict=synthon_ring_connector_dict,
        current_synthons=synthons,
        current_synthon_idx=s_idx,
        other_synthons=other_synthons,
        output_file=out_file,
        **kwargs,
    )
    logging.info(
        f"Finished generating conformers for {rxn_id} synthon {s_idx}-th reactants, chunk size: {len(synthons)}"
    )


# Task generators for dask
def generate_library_instantiation_tasks(
    grouped_synthons: dict[str, dict],
    synthon_handler: SynthonHandler,
    reaction_ids: list[str],
    chunk_size: int,
):
    """Generate tasks for instantiating library products.

    Tasks are parallelized on the "1st dimension" of the synthon lists. i.e. for a
    reaction with n x m x k synthons, the n synthons are split such that each chunk
    contains ~`chunk_size` products.

    Parameters
    ----------
    synthon_handler
        SynthonHandler object.
    reaction_ids
        List of reaction ids.
    chunk_size
        Desired number of products to instantiate per chunk. Actual number will vary.
    """
    n = 0
    random.shuffle(reaction_ids)
    for rxn_id in reaction_ids:
        smirks = synthon_handler.get_reaction_smirks_by_id(rxn_id)
        reactants = grouped_synthons[rxn_id]
        num_products = math.prod([len(val) for val in reactants.values()])
        num_chunks = min(len(reactants[0]), math.ceil(num_products / chunk_size))
        this_chunk_size = math.ceil(len(reactants[0]) / num_chunks)
        other_reactant_sizes = [len(val) for key, val in reactants.items() if key != 0]
        other_size_text = "".join([f" x {val}" for val in other_reactant_sizes])
        logging.info(f"Splitting {rxn_id} into ~{num_chunks} chunks of size {this_chunk_size}{other_size_text}.")

        chunk = {key: val for key, val in reactants.items() if key != 0}
        for i in range(0, len(reactants[0]), this_chunk_size):
            chunk[0] = reactants[0][i : i + this_chunk_size]
            yield (chunk, smirks, n)
            n += 1
    logging.info(f"Instantiating a total of {n} chunks.")


def generate_instantiation_tasks(
    product_list: list,
    chunk_size: int,
    shandler: SynthonHandler,
    synthon_data: dict,
    warts_separator: str,
    title_separator: str,
):
    # Load shandler and synthon_data on each dask worker, avoid feeding those data through head node.
    logging.info(f"Splitting pseudo-product list into {chunk_size}/chunk for instantiation.")
    chunk = []
    n = 0
    for data in product_list:
        pid = data[1]
        pid = extract_base_id(pid, warts_separator=warts_separator, keep_isomer_warts=False)
        sids = pid.split(title_separator)
        reactants = []
        rids = []
        for sid in sids:
            this_data = synthon_data[sid]
            smi, sorder, rxn_id = this_data["SMILES"], this_data["s_idx"], this_data["rxn_id"]
            reactants.append((sid, smi, sorder))
            rids.append(rxn_id)
        rxn_id = determine_rxn_id(rids)
        this_smirks = shandler.get_reaction_smirks_by_id(rxn_id)
        chunk.append((reactants, this_smirks))
        if len(chunk) >= chunk_size:
            if n % 1000 == 0:
                logging.info(f"Start instantiating chunk {n}")
            yield (chunk, n)
            chunk = []
            n += 1
    if chunk:
        logging.info(f"Start instantiating chunk {n}")
        yield (chunk, n)
        chunk = []
        n += 1
    logging.info("Instantiating a total of %s chunks.", n)


def generate_combine_file_tasks(
    initial_list: list[Path],
    output_dir: Path,
    file_name: str,
):
    """Generate tasks for merging score files.

    Files are merged in pairs, and the merged file is appended to the queue. Total number
    of merge operation is 2 * n, but it allows massive parallelization.

    Parameters
    ----------
    initial_list
        Initial list of files to be merged.
    output_dir
        Directory to write the merged files to.
    file_name
        Base name for the combined file.

    Yields
    ------
        Tuple of two files to be merged, and the name of the merged file.
    """
    q = deque(initial_list)
    n = 0
    while len(q) > 1:
        if len(q) > 2:
            merged_fname = output_dir / f"merged_{file_name}_{n}.pkl"
        else:
            merged_fname = output_dir / f"combined_{file_name}.pkl"
        # logging.info(f'{str(q[0])} + {str(q[1])} --> {str(merged_fname)}')
        yield ((q[0], q[1]), merged_fname)
        for _ in range(2):
            q.popleft()
        q.append(merged_fname)
        n += 1


def generate_synthon_scoring_tasks(
    reactions: list[str],
    shandler: SynthonHandler,
    query_fragments_dict: dict[int, list],
    grouped_synthons: dict[Any, Any],
    synthon_connectors: list[str],
    frag_connectors: list[str],
    cross_score: bool,
):
    n = 0
    for rxn_id in reactions:
        n_component = shandler.get_number_of_components(rxn_id)
        if n_component not in query_fragments_dict:
            raise ValueError(f"{n_component}-fragment sets were not generated from query.")
        # Take the 1st synthon SMILES from each reactant set. Do not use the SMARTS
        # from the reaction file, since OETK cannot parse "~" in SMARTS.
        synthon_smiles_eg = {}
        for key, syns in grouped_synthons[rxn_id].items():
            _, smi = syns[0]
            synthon_smiles_eg[key] = smi

        rxn_synthons_lens = [len(val) for val in grouped_synthons[rxn_id].values()]

        frag_sets = query_fragments_dict[n_component]
        if len(frag_sets) > 0:
            logging.info(f"Start scoring {rxn_id} synthons ({rxn_synthons_lens}); frag_set_count: {len(frag_sets)}")

            for fset_idx, frags in enumerate(frag_sets):
                (
                    synthon_order,
                    ordered_frags,
                ) = order_and_substitute_fragments(  # synthon_order is 0-indexed!
                    synthon_smiles_eg=synthon_smiles_eg,  # This contains info on how many component-reaction.
                    synthon_connectors=synthon_connectors,
                    fragments=frags,
                    frag_connectors=frag_connectors,
                    cross_score=cross_score,
                )
                for forder_idx, ordered_f in enumerate(ordered_frags):
                    logging.debug(f"Start ROCS with reaction: {rxn_id}, frag set: {fset_idx}, forder: {forder_idx}.")
                    yield (
                        rxn_id,
                        fset_idx,
                        forder_idx,
                        synthon_order,
                        ordered_f,
                    )
                    n += 1
    logging.info(f"Generated a total of {n} synthon scoring tasks.")


def generate_synthon_confgen_tasks(
    reactions: list[str],
    synthon_handler: SynthonHandler,
    grouped_synthons: dict,
    connector_atoms: list[str],
    chunk_size: int,
):
    """Generate tasks for synthon conformer generation.

    Parameters
    ----------
    reactions
        List of reaction ids to include.
    synthon_handler
        Synthon handler containing synthon and reaction information.
    grouped_synthons
        Synthons grouped by reaction id.
    connector_atoms
        Special atoms denoting a connector atom on synthons.
    chunk_size
        Chunk size of a synthon component list for each dask worker to process.
    """
    n = 0
    for rxn_id in reactions:
        rxn_synthons = grouped_synthons[rxn_id]
        eg_synthon_smiles = {}
        for key, syns in rxn_synthons.items():
            _, smi = syns[0]
            eg_synthon_smiles[key] = smi

        smirks = synthon_handler.get_reaction_smirks_by_id(rxn_id)
        synthon_ring_connector_dict = label_synthon_ring_connector(
            synthon_smiles=eg_synthon_smiles,
            connector_atoms=connector_atoms,
            smirks=smirks,
        )
        logging.info(f"Reaction {rxn_id} ring connector atoms: {synthon_ring_connector_dict}")

        for s_idx, synthons in rxn_synthons.items():
            other_synthons = {key: val for key, val in eg_synthon_smiles.items() if key != s_idx}
            num_chunk = math.ceil(len(synthons) / chunk_size)
            actual_chunk_size = math.ceil(len(synthons) / num_chunk)
            for chunk_idx, i in enumerate(range(0, len(synthons), actual_chunk_size)):
                yield (
                    rxn_id,
                    smirks,
                    s_idx,
                    synthons[i : i + actual_chunk_size],
                    other_synthons,
                    synthon_ring_connector_dict,
                    chunk_idx,
                )
                n += 1
            logging.info(f"Generated {num_chunk} tasks for {rxn_id} synthon {s_idx}.")

    logging.info(f"Generated a total of {n} synthon confgen tasks.")


# Other functions
def score_forder_synthon(
    synthon_order: list[int],
    ordered_f: list[oechem.OEMolBase],
    synthon_conf_dir: Path,
    conf_format: str,
    rxn_id: int,
    align_connector_atom: bool,
    synthon_connectors: list[str],
    rocs_key_opt: int | None,
    rocs_key_score: int | None,
    opt_shape_func: str,
    opt_color_func: str | None,
    overlap_shape_func: str,
    overlap_color_func: str | None,
    warts_separator: str,
    color_ff_dir: Path,
) -> dict:
    """Scores all components within a particular f-order.

    Return: dict[int, list[tuple[sid, score, etc.]]].
    """
    scores = {}
    for s_idx, _frag in zip(synthon_order, ordered_f):
        conf_f = Path(synthon_conf_dir) / f"{rxn_id}_synthon_{s_idx}_conf.{conf_format}"
        if not conf_f.is_file():
            # oechem does not raise error, only warning, if file not found.
            logging.error(f"Conformer file {conf_f} does not exist.")
            continue

        if align_connector_atom is False:
            _rocs_key_opt = _rocs_key_score = 0
        else:
            num_connector = len(get_conn_symbols(oechem.OEMolToSmiles(_frag), synthon_connectors))
            _rocs_key_opt = num_connector if rocs_key_opt is None else rocs_key_opt
            _rocs_key_score = num_connector if rocs_key_score is None else rocs_key_score

        score = sp_simple_oeoverlay(
            ref_mol=_frag,
            conf_file=conf_f,
            overlay_opts=None,
            rocs_key_opt=_rocs_key_opt,
            rocs_key_score=_rocs_key_score,
            opt_shape_func=opt_shape_func,
            opt_color_func=opt_color_func,
            overlap_shape_func=overlap_shape_func,
            overlap_color_func=overlap_color_func,
            color_ff_dir=color_ff_dir,
        )

        score = deduplicate_scores(score, sort_scores=True, warts_separator=warts_separator)

        scores[s_idx] = score
    return scores


def combine_forder_scores(
    scores: dict[int, list[tuple[str, float, Any]]],
    top_m: int,
    dedup_limit: int,
    warts_separator: str,
    sass_separator: str,
    synthon_comb_limit: int,
    agg_method: str,
) -> tuple[list, int]:

    def backtrack(score_list: list, sid_list: list) -> None:
        nonlocal res

        if len(res) > dedup_limit:
            res = deduplicate_scores(res, sort_scores=True, limit=top_m, warts_separator=warts_separator)

        if len(sid_list) == n_component:
            # comb_score = aggregate_scores(agg_method, scores, sid_list)
            if agg_method == "simple_average":
                # Separate out this case just for speed reasons. Slow to load in `synthon_data`.
                comb_score = simple_average(score_list)
            else:
                comb_score = aggregate_scores(agg_method, score_list, sids=sid_list, synthon_data=None)

            res.append((comb_score, sass_separator.join(sorted(sid_list))))
            return

        cur_s_idx = len(sid_list)  # use result list to track which s-score is on.
        cur_s_limit = s_limit_dict[score_keys[cur_s_idx]]

        for i in range(len(scores[score_keys[cur_s_idx]])):
            if i > cur_s_limit:
                break
            else:
                _score, _sid = scores[score_keys[cur_s_idx]][i]
                score_list.append(_score)
                sid_list.append(_sid)
                backtrack(score_list, sid_list)
                score_list.pop()
                sid_list.pop()

    syn_counts = {key: len(val) for (key, val) in scores.items()}
    if synthon_comb_limit is None:
        s_limit_dict = syn_counts
    else:
        s_limit_dict = allocate_synthons(syn_counts, max_total=synthon_comb_limit)
    logging.info(f"Limit for each synthon score list: {s_limit_dict}.")

    n_component = len(scores)

    res = []
    score_keys = list(scores.keys())
    backtrack([], [])
    res.sort(reverse=True)

    # 20241030: use the total number of products instead of combinations evaluated.
    num_prod = math.prod(syn_counts.values())
    return res[:top_m], num_prod
