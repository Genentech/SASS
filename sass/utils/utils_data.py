"""Utils functions for synthon/reaction file handling, result analysis."""

# Standard Library
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import seaborn as sns

# Third Party Library
from matplotlib import pyplot as plt
from openeye import oechem
from sklearn.metrics import auc

# Genentech Library
from sass.utils.utils_general import extract_base_id
from sass.utils.utils_mol import cleanup_smiles


class SynthonHandler:
    """Handle for Enamine synthon and reaction files."""

    def __init__(
        self,
        reaction_file: Optional[Union[str, Path]] = None,
        synthon_file: Optional[Union[str, Path]] = None,
        delimiter: str = "\t",
        reaction_df: Optional[pd.DataFrame] = None,
        synthon_df: Optional[pd.DataFrame] = None,
        rxn_id_col: str = "reaction_id",
        rxn_component_col: str = "components",
        rxn_smirks_col: str = "Reaction",
        rxn_smarts_cols: tuple[str] = ("R1", "R2", "R3", "R4"),
        syn_smiles_col: str = "SMILES",
        syn_id_col: str = "synton_id",
        syn_num_col: str = "synton#",
    ) -> None:
        """
        Initialize the SynthonHandler class.

        Parameters
        ----------
        reaction_file
            File containing reaction id and SMIRKS.
        synthon_file 
            File containing synthon id, reaction type, SMILES.
        delimiter
            Delimiter in the files.
        reaction_df
            Alternative to providing a reaction text file, provide a pandas DataFrame directly.
        synthon_df
            Alternative to providing a synthon text file, provide a pandas DataFrame directly.
        rxn_id_col
            Name of the column containing reaction ids.
        rxn_component_col
            Name of the column for how many components a reaction has.
        rxn_smirks_col
            Name of the column containing reaction SMIRKS.
        rxn_smarts_cols
            Name of the columns containing SMARTS of each reactants.
        syn_smiles_col
            Name of the column containing synthon SMILES.
        syn_id_col
            Name of the column containing synthon ids.
        syn_num_col
            Name of the column for the position of a synthon in a reaction. Used to assign
            reactant position when enumerating products.
        """
        if reaction_df is not None:
            self.rxn_df = reaction_df
        else:
            self.rxn_df = pd.read_csv(
                reaction_file,
                sep=delimiter,
                dtype={
                    rxn_id_col: "string",
                },
            )
        if synthon_df is not None:
            self.syn_df = reaction_df
        else:
            self.syn_df = pd.read_csv(
                synthon_file,
                sep=delimiter,
                dtype={
                    syn_id_col: "string",
                },
            )

        for df in (self.rxn_df, self.syn_df):
            if df is None:
                raise ValueError("Must provide valid DataFrame or input files.")

        self.rxn_id_col = rxn_id_col
        self.rxn_component_col = rxn_component_col
        self.rxn_smirks_col = rxn_smirks_col
        self.rxn_smarts_cols = rxn_smarts_cols
        self.syn_smiles_col = syn_smiles_col
        self.syn_id_col = syn_id_col
        self.syn_num_col = syn_num_col

    def get_n_component_rxn_id(self, n_component: int) -> list:
        """
        Get ids for all n-component reactions.

        Parameters
        ----------
        n_component
            Number of components of reactions to be selected.
        """
        res = self.rxn_df.loc[self.rxn_df[self.rxn_component_col] == n_component, self.rxn_id_col].tolist()
        if len(res) == 0:
            raise ValueError(f"There is no {n_component}-component reaction.")
        return res

    def get_synthon_by_n_component(
        self,
        n_component: int,
        cleanup: bool = True,
    ) -> list:
        """
        Get all synthons used in n-component reactions.

        Parameters
        ----------
        n_component
            Number of components of reactions to be selected.
        cleanup
            Whether to clean up the input SMILES.

        Returns
        -------
        List of tuples containing synthon id and SMILES.
        """
        rxn_ids = self.get_n_component_rxn_id(n_component)
        temp = self.syn_df[self.syn_df[self.rxn_id_col].isin(rxn_ids)].copy()
        if cleanup:
            temp.loc[:, self.syn_smiles_col] = temp[self.syn_smiles_col].apply(cleanup_smiles)
        return list(zip(temp[self.syn_id_col].tolist(), temp[self.syn_smiles_col].tolist()))

    def group_synthon_by_rxn_id(

        self,
        n_components: Optional[list[int]] = None,
        rxn_ids: Optional[list] = None,
        limit: Optional[int] = None,
        random_seed: Optional[int] = None,
        cleanup: bool = False,
    ) -> dict:
        """
        Group synthons used in reactions by reaction ids.

        Parameters
        ----------
        n_components
            Number of components of reactions to be included. If None, include all numbers
            of components (2, 3, 4).
        rxn_ids
            Limit the selection to specific reaction types. If None, select all reactions
            that are within `n_components`.
        limit
            If not `None`, use a subset of the synthons.
        random_seed
            If not `None`, select a random subset of synthons using the input seed. Else,
            select the first `limit` number of synthons.
        cleanup
            Whether to clean up the SMILES strings of synthons. Used mostly for synthon
            scoring (since enumerated product are also cleaned up).

        Returns
        -------
        dict
            A dictionary where keys are reaction ids and values are dictionaries with
            keys as component indices and values as lists of tuples.
            Each tuple contains a synthon id and its corresponding SMILES string.

            Example:
           {
                reaction_id1: {
                    0: [(sid1, smi1), (sid2, smi2), ...],
                    1: [(sid1, smi1), (sid2, smi2), ...],
                    ...
                },
                reaction_id2: {...},
                ...
           }

        Notes
        -----
        The keys (0, 1, ...) under each rxn_id are based on the number of unique "synton#"
        in the Enamine file. The "synton#" in the files are 1-indexed,
        and the ordering matters for the SMIRKS pattern when instantiating products.
        """
        output = {}
        if rxn_ids is None:
            if n_components is None:
                n_components = self.rxn_df["components"].unique()
            rxn_ids = []
            for n in n_components:
                rxn_ids.extend(self.get_n_component_rxn_id(n))
        for rxn_id in rxn_ids:
            output[rxn_id] = {}
            temp = self.syn_df[self.syn_df[self.rxn_id_col] == rxn_id].copy()
            temp = temp.drop_duplicates(subset=[self.syn_id_col])

            if cleanup:
                temp.loc[:, self.syn_smiles_col] = temp[self.syn_smiles_col].apply(cleanup_smiles)

            # Infer the number of components.
            component_labels = sorted(temp[self.syn_num_col].unique())
            # Sorted here, but still relies on the synthon order being labeld sensibly.

            for i, comp_label in enumerate(component_labels):
                sub_df = temp[temp[self.syn_num_col] == comp_label]
                if limit is not None and len(sub_df) > limit:
                    if random_seed is None:
                        sub_df = sub_df.head(limit)
                    else:
                        sub_df = sub_df.sample(limit, random_state=random_seed, axis=0)
                output[rxn_id][i] = list(zip(sub_df[self.syn_id_col].tolist(), sub_df[self.syn_smiles_col].tolist()))

        return output

    def get_reaction_smirks_by_id(self, rxn_id: str) -> str:
        """Get the reaction SMIRKS by reaction id."""
        smirks = self.rxn_df.loc[self.rxn_df[self.rxn_id_col] == rxn_id, self.rxn_smirks_col].item()

        return smirks

    # To deprecate
    def get_reactant_smarts_by_id(self, rxn_id: str) -> list[str]:
        """Get all reactant SMARTS patterns, in order."""
        row = self.rxn_df.loc[self.rxn_df[self.rxn_id_col] == rxn_id]
        smarts = []
        for col_name in self.rxn_smarts_cols:
            _smarts = row[col_name].item()
            if pd.isna(_smarts) or _smarts == "-":
                break
            smarts.append(_smarts)
        return smarts

    def get_number_of_components(self, rxn_id: str) -> int:
        """Get the number of components of a reaction."""
        return int(self.rxn_df.loc[self.rxn_df[self.rxn_id_col] == rxn_id, self.rxn_component_col].item())

    def get_rxn_id_by_synthon(self, syn_id: int | str) -> list:
        """
        Get the reaction id by synthon id.

        Returns:
            A list of reaction ids (one synthon may participate in multiple reactions).
        """
        if not isinstance(syn_id, str):
            syn_id = str(syn_id)

        return self.syn_df.loc[self.syn_df[self.syn_id_col] == syn_id, self.rxn_id_col].tolist()

    def get_synthon_smi(self, syn_id: int | str) -> str:
        syn_id = str(syn_id)
        return self.syn_df.loc[self.syn_df[self.syn_id_col] == syn_id, self.syn_smiles_col].tolist()[0]


class ResultAnalysis:
    """For analysis of scores calculated from a query, compared to ground truth."""

    def __init__(
        self,
        true_scores: list[tuple[float, str]],
        exp_scores: Optional[list[tuple[float, str]]] = None,
        exp_product_file: Optional[str] = None,
        max_exp_product_n: Optional[int] = None,
        sort_scores: bool = False,
        warts_separator: str = "_",
    ) -> None:
        """
        Initialize the class.

        Parameters
        ----------
        true_scores
            The reference scores to compare to. Assumes that the 1st element of the tuple
            is the score and the 2nd element is the molecule id.
        exp_scores
            Query results to evaluate.
        exp_product_file
            Alternative to the `exp_scores` array, input an oeb.gz file containing products 
            from selected synthons. Only extracts the titles of these molecules.
        max_exp_product_n
            Include only up to this many top scores when calculating metrics.
        sort_scores
            Whether to sort the input `true_scores` and `exp_scores`.
        warts_separator
            String used to separate molecule id and the isomer/conformer suffix in the 
            molecule title.

        Raises
        ------
        ValueError
            If neither `exp_scores` nor `exp_product_file` is provided.
        """
        self.true_scores = true_scores
        self.is_raw_file = False
        if exp_scores is not None:
            if max_exp_product_n is not None:
                exp_scores = exp_scores[:max_exp_product_n]
            self.exp_scores = exp_scores
        elif exp_product_file is not None:
            self.is_raw_file = True
            self.exp_scores = []
            for n, mol in enumerate(oechem.oemolistream(str(exp_product_file)).GetOEMols()):
                self.exp_scores.append(("", mol.GetTitle()))
                if max_exp_product_n is not None and n == max_exp_product_n:
                    # Limit time/mem use when reading in exp_product_file.
                    break
        else:
            raise ValueError("Must provide either score array or `exp_product_file`.")

        if sort_scores:
            self.true_scores.sort(reverse=True)
            self.exp_scores.sort(reverse=True)
        self.true_ids = self.extract_ids(self.true_scores, warts_separator)
        self.exp_ids = self.extract_ids(self.exp_scores, warts_separator)

        # Extract base product number while maintaining the order.
        base_true_ids = []
        base_true_id_set = set()
        for pid in self.true_ids:
            _pid = extract_base_id(pid, warts_separator=warts_separator)
            if _pid not in base_true_id_set:
                base_true_id_set.add(_pid)
                base_true_ids.append(_pid)
        self.true_ids = base_true_ids.copy()

        # self.exp_ids = [extract_base_id(pid, warts_separator) for pid in self.exp_ids]
        # Don't need to deduplicate here since taking a set afterwards.

        self.exp_ids_set = set(self.exp_ids)

    def extract_ids(self, scores: list[tuple[float, str]], warts_separator: str) -> list[str]:
        """
        Extract product ids from raw ids.

        Raw scores files are arrays of (score, compound_id). Remove the conformer #
        of in the compound_id (i.e. xxx_yyy_isomer_conformer) and output just the
        [compound_id] sorted by score in reverse.

        Note
        ----
        This function assumes the compound_id is the second element of the tuple.
        """
        return [extract_base_id(data[1], warts_separator=warts_separator) for data in scores]

    def calc_single_recall(self, top_n: int):
        """
        Calculate the recall value.

        Calculate the recall of top_n ground truth molecules in the top_n of
        query results at a given top_n value. i.e. if among top 20 molecules of
        the query result, 15 of those are actually among the top 20 molecules of
        the ground truth results, the recall at n=20 is 0.75.
        """
        max_n = max(len(self.true_ids), len(self.exp_ids_set))
        if top_n > max_n:
            logging.error(
                (
                    f"`top_n` parameter of {top_n} exceeds the max number of results. "
                    "Resetting `top_n` to max n of {max_n}"
                )
            )
            top_n = max_n

        recall = sum(tid in self.exp_ids_set for tid in self.true_ids[:top_n])
        return round(recall / top_n, 3)

    def calc_recall_auc(self, top_n: int):
        """
        Calculate the AUC of recall curve.

        Calculate the AUC for all recall values from 1 to `top_n` using the
        trapezoidal rule. The array of recall values here is calculated using an
        O(n) method, not by calling `calc_single_recall` for every n.
        """
        self.top_n = top_n

        if top_n > len(self.exp_ids):
            raise ValueError("Arg `pred_top_n` cannot exceed the number of `pred_labels`.")

        self.recall_rate = [1]  # Define recall at 0 to be 1, such that AUC=1 if all other recall rates are 1.
        total_hit = 0
        for i in range(top_n):
            # This works because if ground truth is sorted by score, and any
            # ground truth molecules not found in exp_scores[:n] will not be found
            # further down the list.

            total_hit += int(self.true_ids[i] in self.exp_ids_set)

            self.recall_rate.append(round(total_hit / (i + 1), 3))

        # x-axis
        self.frac = [n / top_n for n in range(top_n + 1)]
        self.recall_auc = auc(self.frac, self.recall_rate)

        return self.recall_auc

    def plot_recall_auc(self, title: str, ax: plt.Axes):

        if self.recall_rate is None:
            logging.error("Call `calc_recall_auc` first before plotting.")
            return

        if ax is None:
            ax = plt.axes()
        sns.lineplot(
            x=self.frac[1:],  # 20231103: Do not plot the point at 0, only there for auc calc.
            y=self.recall_rate[1:],
            ax=ax,
            marker=".",
            markeredgecolor=None,
            markersize=1,
        )
        # ax.figure.set_size_inches((4, 3))
        ax.set_xlabel(f'Fraction "screened" out of {self.top_n}')
        ax.set_ylabel("Recall at given fraction")
        title_txt = f"AUC: {self.recall_auc:.3f}"
        if title:
            title_txt = title + "; " + title_txt
        ax.set_title(title_txt)
        ax.set_ylim((-0.05, 1.05))
        # ax.legend(labels=[f'{len(self.exp_scores)}'])
        ax.annotate(
            text=f"top-m={len(self.exp_scores)}",
            xy=(0.4, 0.05),
            xycoords="axes fraction",
        )
