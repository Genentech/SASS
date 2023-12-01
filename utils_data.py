"""
Utils functions for synthon/reaction file handling, result analysis.
"""

# Standard Library
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Third Party Library
from matplotlib import pyplot as plt
from openeye import oechem
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc

# Genentech Library
from utils_general import extract_base_id
from utils_mol import cleanup_smiles, substitute_smiles_atoms


class Synthon_handler:
    """
    Handle for Enamine synthon and reaction files.
    """

    def __init__(
        self,
        reaction_file: Optional[Union[str, Path]] = None,
        synthon_file: Optional[Union[str, Path]] = None,
        delimiter: str = "\t",
        reaction_df: Optional[pd.DataFrame] = None,
        synthon_df: Optional[pd.DataFrame] = None,
        sub_dict: dict = None,
    ) -> None:
        """
        Args:
            reaction_file: File containing reaction id and SMIRKS.
            synthon_file: File containing synthon id, reaction type, SMILES.
            delimiter: Delimiter in the files.
            reaction_df: Alternative to providing a reaction text file, provide
                a pandas DataFrame directly.
            synthon_df: Alternative to providing a synthon text file, provide
                a pandas DataFrame directly.
            sub_dict: Mapping for substituting atoms in the synthon files.
                The reason is that Openeye doesn't have forcefield parameters
                for transuranium elements. e.g. CCC[U] --> CCC[13C].
                However, this is largely unused here, due to the string-replacement
                implementation. Substituting the atoms in mol objects is more precise.
        """

        if reaction_df is not None:
            self.rxn_df = reaction_df
        else:
            self.rxn_df = pd.read_csv(reaction_file, sep=delimiter)
        if synthon_df is not None:
            self.syn_df = reaction_df
        else:
            self.syn_df = pd.read_csv(synthon_file, sep=delimiter)

        for df in (self.rxn_df, self.syn_df):
            if df is None:
                raise ValueError("Must provide valid DataFrame or input files.")
        self.sub_dict = sub_dict

    def get_n_component_rxn_id(self, n_component: int) -> list:
        """
        Get ids for all n-component reactions.

        Args:
            n_component: Number of components of reactions to be selected.
        """

        res = self.rxn_df.loc[self.rxn_df["components"] == n_component, "reaction_id"].tolist()
        if len(res) == 0:
            raise ValueError(f"There is no {n_component}-component reaction.")
        return res

    def get_synthon_by_n_component(
        self,
        n_component: int,
        cleanup: bool = True,
    ) -> list:
        """
        Get all synthons used in n-component reactions:

        Args:
            n_component: Number of components of reactions to be selected.

        Returns:
            [(sid1, smi1), (sid2, smi2), ...]
        """

        rxn_ids = self.get_n_component_rxn_id(n_component)
        temp = self.syn_df[self.syn_df["reaction_id"].isin(rxn_ids)].copy()
        if cleanup:
            temp.loc[:, "SMILES"] = temp["SMILES"].apply(cleanup_smiles)
        if self.sub_dict is not None:
            temp.loc[:, "SMILES"] = temp["SMILES"].apply(substitute_smiles_atoms, sub_dict=self.sub_dict)
        return [(sid, smiles) for sid, smiles in zip(temp["synton_id"].tolist(), temp["SMILES"].tolist())]

    def group_synthon_by_rxn_id(
        self,
        n_components: List[int] = None,
        rxn_ids: list = None,
        sub_dummy_atom: bool = False,
        limit: int = None,
        random_seed: int = None,
        cleanup: bool = False,
    ) -> dict:
        """
        Group synthons used in reactions by reaction ids.

        Args:
            n_components: Number of components of reactions to be included. If None,
                include all numbers of components (2, 3, 4).
            rxn_ids: Limit the selection to specific reaction types. If None,
                select all reactions that are within `n_components`.
            sub_dummy_atom: Whether to substitute heavy element dummy atom based on the
                `sub_dict`. Default to not do the string substitution. Instead, subsitute
                in the mol objects later.
            limit: If not `None`, use a subset of the synthons.
            random_seed: If not `None`, select a random subset of synthons using the input
                seed. Else, select the first `limit` number of synthons.
            cleanup: Whether to clean up the SMILES strings of synthons. Used
                mostly for synthon scoring (since enumerated product are also
                cleaned up.)

        Returns:
           {
                reaction_id1: {
                    0: [(sid1, smi1), (sid2, smi2), ...],
                    1: [(sid1, smi1), (sid2, smi2), ...],
                    ...
                },
                reaction_id2: {...},
                ...
           }
           Note: The keys (0, 1, ...) under each rxn_id is based on number of unique "synton#"
           in the Enamine file. The "synton#" in the files are 1-indexed, and the ordering matters
           for the SMIRKS pattern when instantiating products.
        """

        output = {}
        if rxn_ids is None:
            if n_components is None:
                n_components = [2, 3, 4]  # The most Enamine has are 4 component reactions.
            rxn_ids = []
            for n in n_components:
                rxn_ids.extend(self.get_n_component_rxn_id(n))
        for rxn_id in rxn_ids:
            output[rxn_id] = {}
            temp = self.syn_df[self.syn_df["reaction_id"] == rxn_id].copy()
            temp.drop_duplicates(
                subset=["synton_id"], inplace=True
            )  # deduplicate here, because some synthon lists have duplicate entries.

            if cleanup:
                temp.loc[:, "SMILES"] = temp["SMILES"].apply(cleanup_smiles)
            if sub_dummy_atom:
                temp.loc[:, "SMILES"] = temp["SMILES"].apply(substitute_smiles_atoms, sub_dict=self.sub_dict)

            # Infer the number of components.
            component_labels = sorted(temp["synton#"].unique())

            for i, comp_label in enumerate(component_labels):
                sub_df = temp[temp["synton#"] == comp_label]
                if limit is not None and len(sub_df) > limit:
                    if random_seed is None:
                        sub_df = sub_df.head(limit)
                    else:
                        sub_df = sub_df.sample(limit, random_state=random_seed, axis=0)
                output[rxn_id][i] = [
                    (sid, smiles) for sid, smiles in zip(sub_df["synton_id"].tolist(), sub_df["SMILES"].tolist())
                ]
        return output

    def get_reaction_smirks_by_id(self, rxn_id: str, sub_dummy_atom: bool = False) -> str:
        """
        Get the reaction SMIRKS by reaction id.
        """

        smirks = self.rxn_df.loc[self.rxn_df["reaction_id"] == rxn_id, "Reaction"].item()

        if sub_dummy_atom:
            smirks = substitute_smiles_atoms(smirks, self.sub_dict)

        return smirks

    def get_reactant_smarts_by_id(self, rxn_id: str) -> List[str]:
        """ """

        row = self.rxn_df.loc[self.rxn_df["reaction_id"] == rxn_id]
        smarts = []
        for i in range(1, 5):  # max 4 component reaction in Enamine REAL so far.
            col_name = f"R{i}"
            _smarts = row[col_name].item()
            if pd.isna(_smarts) or _smarts == "-":
                break
            else:
                smarts.append(_smarts)
        return smarts

    def get_number_of_components(self, rxn_id: str) -> int:

        return int(self.rxn_df.loc[self.rxn_df["reaction_id"] == rxn_id, "components"].item())

    def get_rxn_id_by_synthon(self, syn_id: Union[int, str]) -> str:
        """
        Get the reaction id by synthon id.
        """

        if not isinstance(syn_id, int):
            syn_id = int(syn_id)

        return self.syn_df.loc[self.syn_df["synton_id"] == syn_id, "reaction_id"].tolist()

    def get_synthon_smi(self, syn_id: int) -> str:

        syn_id = int(syn_id)
        return self.syn_df.loc[self.syn_df["synton_id"] == syn_id, "SMILES"].tolist()[0]


class ResultAnalysis:
    """
    For analysis of scores calculated from a query, compared to ground truth.
    """

    def __init__(
        self,
        true_scores: List[Tuple[float, str]],
        exp_scores: List[Tuple[float, str]] = None,
        exp_product_file: str = None,
        max_exp_product_n: int = None,
        sort_scores: bool = False,
        warts_separator: str = "_",
    ):
        """
        Args:
            true_scores: The scores of all pair-wise synthon products.
            exp_scores: The scores from a query.
            exp_product_file: Alternative to the `exp_scores` array, input an oeb.gz
                file containing products from selected synthons. Only extract the
                titles of these molecules.
            extract_id: See docstring of `extract_ids` function.
            sort_scores: Whether to sort the input `true_scores` and `exp_scores`.
                Use with caution, when only the numerical scores are present.
        """

        self.true_scores = true_scores
        self.is_raw_file = False
        if exp_scores is not None:
            exp_scores = exp_scores[:max_exp_product_n]
            self.exp_scores = exp_scores
        elif exp_product_file is not None:
            self.is_raw_file = True
            self.exp_scores = []
            n = 0
            for mol in oechem.oemolistream(exp_product_file).GetOEMols():
                self.exp_scores.append(("", mol.GetTitle()))
                n += 1
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
            pid = extract_base_id(pid, warts_separator=warts_separator)
            if pid not in base_true_id_set:
                base_true_id_set.add(pid)
                base_true_ids.append(pid)
        self.true_ids = base_true_ids.copy()
        self.exp_ids_set = set(self.exp_ids)

    def extract_ids(self, scores: List[Tuple[float, str]], warts_separator) -> List[str]:
        """
        Raw scores files are arrays of (score, compound_id). Remove the conformer #
        of in the compound_id (i.e. xxx_yyy_isomer_conformer) and output just the
        [compound_id] sorted by score in reverse.
        """
        return [extract_base_id(data[1], warts_separator=warts_separator) for data in scores]

    def calc_single_recall(self, top_n: int):
        """
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
            total_hit += int(self.true_ids[i] in self.exp_ids_set)
            self.recall_rate.append(round(total_hit / (i + 1), 3))

        # x-axis
        self.frac = [n / top_n for n in range(top_n + 1)]
        self.recall_auc = auc(self.frac, self.recall_rate)

        return self.recall_auc

    def plot_recall_auc(self, title: str = None, ax: plt.Axes = None):

        if self.recall_rate is None:
            logging.error("Call `calc_recall_auc` first before plotting.")
            return

        if ax is None:
            ax = plt.axes()
        sns.lineplot(
            x=self.frac[1:],
            y=self.recall_rate[1:],
            ax=ax,
            marker=".",
            markeredgecolor=None,
            markersize=1,
        )
        ax.set_xlabel(f'Fraction "screened" out of {self.top_n}')
        ax.set_ylabel("Recall at given fraction")
        title_txt = f"AUC: {self.recall_auc:.3f}"
        if title:
            title_txt = title + "; " + title_txt
        ax.set_title(title_txt)
        ax.set_ylim((-0.05, 1.05))
        ax.annotate(
            text=f"top-m={len(self.exp_scores)}",
            xy=(0.4, 0.05),
            xycoords="axes fraction",
        )
