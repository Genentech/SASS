"""Result analysis and plotting for SASS runs."""

import logging
from typing import Optional

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import auc

from sass.chem.mol import read_mols
from sass.ids import extract_base_id

logger = logging.getLogger(__name__)


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
            with read_mols(exp_product_file) as mols:
                for n, mol in enumerate(mols):
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

        # `exp_ids` need no deduplication, since only their set is used.

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
            logger.error(
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
            logger.error("Call `calc_recall_auc` first before plotting.")
            return

        if ax is None:
            ax = plt.axes()
        sns.lineplot(
            x=self.frac[1:],  # Do not plot the point at 0, only there for auc calc.
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
