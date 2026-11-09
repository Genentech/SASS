"""Synthon and reaction file handling.

Keep imports here minimal, since dask workers import this module.
"""

import functools
import logging
import pickle
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from sass.chem.mol import cleanup_smiles, count_non_H_atom, smiles_to_mol

logger = logging.getLogger(__name__)


class SynthonHandler:
    """Handle for synthons and reactions."""

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
        synthon_id_index: list[str] | str | Path | None = None,
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
        synthon_id_index
            Optional `list[str]` (or a path to a pickled one) mapping integer synthon ids to
            the original strings. When given, the synthon id column is converted to those
            integers on load.
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
            self.syn_df = synthon_df
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

        if synthon_id_index is not None:
            self._apply_synthon_id_index(synthon_id_index, syn_id_col)

        self.rxn_id_col = rxn_id_col
        self.rxn_component_col = rxn_component_col
        self.rxn_smirks_col = rxn_smirks_col
        self.rxn_smarts_cols = rxn_smarts_cols
        self.syn_smiles_col = syn_smiles_col
        self.syn_id_col = syn_id_col
        self.syn_num_col = syn_num_col

    def _apply_synthon_id_index(self, synthon_id_index: list[str] | str | Path, syn_id_col: str) -> None:
        """Replace string synthon ids with their integer index in place.

        Raises if any id is absent from the index, which means the index and the synthon file
        come from different releases.
        """
        if not isinstance(synthon_id_index, list):
            with Path(synthon_id_index).open("rb") as f:
                synthon_id_index = pickle.load(f)  # Internal artifact, never external input.

        id_to_int = {sid: i for i, sid in enumerate(synthon_id_index)}
        mapped = self.syn_df[syn_id_col].map(id_to_int)
        missing = self.syn_df.loc[mapped.isna(), syn_id_col]
        if len(missing) > 0:
            raise ValueError(
                f"{len(missing)} synthon ids are absent from the id index "
                f"(e.g. {list(missing.head(3))}). The index and the synthon file must come "
                "from the same library release."
            )
        # int32 holds any realistic synthon count and halves the DataFrame's memory.
        self.syn_df[syn_id_col] = mapped.astype("int32")
        logger.info("Mapped %s synthon ids to integers via the id index.", len(self.syn_df))

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
        return list(zip(temp[self.syn_id_col].tolist(), temp[self.syn_smiles_col].tolist(), strict=False))

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
        # Group once rather than scanning the synthon table per reaction.
        by_reaction = dict(tuple(self.syn_df.groupby(self.rxn_id_col, sort=False)))

        for rxn_id in rxn_ids:
            output[rxn_id] = {}
            temp = by_reaction.get(rxn_id)
            if temp is None:
                continue
            temp = temp.drop_duplicates(subset=[self.syn_id_col]).copy()

            if cleanup:
                temp.loc[:, self.syn_smiles_col] = temp[self.syn_smiles_col].apply(cleanup_smiles)

            # Infer the number of components.
            component_labels = sorted(temp[self.syn_num_col].unique())
            # Sorted here, but still relies on the synthon order being labeled sensibly.

            for i, comp_label in enumerate(component_labels):
                sub_df = temp[temp[self.syn_num_col] == comp_label]
                if limit is not None and len(sub_df) > limit:
                    if random_seed is None:
                        sub_df = sub_df.head(limit)
                    else:
                        sub_df = sub_df.sample(limit, random_state=random_seed, axis=0)
                output[rxn_id][i] = list(
                    zip(sub_df[self.syn_id_col].tolist(), sub_df[self.syn_smiles_col].tolist(), strict=False)
                )

        return output

    @functools.cached_property
    def _smirks_by_rxn(self) -> dict:
        """Reaction id -> SMIRKS, built once."""
        return dict(zip(self.rxn_df[self.rxn_id_col], self.rxn_df[self.rxn_smirks_col], strict=True))

    @functools.cached_property
    def _components_by_rxn(self) -> dict:
        """Reaction id -> number of components, built once."""
        return dict(zip(self.rxn_df[self.rxn_id_col], self.rxn_df[self.rxn_component_col], strict=True))

    def get_reaction_smirks_by_id(self, rxn_id: str) -> str:
        """Get the reaction SMIRKS by reaction id."""
        return self._smirks_by_rxn[rxn_id]

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
        return int(self._components_by_rxn[rxn_id])

    def _id_mask(self, syn_id: int | str) -> "pd.Series":
        """Compare `syn_id` against the id column in that column's own type.

        The column is `int32` after `_apply_synthon_id_index`, and string otherwise.
        """
        column = self.syn_df[self.syn_id_col]
        key = int(syn_id) if pd.api.types.is_integer_dtype(column) else str(syn_id)
        return column == key

    def get_rxn_id_by_synthon(self, syn_id: int | str) -> list:
        """
        Get the reaction id by synthon id.

        Returns:
            A list of reaction ids (one synthon may participate in multiple reactions).
        """
        return self.syn_df.loc[self._id_mask(syn_id), self.rxn_id_col].tolist()

    def get_synthon_smi(self, syn_id: int | str) -> str:
        smiles = self.syn_df.loc[self._id_mask(syn_id), self.syn_smiles_col].tolist()
        if not smiles:
            raise KeyError(f"No synthon with id {syn_id!r} in the synthon table.")
        return smiles[0]


def map_synthon_ids(
    reactions: list[str],
    grouped_synthons: dict[str, dict[int, list[tuple[int, str]]]],
    synthon_connectors: list[str],
) -> dict:
    """Map synthon ids to SMILES, synthon index, heavy atom count, and reaction ids."""
    synthon_data = {}
    for rxn_id in reactions:
        for s_idx, synthons in grouped_synthons[rxn_id].items():
            for sid, smi in synthons:
                # heavy atom count
                mol = smiles_to_mol(smi)
                ha_count = count_non_H_atom(mol, set(synthon_connectors))
                synthon_data.setdefault(sid, {})
                for key, val in {"heavy_atom_count": ha_count, "SMILES": smi}.items():
                    if key not in synthon_data[sid]:
                        synthon_data[sid][key] = val
                    elif synthon_data[sid][key] != val:
                        raise ValueError(f"Synthon {sid} has different {key=} values.")
                # Sanity check
                if rxn_id in synthon_data[sid]:
                    raise ValueError(f"Duplicate synthon id {sid} found in reaction {rxn_id}.")
                synthon_data[sid][rxn_id] = {}
                synthon_data[sid][rxn_id]["s_idx"] = s_idx
    return synthon_data


#: A synthon id must fit an int32 once numbered; the index positions become the ids.
_INT32_MAX = 2**31 - 1


def build_synthon_id_index(synthon_file: Path | str, syn_id_col: str = "synton_id") -> list[str]:
    """Build the string -> integer synthon id index for a library.

    Returns a `list[str]` whose positions are the integer ids. Ids are read from the synthon
    file, not from the possibly filtered `grouped_synthons.pkl`, and sorted before numbering,
    so the index depends only on the set of ids.

    Raises
    ------
    ValueError
        If an id is null, or there are more ids than an int32 can hold.
    """
    df = pd.read_csv(synthon_file, sep="\t", dtype={syn_id_col: "string"})

    # Check for nulls before sorting, which would fail on them with an unclear `TypeError`.
    if df[syn_id_col].isna().any():
        n = int(df[syn_id_col].isna().sum())
        raise ValueError(f"{synthon_file} has {n} null value(s) in the {syn_id_col!r} column.")

    ids = sorted(set(df[syn_id_col]))
    if len(ids) > _INT32_MAX:
        raise ValueError(f"{len(ids):,} synthon ids exceeds int32; the id scheme assumes int32.")

    # Harmless under integer ids, but worth reporting.
    collide = [i for i in ids if "_" in i or "&" in i]
    if collide:
        logger.warning(
            "%s synthon ids contain '_' or '&', e.g. %s. Harmless under integer ids, but the "
            "string-id pipeline would have mis-parsed them.",
            len(collide),
            collide[:3],
        )

    logger.info("Built id index from %s: %s unique synthon ids.", synthon_file, f"{len(ids):,}")
    return ids
