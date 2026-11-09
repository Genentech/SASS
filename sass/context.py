"""The resolved state of one run, built by `sass.main` and shared by every task module."""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sass import COLOR_FF_DIR, EXTEND_ATOM_ISOTOPE, EXTEND_ATOM_SYMBOL
from sass.chem.overlay import overlay_opt_builder, write_color_force_fields
from sass.library.registry import LIBRARY_KEYS
from sass.library.synthons import SynthonHandler


@dataclass
class RunContext:
    """Everything a task module needs, resolved once.

    In each `_w` / `_r` pair, a task writes to the canonical location under `exp_dir` and
    reads from wherever the config points. This lets a run resume from a previous run's output.
    """

    config: dict[str, Any]
    args: argparse.Namespace
    logger: logging.Logger
    exp_dir: Path

    # Scoring parameters, unpacked from the config because every task reads them.
    top_m: int
    dedup_limit: float
    reactions: list[str] | None
    max_stereocenter: int
    conf_format: str
    synthon_connectors: list[str]
    frag_connectors: list[str]
    warts_id_sep: str
    product_id_sep: str
    opt_shape_func: str
    opt_color_func: str | None
    overlap_shape_func: str
    overlap_color_func: str | None
    rocs_key_opt: int | None
    rocs_key_score: int | None
    synthon_comb_limit: int | None

    # Paths, all absolute because several are shipped to dask workers.
    pseudo_res_dir: Path
    synthon_conf_dir: Path
    rescore_file_w: Path
    rescore_file_r: Path
    combined_pseudo_res_file_w: Path
    combined_pseudo_res_file_r: Path
    synthon_data_w: Path
    synthon_data_r: Path
    grouped_synthon_w: Path
    grouped_synthon_r: Path
    rescore_res_dir_w: Path
    full_enumeration_product_dir_w: Path
    full_enumeration_product_dir_r: Path
    full_enumeration_conf_dir_w: Path
    full_enumeration_conf_dir_r: Path
    full_enumeration_score_dir_w: Path
    single_frag_synthon_conf_dir: Path
    single_frag_pseudo_res_file_w: Path
    single_frag_pseudo_res_file_r: Path

    # Loaded data. Populated only when the requested tasks need them.
    shandler: SynthonHandler | None = None
    grouped_synthons: dict | None = None
    synthon_data: dict | None = None
    query_mol: Any = None
    #: Written on first access. See `color_ff_dir`.
    _color_ff_dir: Path | None = None
    query_mol_file: str | None = None
    tasks: list[str] = field(default_factory=list)

    @property
    def color_ff_dir(self) -> Path:
        """Directory of color force field files, built on first access and cached for the run."""
        if self._color_ff_dir is None:
            if self.config["color_ff_dir"] is not None:
                self._color_ff_dir = Path(self.config["color_ff_dir"])
            else:
                self._color_ff_dir = write_color_force_fields(
                    overlay_opt_builder(
                        align_on_dummy_atom=self.config["align_on_dummy_atom"],
                        color_weight=self.config["custom_color_weight"],
                        color_radius=self.config["custom_color_radius"],
                        interaction_type=self.config["color_interaction_type"],
                        color_patterns=self.config["custom_color_ff"],
                        additional_colors=self.config["additional_color_patterns"],
                        scale_weight_per_atom=self.config["scale_weight_per_atom"],
                        no_default_color=self.config["no_default_color"],
                        align_on_bond=self.config["align_on_bond"],
                        extend_bond_atom=EXTEND_ATOM_SYMBOL,
                        extend_bond_isotope=EXTEND_ATOM_ISOTOPE,
                        base_cff_file=self.config["base_cff_file"],
                    ),
                    self.exp_dir / COLOR_FF_DIR,
                )
        return self._color_ff_dir

    @property
    def library_artifact_paths(self) -> list[Path]:
        """Every path that is a library artifact for this run, whether read or written.

        Includes both the configured paths and the default `exp_dir` ones. These are excluded
        from `compress_pkl_files`, since configs resolve library artifacts by exact path.
        """
        paths = [Path(self.config[key]) for key in LIBRARY_KEYS if self.config.get(key)]
        paths += [self.synthon_data_w, self.synthon_data_r, self.grouped_synthon_w, self.grouped_synthon_r]
        if self.config.get("synthon_id_index"):
            index = Path(self.config["synthon_id_index"])
            paths.append(index if index.is_absolute() else self.exp_dir / index)
        return paths
