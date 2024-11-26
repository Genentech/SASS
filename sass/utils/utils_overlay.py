"""Util functions for overlaying molecules using ROCS."""

import logging
from pathlib import Path
from typing import Iterable

from openeye import oechem, oeshape


def sp_simple_oeoverlay(
    ref_mol: oechem.OEMol,
    conf_file: Path | str | None,
    overlay_opts: dict[int, oeshape.OEOverlayOptions] | None,
    rocs_key_opt: int,
    rocs_key_score: int,
    opt_shape_func: str,
    opt_color_func: str | None,
    overlap_shape_func: str,
    overlap_color_func: str | None,
    num_hits: int | None = None,
    write_out_overlay: bool = False,
    out_file: Path | str | None = None,
    keep_color: bool = False,
    color_ff_dir: Path | None = None,
    fit_mols: Iterable[oechem.OEMol] | None = None,
    write_out_ref: bool = False,
    deduplicate_mols: bool = False,
    sep_shape_color_score: bool = False,
) -> list[tuple[float, str]]:
    """
    Overlay molecules using ROCS with one reference conformer.

    Optionally use different color features for the overlay optimization and scoring, and
    different shape and color functions for overlay optimization and scoring.

    Parameters
    ----------
    ref_mol
        Reference (query) molecule.
    conf_file
        Conformer file containing molecules to be overlaid and scored.
    overlay_opts
        Dictionary of ROCS options to use for overlay optimization and scoring.
    rocs_key_opt
        Key of the ROCS option to load from `color_ff_dir` for overlay optimization.
    rocs_key_score
        Key of the ROCS option for in-place scoring after overlay optimization, if a
        different overlap method is desired.
    opt_shape_func
        The shape score function to be used in the overlay optimization.
    opt_color_func
        The color score function to be used in the overlay optimization.
    overlap_shape_func
        The shape score function to be used in overlap calculation (in-place).
    overlap_color_func
        The color score function to be used in overlap calculation (in-place).
    num_hits
        Optionally limit the number of output.
    write_out_overlay
        Whether to save the overlay poses to file.
    out_file
        File path to save the overlay poses.
    keep_color
        Whether to keep the color atoms in the poses to be written out. Color atoms
        cannot be opened in MOE e.g.
    color_ff_dir
        Directory containing color ff files to use.
    fit_mols
        List of molecules to overlay. If `conf_file` is not provided, this list will be used.
    write_out_ref
        Whether to write out the reference molecule.
    deduplicate_mols
        Whether to deduplicate the fit mols in the output. Do not use for synthon scoring.
    sep_shape_color_score
        Whether to write out separate shape and color scores in the output.

    Returns
    -------
    list[tuple[float, str]]
        A list of tuples containing the score and the title of the molecules.
    """

    def _get_score(score: oeshape.OEOverlapResults, shape_func: str, color_func: str | None) -> float:
        res = 0
        for func in (shape_func, color_func):
            if func:
                res += getattr(score, func)()
        return res

    ref_mol = oechem.OEMol(ref_mol)
    if write_out_overlay is True:
        ostream = oechem.oemolostream(str(out_file))
        if write_out_ref is True:
            oechem.OEWriteMolecule(ostream, ref_mol)

    # Set overlay optimization parameters.
    if overlay_opts is not None:
        # Make copies to avoid modifying the original overlay options.
        overlay_opt = oeshape.OEOverlayOptions(overlay_opts[rocs_key_opt])
        score_opt = oeshape.OEOverlayOptions(overlay_opts[rocs_key_score])
    else:
        overlay_opt = get_overlay_opt_from_file(color_ff_dir / f"custom_color_ff_{rocs_key_opt}.txt")
        score_opt = get_overlay_opt_from_file(color_ff_dir / f"custom_color_ff_{rocs_key_score}.txt")

    cff_opt = overlay_opt.GetColorOptions().GetColorForceField()
    prep_overlay = oeshape.OEOverlapPrep()
    if opt_color_func is None:
        prep_overlay.GetColorForceField().Clear()  # Technially don't need to, i.e. can
        # still just color the atoms, but for readability, clear the colors in Prep.
        cff_opt.Clear()
        # This clears the cff of the `overlay_opt`.abs
        # For some reason, you cannot Set a cleared `cff_opt` to `prep`.
    else:
        prep_overlay.SetColorForceField(cff_opt)

    oeshape.OERemoveColorAtoms(ref_mol)
    prep_overlay.Prep(ref_mol)
    overlay = oeshape.OEOverlay(overlay_opt)
    overlay.SetupRef(ref_mol)

    # Set up overlap function for in-place scoring (after overlay optimization).
    ref_mol_copy = oechem.OEMol(ref_mol)
    cff_score = score_opt.GetColorOptions().GetColorForceField()
    prep_score = oeshape.OEOverlapPrep()
    if overlap_color_func is None:
        prep_score.GetColorForceField().Clear()
        cff_score.Clear()
    else:
        prep_score.SetColorForceField(cff_score)

    oeshape.OERemoveColorAtoms(ref_mol_copy)
    prep_score.Prep(ref_mol_copy)
    overlap_func = score_opt.GetOverlapFunc()
    overlap_func.SetupRef(ref_mol_copy)

    # Get the predicate and score methods
    overlay_predicate, opt_method1, opt_method2 = get_overlay_score_methods(opt_shape_func, opt_color_func)
    _, overlap_method1, overlap_method2 = get_overlay_score_methods(overlap_shape_func, overlap_color_func)

    out = []

    if conf_file is not None:
        fitmols_iter = oechem.oemolistream(str(conf_file)).GetOEMols()
    else:
        if fit_mols is None:
            raise ValueError("Must either provide a file or list of molecules for overlay.")
        fitmols_iter = fit_mols

    if deduplicate_mols:
        seen_smi = set()
    for fitmol in fitmols_iter:
        if deduplicate_mols:
            smi = oechem.OEMolToSmiles(fitmol)
            if smi in seen_smi:
                continue
            seen_smi.add(smi)
        oeshape.OERemoveColorAtoms(fitmol)
        prep_overlay.Prep(fitmol)
        score = oeshape.OEBestOverlayScore()
        overlay.BestOverlay(score, fitmol, overlay_predicate)

        if (
            (rocs_key_opt == rocs_key_score)
            and (opt_shape_func == overlap_shape_func)
            and (opt_color_func == overlap_color_func)
        ):
            # No re-scoring needed. Use scores directly.
            _score = _get_score(score, opt_method1, opt_method2)
            if sep_shape_color_score:
                shape_score = _get_score(score, opt_shape_func, None)
                color_score = _get_score(score, None, opt_color_func)
            out.append([_score, fitmol.GetTitle()])
        else:
            # Transform to the overlay pose and re-score.
            # Get the best conformer
            best_conf = oechem.OEGraphMol(fitmol.GetConf(oechem.OEHasConfIdx(score.GetFitConfIdx())))
            score.Transform(best_conf)
            oeshape.OERemoveColorAtoms(best_conf)
            prep_score.Prep(best_conf)
            rescore_res = oeshape.OEOverlapResults()
            overlap_func.Overlap(best_conf, rescore_res)
            _score = _get_score(rescore_res, overlap_method1, overlap_method2)
            if sep_shape_color_score:
                shape_score = _get_score(rescore_res, overlap_shape_func, None)
                color_score = _get_score(rescore_res, None, overlap_color_func)
            out.append([_score, best_conf.GetTitle()])

        if write_out_overlay is True:
            if (
                (rocs_key_opt == rocs_key_score)
                and (opt_shape_func == overlap_shape_func)
                and (opt_color_func == overlap_color_func)
            ):
                # Do not double transform! Transforms are cumulative. 2nd transform
                # will move the `fitmol` out of the optimized overlay position!
                best_conf = oechem.OEGraphMol(fitmol.GetConf(oechem.OEHasConfIdx(score.GetFitConfIdx())))
                score.Transform(best_conf)
            if keep_color is False:
                oeshape.OERemoveColorAtoms(best_conf)
            oechem.OESetSDData(best_conf, "Score", str(_score))
            if sep_shape_color_score:
                oechem.OESetSDData(best_conf, "ShapeScore", str(shape_score))
                oechem.OESetSDData(best_conf, "ColorScore", str(color_score))
            oechem.OEWriteMolecule(ostream, best_conf)

    if write_out_overlay is True:
        ostream.close()

    out.sort(reverse=True)
    if num_hits is not None:
        return out[:num_hits]
    else:
        return out


def get_overlay_score_methods(
    shape_func: str, color_func: str | None
) -> tuple[oeshape.OEBestOverlayScoreSorterPred, str, str | None]:
    """Return the overlay BinaryPredicate and the score methods to be used.

    For specific cases like using TanimotoCombo or Tanimoto shape-only scores, use the
    built-in Predicates and score methods (potentially faster).

    Parameters
    ----------
    shape_func
        The shape score function to be used in the comparison.
    color_func
        The color score function to be used in the comparison.

    Returns
    -------
        tuple[oeshape.OEBestOverlayScoreSorterPred, str, str | None]
    """
    if color_func == str(None):  # To cover "None" args passed in as shell scripts (non-dask).
        color_func = None
    if shape_func == "GetTanimoto" and color_func == "GetColorTanimoto":
        return oeshape.OEHighestTanimotoCombo(), "GetTanimotoCombo", None
    if shape_func == "GetTanimoto" and color_func is None:
        return oeshape.OEHighestTanimoto(), "GetTanimoto", None
    return CustomPredicate(shape_func, color_func), shape_func, color_func


class CustomPredicate(oeshape.OEBestOverlayScoreSorterPred):
    """Custom BinaryPredicate for comparing two OEBestOverlayScore objects."""

    def __init__(self, shape_func: str | None, color_func: str | None) -> None:
        """Set up the shape and color score functions to be used in the comparison.

        Score functions must be valid public methods of oeshape.OEOverlapResults class. See:
        https://docs.eyesopen.com/toolkits/python/shapetk/OEShapeClasses/OEOverlapResults.html

        Parameters
        ----------
        shape_func
            The shape score function to be used in the comparison.
        color_func
            The color score function to be used in the comparison.
        """
        super().__init__()
        self.shape_func = shape_func
        self.color_func = color_func
        if not self.shape_func and not self.color_func:
            raise ValueError("At least one of shape or color function must be provided.")

    def __call__(self, score1: oeshape.OEBestOverlayScore, score2: oeshape.OEBestOverlayScore) -> bool:
        try:
            shape1 = getattr(score1, self.shape_func)() if self.shape_func else 0
            shape2 = getattr(score2, self.shape_func)() if self.shape_func else 0
            color1 = getattr(score1, self.color_func)() if self.color_func else 0
            color2 = getattr(score2, self.color_func)() if self.color_func else 0
            return shape1 + color1 > shape2 + color2
        except AttributeError:
            logging.exception("Invalid shape or color function: %s, %s.", self.shape_func, self.color_func)
            raise

    def CreateCopy(self):
        return CustomPredicate(self.shape_func, self.color_func).__disown__()
        # `__disown__` method is needed to release the object to C++ layer.


def overlay_opt_builder(
    align_on_dummy_atom: bool = True,
    color_weight: int = -1,
    color_radius: int = 3,
    interaction_type: str = "gaussian",
    color_patterns: Iterable[str] = ("[U]", "[Np]", "[Pu]", "[Am]"),
    additional_colors: Iterable[dict] | None = None,
    scale_weight_per_atom: bool = True,
    no_default_color: bool = False,
    align_on_bond: bool = False,
    bond_dummy_atom: str | None = None,
    base_cff_file: str | None = None,
) -> dict[int, oeshape.OEOverlayOptions]:
    """
    Create custom ROCS overlay options.

    Parameters
    ----------
    align_on_dummy_atom
        Whether to apply custom color FF on connector atoms.
    color_weight
        Weight parameter of the custom color FF.
    color_radius
        Color parameter of the custom color FF.
    interaction_type
        Type of the custom color interaction (either "discrete" or "gaussian").
    color_patterns
        SMARTS patterns of the connector atoms.
    additional_colors
        User-specified color features. Useful for overweighing certain functional groups.
    scale_weight_per_atom
        Whether to normalize the custom interaction weights by the number of connector atoms.
    no_default_color
        Whether to run ROCS with shape and only custom color (from connector atoms) i.e.
        no default color features such as rings, hydrophobes, HBD.
    align_on_bond
        Whether to apply custom color FF on the bond along the connector atom vector.
    bond_dummy_atom
        Dummy atom used to construct the connector bond together with the connector atom.
    base_cff_file
        Path to a base color force field file to use instead of the default Mills-Dean.

    Returns
    -------
    dict[int, oeshape.OEOverlayOptions]
        A dictionary of ROCS options.
    """
    color_ff_main = oeshape.OEColorForceField()
    color_ff_main.Clear()
    if not no_default_color:
        if base_cff_file is not None:
            color_ff_main.Init(oechem.oeifstream(str(base_cff_file)))
        else:
            color_ff_main.Init(oeshape.OEColorFFType_ImplicitMillsDean)
    if additional_colors:
        for color in additional_colors:
            color_ff_main = build_custom_FF(
                color_ff_main,
                color["pattern"],
                None,
                color["weight"],
                color["radius"],
                color["interaction"],
            )

    overlay_opts = {}  # e.g. {int: oeshape.OEOverlayOptions}, where `int` is the # of connector atoms in the synthon.
    # This is to normalize the color score of synthons regardless of number of connector atoms.
    # `0` means no custom color features.
    for i in range(len(color_patterns) + 1):
        color_ff = oeshape.OEColorForceField(color_ff_main)  # copy
        if align_on_dummy_atom and i != 0:  # when i == 0, use a color-ff without connector atoms
            color_ff = add_multi_custom_FF(
                color_ff,
                color_patterns,
                weight=round(color_weight / i if scale_weight_per_atom else color_weight, 3),
                radius=color_radius,
                interaction=interaction_type,
            )
            if align_on_bond:
                color_ff = add_multi_custom_FF(
                    color_ff,
                    [f"[{isotope}{bond_dummy_atom}]" for isotope in range(33, 33 + len(color_patterns))],
                    weight=round(color_weight / i if scale_weight_per_atom else color_weight, 3),
                    radius=color_radius,
                    interaction=interaction_type,
                )
        color_opt = oeshape.OEColorOptions()
        if color_opt.SetColorForceField(color_ff) is False:
            # If `no_default_color` and no `additional_colors`, add a dummy color feature
            # to avoid unable to set color FF and defaulting to default color ff.
            color_ff = build_custom_FF(
                color_ff,
                smarts_pattern="[Xe][Xe][Xe]",
                smarts_pattern2=None,
                weight=-1,
                radius=1,
                interaction="gaussian",
            )
            color_opt.SetColorForceField(color_ff)
        overlay_opt = oeshape.OEOverlayOptions()
        overlay_opt.SetColorOptions(color_opt)
        overlay_opts[i] = overlay_opt

    return overlay_opts


def build_custom_FF(
    cff: oeshape.OEColorForceField | None,
    smarts_pattern: str,
    smarts_pattern2: str | None,
    weight: int,
    radius: int,
    interaction: str,
) -> oeshape.OEColorForceField:
    """
    Create a custom color force field, for ROCS scoring.

    Parameters
    ----------
    cff
        An existing color force field object. If `None`, a new color force field will be created.
    smarts_pattern
        SMARTS pattern of the substructure to be assigned a particular "color".
        Patterns must be enclosed in "[]".
    smarts_pattern2
        SMARTS pattern of a second substructure. If provided, interaction between `smarts_pattern`
        and `smarts_pattern2` will be added. Otherwise, self-interaction between `smarts_pattern`
        will be added.
    weight
        Weight of the color in scoring. See: https://docs.eyesopen.com/toolkits/python/shapetk/shape_theory.html#color-features
    radius
        Radius of the color interaction.
    interaction
        Type of color feature interaction, i.e. "gaussian" or "discrete".
    """
    if cff is None:
        cff = oeshape.OEColorForceField()
        cff.Init(oeshape.OEColorFFType_ImplicitMillsDean)

    cur_type_count = len(cff.GetTypes())
    new_type = f"new_type_{cur_type_count + 1}"
    cff.AddType(new_type)
    int_new_type = cff.GetType(new_type)
    cff.AddColorer(int_new_type, smarts_pattern)

    if smarts_pattern2 is not None:
        cur_type_count = len(cff.GetTypes())
        new_type2 = f"new_type_{cur_type_count + 1}"
        cff.AddType(new_type2)
        int_new_type2 = cff.GetType(new_type2)
        cff.AddColorer(int_new_type2, smarts_pattern2)

    if smarts_pattern2 is None:
        cff.AddInteraction(int_new_type, int_new_type, interaction, weight, radius)
    else:
        cff.AddInteraction(int_new_type, int_new_type2, interaction, weight, radius)

    return cff


def add_multi_custom_FF(  # TODO: Move to utils_query!!
    color_ff: oeshape.OEColorForceField,
    patterns: Iterable[str],
    weight: int,
    radius: int,
    interaction: str,
) -> oeshape.OEColorForceField:
    """
    Add self-interaction of input patterns to the custom color FF.

    Parameters
    ----------
    color_ff
        Input color FF to be modified.
    patterns
        An iterable of SMARTS patterns.
    weight
        Weight of the color FF for a pattern.
    radius
        Radius of the color FF for a pattern.
    interaction
        Type of color feature interaction, i.e. "gaussian" or "discrete".
    """
    for pattern in patterns:
        color_ff = build_custom_FF(
            color_ff,
            smarts_pattern=pattern,
            smarts_pattern2=None,
            weight=weight,
            radius=radius,
            interaction=interaction,
        )

    return color_ff


def get_overlay_opt_from_file(file: Path):
    """Load color ff from file and return overlay options."""
    color_ff = oeshape.OEColorForceField()
    color_ff.Init(oechem.oeifstream(str(file)))
    color_opt = oeshape.OEColorOptions()
    color_opt.SetColorForceField(color_ff)
    overlay_opt = oeshape.OEOverlayOptions()
    overlay_opt.SetColorOptions(color_opt)
    return overlay_opt
