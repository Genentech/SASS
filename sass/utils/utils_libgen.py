"""Util functions for molecule instantiation and isomer enumeration."""

import logging
from pathlib import Path
from typing import Any, Iterator

from openeye import oechem, oeomega

from sass.utils.utils_mol import cleanup_mol, smiles_to_mol


def generate_library_products(
    reactants: dict[int, list[oechem.OEMol | tuple[int, str]]],
    smirks: str,
    sort_title: bool,
    title_separator: str,
) -> Iterator[oechem.OEMol]:
    """
    Yield all combination products from lists of synthons.

    Parameters
    ----------
    reactants
        Dict of reactants, grouped by the reaction position index. e.g.
        {0: [mol1 | (rid1, smi1), mol2 | (rid2, smi2), ...],
        1: [mol10 | (rid10, smi10), (mol11 | rid11, smi11), ...],
        2: ...
        ...}.
        Reactant position index is 0-indexed!
    smirks
        SMIRKS string of a reaction.
    sort_title
        Whether to rename products by sorting the individual synthon ids (by
        default, a new product is named <reac1_id>_<reac2_id>).
    title_separator
        String separator for joining the reactant ids to form product ids.

    Returns
    -------
    Iterator[oechem.OEMol]
        A generator that yields the pairwise product between two lists of synthons.
    """
    libGen = oechem.OELibraryGen(smirks)
    libGen.SetExplicitHydrogens(False)  # May not be needed.
    libGen.SetClearCoordinates(True)
    # Clears StartingMaterial cooridnates to prevent assignment of unspecified
    # stereochemistry based on 3D information.
    libGen.SetTitleSeparator(title_separator)

    for s_idx, data in reactants.items():
        for d in data:
            if isinstance(d, oechem.OEMolBase):
                mol = oechem.OEMol(d)
            else:
                sid, smi = d
                mol = smiles_to_mol(smi, str(sid))
            libGen.AddStartingMaterial(mol, s_idx)

    for prod in libGen.GetProducts():
        prod_mol = cleanup_mol(prod)  # Output is an OEMolBase object.
        prod_mol = oechem.OEGraphMol(prod)  # Set to OEGraphMol to not having to deal with conformer titles.

        if sort_title:
            sids = prod_mol.GetTitle().split(title_separator)
            if len(sids) > 1:
                title = title_separator.join(sorted(sids))
                prod_mol.SetTitle(title)
        yield prod_mol


def instantiate_singleton_product(
    reactants: list[tuple[int, str, int]],
    smirks: str,
    sort_title: bool,
    title_separator: str,
) -> oechem.OEGraphMol:
    """
    Instantiate a single product from one set of reactants.

    Parameters
    ----------
    reactants
        List of reactants in the format: [(id, SMILES, reaction position index), ...].
    smirks
        SMIRKS string of the reaction.
    sort_title
        Whether to name the product based on sorted or unsorted ids of the reactants.
    title_separator
        String separator for joining the reactant ids to form product ids.

    Returns
    -------
    oechem.OEGraphMol
        The product molecule.

    """
    libGen = oechem.OELibraryGen(smirks)
    libGen.SetExplicitHydrogens(False)  # May not be needed.
    libGen.SetClearCoordinates(True)
    # Clear StartingMaterial cooridnates to prevent assignment of unspecified stereocenters.
    libGen.SetTitleSeparator(title_separator)

    for reactant in reactants:
        sid, smi, sorder = reactant
        mol = smiles_to_mol(smi, str(sid))
        ret_code = libGen.SetStartingMaterial(mol, sorder)
        if ret_code != 1:
            raise ValueError(f"Reactant {sorder}:{sid} are not in the correct order!")

    prod = next(libGen.GetProducts())  # Should generate only one product.
    prod_mol = cleanup_mol(prod)  # Output is an OEMolBase object.
    prod_mol = oechem.OEGraphMol(prod_mol)  # Convert to OEGraphMol.

    if sort_title:
        sids = prod_mol.GetTitle().split(title_separator)
        if len(sids) > 1:
            title = title_separator.join(sorted(sids))
            prod_mol.SetTitle(title)
    return prod_mol


def generate_stereoisomers(
    mol: oechem.OEMolBase,
    flipper_options: oeomega.OEFlipperOptions | None,
) -> Iterator[oechem.OEMol]:
    """
    A generator that yields stereoisomers from the input molecule.

    Parameters
    ----------
    mol
        Input molecule.
    flipper_options
        Options for the FLIPPER module. Will use default FLIPPER options if `None`.
        Some options such as `SetWarts` (OEConfGen::OEFlipperOptions::SetWarts) are important.

    Yields
    ------
    oechem.OEMolBase
        The generated stereoisomers.
    """
    # Load default Flipper options if not provided.
    if flipper_options is None:
        # logging.warning('Flipper option should be set explicitly by the user!')
        flipper_options = oeomega.OEFlipperOptions()

        # Copy/pasted from https://docs.eyesopen.com/toolkits/python/omegatk/omegaexamples.html#generating-stereoisomers
        opts = oechem.OESimpleAppOptions(
            flipper_options,
            "stereo_and_torsion",
            oechem.OEFileStringType_Mol,
            oechem.OEFileStringType_Mol,
        )
        flipper_options.UpdateValues(opts)

    for isomer in oeomega.OEFlipper(mol, flipper_options):
        fmol = oechem.OEMol(isomer)  # Create a new mol, safe from mutating pointers.
        yield fmol


def write_singleton_products(
    pid_data: list[tuple[dict, str]],
    out_file: Path,
    title_separator: str,
    max_stereocenter: int,
) -> None:
    """
    Instantiate and write products to file.

    Parameters
    ----------
    pid_data
        List of tuples containing reactants and smirks.
    out_file
        Output file path.
    title_separator
        String separator for joining the reactant ids to form product ids.
    max_stereocenter
        Maximum number of stereocenters to consider.
    """
    ostream = oechem.oemolostream(str(out_file))
    flipper_opt = flipper_opt_builder(max_stereocenter=max_stereocenter)

    n = 0
    iso = 0
    for reactants, smirks in pid_data:
        mol = instantiate_singleton_product(reactants, smirks, sort_title=True, title_separator=title_separator)
        n += 1
        for isomer in generate_stereoisomers(mol, flipper_opt):
            iso += 1
            oechem.OEWriteMolecule(ostream, isomer)
    ostream.close()
    logging.info(f"Enumerated {n} molecules, {iso} isomers. File written to {out_file}.")


def write_singleton_products_chunks(
    pid_data: list[tuple[dict, str]],
    out_file_stem: str,
    out_file_dir: Path,
    out_file_suffix: str,
    title_separator: str,
    max_stereocenter: int,
    split_chunk_size: int | None = None,
) -> None:
    """
    Instantiate and write products to chunks.

    Parameters
    ----------
    pid_data
        List of tuples containing reactants and smirks.
    out_file_stem
        Output file stem.
    out_file_dir
        Output file directory.
    out_file_suffix
        Output file suffix.
    title_separator
        String separator for joining the reactant ids to form product ids.
    split_chunk_size
        Number of molecules to write to each file. If `None`, write all products to one file.
        This number refers to base molecule number, not isomers.
    """
    if split_chunk_size is None:
        split_chunk_size = len(pid_data)

    chunk_idx = 0
    for chunk_idx, i in enumerate(range(0, len(pid_data), split_chunk_size)):
        pid_chunk = pid_data[i : i + split_chunk_size]
        out_file = out_file_dir / f"{out_file_stem}_{chunk_idx}.{out_file_suffix}"
        write_singleton_products(pid_chunk, out_file, title_separator, max_stereocenter=max_stereocenter)


def write_library_products_chunks(
    out_file: Path,
    max_stereocenter: int,
    chunk_size: int,
    **kwargs: Any,
):
    """
    Write library products to file.

    Parameters
    ----------
    out_file
        Output file path.
    kwargs
        Keyword arguments to pass to `generate_library_products`.
    chunk_size
        Desired chunk size for writing product isomers to file.
    """
    flipper_opt = flipper_opt_builder(max_stereocenter=max_stereocenter)
    file_name = out_file.name.split(".")[0]
    output_dir = out_file.parent
    out_file_suffix = ".".join(out_file.name.split(".")[1:])

    chunk = 0
    n = 0
    iso = 0
    ostream = oechem.oemolostream(str(output_dir / f"{file_name}_{chunk}.{out_file_suffix}"))
    for prod in generate_library_products(**kwargs):
        for isomer in generate_stereoisomers(prod, flipper_opt):
            if (iso + 1) % chunk_size == 0:
                ostream.close()
                chunk += 1
                ostream = oechem.oemolostream(str(output_dir / f"{file_name}_{chunk}.{out_file_suffix}"))
            oechem.OEWriteMolecule(ostream, isomer)
            iso += 1
        n += 1
    ostream.close()
    logging.info(f"Enumerated {n} molecules, {iso} isomers. File written to {chunk + 1} chunks.")
    return n, iso


def flipper_opt_builder(
    set_warts: bool = True,
    enum_nitrogen: bool = True,
    enum_specified_stereo: bool = False,
    max_stereocenter: int | None = None,
):
    """
    Build flipper options.

    Parameters
    ----------
    set_warts
        Whether to set Warts for isomers.
    enum_nitrogen
        Whether to enumerate nitrogen stereocenters.
    enum_specified_stereo
        Whether to enumerate specified stereocenters.
    max_stereocenter
        Maximum number of stereocenters to consider.
    """
    flipper_opt = oeomega.OEFlipperOptions()
    flipper_opt.SetWarts(set_warts)
    flipper_opt.SetEnumNitrogen(enum_nitrogen)
    flipper_opt.SetEnumSpecifiedStereo(enum_specified_stereo)
    if max_stereocenter is not None:
        flipper_opt.SetMaxCenters(max_stereocenter)
    opts = oechem.OESimpleAppOptions(
        flipper_opt, "stereo_and_torsion", oechem.OEFileStringType_Mol, oechem.OEFileStringType_Mol
    )
    flipper_opt.UpdateValues(opts)
    return flipper_opt
