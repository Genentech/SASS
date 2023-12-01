"""
Set up global variables for multiprocessing functions (Flipper, OMEGA, ROCS).
"""

# Standard Library
from typing import Iterable

# Third Party Library
from openeye import oechem, oeomega, oeshape

# Genentech Library
from utils_mol import add_multi_custom_FF


def init(
    synthon_omega_max_conf: int = 50,
    synthon_omega_max_time: int = 120,
    synthon_omega_energy: int = 10,
    product_omega_max_conf: int = 500,
    product_omega_max_time: int = 300,
    product_omega_energy: int = 10,
    align_on_dummy_atom: bool = True,
    color_weight: int = -1,
    color_radius: int = 3,
    interaction_type: str = "gaussian",
    color_patterns: Iterable[str] = ["[U]", "[Np]", "[Pu]", "[Am]"],
    rocs_num_hits: int = 3000,
    scale_weight_per_atom: bool = True,
    ray_head_node: str = None,
) -> dict:

    global flipper_option, rocs_opts, ray_head, omega_opts

    # Ray enabled or not
    ray_head = ray_head_node

    # Flipper
    flipper_option = oeomega.OEFlipperOptions()
    flipper_option.SetWarts(True)
    flipper_option.SetEnumNitrogen(False)
    flipper_option.SetEnumSpecifiedStereo(False)
    opts = oechem.OESimpleAppOptions(
        flipper_option, "stereo_and_torsion", oechem.OEFileStringType_Mol, oechem.OEFileStringType_Mol
    )
    flipper_option.UpdateValues(opts)

    # OMEGA
    omega_opts = {}
    omega_opts["synthon"] = omega_opt_builder(
        synthon_omega_max_conf,
        synthon_omega_max_time,
        synthon_omega_energy,
    )
    omega_opts["product"] = omega_opt_builder(
        product_omega_max_conf,
        product_omega_max_time,
        product_omega_energy,
    )

    # ROCS
    rocs_opts = {}  # e.g. {int: oeshape.OEROCSOptions}, where `int` is the # of connector atoms in the synthon.
    # This is to normalize the color score of synthons regardless of number of connector atoms.
    # `0` means no custom color features.

    if align_on_dummy_atom:
        for i in range(1, len(color_patterns) + 1):
            color_ff = oeshape.OEColorForceField()
            color_ff.Init(oeshape.OEColorFFType_ImplicitMillsDean)
            color_ff = add_multi_custom_FF(
                color_ff,
                color_patterns,
                weight=round(color_weight / i if scale_weight_per_atom else color_weight, 3),
                radius=color_radius,
                interaction=interaction_type,
            )
            color_opt = oeshape.OEColorOptions()
            color_opt.SetColorForceField(color_ff)
            rocs_opts[i] = rocs_opt_builder(color_opt, rocs_num_hits)

    rocs_opts[0] = rocs_opt_builder(None, rocs_num_hits)

    return rocs_opts


def rocs_opt_builder(
    color_options: oeshape.OEColorOptions,
    rocs_num_hits: int,
) -> oeshape.OEROCS:

    overlay_opt = oeshape.OEOverlayOptions()
    if color_options is not None:
        overlay_opt.SetColorOptions(color_options)
    rocs_opt = oeshape.OEROCSOptions()
    rocs_opt.SetOverlayOptions(overlay_opt)
    rocs_opt.SetNumBestHits(rocs_num_hits)
    return rocs_opt


def omega_opt_builder(
    max_conf: int,
    max_search_time: int,
    energy_window: int,
) -> oeomega.OEOmegaOptions:

    omega_options = oeomega.OEOmegaOptions()
    mol_builder_options = oeomega.OEMolBuilderOptions()
    mol_builder_options.SetEnumNitrogen(oeomega.OENitrogenEnumeration_Unspecified)
    omega_options.SetMolBuilderOptions(mol_builder_options)
    omega_options.SetMaxConfs(max_conf)
    omega_options.GetTorDriveOptions().SetMaxSearchTime(max_search_time)
    omega_options.SetEnergyWindow(energy_window)
    # Copied from https://docs.eyesopen.com/toolkits/python/omegatk/omegaexamples.html#generating-conformers
    opts = oechem.OESimpleAppOptions(
        omega_options,
        "Omega",
        oechem.OEFileStringType_Mol,
        oechem.OEFileStringType_Mol3D,
    )
    omega_options.UpdateValues(opts)

    return omega_options
