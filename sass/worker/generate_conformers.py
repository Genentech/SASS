"""OMEGA conformer generation for synthons, run on a worker."""

import logging
from pathlib import Path
from typing import Any

from openeye import oechem

from sass.chem.confgen import sample_general_mixed_synthon_conf, write_conformers
from sass.chem.mol import delete_atom, get_conn_symbols, read_mols, substitute_atoms
from sass.chem.synthon import extend_dummy_bond
from sass.io import file_size

logger = logging.getLogger(__name__)


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
    logger.info(
        f"Finished generating conformers for {rxn_id} synthon {s_idx}-th reactants, chunk size: {len(synthons)}"
    )


def write_conformers_wrapper(
    data: Any,
    output_dir: Path,
    delete_input_file: bool,
    **kwargs: Any,  # OMEGA parameters
) -> int:
    """Generate conformers for one product chunk.

    Returns
    -------
        Net change in bytes on disk (conformer file written, product file optionally freed).
    """
    logger.info("Starting conformer generation.")
    mol_file = data
    base_name = mol_file.name.split(".")[0]
    out_file = output_dir / f"{base_name}.conf.oez"
    freed = file_size(mol_file) if delete_input_file else 0
    write_conformers(
        in_file=mol_file,
        out_file=out_file,
        **kwargs,
    )
    written = file_size(out_file)
    if delete_input_file:
        mol_file.unlink()
    return written - freed


def generate_single_frag_synthon_conformers(
    synthon_file: Path,
    output_dir: Path,
    synthon_connectors: list[str],
    target_connector_atom: str,
    extend_bond: bool,
    extend_atom_symbol: str,
    extend_atom_isotope: int,
    extend_bond_length: float,
    chunk_size: int,
):
    input_base = synthon_file.stem
    chunk_idx = 0
    cur_chunk_size = 0

    fail_dir = output_dir / "failed"
    fail_dir.mkdir(parents=True, exist_ok=True)
    fail_ostream = oechem.oemolostream(str(fail_dir / f"{input_base}_fail.oez"))
    fail_encountered = 0

    ostream = oechem.oemolostream(str(output_dir / f"{input_base}_single_frag_{chunk_idx}.oez"))
    with read_mols(synthon_file) as synthons:
        for synthon in synthons:
            conns = get_conn_symbols(oechem.OEMolToSmiles(synthon), set(synthon_connectors))
            for conn in conns:
                new_synthon = oechem.OEMol(synthon)
                new_synthon.SetTitle(f"{synthon.GetTitle()}[{conn}]")
                # delete all other connector atoms
                for _conn in conns:
                    if _conn != conn:
                        delete_atom(new_synthon, _conn, 0)
                # substitute remaining `conn` with `target_connector_atom`
                substitute_atoms(
                    mol=new_synthon,
                    sub_dict={(conn, 0): (target_connector_atom, 0)},
                    delete_H=True,
                )
                # optionally extend the bond
                if extend_bond:
                    try:
                        new_synthon = extend_dummy_bond(
                            mol=new_synthon,
                            dummy_atom_symbol=target_connector_atom,
                            dummy_isotope=0,
                            new_dummy_symbol=extend_atom_symbol,
                            new_isotope=extend_atom_isotope,
                            distance=extend_bond_length,
                            explicit_bond=False,
                        )
                    except ZeroDivisionError as e:
                        # Some confgen failed but output a structure with all atoms at (0, 0, 0)
                        # This results in dummy bond distance being 0.
                        fail_encountered += 1
                        oechem.OEWriteMolecule(fail_ostream, new_synthon)
                        logger.warning(
                            f"Failed to extend bond for {new_synthon.GetTitle()} from file {synthon_file}: {e}"
                        )
                        # Skip it: without the extension atom it cannot align on the bond.
                        continue

                # write to file.
                if cur_chunk_size >= chunk_size:
                    ostream.close()
                    chunk_idx += 1
                    cur_chunk_size = 0
                    ostream = oechem.oemolostream(str(output_dir / f"{input_base}_single_frag_{chunk_idx}.oez"))
                oechem.OEWriteMolecule(ostream, new_synthon)
                cur_chunk_size += 1
    ostream.close()
    fail_ostream.close()
    if fail_encountered == 0:
        (fail_dir / f"{input_base}_fail.oez").unlink(missing_ok=True)  # Remove fail file if no failures encountered.
