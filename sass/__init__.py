"""
Copyright 2026 Genentech, Inc. Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

Top-level package for SASS.

"""

__author__ = """Chen Cheng"""
__email__ = "cheng.chen.cc6@gene.com"
__version__ = "2.2.8"


from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / "data"
TEST_DIR = SRC_DIR.parent / "tests"

COLOR_FF_DIR = "color_ffs"
SYNTHON_CONF_DIR = "synthon_conformers"
SYNTHON_SCORE_DIR = "synthon_scores"
PSEUDO_RES_DIR = "pseudo_res"

# Extra dummy atoms that define the connection bond vector. Keep these fixed across runs.
# An unused Si isotope cannot clash with real atoms in the molecule.
EXTEND_ATOM_SYMBOL = "Si"
EXTEND_ATOM_ISOTOPE = 33  # Starting isotope. Incremented by 1 for each new atom.
