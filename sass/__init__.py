"""
(C) 2024 Genentech. All rights reserved.

Top-level package for SASS.

"""

__author__ = """Chen Cheng"""
__email__ = "cheng.chen.cc6@gene.com"
__version__ = "1.1.0"


from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / "data"
UTILS_DIR = SRC_DIR / "utils"
TEST_DIR = SRC_DIR.parent / "tests"

COLOR_FF_DIR = "color_ffs"
SYNTHON_CONF_DIR = "synthon_conformers"
SYNTHON_SCORE_DIR = "synthon_scores"
PSEUDO_RES_DIR = "pseudo_res"
