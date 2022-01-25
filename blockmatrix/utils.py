from pathlib import Path

import numpy as np

from blockmatrix import blockmatrix


def get_resource_dir():
    p = Path(blockmatrix.__file__).parent.parent
    return p / "res"


def get_example_raw():
    saved = np.load(get_resource_dir() / "example_dat.npz")
    return saved["raw"], saved["nch"], saved["ntim"]
