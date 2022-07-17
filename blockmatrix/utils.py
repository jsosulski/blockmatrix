from pathlib import Path

import numpy as np

from blockmatrix import blockmatrix


def get_resource_dir():
    p = Path(blockmatrix.__file__).parent.parent
    return p / "res"


def get_example_data(name="simple"):
    if name == "simple":
        saved = np.load(get_resource_dir() / "example_dat.npz")
    elif name == "with_alpha":
        saved = np.load(get_resource_dir() / "example_dat_with_alpha.npz")
    elif name == "with_alpha_targetswithmeans":
        saved = np.load(get_resource_dir() / "example_dat_with_alpha_targetswithmeans.npz")
    elif name == "with_alpha_200hz":
        saved = np.load(get_resource_dir() / "example_dat_with_alpha_large.npz")
    elif name == "no_alpha":
        saved = np.load(get_resource_dir() / "example_dat_no_alpha.npz")
    else:
        raise ValueError("Unknown example data.")
    return saved["raw"], saved["nch"], saved["ntim"]
