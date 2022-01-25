import numpy as np

from blockmatrix import SpatioTemporalData
from blockmatrix.utils import get_example_raw

raw, nch, ntim = get_example_raw()
std = SpatioTemporalData(raw)

conds = []
stm = std.get_global_scm()
conds.append(np.linalg.cond(stm.mat))
stm2 = std.swap_primeness().get_global_scm()
stm.plot(True)
stm.swap_primeness().plot(True)
stm.swap_primeness().force_toeplitz_offdiagonals()
conds.append(np.linalg.cond(stm.mat))
stm.swap_primeness().force_toeplitz_offdiagonals(raise_spatial=False)
conds.append(np.linalg.cond(stm.mat))
stm.swap_primeness().taper_offdiagonals().plot(True)
conds.append(np.linalg.cond(stm.mat))
stm.swap_primeness().plot(True)
stm.taper_offdiagonals().plot(True)
conds.append(np.linalg.cond(stm.mat))
stm.swap_primeness().plot(True)

stm = std.get_global_scm()
lu_mat = np.ones((stm.block_dim[1], stm.block_dim[1]))
before = stm.mat.copy()
lookup_block_taper = lambda b1, b2, _: lu_mat[b1, b2]
stm.taper_blocks(lookup_block_taper)
print(np.all(stm.mat == before))
lu_mat = np.zeros((stm.block_dim[1], stm.block_dim[1]))
lookup_block_taper = lambda b1, b2, _: lu_mat[b1, b2]
stm.taper_blocks(lookup_block_taper)
print(np.all(stm.mat == 0))
