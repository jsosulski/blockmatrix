from blockmatrix.utils import get_example_data
from blockmatrix.blockmatrix import SpatioTemporalData
import time

dat, nch, ntim = get_example_data(name="with_alpha_200hz")
std = SpatioTemporalData(dat)
now = time.time()
stm = std.get_global_scm()
print(f"Time for scm calculation: {time.time() - now}")
