from mne.channels import DigMontage

from blockmatrix import SpatioTemporalData
from blockmatrix.utils import get_example_data

import matplotlib.pyplot as plt

# The data is from one subject of a visual ERP paradigm
# with 4760 epochs recorded.
dat, nch, ntim = get_example_data(name="with_alpha_large")
pleg = False
fylim = (-60, 140)
# fylim = (-25, 80)
channels = ["F3", "C3", "Cz", "C4", "T8", "P7", "Pz", "O2"]
use_chs = [0, 2]
clabs = [channels[ci] for ci in use_chs]
# Block offset -> 0 show within channel-blocks, 1 show channels with index-neighbors, 2 ...
bo = 0


def get_ax():
    f, a = plt.subplots(1, 2, sharex=True, figsize=(8, 4), dpi=450)
    return a


def plot_stat(stm):
    # Add stub montage
    dm = DigMontage()
    dm.ch_names = clabs
    stm.montage = dm
    stm.sfreq = 200  # Hz
    fig, a = stm.plot_stationarity(
        plot_legend=pleg, fixed_ylim=fylim, axes=get_ax(),
        oneside=True, block_offset=bo
    )
    return fig, a


# Full data
std = SpatioTemporalData(dat[use_chs, :, :])
stm = std.get_global_scm()
nepo = std.data.shape[2]
# Check if data looks somewhat stationary
fig, _ = plot_stat(stm)
fig.suptitle(f"Covariance over time, shrinkage ({nepo:04d} epochs)")
fig.savefig(f"/home/jan/Desktop/asd/stats/nepo{nepo:04d}.png")
fig, _ = plot_stat(stm.force_toeplitz_offdiagonals())
fig.suptitle(f"Covariance over time, Toeplitz ({nepo:04d} epochs)")
fig.savefig(f"/home/jan/Desktop/asd/stats/nepo{nepo:04d}_toep.png")

# Medium size, pick some epochs from the middle
std = SpatioTemporalData(dat[use_chs, :, 1000:1500])
stm = std.get_global_scm()
nepo = std.data.shape[2]
# Check if data looks somewhat stationary
fig, _ = plot_stat(stm)
fig.suptitle(f"Covariance over time, shrinkage ({nepo:04d} epochs)")
fig.savefig(f"/home/jan/Desktop/asd/stats/nepo{nepo:04d}.png")
fig, _ = plot_stat(stm.force_toeplitz_offdiagonals())
fig.suptitle(f"Covariance over time, Toeplitz ({nepo:04d} epochs)")
fig.savefig(f"/home/jan/Desktop/asd/stats/nepo{nepo:04d}_toep.png")

# Tiny size, pick few epochs from the middle
std = SpatioTemporalData(dat[use_chs, :, 1000:1050])
stm = std.get_global_scm()
nepo = std.data.shape[2]
# Check if data looks somewhat stationary
fig, _ = plot_stat(stm)
fig.suptitle(f"Covariance over time, shrinkage ({nepo:04d} epochs)")
fig.savefig(f"/home/jan/Desktop/asd/stats/nepo{nepo:04d}.png")
fig, _ = plot_stat(stm.force_toeplitz_offdiagonals())
fig.suptitle(f"Covariance over time, Toeplitz ({nepo:04d} epochs)")
fig.savefig(f"/home/jan/Desktop/asd/stats/nepo{nepo:04d}_toep.png")
plt.show()