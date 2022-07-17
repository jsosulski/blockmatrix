from __future__ import annotations

import warnings
from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple, Union

# only needed for plt show!!!
import matplotlib.pyplot as plt
import numpy as np
import vg

from blockmatrix.visualization import plot_covmat

if TYPE_CHECKING:
    from mne.channels import DigMontage
try:
    import toeplitz

    _has_toeplitz = True
except ImportError:
    _has_toeplitz = False


class MatrixProperty(Enum):
    TOEPLITZ = 1
    BANDED = 2
    TAPERED = 3  # cannot really be exploited I think


def block_levinson(block_column, block_row, x):
    """
    A naive implementation using default numpy operations of the levinson recursion
    for block Toeplitz matrices.

    Implementation based on:
     Tobin Fricke (2021). Block Levinson solver
     (https://www.mathworks.com/matlabcentral/fileexchange/30931-block-levinson-solver),
     MATLAB Central File Exchange. Retrieved July 14, 2021.
    """
    L = block_column
    y = x
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    d = L.shape[1]
    N = L.shape[0] // d

    # TODO calculate B from L
    B = block_row

    f = np.linalg.pinv(L[:d, :])
    b = np.copy(f)
    x = f @ y[:d]

    for n in range(1, N):
        ef = B[:, ((N - n - 1) * d) :] @ np.vstack([f, np.zeros((d, d))])
        eb = L[: (n + 1) * d, :].T @ np.vstack([np.zeros((d, d)), b])
        ex = B[:, (N - n - 1) * d :] @ np.vstack([x, np.zeros((d, 1))])

        A = np.linalg.pinv(
            np.vstack([np.hstack([np.eye(d), eb]), np.hstack([ef, np.eye(d)])])
        )
        fn = (
            np.hstack(
                [np.vstack([f, np.zeros((d, d))]), np.vstack([np.zeros((d, d)), b])]
            )
            @ A[:, :d]
        )
        bn = (
            np.hstack(
                [np.vstack([f, np.zeros((d, d))]), np.vstack([np.zeros((d, d)), b])]
            )
            @ A[:, d:]
        )

        f = fn
        b = bn
        x = np.vstack([x, np.zeros((d, 1))]) + b @ (y[n * d : (n + 1) * d] - ex)

    return np.squeeze(x)


def linear_taper(d, dmax):
    return (dmax - np.abs(d)) / dmax


def banding_taper_factory(bands):
    def banding_taper(offset, _):
        return 1 if np.abs(offset) <= bands else 0

    return banding_taper


def fortran_cov_mean_transformation(
    A: np.ndarray, mean: np.ndarray, nch: int, ntim: int
):
    """
    This function can be used to perform the transformation of the covariance into the form
    required by the Fortran library, e.g. when you only want to benchmark the actual solving of
    the equation and not the transformations necessary.
    Use the returned matrix together with fortran_block_levinson(..., transform_A=False)

    Parameters
    ----------
    A: The covariance matrix in default form.
    mean: The mean in default form.
    nch: number of channels (general matrix over this dimension)
    ntim: number of time samples (Toeplitz form over this dimension)

    Returns
    -------
    a: the row/column of the block-Toeplitz covariance matrix stacked as needed for Fortran.

    """
    if nch is None or ntim is None:
        raise ValueError(f"nch and ntim need to be supplied.")
    # leave out first block-matrix, because it is already given in the row array
    col = A[nch:, :nch]
    row = A[:nch, :]

    # # get array, where each entry contains one block of the column and each block is flatten
    # # NOTE: first block is left out
    newcol = (
        np.array([l.T for l in np.vsplit(col, ntim - 1)])
        .reshape((ntim - 1, nch ** 2))
        .T
    )
    # # Each block in row is flatten and horizontally concatenated
    newrow = np.array([l.T for l in np.hsplit(row, ntim)]).reshape(ntim, nch ** 2).T

    # Concat row and column. Start with row, because it contains the first block (column does not)
    a = np.hstack((newrow, newcol))
    fortran_mean = mean.reshape((ntim, nch)).T
    return a, fortran_mean


def fortran_block_levinson(
    A: np.ndarray,
    mean: np.ndarray,
    transform_A=True,
    nch: int = None,
    ntim: int = None,
):
    """
    Solve a block-Toeplitz system using the fortran library.

    Parameters
    ----------
    A: The normal numpy covariance matrix. If you have the covariance
    matrix already in Fortran form, use transform_A=False.

    mean:
    nch
    ntim
    transform_A

    Returns
    -------

    """
    if not _has_toeplitz:
        raise ValueError(
            f"Cannot use fortran solver as toeplitz solver package is not "
            f"installed. Consider installing blockmatrix[solver]."
        )
    if transform_A:
        a, fortran_mean = fortran_cov_mean_transformation(
            A, mean=mean, nch=nch, ntim=ntim
        )
        w_fortr = toeplitz.ctg_sl(a, fortran_mean)
    else:
        w_fortr = toeplitz.ctg_sl(A, mean)
    return w_fortr.real.T.flatten()


def calc_scm(
    x_0: np.ndarray,
    x_1: Optional[np.ndarray] = None,
    return_means: bool = False,
):
    if x_1 is None:
        x_1 = x_0
    p, n = x_0.shape
    mu_0 = np.repeat(np.mean(x_0, axis=1, keepdims=True), n, axis=1)
    mu_1 = np.repeat(np.mean(x_1, axis=1, keepdims=True), n, axis=1)
    Xn_0 = x_0 - mu_0
    Xn_1 = x_1 - mu_1
    S = np.matmul(Xn_0, Xn_1.T)
    Cstar = S / (n - 1)
    if not return_means:
        return Cstar
    else:
        return Cstar, (mu_0, mu_1)


# Simple helper, not nice for the stack_trace though
def check(b: bool, error_msg: str):
    if not b:
        raise ValueError(error_msg)


def get_channel_distance_matrix(mnt, distance="3d"):
    ch_pos = mnt.get_positions()["ch_pos"]
    n_chans = len(ch_pos)
    mat = np.zeros((n_chans, n_chans))
    # tri = Delaunay([ch_pos[k] for k in ch_pos])  # or ConvexHull
    # fig, _ = plt.subplots()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_trisurf(tri.points[:, 0], tri.points[:, 1], tri.points[:, 2], triangles=tri.simplices,
    #                 cmap=plt.cm.Spectral)
    # plt.show()
    for c in range(n_chans):
        cur_ch_name = mnt.ch_names[c]
        cur_ch_pos = ch_pos[cur_ch_name]
        if distance == "3d":
            x = np.array([np.linalg.norm(cur_ch_pos - ch_pos[k]) for k in ch_pos])
        elif distance == "surf":
            pass
        mat[:, c] = x
    return mat


class BlockBased(ABC):
    def __init__(self, block_dim: Sequence, block_label: Optional[Sequence] = None):
        if block_label is None:
            block_label = ["b_0", "b_1"]
        check(len(block_label) == 2, "So far only 2D Block labels are supported")
        check(len(block_dim) == 2, "So far only 2D Block structures are supported")
        self.block_dim = block_dim
        self.block_label = block_label

    def _swap_primeness(self):
        self.block_dim = self.block_dim[::-1]
        self.block_label = self.block_label[::-1]

    def swap_primeness(self):
        raise NotImplementedError(
            f"This method is not implemented for {self.__class__}"
        )

    def block_dims(self, *args: int):
        """Helper function, useful for, e.g., reshaping of matrices.

        Instead of:
            mat = np.reshape(mat, (self.block_dim[0], self.block_dim[1], self.block_dim[0], self.block_dim[1]))
        You can use:
            mat = np.reshape(mat, self.block_dims(0, 1, 0, 1))
        """
        dims = []
        for i in args:
            dims.append(self.block_dim[i])
        return tuple(dims)

    @property
    def primeness(self):
        return self.block_label[0]

    def __repr__(self):
        domain_descr = ""
        for bl, bd in zip(self.block_label, self.block_dim):
            domain_descr += f"{bl}={bd}, "
        domain_descr = domain_descr[:-2]
        return f"{self.__class__.__name__} (with {len(self.block_dim)} domains), dims: {domain_descr}"


# TODO: make BlockMatrix and maybe BlockData accept itself and np.ndarray as inputs
class TwoDomainData(BlockBased):
    """
    The BlockData class facilitates handling of block matrix, e.g. in conjunction with numpy.
    It assumes that the data inherently covers two different domains, which, when stacked, yield
    a covariance matrix with within-domain blocks. For example when stacking spatial features
    across time dimensions.
    """

    def __init__(self, data: np.ndarray, block_label: Optional[list] = None):
        check(len(data.shape) == 3, "Currently only 3-D data is supported.")
        block_dim = data.shape[0:2]
        super().__init__(block_dim=block_dim, block_label=block_label)
        self.data = data

    def swap_primeness(self):
        self._swap_primeness()
        self.data = np.transpose(self.data, (1, 0, 2))
        return self

    def get_swapped_data(self):
        self.swap_primeness()
        data_copy = np.copy(self.data)
        self.swap_primeness()
        return data_copy

    def get_block_vec(self, b: int, block: Optional[str] = None):
        # FIXME: I think this is wrong
        block = self.primeness if block is None else block
        check(
            block in self.block_label,
            f"Invalid {block=} label. Pick one of {self.block_label}",
        )
        data = self.data if self.primeness == block else self.get_swapped_data()
        return np.squeeze(data[b, :])

    def get_pooled_shift_vec(
        self, block_shift: int, side_crop: int = 0, block: Optional[str] = None
    ):
        block = self.primeness if block is None else block
        check(
            block in self.block_label,
            f"Invalid {block=} label. Pick one of {self.block_label}",
        )
        dim, shift_dim = (
            self.block_dim if self.primeness == block else self.block_dim[::-1]
        )
        n_pooled_blocks = shift_dim - block_shift
        check(n_pooled_blocks > 0, "Invalid block dist")
        data = self.data if self.primeness == block else self.get_swapped_data()
        pooled_0 = data[:, : (shift_dim - block_shift), :].reshape((dim, -1), order="F")
        pooled_1 = data[:, block_shift:, :].reshape((dim, -1), order="F")

        return pooled_0, pooled_1, n_pooled_blocks

    def get_flattened(self, side_crop: int = 0):
        crop_max = self.data.shape[1]
        dat = self.data[:, side_crop : crop_max - side_crop, :]
        return dat.reshape(
            ((self.block_dim[0] * (self.block_dim[1] - 2 * side_crop)), -1), order="F"
        )


class SpatioTemporalData(TwoDomainData):
    @staticmethod
    def from_stacked_channel_prime(
        data,
        n_chans: int,
        n_times: Optional[int] = None,
        montage: Optional[DigMontage] = None,
    ):
        if n_times is None:
            div, mod = divmod(data.shape[0], n_chans)
            if mod != 0:
                raise ValueError(
                    "Data cannot be interpreted as a SpatioTemporal vector with the "
                    "given dimensions"
                )
            n_times = div
        data = data.reshape((n_chans, n_times, -1), order="F")
        return SpatioTemporalData(data, montage=montage)

    def __init__(self, data: np.ndarray, montage: Optional[DigMontage] = None, sfreq: Optional[float] = None):
        super().__init__(data, ["channel", "time"])
        self.spatial_variance = None
        self.spatial_means = None
        self.montage = montage
        self.sfreq = sfreq

    @property
    def n_chans(self):
        return self.block_dim[0] if self.primeness == "channel" else self.block_dim[1]

    @property
    def n_times(self):
        return self.block_dim[0] if self.primeness == "time" else self.block_dim[1]

    def get_channel_vec(self, idx: int):
        return self.get_block_vec(idx, block="channel")

    def pool_time_independent(self, t_dist: int, side_crop: int = 0):
        return self.get_pooled_shift_vec(t_dist, block="channel", side_crop=side_crop)

    def get_global_scm(self):
        flata = self.get_flattened()
        stm = SpatioTemporalMatrix(
            calc_scm(flata, flata), self.n_chans, self.n_times, montage=self.montage, sfreq=self.sfreq
        )
        # FIXME: SpatioTemporalMatrix should have an option in the constructor
        if self.primeness != "channel":
            stm._swap_primeness()
        return stm

    def get_spatial_scm(self):
        s_data = self.pool_time_independent(0)[0]
        spatial_scm, mu = calc_scm(s_data, s_data, return_means=True)
        self.spatial_variance = np.diag(spatial_scm)
        self.spatial_means = mu[0]
        return SpatioTemporalMatrix(spatial_scm, self.n_chans, 1, montage=self.montage)

    def get_spatial_crosscov(self, t_diff):
        s_dat0, s_dat1, n_pool = self.pool_time_independent(t_diff)
        spatial_scm = calc_scm(s_dat0, s_dat1, return_means=True)
        return SpatioTemporalMatrix(spatial_scm, self.n_chans, 1, montage=self.montage)

    def get_temporal_scm(
        self, calc_t_diffs: bool = False, standardize: bool = True
    ) -> SpatioTemporalMatrix:
        flip_primeness = self.primeness != "channel"
        if flip_primeness:
            self.swap_primeness()
        if self.spatial_variance is None:
            self.get_spatial_scm()
        t_data = self.get_pooled_shift_vec(0, 0, "time")[0]
        if calc_t_diffs:
            temporal_scm = np.zeros((self.n_times, self.n_times))
            st_temp = SpatioTemporalData(t_data)
            for t in range(self.n_times):
                pool = st_temp.pool_time_independent(t)
                for diff in range(self.n_times - t):
                    var = calc_scm(pool[0], pool[1])
                    temporal_scm[diff, t + diff] = var
                    if t > 0:
                        temporal_scm[t + diff, diff] = var
        else:
            temporal_scm = calc_scm(t_data, t_data)

        stm = SpatioTemporalMatrix(temporal_scm, 1, self.n_times, montage=self.montage)
        if flip_primeness:
            stm._swap_primeness()
            self.swap_primeness()
        return stm


class BlockMatrix(BlockBased):
    def __init__(
        self, matrix: np.ndarray, block_dim: list, block_label: Optional[list] = None
    ):
        super().__init__(block_dim, block_label)
        check(
            matrix.shape[0] == np.product(block_dim),
            f"Incompatible dimensions: {matrix.shape[0]} != {block_dim[0]}*{block_dim[1]}",
        )
        self.mat = matrix
        self.properties = []

    def get_block(self, b0: int, b1: int, return_blockmatrix: bool = False):
        self._to_2dblockmat()
        check(b0 < self.block_dim[1], "Invalid first index")
        check(b1 < self.block_dim[1], "Invalid second index")
        start_b0 = self.block_dim[0] * b0
        end_b0 = start_b0 + self.block_dim[0]
        start_b1 = self.block_dim[0] * b1
        end_b1 = start_b1 + self.block_dim[0]
        if return_blockmatrix:
            return BlockMatrix(
                self.mat[start_b0:end_b0, start_b1:end_b1], [self.block_dim[0], 1]
            )
        else:
            return self.mat[start_b0:end_b0, start_b1:end_b1]

    def set_block(self, b0: int, b1: int, blockmat: np.ndarray):
        self._to_2dblockmat()
        check(
            blockmat.shape[0] == blockmat.shape[1], "Matrix to be set is not quadratic."
        )
        check(
            blockmat.shape[0] == self.block_dim[0],
            "Matrix to be set has wrong dimensions.",
        )
        bl = self.get_block(b0, b1)
        bl[:] = blockmat[:]

    def swap_primeness(self):
        self._to_4dblockmat()
        pre_shape = self.mat.shape
        self.mat = self.mat.reshape(
            (
                self.block_dim[0] * self.block_dim[0],
                self.block_dim[1] * self.block_dim[1],
            )
        )
        self.mat = self.mat.T.reshape(self.block_dims(0, 0, 1, 1), order="F")
        post_shape = self.mat.shape
        assert post_shape[::-1] == pre_shape
        self._swap_primeness()
        self._to_2dblockmat()
        return self

    def _to_4dblockmat(self):
        if self.mat.ndim != 2:
            return
        stacked_mat = (
            np.reshape(
                self.mat,
                self.block_dims(0, 1, 0, 1),
                order="F",
            )
            .transpose((0, 2, 1, 3))
            .T
        )
        self.mat = stacked_mat

    def _to_2dblockmat(self):
        if self.mat.ndim != 4:
            return
        shape = np.product(self.block_dim)
        self.mat = self.mat.T.transpose(2, 0, 3, 1).reshape((shape, shape))

    def get_block_diagonal(
        self, diagonal_offset: int = 0, writable: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        check(
            diagonal_offset < self.block_dim[1],
            f"Offdiagonal at {diagonal_offset} does not exist",
        )
        self._to_4dblockmat()
        d = np.diagonal(self.mat, diagonal_offset)
        d.setflags(write=writable)
        return d

    def set_block_diagonal(self, block_diagonal: np.ndarray, diagonal_offset: int = 0):
        # TODO program checks
        d = self.get_block_diagonal(diagonal_offset, writable=True)
        d[:] = block_diagonal
        return self

    def set_block_diagonal_blockmat(
        self, blockmat: np.ndarray, diagonal_offset: int = 0
    ):
        check(
            blockmat.shape[0] == blockmat.shape[1], "Matrix to be set is not quadratic."
        )
        check(
            blockmat.shape[0] == self.block_dim[0],
            "Matrix to be set has wrong dimensions.",
        )
        check(
            diagonal_offset < self.block_dim[1],
            f"Offdiagonal at {diagonal_offset} does not exist",
        )
        if blockmat.ndim == 3:
            check(
                blockmat.shape[2] == (self.block_dim[1] - np.abs(diagonal_offset)),
                f"Invalid blockmat shape {blockmat.shape}",
            )
        else:
            blockmat = blockmat[:, :, np.newaxis]
        repeats = self.block_dim[1] - np.abs(diagonal_offset)
        if blockmat.shape[2] == 1:
            blockmat = np.repeat(a=blockmat, repeats=repeats, axis=2)
        d = self.get_block_diagonal(diagonal_offset, writable=True)
        d[:] = blockmat
        return self

    def plot(
        self,
        show=False,
        scaling=None,
        axes=None,
        title=None,
        tick_labels=None,
        skip_tick_labels=0,
        subtick_labels=None,
        plot_correlations=False,
        **kwargs,
    ):
        self._to_2dblockmat()
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(12, 12))
        else:
            fig = axes.figure
        title = f"Primeness: {self.primeness}" if title is None else title
        if plot_correlations:
            dg = np.linalg.inv(np.sqrt(np.diag(np.diag(self.mat))))
            mat = dg @ self.mat @ dg
        else:
            mat = self.mat
        if tick_labels is None and self.montage is not None and self.primeness == "time":
            tick_labels = self.montage.ch_names
        plot_covmat(
            mat,
            dim1=self.block_dim[0],
            dim2=self.block_dim[1],
            scaling=scaling,
            axes=axes,
            title=title,
            tick_labels=tick_labels,
            skip_tick_labels=skip_tick_labels,
            subtick_labels=subtick_labels,
            primeness=self.primeness,
            **kwargs,
        )
        if show:
            plt.show()
        return fig, axes


class SpatioTemporalMatrix(BlockMatrix):
    def __init__(
        self,
        matrix: np.ndarray,
        n_chans: int,
        n_times: int,
        channel_prime: Optional[bool] = True,
        montage: Optional[DigMontage] = None,
        sfreq: Optional[float] = None,
    ):
        if channel_prime:
            super().__init__(
                matrix=matrix,
                block_dim=[n_chans, n_times],
                block_label=["channel", "time"],
            )
        else:
            super().__init__(
                matrix=matrix,
                block_dim=[n_times, n_chans],
                block_label=["time", "channel"],
            )

        self.is_inverted = False
        self.montage = montage
        self.sfreq = sfreq

    def get_channel_block(self, t0: int, t1: int):
        return self.get_block(t0, t1)

    def set_channel_block(self, t0: int, t1: int, blockmat: np.ndarray):
        self.set_block(t0, t1, blockmat)

    def force_toeplitz_offdiagonals(
        self,
        average_blocks: bool = True,
        raise_spatial: bool = True,
        normalize_within: bool = False,
    ) -> SpatioTemporalMatrix:
        """Force the matrix to be block-Toeplitz

        Parameters
        ----------
        average_blocks : bool
            When this flag is true, diagonal values are averaged. Otherwise
            the sum is taken. Note: using sum corresponds to linear tapering
            (the matrix is by 'num_blocks' scaled compared to avg+taper)
        raise_spatial : bool
            Set this to false if you want to force toeplitz offdiagonals even
            along the spatial domain. See [TODO] for more information.
        """
        if self.primeness != "channel" and raise_spatial:
            raise ValueError(
                "Attempting to force toeplitz structure across channels instead of time. If you "
                "are sure, set raise_spatial=False, otherwise swap_primeness() before calling "
                "this function."
            )
        dim = self.block_dim[1]
        for di in range(-dim + 1, dim):
            d = self.get_block_diagonal(di)
            if average_blocks:
                new_d = np.mean(d, axis=2)
            else:
                new_d = np.sum(d, axis=2)
            self.set_block_diagonal_blockmat(new_d, di)
        self._to_2dblockmat()
        return self

    def taper_offdiagonals(self, taper_f: Callable[[int, int], float] = None):
        """Apply uniform taper along the block diagonals

        Parameters
        ----------
        taper_f : A callable that takes two arguments
            The taper_f is called for each block-diagonal with the
            block-diagonal index (from -block_dim to +block_dim).
            The default is to apply a linear tapering. See [TODO].
        """
        taper_f = linear_taper if taper_f is None else taper_f
        dim = self.block_dim[1]
        for di in range(-dim + 1, dim):
            d = self.get_block_diagonal(di)
            new_d = d * taper_f(di, dim)
            self.set_block_diagonal_blockmat(new_d, di)
        self._to_2dblockmat()
        return self

    def taper_blocks(self, block_taper_f: Callable[[int, int, int], float]):
        """Apply individual tapering for each block

        Parameters
        ----------
        block_taper_f : Callable that takes three arguments
            The block_taper_f is called for each block and should return
            a scaling factor that depends on block index and the maximum
            outer dimension.

        Examples
        --------
        Using a simple taper depending on the sum of the indices

        >>> stm = SpatioTemporalMatrix(...)
        >>> def block_taper(b0, b1, b_max):
        ...     return (x0+x1) / x_max**2
        >>> stm.taper_blocks(block_taper)

        Using a lookup matrix

        >>> stm = SpatioTemporalMatrix(...)
        >>> lu_mat = np.random.normal(size=(stm.block_dim[1], stm.block_dim[1]))
        >>> def lookup_block_taper(b0, b1, b_max):
        ...     return lu_mat[b0, b1]
        >>> stm.taper_blocks(lookup_block_taper)
        """
        dim = self.block_dim[1]
        for b0 in range(dim):
            for b1 in range(dim):
                bl = self.get_block(b0, b1)
                new_bl = bl * block_taper_f(b0, b1, dim)
                self.set_block(b0, b1, new_bl)
        self._to_2dblockmat()
        return self

    def band_offdiagonals(self, bands: int = 0):
        """Helper function that applies a band-wise tapering

        Parameters
        ----------
        bands : int
            Block-diagonals with abs(index) > bands are set to 0.
        """
        dim = self.block_dim[1]
        check(np.abs(bands) < dim, f"{bands=} cannot be greater than {dim - 1}")
        banding_taper = banding_taper_factory(bands)
        self.taper_offdiagonals(banding_taper)
        return self

    def plot_stationarity(
        self,
        axes=None,
        figsize: tuple = (12, 10),
        oneside: bool = True,
        sharey: bool = True,
        fixed_ylim: tuple = None,
        show: bool = False,
        plot_legend: bool = True,
        block_offset: int = 0,
    ):
        self._to_4dblockmat()
        check(
            self.primeness == "channel",
            "Can only plot stationarity in channel prime form. Consider calling "
            ".swap_primeness() first.",
        )
        n_chans = self.mat.shape[2]
        sharey = False if fixed_ylim is not None else sharey
        if block_offset > 0 and oneside:
            print("WARNING: Plotting with block_offset != 0 and oneside will hide "
                  "information as the off-diagonal blocks are not symmetric.")
        if axes is None:
            n_subp = np.ceil(np.sqrt(n_chans + int(plot_legend))).astype(int)
            fig, axes = plt.subplots(
                n_subp,
                n_subp,
                sharey="all" if sharey else "none",
                sharex="all",
                figsize=figsize,
            )
        else:
            fig = axes.ravel()[0].get_figure()
        [a.set_visible(False) for a in axes.ravel()]
        ch_names = (
            self.montage.ch_names if self.montage is not None else range(1, n_chans + 1)
        )
        n_times = self.mat.shape[1]
        cm = plt.cm.plasma(np.linspace(0, 1, n_times))
        bo = block_offset
        for c in (range(n_chans - bo)):
            submat = self.mat[:, :, c, c+bo]
            ax = axes.ravel()[c]
            ax.set_visible(True)
            ax.axhline(0, linestyle="--", color="k")
            for t in range(n_times):
                x = np.array(range(n_times)) - t
                x = x if self.sfreq is None else x / self.sfreq
                y = submat[t, :]
                ax.plot(x, y, c=cm[t, :])
                ch = c if ch_names is None else ch_names[c]
                ax.set_title(ch)
                if oneside:
                    ax.set_xlim((0, None))
            if fixed_ylim is not None:
                ax.set_ylim(fixed_ylim)
        if plot_legend:
            ax = axes.ravel()[-1]
            ax.set_visible(True)
            ax.get_shared_x_axes().remove(ax)
            ax.get_shared_y_axes().remove(ax)
            fake_dat = np.repeat(
                np.linspace(1, 2, n_times)[:, np.newaxis], n_times, axis=1
            )
            if oneside:
                fake_dat = np.triu(fake_dat)
                fake_dat[fake_dat == 0] = np.nan
            fake_dat -= 1.5
            ax.imshow(fake_dat, cmap="plasma", clip_on=False)
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("Time")
            ax.set_ylabel("Time")
            [l.set_visible(False) for l in ax.get_yticklabels()]
            [s.set_visible(False) for s in ax.spines.values()]
            [t.set_visible(False) for t in ax.get_xticklines()]
            [t.set_visible(False) for t in ax.get_yticklines()]
            ax.set_title("Within channel")
            ax.grid(False)

        matrix_type = "Precision" if self.is_inverted else "Covariance"
        time_unit = "(in samples)" if self.sfreq is None else "(in seconds)"
        fig.text(0.5, 0.01, f"Time offset {time_unit}", ha="center")
        fig.text(0.01, 0.5, f"{matrix_type}", va="center", rotation="vertical")
        fig.suptitle(f"{matrix_type} - Temporal stationarity analysis")
        if show:
            plt.show()
        self._to_2dblockmat()
        return fig, axes

    def plot_channel_covariance(
        self,
        axes=None,
        distance: str = "3d",
        pool_times: bool = True,
        normalize: Optional[str] = "row",
        sharey: bool = True,
        show: bool = False,
    ):
        if self.montage is None and distance != "index":
            warnings.warn("Can only use channel index without montage information.")
            distance = "index"
        self._to_4dblockmat()
        n_times = 1 if pool_times else self.mat.shape[0]
        if axes is None:
            n_subp = np.ceil(np.sqrt(n_times)).astype(int)
            fig, axes = plt.subplots(
                n_subp,
                n_subp,
                sharey="all" if sharey else "none",
                sharex="all",
                figsize=(12, 10),
                squeeze=False,
            )
        else:
            fig = axes.ravel()[0].get_figure()
        [a.set_visible(False) for a in axes.ravel()]
        time_points = (
            ["$T_{all}$"] if pool_times else [f"$T_{{{i}}}$" for i in range(n_times)]
        )
        ch_pos = self.montage.get_positions()["ch_pos"]
        origin_ref = np.mean([ch_pos[k] for k in ch_pos], axis=0)
        for t in range(n_times):
            if pool_times:
                submats = [self.mat[ti, ti, :, :] for ti in range(self.mat.shape[0])]
                submat = np.mean(submats, axis=0)
            else:
                submat = self.mat[t, t, :, :]
            # submat = np.linalg.inv(submat)
            n_chans = submat.shape[1]
            cm = plt.cm.Dark2(np.linspace(0, 1, n_chans))
            ax = axes.ravel()[t]
            ax.set_visible(True)
            ax.axhline(0, linestyle="--", color="k")
            X = np.zeros((n_chans, n_chans))
            Y = np.zeros((n_chans, n_chans))
            stdevs = np.sqrt(np.diag(submat))
            for c in range(n_chans):
                cur_ch_name = self.montage.ch_names[c]
                cur_ch_pos = ch_pos[cur_ch_name]
                if distance == "3d":
                    x = np.array(
                        [np.linalg.norm(cur_ch_pos - ch_pos[k]) for k in ch_pos]
                    )
                elif distance == "angle":
                    x = np.array(
                        [vg.angle(cur_ch_pos, ch_pos[k], origin_ref) for k in ch_pos]
                    )
                elif distance == "index":
                    x = np.abs(np.array(range(n_chans)) - c)
                elif distance == "surface":
                    adj_mat = np.array(
                        [np.linalg.norm(cur_ch_pos - ch_pos[k]) for k in ch_pos]
                    )
                    x = None
                else:
                    raise ValueError(f"Unkown distance {distance}")
                # xleft = -x[:c]
                # xleft_sort = np.argsort(xleft)
                # xright = x[(c + 1):]
                # xright_sort = np.argsort(xright)
                # x = np.hstack([np.sort(xleft), [0], np.sort(xright)])
                # sort_idx = np.hstack([xleft_sort, [c], 1 + c + xright_sort])
                sort_idx = np.argsort(x)
                x = x[sort_idx]
                y = np.copy(submat[c, :])
                if normalize == "row":
                    y /= np.mean(np.abs(y))
                elif normalize == "first":
                    y /= y[c]
                elif normalize == "correlation":
                    y /= stdevs[c] * stdevs
                y_sorted = y[sort_idx]
                # ax.plot(x, y_sorted, c=cm[c, :], marker='.')
                X[c, :] = x
                Y[c, :] = y_sorted
                ax.plot(x, y_sorted, ".:", c=cm[c, :])
                tp = time_points[t]
                ax.set_title(tp)

                ax.set_xlim((0, None))
        addstr = ""
        if normalize is not None:
            addstr += f" (normalized by {normalize})"
        matrix_type = "Precision" if self.is_inverted else "Covariance"
        fig.text(0.5, 0.04, f"Channel distance ({distance})", ha="center")
        fig.text(0.04, 0.5, f"{matrix_type}{addstr}", va="center", rotation="vertical")
        fig.suptitle(f"{matrix_type} - Spatial analysis")
        if show:
            plt.show()
        self._to_2dblockmat()
        return fig, axes

    def invert(self):
        self._to_2dblockmat()
        self.mat = np.linalg.pinv(self.mat)
        self.is_inverted = not self.is_inverted
        return self
