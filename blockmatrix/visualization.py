from math import ceil

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def _symlog(arr):
    arr = np.where((arr > 1), np.log(arr) + 1, arr)
    arr = np.where((arr < -1), -np.log(-arr) - 1, arr)
    return arr


def plot_covmat(
    covmat,
    dim1,
    dim2,
    title="Covariance matrix",
    unit=None,
    add_zoom=False,
    show_colorbar=False,
    channel_names=None,
    scaling=(2, 98),
    axes=None,
    tick_labels=None,
    skip_tick_labels=0,
    subtick_labels=None,
    emphasize_offblocks=True,
    delineate_domains=True,
    primeness="channel",
    highlight_free_params=None,  # None, "scm", "toep", or "both"
    show_averaging=False,
    annotate_dims=True,
):
    # scaling can be 'symlog' or a tuple which defines the percentiles
    cp = sns.color_palette()
    assert covmat.shape[0] == dim1 * dim2, (
        "Covariance does not correspond to feature dimensions.\n"
        f"Cov-dim: {covmat.shape} vs. dim1 ({primeness}): {dim1} and dim2: {dim2}"
    )
    if scaling == "symlog":
        data_range = (np.min(covmat), np.max(covmat))
        covmat = _symlog(covmat)
        color_lim = np.max(np.abs(covmat))
    elif scaling is None:
        data_range = (np.nanmin(covmat), np.nanmax(covmat))
        color_lim = np.nanmax(np.abs(data_range))
    else:
        data_range = (np.nanmin(covmat), np.nanmax(covmat))
        percs = np.percentile(covmat, scaling)
        color_lim = np.nanmax(np.abs(percs))
    base_fig_width, base_fig_height = 4, 3
    if add_zoom:
        if axes is None:
            fig, axes = plt.subplots(
                1, 2, figsize=(2 * base_fig_width, base_fig_height)
            )
        else:
            fig = axes[0].figure
            ax = axes[0]
    else:
        if axes is None:
            fig, ax = plt.subplots(1, 1, figsize=(base_fig_width, base_fig_height))
        else:
            fig = axes.figure
            ax = axes
    hm_cmap = "RdBu_r"
    # hm_cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    # hm_cmap = sns.diverging_palette(275, 150, s=80, as_cmap=True)
    im_map = ax.imshow(covmat, cmap=hm_cmap, vmin=-color_lim, vmax=color_lim)
    if title is not None:
        # title_str = title + "\n"
        # title_str += f"Original datarange: {data_range}\n"
        # if scaling != "symlog" and scaling is not None:
        #     title_str += f"Percentiles: {percs}"
        ax.set_title(title)
    ax.grid(False)
    offset = 0.5 if dim1%2 == 0 else 1
    xticks = [i * dim1 + ceil(dim1 / 2) - offset for i in range(dim2)]
    if tick_labels is None:
        lab = "t" if primeness == "channel" else "c"
        xtick_labels = [f"$\\mathrm{{{lab}_{{{i+1}}}}}$" for i in range(dim2)]
        if skip_tick_labels > 0:
            new_xtick_labels = list()
            sc = skip_tick_labels  # always show first
            for xtl in xtick_labels:
                sc += 1
                if sc <= skip_tick_labels:
                    new_xtick_labels.append("")
                else:
                    new_xtick_labels.append(xtl)
                    sc = 0
            xtick_labels = new_xtick_labels

    else:
        xtick_labels = tick_labels
    if subtick_labels is not None:
        yticks = [*range(dim1), *xticks[1:]]
        ytick_labels = [*subtick_labels, *xtick_labels[1:]]
    else:
        yticks, ytick_labels = xticks, xtick_labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    emph_cp = "k"  # cp[2]
    if delineate_domains:
        for i in range(dim2):
            x1 = i * dim1 - 0.5
            x2 = x1 + dim1
            y1 = x1
            y2 = x2
            if emphasize_offblocks:
                ax.axhline(y1, linestyle="-", color=emph_cp, clip_on=True, linewidth=1)
                ax.axhline(y2, linestyle="-", color=emph_cp, clip_on=True, linewidth=1)
                ax.axvline(x1, linestyle="-", color=emph_cp, clip_on=True, linewidth=1)
                ax.axvline(x2, linestyle="-", color=emph_cp, clip_on=True, linewidth=1)
            else:
                ax.plot(
                    [x1, x2, x2, x1, x1],
                    [y1, y1, y2, y2, y1],
                    clip_on=True,
                    color=emph_cp,
                    linewidth=1,
                )
    if show_colorbar:
        fc = fig.colorbar(im_map, ax=fig.axes, fraction=0.026, pad=0.04)
        if unit is not None:
            fc.ax.set_title(unit, rotation=0)
    highlight_dash_width = 2
    if highlight_free_params in ["both", "toep"]:
        hdw = highlight_dash_width
        xdat = [-0.5 + dim1, -0.5, -0.5]
        # ydat = [-0.5 + margin, dim1*dim2-0.5 - margin]
        ydat = [dim1 * dim2 - 0.5, dim1 * dim2 - 0.5, -0.5]
        for i in range(dim1):
            xdat.extend([i - 0.5, i + 0.5])
            ydat.extend([i - 0.5, i - 0.5])
        xdat.extend([dim1 - 0.5, -0.5])
        ydat.extend([dim1 * dim2 - 0.5, dim1 * dim2 - 0.5])
        ax.plot(
            xdat,
            ydat,
            linestyle=(hdw, (hdw, hdw)),
            color=cp[0],
            clip_on=False,
            zorder=123123,
            linewidth=3,
        )
        # pass # STUFF
    if highlight_free_params in ["both", "scm"]:
        xdat = [-0.5 + dim1, -0.5, -0.5]
        # ydat = [-0.5 + margin, dim1*dim2-0.5 - margin]
        ydat = [dim1 * dim2 - 0.5, dim1 * dim2 - 0.5, -0.5]
        for ij in range(dim1 * dim2):
            xdat.extend([ij - 0.5, ij + 0.5])
            ydat.extend([ij - 0.5, ij - 0.5])
        xdat.extend([dim1 * dim2 - 0.5, -0.5])
        ydat.extend([dim1 * dim2 - 0.5, dim1 * dim2 - 0.5])
        ax.plot(
            xdat,
            ydat,
            linestyle=(0, (hdw, hdw)),
            color=cp[1],
            clip_on=False,
            zorder=123123,
            linewidth=3,
        )
        # pass # STUFF
    # Paper plot: from SCM to block-Toeplitz
    if show_averaging:
        local_cp = sns.color_palette("Set1", dim2)
        m = 0.2
        for i in range(dim2):
            xdat = [1.5, -2 + dim1 * dim2 - i * dim1]
            ydat = [1.5 + i * dim1, -2 + dim1 * dim2]
            # ax.plot(xdat, ydat, color=local_cp[i], clip_on=False, zorder=100, linewidth=2)
            for j in range(dim2 - i):
                ls = "-"  # if j == 0 else ':'
                r = plt.Rectangle(
                    (-0.5 + j * dim1 + m / 2, -0.5 + j * dim1 + i * dim1 + m / 2),
                    dim1 - m,
                    dim1 - m,
                    edgecolor=local_cp[i],
                    clip_on=False,
                    fill=False,
                    linewidth=3,
                    linestyle=ls,
                )
                ax.add_patch(r)

    if add_zoom:
        covmat_zoom = covmat[0:dim1, 0:dim1]
        ax = axes[1]
        im_map = ax.imshow(covmat_zoom, cmap="RdBu_r", vmin=-color_lim, vmax=color_lim)
        ax.set_title(f"Zoom on first main diagonal block $B_1$")
        ax.grid(False)
        xticks = list(range(dim1))
        if channel_names is None:
            xtick_labels = [f"ch_{i}" for i in range(dim1)]
        else:
            if len(channel_names) != dim1:
                raise ValueError(
                    "Number of channel names do not correspond to the number of channels."
                )
            xtick_labels = channel_names
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=2.5)
        ax.set_yticks(xticks)
        ax.set_yticklabels(xtick_labels, fontsize=2.5)
        [s.set_color(cp[2]) for s in ax.spines.values()]
        if show_colorbar:
            fig.colorbar(im_map)
    # fig.tight_layout()
