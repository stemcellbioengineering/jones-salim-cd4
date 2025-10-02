import matplotlib.pyplot as plt
from itertools import chain
import scanpy as sc
from anndata import AnnData

def remove_colorbars(fig, N):
    """
    Remove colorbars from N axes of a UMAP figure, in order of appearance.
    """
    num_removed = 0
    for ax in fig.axes:
        if (
            ax.collections
            and hasattr(ax.collections[-1], "colorbar")
            and ax.collections[-1].colorbar
        ):
            ax.collections[-1].colorbar.remove()
            num_removed += 1
        if num_removed == N:
            break  # didn't work to iterate to axes[:-1] so manually check

    return fig


def add_colorbar(fig, position, **kwargs):
    """
    Easier addition of a master colorbar for a grid of UMAP plots or similar
    """
    cmap = kwargs.pop("cmap", "viridis")
    label = kwargs.pop("label", "Expression")
    label_rotation = kwargs.pop("label_rotation", -90)

    cbar = fig.axes[-1].collections[-1]  # type: ignore # grabs useful info from existing plots, but not necessarily a colorbar itself
    cbar_ax = fig.add_axes(position)  # type: ignore  # [left, bottom, width, height]
    cbar_ax.set_axis_off()
    cbar = fig.colorbar(
        cbar,
        ax=cbar_ax,
        cmap=cmap,
        **kwargs,  # pass through remaining keyword arguments
    )  # cbar is actually pointing to the colorbar now

    # Adjust paddings/alignments to look nice
    match label_rotation:
        case 90:
            cbar.ax.set_ylabel(label, rotation=90)  # default labelpad
        case 0:
            cbar.ax.set_ylabel(label, rotation=0, ha="left", va="center")
        case -90:
            cbar.ax.set_ylabel(label, rotation=-90, ha="center", va="bottom")
        case _:
            cbar.ax.set_ylabel(label, rotation=label_rotation)


def update_titles(fig, titles):
    """
    Update titles of a UMAP figure with a list of titles.
    """
    axs = fig.get_axes()

    # Update titles of all axes in the main plot
    i = 0
    for ax in fig.axes:
        if ax.get_title():  # Check if the title is already set
            ax.set_title(titles[i])
            i += 1
            if i >= len(titles):
                break  # safety in case titles is shorter than # axes

    return fig


def rotate_labels(figdp, xticklabels=None, xtickrotation=45, fontsize=None):
    """
    Rotate x-axis labels and group labels of a dotplot figure for better readability.
    """
    ax = figdp.get_axes()

    if not fontsize:
        fontsize = plt.rcParams["font.size"]

    # Rotate x-axis labels to 45deg for readability
    if xticklabels is not None:
        ax["mainplot_ax"].set_xticklabels(xticklabels)
    plt.setp(
        ax["mainplot_ax"].get_xticklabels(),
        rotation=xtickrotation,
        ha="right",
        fontsize=fontsize,
    )

    # Change rotation of all text elements in ax['gene_group_ax'] children inplace
    group_labels = [ch for ch in ax["gene_group_ax"].get_children() if isinstance(ch, plt.Text)]  # type: ignore
    for label in group_labels:
        if hasattr(label, "set_rotation"):
            label.set_rotation(0)
            label.set_fontsize(fontsize)

    ax["mainplot_ax"].set_yticklabels(
        ax["mainplot_ax"].get_yticklabels(), fontsize=fontsize
    )

    return figdp


def compute_dendrogram(adata_in: AnnData, signats: dict[str, list[str]] | list[str], groupby: str):
    """
    Compute dendrogram for a subset of genes/proteins (usually to be plotted)

    Inputs:
    - adata_in: AnnData object with data
    - signats: either a list of genes/proteins or a dictionary of groups of genes/proteins
    - groupby: observation to group by for dendrogram computation
    
    """

    # If genes are grouped, this flattens them and necessarily gets unique ones
    if isinstance(signats, dict):
        genes = list(chain.from_iterable(signats.values()))
    
    # Get unique genes
    unique_genes = list(set(genes))
    
    # Don't modify existing adata as it will cause issues w/ storing the dendrogram
    adata_out = adata_in[:, unique_genes].copy()  # type: ignore # defaults to using the log1p layer
    sc.tl.dendrogram(adata_out, groupby=groupby, var_names=unique_genes)  # type: ignore

    return adata_out
