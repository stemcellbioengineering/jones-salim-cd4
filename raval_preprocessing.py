"""
Functions I utilize (broadly speaking) when working with single cell omics data
(usually scRNA-seq). I have written these to be relatively general, such that
they can be utilized for many datasets.

Written by: Divy Raval
Creation Date: 2025/03/21
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import anndata as ad
import scanpy as sc


def identify_rna_outliers(
    adata: ad.AnnData,
    scale_mad: float = 3,
    inplace: bool = True,
    automated: bool = True,
    q_threshold: float = 0.01,  # percent of upper and lower bound
    pct_mito_threshold: float = 15,
):
    """
    Identify possible RNA outliers and include in .obs. Does not remove any
    barcode-observations - that comes later.

    Inspired by https://github.com/mousepixels/sanbomics_scripts/blob/main/sc2024/iterative_preprocessing.ipynb
    """

    ## TODO: automated = False not implemented (no manual identifying within function)
    if not automated:
        raise ValueError("automated must be True. Manual identifying not implemented.")

    # if user has indicated to not store within the passed adata object
    if not inplace:
        adata = adata.copy()

    ## compute general metrics
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mito", "ribo"], percent_top=(20,), inplace=True, log1p=False
    )

    ## compute outliers
    # empty cells
    metrics = ["total_counts", "n_genes_by_counts"]
    outliers_empty = np.zeros(shape=(adata.shape[0]), dtype=bool)
    for e in metrics:
        # logical OR as either case is sufficient
        outliers_empty = outliers_empty | (
            adata.obs[e]
            < max(
                np.median(adata.obs[e])
                - scale_mad * stats.median_abs_deviation(adata.obs[e]),
                adata.obs[e].quantile(q_threshold),
            )
        )
    adata.obs["outliers_empty"] = outliers_empty

    # other outliers -- multiplets, poorly sequenced cells
    metrics = ["total_counts", "n_genes_by_counts", "pct_counts_in_top_20_genes"]
    outliers_multiplets = np.zeros(shape=(adata.shape[0]), dtype=bool)
    for e in metrics:
        # logical OR as either case is sufficient
        outliers_multiplets = outliers_multiplets | (
            adata.obs[e]
            > min(
                np.median(adata.obs[e])
                + scale_mad * stats.median_abs_deviation(adata.obs[e]),
                adata.obs[e].quantile(1 - q_threshold),
            )
        )
    adata.obs["outliers_multiplets"] = outliers_multiplets

    # include hard coding mitocondrial threshold just incase
    outliers_mito = (adata.obs["pct_counts_mito"] > pct_mito_threshold) | (
        adata.obs["pct_counts_mito"]
        > np.median(adata.obs["pct_counts_mito"])
        + scale_mad * stats.median_abs_deviation(adata.obs["pct_counts_mito"])
    )

    adata.obs["outliers_mito"] = outliers_mito

    outliers_ribo = adata.obs["pct_counts_ribo"] > np.median(
        adata.obs["pct_counts_ribo"]
    ) + scale_mad * stats.median_abs_deviation(adata.obs["pct_counts_ribo"])
    adata.obs["outliers_ribo"] = outliers_ribo

    return adata


def visualize_rna_outliers(
    adata: ad.AnnData,
    without_outliers: list[str] = [],
    show: bool = True,
    save_suffix: str | None = None,
    save_dir: str | None = None,
):
    if without_outliers:
        adata = adata.copy()
        outliers = np.zeros(shape=(adata.shape[0]), dtype=bool)
        for e in without_outliers:
            outliers = outliers | (adata.obs[e])
        adata = adata[~outliers]

    save_name = (
        f"{save_dir}/dotplot_rna_outliers_1_{save_suffix}.png" if save_suffix else None
    )
    # Create the violin plot without showing it or saving automatically
    g = sc.pl.violin(
        adata,
        keys=[
            "total_counts",
            "n_genes_by_counts",
            "pct_counts_in_top_20_genes",
            "pct_counts_mito",
            "pct_counts_ribo",
        ],
        groupby=None,  # or groupby='some_obs_column'
        multi_panel=True,
        show=False,
        save=None,  # disable auto-save to control file saving manually
        stripplot=True,  # keep jittered points
    )

    # g is a Seaborn FacetGrid → access and customize it
    for ax in g.axes.flat:
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_title(ax.get_title(), fontsize=15)
        ax.set_xlabel(ax.get_xlabel(), fontsize=15)
        ax.set_ylabel(ax.get_ylabel(), fontsize=15)

    g.fig.tight_layout()

    # Save manually
    g.fig.savefig(f"{save_name}", dpi=600, bbox_inches="tight", transparent=True)

    save_name = f"_rna_outliers_2_{save_suffix}.svg" if save_suffix else None
    sc.pl.scatter(
        adata,
        x="pct_counts_mito",
        y="pct_counts_in_top_20_genes",
        color="total_counts",
        show=show,
        save=save_name,
    )

    save_name = f"_rna_outliers_3_{save_suffix}.svg" if save_suffix else None
    sc.pl.scatter(
        adata,
        x="pct_counts_mito",
        y="pct_counts_ribo",
        color="total_counts",
        show=show,
        save=save_name,
    )

    save_name = f"_rna_outliers_4_{save_suffix}.svg" if save_suffix else None
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        color="pct_counts_mito",
        show=show,
        save=save_name,
    )
    return


def normalize_rna_sample(adata: ad.AnnData, inplace: bool, target_counts=1e4, base=2):
    if not inplace:
        adata = adata.copy()

    # perform normalization, each cell will sum to L
    sc.pp.normalize_total(
        adata, target_sum=target_counts, inplace=True, exclude_highly_expressed=False
    )

    # perform log_base + 1 transformation
    sc.pp.log1p(adata, base=base, copy=False)
    return adata


def cell_cycle_regress_rna_sample(
    adata: ad.AnnData, genes: dict, inplace: bool = True, njobs: int = 16
):
    """ """
    if not inplace:
        adata = adata.copy()

    # score cell cycle phases
    sc.tl.score_genes_cell_cycle(
        adata, s_genes=genes["s_genes"], g2m_genes=genes["g2m_genes"]
    )

    # regress out cell cycle effects
    sc.pp.regress_out(adata, ["S_score", "G2M_score"], n_jobs=njobs)

    # scale the data after regression
    sc.pp.scale(adata, max_value=10)
    return adata


def identify_adt_outliers(
    adata: ad.AnnData,
    scale_mad: float = 3,
    inplace: bool = True,
    automated: bool = True,
    q_threshold: float = 0.01,  # percent of upper and lower bound
    pct_mito_threshold: float = 15,
):
    """
    Identify possible ADT outliers and include in .obs. Does not remove any
    barcode-observations - that comes later.

    Inspired by https://github.com/mousepixels/sanbomics_scripts/blob/main/sc2024/iterative_preprocessing.ipynb
    """

    ## TODO: automated = False not implemented (no manual identifying within function)
    if not automated:
        raise ValueError("automated must be True. Manual identifying not implemented.")

    # if user has indicated to not store within the passed adata object
    if not inplace:
        adata = adata.copy()

    ## compute general metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=(20,), inplace=True, log1p=False)

    ## compute outliers
    # empty cells
    metrics = ["total_counts"]
    outliers_empty = np.zeros(shape=(adata.shape[0]), dtype=bool)
    for e in metrics:
        # logical OR as either case is sufficient
        outliers_empty = outliers_empty | (
            adata.obs[e]
            < max(
                np.median(adata.obs[e])
                - scale_mad * stats.median_abs_deviation(adata.obs[e]),
                adata.obs[e].quantile(q_threshold),
            )
        )
    adata.obs["outliers_empty"] = outliers_empty

    # other outliers -- multiplets, poorly sequenced cells
    metrics = ["total_counts", "pct_counts_in_top_20_genes"]
    outliers_multiplets = np.zeros(shape=(adata.shape[0]), dtype=bool)
    for e in metrics:
        # logical OR as either case is sufficient
        outliers_multiplets = outliers_multiplets | (
            adata.obs[e]
            > min(
                np.median(adata.obs[e])
                + scale_mad * stats.median_abs_deviation(adata.obs[e]),
                adata.obs[e].quantile(1 - q_threshold),
            )
        )
    adata.obs["outliers_multiplets"] = outliers_multiplets

    return adata


def visualize_adt_outliers(
    adata: ad.AnnData,
    without_outliers: list[str] = [],
    show: bool = True,
    save_prefix=None,
):
    if without_outliers:
        adata = adata.copy()
        outliers = np.zeros(shape=(adata.shape[0]), dtype=bool)
        for e in without_outliers:
            outliers = outliers | (adata.obs[e])
        adata = adata[~outliers]

    save_name = f"_adt_outliers_1_{save_prefix}" if save_prefix else None
    sc.pl.violin(
        adata,
        keys=[
            "total_counts",
            "n_genes_by_counts",
            "pct_counts_in_top_20_genes",
        ],
        multi_panel=True,
        show=show,
        save=save_name,
    )

    save_name = f"_adt_outliers_2_{save_prefix}" if save_prefix else None
    sc.pl.scatter(
        adata,
        x="n_genes_by_counts",
        y="pct_counts_in_top_20_genes",
        color="total_counts",
        show=show,
        save=save_name,
    )

    save_name = f"_adt_outliers_3_{save_prefix}" if save_prefix else None
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        color="pct_counts_in_top_20_genes",
        show=show,
        save=save_name,
    )
    return
