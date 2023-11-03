import scanpy as sc
import numpy as np
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KernelDensity


def select_genes(
    adata,
    n_top_genes: int,
    preselected_genes: int = None,
    seed: int = 0,
):
    """Selects genes for analysis that are either preselected or highly variable, based on a clustering heuristic.

    Parameters
    ----------
    adata :
    n_top_genes : int
        Number of genes to keep
    preselected_genes : int, optional
        Names of genes to keep, by default None
    seed : int, optional
        Seed, by default 0
    """
    adata_ = adata.copy()
    preselected_genes = preselected_genes if preselected_genes is not None else []

    adata_log = adata_.copy()
    sc.pp.normalize_total(adata_log, target_sum=1e6)
    sc.pp.log1p(adata_log)
    sc.pp.pca(adata_log, n_comps=200)
    sc.pp.highly_variable_genes(
        adata_log, n_top_genes=n_top_genes, flavor="cell_ranger"
    )

    pcs = adata_log.varm["PCs"]
    clusters = KMeans(n_clusters=n_top_genes, random_state=seed).fit_predict(pcs)
    adata_log.var.loc[:, "clusters"] = clusters

    selected_genes = (
        adata_log.var.reset_index()
        .groupby("clusters")
        .apply(lambda x: x.sort_values("dispersions_norm").iloc[-1]["index"])
        .values
    )
    union_genes = np.union1d(selected_genes, preselected_genes)
    return adata[:, adata.var.index.isin(union_genes)].copy()

def one_hot(x):
    return np.eye(x.max() + 1)[x]


def select_architecture(
    model_cls,
    adata,
    xy_base_params: dict,
    xy_grid: dict,
    return_params=True
):
    """Selects architecture for feature selection"""
    all_losses = pd.DataFrame()
    parameter_grid = list(ParameterGrid(xy_grid))
    keys_of_interest = None
    for grid_param in parameter_grid:
        params = {**xy_base_params, **grid_param}
        crt = model_cls(adata=adata, **params)
        crt.train_statistic()
        losses = crt.losses.assign(**grid_param)
        all_losses = pd.concat([all_losses, losses])
        if keys_of_interest is None:
            keys_of_interest = list(grid_param.keys())
    relevant_losses = all_losses.loc[lambda x: x.metric == "stat_val_loss"]
    if return_params:
        best_params = (
            relevant_losses.groupby(keys_of_interest)["value"]
            .min()
            .sort_values()
            .reset_index()
            .loc[:, keys_of_interest]
            .iloc[0]
            .to_dict()
        )
        return {**xy_base_params, **best_params}
    else:
        return relevant_losses


def spatial_get_density(adata, spatial_coor, subpopulation_of_interest, bandwidth=500, compute_relative_density=True, new_obs_name=None):
    bandwidth_str = new_obs_name if new_obs_name is not None else "density_bw{}".format(bandwidth)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
        spatial_coor[subpopulation_of_interest]
    )
    log_density = kde.score_samples(spatial_coor)
    density = np.exp(log_density)
    adata.obs.loc[adata.obs.index, bandwidth_str] = density

    if compute_relative_density:
        kde_overall = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
            spatial_coor
        )
        log_density_all = kde_overall.score_samples(spatial_coor)
        rel_density = np.exp(log_density - log_density_all)
        adata.obs.loc[adata.obs.index, "rel_" + bandwidth_str] = rel_density
