from itertools import combinations

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scvi.data import pbmc_seurat_v4_cite_seq
from sklearn.metrics import pairwise_distances

from cpt._utils import select_genes

# from scvi.data import setup_anndata


def get_gene_groupings(
    adata_log,
    max_corr=None,
):
    n_genes = adata_log.X.shape[1]
    if max_corr is None:
        return np.arange(n_genes).tolist()

    corr_structure = np.corrcoef(adata_log.X.T)
    corr_structure[np.isnan(corr_structure)] = 0.0
    all_pairs = (
        pd.DataFrame(corr_structure)
        .unstack()
        .to_frame("score")
        .reset_index()
        .rename(columns={"level_0": "gene_a", "level_1": "gene_b"})
        .astype(dict(gene_a=int, gene_b=int))
        .set_index(["gene_a", "gene_b"])
    )

    bad_pairs = (
        all_pairs.reset_index()
        .query("gene_a > gene_b")
        .loc[lambda x: np.abs(x.score) >= max_corr]
    )

    def get_corr_module(all_pairs, gene_list):
        combs = list(combinations(gene_list, 2))
        return all_pairs.loc[combs]

    gene_assignments = np.arange(n_genes).tolist()
    gene_assignments = [[gene] for gene in gene_assignments]
    for _, pair in bad_pairs.iterrows():
        gene_a = int(pair.gene_a)
        gene_b = int(pair.gene_b)

        idx_a = [idx for (idx, li) in enumerate(gene_assignments) if (gene_a in li)]
        assert len(idx_a) == 1
        if gene_b in gene_assignments[idx_a[0]]:
            continue
        li_a = gene_assignments.pop(idx_a[0])
        idx_b = [idx for (idx, li) in enumerate(gene_assignments) if (gene_b in li)]

        assert len(idx_b) == 1
        li_b = gene_assignments.pop(idx_b[0])
        li = li_a + li_b

        scores = get_corr_module(all_pairs, li).score.abs()
        if (scores >= max_corr).all():
            gene_assignments.append(li)
        else:
            gene_assignments.append(li_a)
            gene_assignments.append(li_b)
    return gene_assignments


def generate_totalseq_semisynth(
    frac=0.05,
    snr=2.0,
    seed=0,
    n_top_genes=1000,
    mode="linear",
    noise="gaussian",
    n_proteins=100,
):
    """Generates total-seq synthetic data

    Parameters
    ----------
    frac : float, optional
        Sparsity level, in fraction, by default 0.05
    snr : float, optional
        Signal to noise ration, by default 2.0
    seed : int, optional
        Default seed, by default 0
    n_top_genes : int, optional
        by default 1000
    mode : str, optional
        Non-linearity to use, by default "linear"

    """
    np.random.seed(seed)
    adata = sc.read(
        "pbmc_10k_protein_v3.h5ad",
        backup_url="https://github.com/YosefLab/scVI-data/raw/master/pbmc_10k_protein_v3.h5ad",
    )
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
    adata = adata[:, adata.var.highly_variable].copy()

    n_obs = adata.obsm["protein_expression"].shape[0]
    n_genes = adata.X.shape[1]

    _adata_log = adata.copy()
    sc.pp.normalize_total(_adata_log, target_sum=1e6)
    sc.pp.log1p(_adata_log)

    # For foreground only
    gene_has_influence = np.random.random(size=(n_genes, n_proteins)) <= frac
    # rescaling
    if mode == "linear":
        coefficients = np.random.randn(n_genes, n_proteins)
        coefficients = gene_has_influence * coefficients
        protein_concentrations = _adata_log.X @ coefficients
        var_signal = protein_concentrations.var(0)
    elif mode == "polynomial":
        coefficients2 = np.random.randn(n_genes, n_proteins)
        coefficients2 = gene_has_influence * coefficients2
        protein_concentrations = (_adata_log.X**2) @ coefficients2

        coefficients1 = np.random.randn(n_genes, n_proteins)
        coefficients1 = gene_has_influence * coefficients1
        protein_concentrations += _adata_log.X @ coefficients1

        var_signal = np.maximum(
            ((_adata_log.X**2) @ coefficients2).var(0),
            (_adata_log.X @ coefficients1).var(0),
        )
    elif mode == "nn":
        protein_concentrations = np.zeros((n_obs, n_proteins))
        for prot in range(n_proteins):
            coefficients = np.random.randn(n_genes, 32)
            a_mat = gene_has_influence[:, [prot]] * coefficients
            y_rep = _adata_log.X @ a_mat
            y_rep = np.maximum(y_rep, 0)
            b_mat = np.random.randn(32, 1)
            y_rep = y_rep @ b_mat

            y_rep = y_rep.squeeze()
            y_rep = (y_rep - y_rep.mean()) / y_rep.std()
            protein_concentrations[:, prot] = y_rep
        var_signal = protein_concentrations.var(0)

    else:
        raise ValueError("Mode {} not implemented".format(mode))

    std_noise = np.sqrt(var_signal / snr)
    if noise == "gaussian":
        noise_ = std_noise * np.random.randn(n_obs, n_proteins)
    else:
        # var = 2 b^2
        b = std_noise / np.sqrt(2)
        noise_ = np.random.laplace(0, scale=b, size=(n_obs, n_proteins))

    protein_concentrations = protein_concentrations + noise_
    protein_concentrations = protein_concentrations - np.mean(protein_concentrations, 0)
    protein_concentrations = protein_concentrations / np.std(protein_concentrations, 0)

    # protein_concentrations = protein_concentrations - protein_concentrations.min(0)
    # protein_concentrations = 1 + 3 * protein_concentrations / protein_concentrations.max(0)

    adata.obsm["protein_expression"] = protein_concentrations

    is_dev = np.random.random(n_obs) >= 0.5
    adata.obs.loc[:, "is_dev"] = is_dev

    # adata_dev = adata[is_dev].copy()
    adata_log = adata.copy()
    sc.pp.normalize_total(adata_log, target_sum=1e6)
    sc.pp.log1p(adata_log)

    return dict(
        adata_log=adata_log,
        adata_main=adata[~is_dev],
        adata_dev=adata[is_dev],
        adata=adata,
        adata_dev_log=adata_log[is_dev],
        n_obs=n_obs,
        n_genes=n_genes,
        n_proteins=n_proteins,
        gene_has_influence=gene_has_influence,
    )


def generate_totalseq_semisynth2(
    frac=0.05,
    seed=0,
    n_top_genes=1000,
    noise="gaussian",
    beta_scale=1.0,
    elementwise_nonlinearity="id",
    nonlinearity="id",
    n_proteins=100,
    largen=False,
    min_cells_expressing=1000,
    sigma=None,
):
    np.random.seed(seed)
    if largen:
        adata = pbmc_seurat_v4_cite_seq(apply_filters=True, aggregate_proteins=True)
    else:
        adata = sc.read(
            "pbmc_10k_protein_v3.h5ad",
            backup_url="https://github.com/YosefLab/scVI-data/raw/master/pbmc_10k_protein_v3.h5ad",
        )
    print(f"beta_scale: {beta_scale}")
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=min_cells_expressing)
    # sc.pp.filter_genes(adata, min_cells=min_cells_expressing)
    # sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
    # adata = adata[:, adata.var.highly_variable].copy()
    adata = select_genes(
        adata,
        n_top_genes=n_top_genes,
        seed=seed,
    )

    if largen:
        adata.X = adata.X.todense()

    n_obs = adata.X.shape[0]
    n_genes = adata.X.shape[1]
    n_positives = int(n_genes * frac)

    _adata_log = adata.copy()
    _adata_log.X = _adata_log.X + 1
    sc.pp.normalize_total(_adata_log, target_sum=1e6)
    sc.pp.log1p(_adata_log)

    X = _adata_log.X.copy()
    nzm = X.sum(0) / (X != 0).sum(0)
    # nzm = np.median(X, 0)
    # X = (X - X.mean(0)) / X.std(0)
    X = (X - nzm) / X.std(0)
    
    # _adata_log = adata.copy()
    # sc.pp.normalize_total(_adata_log)
    # sc.pp.log1p(_adata_log)
    # X = _adata_log.X.copy()
    # nzm = X.sum(0) / (X != 0).sum(0)
    # X = (X - nzm) / X.std(0)

    if elementwise_nonlinearity == "id":
        pass
        adata.layers["X_transformed"] = X
    elif elementwise_nonlinearity == "square":
        X = X**2
        adata.layers["X_transformed"] = X
    elif elementwise_nonlinearity == "squareb":
        X = X**2 + X + X**3
        adata.layers["X_transformed"] = X
    else:
        raise ValueError(elementwise_nonlinearity)
    X = X / n_positives

    beta_ref = np.sqrt(2.0 * np.log(n_genes))
    beta_val = beta_ref * beta_scale
    sigma = np.sqrt(n_obs) if sigma is None else sigma
    mask = [
        np.random.choice(n_genes, n_positives, replace=False) for _ in range(n_proteins)
    ]
    mask = [np.isin(np.arange(n_genes), _mask)[..., None] for _mask in mask]
    mask = np.concatenate(mask, 1)

    if nonlinearity == "id":
        # beta = np.zeros((n_genes, n_proteins))
        # beta[:n_positives] = beta_val
        beta = mask * beta_val
        beta *= 2.0 * (np.random.rand(n_genes, n_proteins) > 0.5) - 1.0
        means = X @ beta
    elif nonlinearity == "square":
        means = []
        for protein in range(n_proteins):
            xs = X[:, mask[:, protein]]
            # n_obs, n_positives
            # amat = np.random.randn(n_positives, n_positives)
            # n_positives, n_positives
            # amat = amat @ amat.T
            # amat = amat - np.diag(np.diag(amat))

            amat = 0.5 * np.ones((n_positives, n_positives))
            amat = amat - np.diag(np.diag(amat)) + np.eye(n_positives)
            beta = amat * beta_val

            _means = xs @ beta
            _means = (xs * _means).sum(1)
            means.append(_means)
        means = np.stack(means, 1)
    elif nonlinearity == "kernel":
        means = []
        for protein in range(n_proteins):
            xs = X[:, mask[:, protein]]
            # n_obs, n_positives
            dists = pairwise_distances(xs, metric="euclidean")
            k = np.exp(-(dists**2) / 2)
            a = np.random.randn(k.shape[0], 1)
            beta = None
            _means = (k @ a).squeeze()
            means.append(_means)
        means = np.stack(means, 1)

        # n, cells, n_genes_subset, n_proteins
        amat = np.random.randn(n_genes, n_proteins)

    # sigma = 1.0
    if noise == "gaussian":
        obs_noise = np.random.randn(n_obs, n_proteins) * sigma
        y = means + obs_noise
    elif noise == "laplace":
        b = sigma / np.sqrt(2)
        obs_noise = np.random.laplace(0, scale=b, size=(n_obs, n_proteins))
        y = means + obs_noise
    elif noise == "poisson":
        # means = means - means.min() + 1.0
        means = means - means.min(0) + 1e-4
        # y = (
        #     torch.distributions.Poisson(torch.tensor(means, dtype=torch.float32))
        #     .sample()
        #     .numpy()
        # )
        y = np.random.poisson(means)
        adata.obsm["protein_counts"] = y
    elif noise == "poissonb":
        means = np.exp(means)
        y = np.random.poisson(means)
    protein_concentrations = y
    protein_concentrations = protein_concentrations - np.mean(protein_concentrations, 0)
    protein_concentrations = protein_concentrations / np.std(protein_concentrations, 0)

    # protein_concentrations = protein_concentrations - protein_concentrations.min(0)
    # protein_concentrations = 1 + 3 * protein_concentrations / protein_concentrations.max(0)

    adata.obsm["protein_expression"] = protein_concentrations

    is_dev = np.random.random(n_obs) >= 0.5
    adata.obs.loc[:, "is_dev"] = is_dev

    # adata_dev = adata[is_dev].copy()
    adata_log = adata.copy()
    sc.pp.normalize_total(adata_log, target_sum=1e6)
    sc.pp.log1p(adata_log)

    # gene_has_influence = np.zeros((n_genes, n_proteins), dtype=bool)
    # gene_has_influence[:n_positives] = True
    gene_has_influence = mask

    return dict(
        adata_log=adata_log,
        adata_main=adata[~is_dev],
        adata_dev=adata[is_dev],
        adata=adata,
        adata_dev_log=adata_log[is_dev],
        n_obs=n_obs,
        n_genes=n_genes,
        n_proteins=n_proteins,
        gene_has_influence=gene_has_influence,
        X=X,
        beta=beta,
        means=means,
    )
