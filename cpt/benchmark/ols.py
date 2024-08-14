import scanpy as sc
import statsmodels.api as sm
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.sparse import issparse
from joblib import Parallel, delayed


class OLS:
    def __init__(
        self,
        adata,
        from_raw=False,
    ):
        self.adata = adata
        adata_log = adata.copy()
        sc.pp.normalize_total(adata_log, target_sum=1e6)
        sc.pp.log1p(adata_log)
        self.adata_log = adata_log
        self.from_raw = from_raw

    def compute(self):
        X = self.adata_log.X
        if self.from_raw:
            X = self.adata.X
        if issparse(X):
            X = X.toarray()
        n_proteins = self.adata_log.obsm["protein_expression"].shape[-1]
        n_genes = X.shape[-1]
        results = pd.DataFrame()
        gene_group_id = np.arange(n_genes)
        gene_grouping = [[gene] for gene in gene_group_id]
        for protein in tqdm(np.arange(n_proteins)):
            y = self.adata_log.obsm["protein_expression"][:, protein]
            X_ = sm.add_constant(X)
            ols_res = sm.OLS(y, X_).fit()
            pvalues = ols_res.pvalues[1:]
            df_ = (
                pd.Series(pvalues)
                .to_frame("pvalue")
                .assign(
                    gene_grouping=gene_grouping,
                    gene_group_id=gene_group_id,
                    n_clusters=n_genes,
                    protein=protein,
                )
            )
            results = pd.concat([results, df_], axis=0)
        return results.assign(model="ols")


class LogisticRegression:
    def __init__(
        self,
        adata,
        from_raw=False,
    ):
        self.adata = adata
        adata_log = adata.copy()
        sc.pp.normalize_total(adata_log, target_sum=1e6)
        sc.pp.log1p(adata_log)
        self.adata_log = adata_log
        self.from_raw = from_raw

    def compute(self):
        X = self.adata_log.X
        if self.from_raw:
            X = self.adata.X
        if issparse(X):
            X = X.toarray()
        n_proteins = self.adata_log.obsm["protein_expression"].shape[-1]
        n_genes = X.shape[-1]
        results = pd.DataFrame()
        gene_group_id = np.arange(n_genes)
        gene_grouping = [[gene] for gene in gene_group_id]
        for protein in tqdm(np.arange(n_proteins)):
            y = self.adata_log.obsm["protein_expression"][:, protein].astype(np.float64)
            X_ = sm.add_constant(X)
            try:
                ols_res = sm.Logit(y, X_).fit()
            except:
                ols_res = sm.OLS(y, X_).fit()
            pvalues = ols_res.pvalues[1:]
            df_ = (
                pd.Series(pvalues)
                .to_frame("pvalue")
                .assign(
                    gene_grouping=gene_grouping,
                    gene_group_id=gene_group_id,
                    n_clusters=n_genes,
                    protein=protein,
                )
            )
            results = pd.concat([results, df_], axis=0)
        return results.assign(model="logistic_regression")


class GLM:
    def __init__(
        self,
        adata,
    ):
        self.adata = adata
        adata_log = adata.copy()
        sc.pp.normalize_total(adata_log, target_sum=1e6)
        sc.pp.log1p(adata_log)
        self.adata_log = adata_log

    def compute(self):
        X = self.adata_log.X
        if issparse(X):
            X = X.toarray()
        n_proteins = self.adata_log.obsm["protein_counts"].shape[-1]
        n_genes = X.shape[-1]
        results = pd.DataFrame()
        gene_group_id = np.arange(n_genes)
        gene_grouping = [[gene] for gene in gene_group_id]
        for protein in tqdm(np.arange(n_proteins)):
            y = self.adata_log.obsm["protein_counts"][:, protein]
            X_ = sm.add_constant(X)
            ols_res = sm.GLM(y, X_, family=sm.families.Poisson()).fit()
            pvalues = ols_res.pvalues[1:]
            df_ = (
                pd.Series(pvalues)
                .to_frame("pvalue")
                .assign(
                    gene_grouping=gene_grouping,
                    gene_group_id=gene_group_id,
                    n_clusters=n_genes,
                    protein=protein,
                )
            )
            results = pd.concat([results, df_], axis=0)
        return results.assign(model="glm_poisson")


class ElementWiseOLS:
    def __init__(
        self,
        adata,
        n_jobs=2,
    ):
        self.adata = adata
        adata_log = adata.copy()
        sc.pp.normalize_total(adata_log, target_sum=1e6)
        sc.pp.log1p(adata_log)
        self.adata_log = adata_log
        self.n_jobs = n_jobs

    # Parallel(n_jobs=2, prefer="threads")(
    # ...     delayed(sqrt)(i ** 2) for i in range(10))

    def compute(self):
        X = self.adata_log.X
        if issparse(X):
            X = X.toarray()
        n_proteins = self.adata_log.obsm["protein_expression"].shape[-1]
        n_genes = X.shape[-1]
        results = pd.DataFrame()
        gene_group_id = np.arange(n_genes)
        gene_grouping = [[gene] for gene in gene_group_id]

        for protein in tqdm(np.arange(n_proteins)):
            y = self.adata_log.obsm["protein_expression"][:, protein]

            def _fit_gene(gene):
                X_ = X[:, gene]
                X_ = sm.add_constant(X_)
                ols_res = sm.OLS(y, X_).fit()
                pvalues = ols_res.pvalues[1]
                beta = ols_res.params[1]
                return pvalues, beta

            res = []
            for gene in tqdm(np.arange(n_genes)):
                pvalues, beta = _fit_gene(gene)
                res.append([pvalues, beta])
            # res = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            #     delayed(_fit_gene)(gene) for gene in np.arange(n_genes)
            # )
            df_  = (
                pd.DataFrame(res, columns=["pvalue", "beta"])
                .assign(
                    gene_grouping=gene_grouping,
                    gene_group_id=gene_group_id,
                    n_clusters=n_genes,
                    protein=protein,
                )
            )
            results = pd.concat([results, df_], axis=0)
        return results.assign(model="elementwise")
