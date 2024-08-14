# %%
import argparse
import os
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.cluster import KMeans
from scipy.sparse import diags
from cpt.benchmark import JAXCRT, OLS, ElementWiseOLS
import pickle
from tqdm import tqdm
from datetime import datetime

from cpt import select_architecture


def save_obj(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default="results/perturbseq2")
    parser.add_argument("--sgrna", type=str, default="CD47")
    parser.add_argument("--xy_mode", type=str)
    parser.add_argument("--n_genes", type=int, default=2000)
    parser.add_argument("--train_frac", type=float, default=0.7)
    return vars(parser.parse_args())


# %%
args = parse_args()
BASEDIR = args["basedir"]
selected_sgrna = args["sgrna"]
xy_mode = args["xy_mode"]
N_GENES = args["n_genes"]
if xy_mode == "nn":
    model_kwargs = {"n_hidden_xy": 8,}
else:
    model_kwargs = {"xy_linear": True}
train_frac = args["train_frac"]
# EXPERIMENT_NAME = datetime.today().strftime('%m%d_%Y')
EXPERIMENT_NAME = f"summed_{selected_sgrna}_{xy_mode}_G{N_GENES}_dev{train_frac}V2"
EXPERIMENT_DIR = os.path.join(BASEDIR, EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# %%
CRT_KWARGS = dict(
    n_epochs=10000,
    # n_epochs=10,
    batch_key=None,
    percent_dev=train_frac,
    n_mc_samples=15000,
    # n_mc_samples=10,
    x_lr=1e-4,
    xy_lr=1e-4,
    batch_size=128,
    xy_batch_size=128,
    n_latent=10,
    n_epochs_kl_warmup=50,
    xy_patience=25,
    x_patience=25,
    xy_dropout_rate=0.0,
    x_dropout_rate=0.0,
)

# %%
# Import data
path_to_data = "frangieh2021.h5ad"
adata_log = sc.read_h5ad(path_to_data)
adata_log = adata_log[adata_log.obs["condition"] == "Control"]

# # %%
library = adata_log.obs["UMI_count"].values.astype(float)
library_diag = diags(library)
y_ = np.expm1(adata_log.X)

counts_ = library_diag.dot(y_) * 1e-6
counts_ = np.round(counts_)
# print((counts_ < 0).sum())
# numerical_errors = (y_.sum(1) - 1e6) / 1e6
# plt.hist(numerical_errors, bins=100);
# new_lib = counts_.sum(1).A1
# plt.scatter(new_lib, library)
adata = sc.AnnData(X=counts_, obs=adata_log.obs, var=adata_log.var)

# %%
sc.pp.filter_genes(adata, min_cells=1e3)
adata_log = adata.copy()
sc.pp.filter_genes(adata_log, min_cells=1e3)
sc.pp.normalize_total(adata_log)
sc.pp.log1p(adata_log)

# %%
# Generating ground-truth

# %%
pd.set_option("display.max_rows", 100)
adata_log.obs["sgRNA"].value_counts().head(20)

# %%
for sgrna in adata_log.obs["sgRNA"].value_counts().head(40).index:
    sgrna_ = sgrna.split("_")[0]
    matches = adata_log.var[adata_log.var.index.str.contains(sgrna_)]
    
    n_matches = adata_log.obs["sgRNA"].value_counts()[sgrna]
    
    print(sgrna, matches.shape[0], n_matches)


# %%
# adata_log.obs["sgRNA"].value_counts().to_csv(os.path.join(EXPERIMENT_DIR, "sgRNA_counts.csv"), sep="\t")

# %%
pop1 = adata_log.obs["sgRNA"].str.startswith(selected_sgrna).fillna(False)
pop2 = adata_log.obs["sgRNA"].isna()
adata1 = adata_log[pop1]
adata2 = adata_log[pop2]
print(adata1.shape)
print(adata2.shape)
a = adata1.X.toarray()
b = adata2.X.toarray()
v = mannwhitneyu(a, b, axis=0)
t = ttest_ind(a, b, axis=0)
de_results = pd.DataFrame(
    {
        "gene": adata_log.var_names,
        "mannwhitney_pvalue": v.pvalue,
        "mannwhitney_statistic": v.statistic,
        "ttest_pvalue": t.pvalue,
        "ttest_statistic": t.statistic,
    }
)

de_results.loc[:, "mannwhitney_padj"] = multipletests(de_results.loc[:, "mannwhitney_pvalue"], method="fdr_bh")[1]
de_results.loc[:, "mannwhitney_is_de"] = de_results["mannwhitney_padj"] <= 0.1
de_results.loc[:, "ttest_padj"] = multipletests(de_results.loc[:, "ttest_pvalue"], method="fdr_bh")[1]
de_results.loc[:, "ttest_is_de"] = de_results["ttest_padj"] <= 0.1
de_results.loc[:, "ttest_sig"] = - np.log10(de_results["ttest_padj"])

(de_results["ttest_is_de"]).sum(), (de_results["mannwhitney_is_de"]).sum()

# %%
# Setting up training data
## Gene clustering
sc.pp.pca(adata_log, n_comps=50)
pcs = adata_log.varm["PCs"]
clusters = KMeans(n_clusters=N_GENES, random_state=0).fit_predict(pcs)
adata.var.loc[:, "cluster"] = clusters
sc.pp.highly_variable_genes(adata, flavor="seurat_v3")
selected_genes = (
    adata.var
    .groupby("cluster")
    .apply(
        lambda x: x.reset_index().sort_values("variances_norm").iloc[0]["gene"]
    )
)


# %%
de_results.loc[:, "cluster_id"] = clusters
de_results_ = (
    de_results
    .merge(selected_genes.to_frame("cluster_gene"), left_on="cluster_id", right_index=True)
)
de_results_.to_csv(os.path.join(EXPERIMENT_DIR, "de_results.csv"), sep="\t")

# %%
selected_cells = adata_log.obs["sgRNA"].isna()
adata_selected = adata[selected_cells]

adata_final = sc.read("frangieh2021_clustered.h5ad")


# %%
target_gene_exp = adata_log[selected_cells, selected_sgrna].X.toarray()
adata_final.obsm["protein_expression"] = target_gene_exp
print(adata_final.shape, target_gene_exp.shape)


# %%
crttool = JAXCRT(
    adata_final,
    **CRT_KWARGS,
    **model_kwargs
)
crttool.train_all()
train_adata = crttool.adata[crttool.adata.obs["is_dev"]].copy()
eval_adata = crttool.adata[(~crttool.adata.obs["is_dev"].values)].copy()


# %%
all_res = crttool.get_hier_importance(
    n_clusters=[100, 200],
    eval_adata=eval_adata,
    batch_size=128,
)
    # %%
save_obj(all_res, os.path.join(EXPERIMENT_DIR, "crt_results.pickle"))

# %%
ols = OLS(adata=adata_final)
ols_res = ols.compute()
save_obj(ols_res, os.path.join(EXPERIMENT_DIR, "ols_results.pickle"))
# %%
em_ols = ElementWiseOLS(adata=adata_final)
em_ols_res = em_ols.compute()
save_obj(em_ols_res, os.path.join(EXPERIMENT_DIR, "em_ols_results.pickle"))
# %%