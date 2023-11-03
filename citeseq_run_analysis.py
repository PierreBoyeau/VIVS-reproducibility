# %%
import os
import uuid

import matplotlib.pyplot as plt
import plotnine as p9
import numpy as np
import xarray as xr
import argparse
import pandas as pd
import scanpy as sc
from scvi.data import pbmc_seurat_v4_cite_seq
from sklearn.cluster import KMeans

from cpt import select_architecture
from cpt.benchmark import JAXCRT, OLS


def parse_kwargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--subsampletzero", action="store_true", default=False)
    parser.add_argument("--percent_dev", type=float, default=0.9)
    return vars(parser.parse_args())

# %%
args = parse_kwargs()
BASEDIR = args["basedir"]
PERCENT_DEV = args["percent_dev"]
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
# %%

N_CLUSTERS = [100, 250, 500, 1000]
RELOAD_MODEL = False

BATCH_KEY = "lane"
CRT_KWARGS = dict(
    n_epochs=10000,
    batch_key=BATCH_KEY,
    percent_dev=PERCENT_DEV,
    n_mc_samples=1000,
    x_lr=1e-3,
    xy_lr=1e-3,
    batch_size=128,
    xy_batch_size=128,
    n_latent=10,
    n_epochs_kl_warmup=50,
    xy_patience=25,
    x_patience=25,
    xy_dropout_rate=0.0,
    x_dropout_rate=0.0,
)
GRID_PARAMETERS = dict(
    n_hidden_xy=[8, 16, 32, 64, 128, 256],
)
EXPERIMENT_ID = str(uuid.uuid4())
os.makedirs(BASEDIR, exist_ok=True)
EXPERIMENT_DIR = os.path.join(BASEDIR, EXPERIMENT_ID)
FIGURE_DIR = "./figures/citeseq"
CELLTYPE_KEY = "celltype.l1"
CRT_BATCH_SIZE = 128
# %%
os.makedirs(EXPERIMENT_DIR)
# %%
processed_adata_path = os.path.join(BASEDIR, "adata.h5ad")
if os.path.exists(processed_adata_path):
    adata = sc.read_h5ad(processed_adata_path)
else:
    adata = pbmc_seurat_v4_cite_seq(apply_filters=True, aggregate_proteins=True)
    adata = adata.copy()  # avoid fragmentation

    if args["subsampletzero"]:
        print("Subsampling t=0")
        adata = adata[adata.obs.time == "0"].copy()
    # adata.obsm["protein_counts"] = adata.obsm["protein_counts"].loc[:, SELECTED_PROTEINS]

    prot_names = adata.obsm["protein_counts"].columns.values
    preselected_genes = []
    for prot in prot_names:
        if (adata.var.index == prot).any():
            preselected_genes.append(prot)
    seed = 0
    np.random.seed(seed)
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=1000)

    adata_log = adata.copy()
    sc.pp.normalize_total(adata_log, target_sum=1e6)
    sc.pp.log1p(adata_log)
    sc.pp.pca(adata_log, n_comps=200)
    sc.pp.highly_variable_genes(adata_log, n_top_genes=1000, flavor="cell_ranger")
    # sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor="seurat_v3")
    pcs = adata_log.varm["PCs"]
    clusters = KMeans(n_clusters=2000, random_state=seed).fit_predict(pcs)
    adata.var.loc[:, "clusters"] = clusters
    adata_log.var.loc[:, "clusters"] = clusters
    selected_genes = (
        adata_log.var.reset_index()
        .groupby("clusters")
        .apply(lambda x: x.sort_values("dispersions_norm").iloc[-1]["index"])
        .values
    )
    # selected_genes = (
    #     adata.var.reset_index()
    #     .groupby("clusters")
    #     .apply(lambda x: x.sort_values("variances_norm").iloc[-1]["index"])
    #     .values

    adata.var.loc[:, "selected_genes"] = adata.var.index.isin(selected_genes)
    adata.var.loc[:, "preselected_genes"] = adata.var.index.isin(preselected_genes)
    VARNAME = "{}/allvars_{}.pickle".format(BASEDIR, EXPERIMENT_ID)
    adata.var.to_pickle(VARNAME)
    good_genes = adata.var.index.isin(preselected_genes) | adata.var.index.isin(
        selected_genes
    )
    adata = adata[:, good_genes].copy()

    adata.obsm["protein_expression"] = adata.obsm["protein_counts"].values
    n_proteins = adata.obsm["protein_expression"].shape[1]
    n_obs = adata.obsm["protein_expression"].shape[0]
    n_genes = adata.X.shape[1]

    # For foreground only
    protein_concentrations = adata.obsm["protein_expression"]
    u_perc = np.percentile(protein_concentrations, 99, axis=0)
    protein_concentrations = np.clip(protein_concentrations, 0, u_perc)
    protein_concentrations = np.log1p(protein_concentrations)
    protein_concentrations = protein_concentrations - np.mean(protein_concentrations, 0)
    protein_concentrations = protein_concentrations / np.std(protein_concentrations, 0)
    protein_concentrations = np.nan_to_num(protein_concentrations)
    adata.obsm["protein_expression"] = protein_concentrations
    adata.write_h5ad(processed_adata_path)

# %%
gene_has_influence = None
n_obs = adata.obsm["protein_expression"].shape[0]
n_genes = adata.X.shape[1]
n_proteins = adata.obsm["protein_expression"].shape[-1]
gene_mapper = adata.var.reset_index().rename(columns={"index": "gene"})
protein_mapper = pd.Series(adata.obsm["protein_counts"].columns.values).to_frame(
    "protein_name"
)
gene_names = adata.var.index.values
protein_names = adata.obsm["protein_counts"].columns

crttool = JAXCRT(
    adata,
    protein_names=protein_names,
    n_hidden_xy=8,
    **CRT_KWARGS
    # **final_parameters,
)
crttool.train_all()
# crttool.save_params(save_path=BASEDIR, save_adata=False)

# %%
train_adata = crttool.adata[crttool.adata.obs["is_dev"]].copy()
eval_adata = adata[(~crttool.adata.obs["is_dev"].values)].copy()
# %%
all_res = crttool.get_hier_importance(
    n_clusters=N_CLUSTERS,
    eval_adata=eval_adata,
    batch_size=CRT_BATCH_SIZE,
)
all_res["gene_results"].to_netcdf(os.path.join(EXPERIMENT_DIR, "crt_results.nc"))
all_res["cluster_results"].to_csv(
    os.path.join(EXPERIMENT_DIR, "crt_cluster_results.csv")
)
all_res["cluster_results"].to_pickle(
    os.path.join(EXPERIMENT_DIR, "crt_cluster_results.pkl")
)
all_res["gene_to_cluster"].to_csv(
    os.path.join(EXPERIMENT_DIR, "crt_gene_to_cluster.csv")
)
all_res["gene_to_cluster"].to_pickle(
    os.path.join(EXPERIMENT_DIR, "crt_gene_to_cluster.pkl")
)
np.save(os.path.join(EXPERIMENT_DIR, "crt_linkage_matrix.npy"), all_res["Z"])

adatab = adata.copy()
ols = OLS(adata=adatab)
ols_res = ols.compute()
ols_res_ = (
    ols_res.merge(
        gene_mapper.loc[:, "gene"],
        left_on="gene_group_id",
        right_index=True,
        how="left",
    )
    .merge(protein_mapper, left_on="protein", right_index=True, how="left")
    .loc[:, ["pvalue", "gene", "protein_name"]]
    .assign(model="ols")
)

res = ols_res_.astype(
    {"gene": "category", "protein_name": "category", "model": "category"}
)
res.to_pickle(os.path.join(EXPERIMENT_DIR, "ols_results.pkl"))


# %%
# res_ct = crttool.get_hier_importance(
#     n_clusters=N_CLUSTERS,
#     eval_adata=eval_adata,
#     batch_size=CRT_BATCH_SIZE,
#     gene_groupings=gene_groupings,
#     group_key=CELLTYPE_KEY,
# )
# res_ct.to_netcdf(os.path.join(EXPERIMENT_DIR, "crt_results_celltype.nc"))

# # %%
# # cell-type specific
# ols_results = []
# for celltype in adata.obs[CELLTYPE_KEY].unique():
#     where_ct = adata.obs[CELLTYPE_KEY] == celltype
#     ct_mask = crttool.adata.obs[CELLTYPE_KEY] == celltype
#     mask = (~crttool.adata.obs["is_dev"]) & ct_mask
#     eval_adata = adata[mask].copy()
#     eval_adata = adata[ct_mask].copy()
#     ols_res = (
#         OLS(eval_adata)
#         .compute()
#         .merge(
#             gene_mapper.loc[:, "gene"],
#             left_on="gene_group_id",
#             right_index=True,
#             how="left",
#         )
#         .merge(protein_mapper, left_on="protein", right_index=True, how="left")
#         .loc[:, ["pvalue", "gene", "protein_name"]]
#         .assign(
#             model="ols",
#             celltype=celltype,
#         )
#     )
#     ols_results.append(ols_res)
# ols_results = pd.concat(ols_results, axis=0)
# ols_results.to_pickle(os.path.join(EXPERIMENT_DIR, "ols_results_celltype.pkl"))
# %%
### Interactive code bits
# # %%
# z = crttool.get_latent()
# adata.obsm["X_scVI"] = z
# sc.pp.neighbors(adata, use_rep="X_scVI")
# sc.tl.umap(adata)
# adata.obs.loc[:, ["UMAP1", "UMAP2"]] = adata.obsm["X_umap"]
# fig = (
#     p9.ggplot(
#         adata.obs,
#         p9.aes(x="UMAP1", y="UMAP2", color="celltype.l1"),
#     )
#     + p9.geom_point(stroke=0.0, size=0.3)
#     + p9.theme_void()
#     + p9.theme(
#         # aspect_ratio=1,
#         legend_position="none"
#     )
# )
# fig.save(os.path.join(FIGURE_DIR, "citeseq_umap_l1.png"), dpi=500, height=3, width=3)
# fig
# # %%
# CM_TO_INCH = 1 / 2.54
# BASE_FIGSIZE = 3.6
# DO_SAVE = True
# save_dir = "figures/citeseq"
# os.makedirs(save_dir, exist_ok=True)

# BASE_THEME = p9.theme(
#     strip_background=p9.element_blank(),
#     subplots_adjust={"wspace": 0.3},
#     panel_background=p9.element_blank(),
#     axis_text=p9.element_text(family="sans-serif", size=7),
#     axis_title=p9.element_text(family="sans-serif", size=8),
#     legend_text=p9.element_text(family="sans-serif", size=5),
# )

# s_obs = adata.obs.groupby("celltype.l1").sample(n=1)
# s_obs

# fig = (
#     p9.ggplot(
#         s_obs,
#         p9.aes(x="UMAP1", y="UMAP2", color="celltype.l1"),
#     )
#     + p9.geom_point(stroke=0.0, size=0.0)
#     + p9.theme_classic()
#     + p9.guides(color=p9.guide_legend(override_aes={"size": 2}))
#     + BASE_THEME
#     + p9.theme(
#         figure_size=(BASE_FIGSIZE * CM_TO_INCH, BASE_FIGSIZE * CM_TO_INCH),
#         axis_ticks=p9.element_blank(),
#         axis_text=p9.element_blank(),
#     )
#     + p9.labs(color="")
# )
# fig.save(os.path.join(FIGURE_DIR, "citeseq_umap_l1_frame.svg"))
# fig

# # %%
# gene_mapper = adata.var.reset_index().rename(columns={"index": "gene"})

# # genes = ["CD19", "CD22", "CD44", "MS4A1"]
# # proteins = ["HLA-DR"]

# genes = ["CD48"]
# proteins = ["CD244"]

# protein_name = proteins[0]

# # %%
# protein_ids = protein_mapper.loc[
#     lambda x: x.protein_name.isin(proteins), :
# ].index.values
# gene_ids = gene_mapper.loc[gene_mapper.gene.isin(genes)].index.values
# eval_adata = adata[(~crttool.adata.obs["is_dev"].values)].copy()

# ts = crttool.get_cell_scores(
#     gene_ids,
#     protein_ids=protein_ids,
#     eval_adata=eval_adata,
#     batch_size=1024,
#     n_mc_samples=100,
# )

# # %%
# ts["tilde_t_mean"].shape, ts["obs_t"].shape
# # %%
# processed_ts = (ts["tilde_t_mean"] - ts["obs_t"]) / ts["obs_t"]
# processed_ts = processed_ts.squeeze()
# quants = np.quantile(processed_ts, 0.80, 0)
# processed_ts = np.clip(processed_ts, 0, quants)
# smoothed_names = [f"{gene} $\\Rightarrow$ {protein_name}" for gene in genes]
# eval_adata.obs.loc[:, smoothed_names] = processed_ts
# smoothed_vals = eval_adata.obs.groupby("celltype.l2")[smoothed_names].median()
# eval_adata.obs.loc[:, smoothed_names] = smoothed_vals.loc[
#     eval_adata.obs["celltype.l2"]
# ].values

# # %%
# for obsname in smoothed_names:
#     fig = (
#         p9.ggplot(
#             eval_adata.obs,
#             p9.aes(x="UMAP1", y="UMAP2", fill=obsname),
#         )
#         + p9.geom_point(colour="black", stroke=0.05, size=0.3)
#         + p9.theme_void()
#         + p9.scale_fill_cmap("Greens")
#         + p9.theme(
#             # aspect_ratio=1,
#             legend_position="none"
#         )
#     )
#     fig.save(
#         os.path.join(FIGURE_DIR, f"citeseq_{obsname}_smoothed.png"),
#         dpi=500,
#         height=3,
#         width=3,
#     )
#     fig.draw()

#     s_obs = eval_adata.obs.groupby("celltype.l1").sample(n=1)
#     fig = (
#         p9.ggplot(
#             s_obs,
#             p9.aes(x="UMAP1", y="UMAP2", fill=obsname),
#         )
#         + p9.geom_point(colour="black", stroke=0.05, size=0.3)
#         + p9.scale_fill_cmap("Greens")
#         + p9.theme_classic()
#         + BASE_THEME
#         + p9.theme(
#             figure_size=(BASE_FIGSIZE * CM_TO_INCH, BASE_FIGSIZE * CM_TO_INCH),
#             axis_ticks=p9.element_blank(),
#             axis_text=p9.element_blank(),
#         )
#     )
#     fig.save(os.path.join(FIGURE_DIR, f"citeseq_{obsname}_smoothed_frame.svg"))
#     fig
# # %%
# plt.show()
# %%
