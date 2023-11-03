# %%
import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc

from cpt.benchmark.jax_crt import JAXCRT
from cpt.benchmark.ols import LogisticRegression
import uuid
from sklearn.neighbors import BallTree
import plotnine as p9


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".45"
# %%
def determine_sample_valid(my_ser):
    has_lymphos = my_ser["lympho"] >= 1
    has_tumors = my_ser["tumor"] >= 1
    return has_lymphos & has_tumors


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--savedir", type=str)
    args.add_argument("--xy_include_batch", action="store_true", default=False)
    args.add_argument("--foreground_genes", type=str, default=None)
    return vars(args.parse_args())


args = parse_args()

# %%
EXPERIMENT_ID = str(uuid.uuid4())
SAVEDIR = args["savedir"]
FOREGROUND_GENES = args["foreground_genes"]
EXPERIMENT_DIR = os.path.join(SAVEDIR, EXPERIMENT_ID)

DATASET = "all"

os.makedirs(EXPERIMENT_DIR)

batch_key = None
adata = sc.read_h5ad(
    "nanostring_finished_hotspot_densities_tumorcells.h5ad"
)
adata.X = np.asarray(adata.layers["counts"].todense())

if FOREGROUND_GENES is not None:
    foreground_genes = pd.read_csv(FOREGROUND_GENES, index_col=0).values.squeeze()
    adata = adata[:, foreground_genes].copy()

adata.obs.loc[:, "dataset"] = "all"
XY_INCLUDE_BATCH = False

mapper_to_lymphoids = lambda x: x["celltypes_coarse_hand"].isin(["Bcell", "Tcell"])
mapper_to_tumors = lambda x: x["celltypes_coarse_hand"].isin(["Tumor_cell"])
spatial_keys = ["x", "y"]


# %%
CRT_KWARGS = dict(
    n_epochs=10000,
    percent_dev=0.5,
    n_mc_samples=5000,
    x_lr=1e-4,
    xy_lr=1e-4,
    batch_size=128,
    xy_batch_size=128,
    n_latent=10,
    n_epochs_kl_warmup=1,
    xy_patience=25,
    x_patience=25,
    xy_dropout_rate=0.0,
    x_dropout_rate=0.0,
    xy_include_batch=args["xy_include_batch"],
    xy_loss="binary",
)
GRID_PARAMETERS = dict(
    n_hidden_xy=[8, 16, 32, 64, 128, 256],
)

# Load data
normalized_counts = adata.X.copy()
adata.X = adata.layers["counts"]

datasets = adata.obs.dataset.unique()
for dataset in datasets:
    adata_ = adata[adata.obs["dataset"] == dataset].copy()
    spatial_coor = adata_.obs[spatial_keys]
    bt_lympho = BallTree(spatial_coor[mapper_to_lymphoids(adata_.obs)])
    ns_lympho = bt_lympho.query_radius(spatial_coor, r=750, count_only=True)
    ns_lympho = ns_lympho >= 100

    bt_tumor = BallTree(spatial_coor[mapper_to_tumors(adata_.obs)])
    ns_tumor = bt_tumor.query_radius(spatial_coor, r=750, count_only=True)
    ns_tumor = ns_tumor >= 100

    regions = np.array(adata_.shape[0] * ["background"])
    regions[ns_lympho & (~ns_tumor)] = "lympho"
    regions[ns_tumor & (~ns_lympho)] = "tumor"
    regions[ns_lympho & ns_tumor] = "lympho"
    adata_.obs["area_type"] = regions
    adata.obs.loc[adata_.obs.index, "area_type"] = regions

    fig = (
        p9.ggplot(
            adata_.obs,
            p9.aes(*spatial_keys, color="area_type"),
        )
        + p9.geom_point(stroke=0.0, size=1.0)
        + p9.labs(title=dataset)
    )
    fig.draw()

# %%
good_datasets = (
    adata.obs.groupby("dataset")
    .area_type.value_counts()
    .unstack()
    .fillna(0)
    .apply(determine_sample_valid, axis=1)
    .loc[lambda x: x]
    .index.values
)

proper_sub = (
    adata.obs["dataset"].isin(good_datasets)
    # & (adata.obs["celltypes_coarse_hand"] == "Myeloid")
    & (adata.obs["area_type"].isin(["lympho", "tumor"]))
)
adata.write_h5ad(os.path.join(EXPERIMENT_DIR, "adata.all.h5ad"))
adata = adata[proper_sub].copy()
adata = adata[adata.obs["celltypes_coarse_hand"] == "Tcell"].copy()
sc.pp.filter_cells(adata, min_counts=10)
adata.obsm["protein_expression"] = (adata.obs.area_type == "lympho").values[:, None]
# %%
# final_parameters = select_architecture(
#     JAXCRT,
#     adata,
#     xy_base_params=CRT_KWARGS,
#     xy_grid=GRID_PARAMETERS,
# )
# final_parameters_ser = pd.Series(final_parameters)
# final_parameters_ser.to_pickle(os.path.join(EXPERIMENT_DIR, "final_parameters.pickle"))

# # Run CPT
crttool = JAXCRT(adata, batch_key=batch_key, n_hidden_xy=8, **CRT_KWARGS)
# crttool = JAXCRT(adata, batch_key=batch_key, n_hidden_xy=8, **CRT_KWARGS)
crttool.train_all()
# crttool.save_params(save_path=EXPERIMENT_DIR, save_adata=False)

# %%
# Compute pvalues for the overall considered dataset
all_res_b = pd.DataFrame()

is_t = adata.obs["celltypes_coarse_hand"] == "Tcell"
for mask, name in zip(
    [is_t,],
    ["Tcell",],
):
    all_mask = mask & (~crttool.adata.obs["is_dev"].values)
    eval_adata = adata[all_mask].copy()
    res = crttool.get_importance(eval_adata)
    res_df = pd.DataFrame(
        dict(
            pvalue=res["pvalues"].squeeze(),
            padj=res["padj"].squeeze(),
            gene=adata.var.index.values,
        )
    ).assign(
        model="CRT",
        subpopulation=name,
        train_data=DATASET,
        experiment_id=EXPERIMENT_ID,
        xy_include_batch=args["xy_include_batch"],
    )
    all_res = crttool.get_hier_importance(
        n_clusters=[25, 50, 100, 150, 200],
        eval_adata=eval_adata,
    )
    res_hier = all_res["gene_results"]
    res_hier.to_netcdf(os.path.join(EXPERIMENT_DIR, f"hier_results_{name}.nc"))
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

    all_res_b = all_res_b.append(res_df)
    all_mask = mask
    adata_ = adata[all_mask].copy()
    ols = LogisticRegression(adata=adata_)
    ols_res = ols.compute().assign(
        gene_id=lambda x: x.gene_group_id,
        gene=adata.var_names,
        model="logistic_regression",
        subpopulation=name,
        train_data=DATASET,
        experiment_id=EXPERIMENT_ID,
        xy_include_batch=args["xy_include_batch"],
    )
    all_res_b = all_res_b.append(ols_res)

all_res_b.to_pickle(os.path.join(EXPERIMENT_DIR, "results_dataset_specific.pkl"))
adata.write_h5ad(os.path.join(EXPERIMENT_DIR, "adata.h5ad"))

# %%
print(EXPERIMENT_DIR)
# %%
