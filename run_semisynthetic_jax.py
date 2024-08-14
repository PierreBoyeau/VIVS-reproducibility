# %%
import argparse
import os
import uuid

import pandas as pd
import numpy as np

from cpt.benchmark import JAXCRT, OLS, ElementWiseOLS
from cpt.data.protein_synth import generate_totalseq_semisynth2

N_CLUSTERS = [50, 100, 200, 300]

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".45"


# %%
def parse_kwargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity", type=float, default=0.05)
    parser.add_argument("--data_noise", type=str, default="gaussian")
    parser.add_argument("--beta_scale", type=float, default=1.0)
    parser.add_argument(
        "--largen",
        dest="largen",
        default=False,
        action="store_true",
    )
    parser.add_argument("--n_genes", type=int, default=1000)
    parser.add_argument("--n_proteins", type=int, default=100)
    parser.add_argument("--nonlinearity", type=str, default="id")
    parser.add_argument("--elementwise_nonlinearity", type=str, default="id")

    parser.add_argument("--n_latent", type=int, default=10)
    parser.add_argument("--importance_statistic", type=str, default="MLP")
    parser.add_argument("--n_hidden_xy", type=int, default=128)
    parser.add_argument("--n_mc_samples", type=int, default=1000)
    parser.add_argument("--gene_likelihood", type=str, default="nb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--xy_train_frac", type=float, default=None)
    parser.add_argument("--percent_dev", type=float, default=0.5)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default="results/semisynthetic")
    parser.add_argument(
        "--randomforest", dest="randomforest", default=False, action="store_true"
    )
    parser.add_argument("--split_seed", type=int, default=0)
    return vars(parser.parse_args())


args = parse_kwargs()
# args = dict(
#     nonlinearity="id",
#     elementwise_nonlinearity="square",
#     percent_dev=0.7,
#     sparsity=0.3,
#     beta_scale=10000.0,
#     n_genes=100,
#     n_proteins=1,
#     data_noise="gaussian",
#     n_mc_samples=100,
#     n_hidden_xy=64,
#     seed=1,
# )


# %%
print(args)
NONLINEARITY = args.get("nonlinearity", "id")
ELEMENTWISE_NONLINEARITY = args.get("elementwise_nonlinearity", "id")
FRAC = args.get("sparsity", 0.05)
BETA_SCALE = args.get("beta_scale", 1.0)
SIGMA = args.get("sigma", None)
N_GENES = args.get("n_genes", 1000)
N_PROTEINS = args.get("n_proteins", 100)
IMPORTANCE_STATISTIC = args.get("importance_statistic", "MLP")
N_HIDDEN_XY = args.get("n_hidden_xy", 128)
SEED = args.get("seed", 0)
N_LATENT = args.get("n_latent", 10)
N_MC_SAMPLES = args.get("n_mc_samples", 1000)
DATA_NOISE = args.get("data_noise", "gaussian")
PERCENT_DEV = args.get("percent_dev", 0.5)
TAG = args.get("tag", "no")
LARGEN = args.get("largen", False)
BASE_DIR = args.get("base_dir", "results/semisynthetic")
RANDOMFOREST = args.get("randomforest", False)
SPLIT_SEED = args.get("split_seed", 0)

UUID = uuid.uuid4().hex
# BASE_DIR = "results/jax_experiments_finalv1"
# BASE_DIR = "results/CHECKDELETE7_betasign"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
FILENAME = "./{}/experiment_tag_{}_{}.pickle".format(BASE_DIR, TAG, UUID)
FILENAME_GeneInfo = "./{}/experimentgene_info_tag_{}_{}.pickle".format(
    BASE_DIR, TAG, UUID
)
FILENAME_ProteinInfo = "./{}/experimentprotein_info_tag_{}_{}.pickle".format(
    BASE_DIR, TAG, UUID
)
FILENAME_GeneInfluence = "./{}/geneinfluences_tag_{}_{}.pickle".format(
    BASE_DIR, TAG, UUID
)
FILENAME_Losses = "./{}/losses_tag_{}_{}.pickle".format(BASE_DIR, TAG, UUID)

data = generate_totalseq_semisynth2(
    frac=FRAC,
    seed=SEED,
    n_top_genes=N_GENES,
    noise=DATA_NOISE,
    beta_scale=BETA_SCALE,
    nonlinearity=NONLINEARITY,
    elementwise_nonlinearity=ELEMENTWISE_NONLINEARITY,
    n_proteins=N_PROTEINS,
    largen=LARGEN,
)
adata = data["adata"]
n_genes = adata.n_vars
gene_has_influence = data["gene_has_influence"]
gene_influences = pd.DataFrame(gene_has_influence)
gene_influences.index.name = "gene_x"
gene_influences.columns.name = "protein"
gene_influences = (
    gene_influences.stack()
    .to_frame("has_influence")
    .reset_index()
    .astype(dict(gene_x=str))
)
n_clusters = [10, 100, 1000, n_genes]

res_df = []
for setup in [
    dict(xy_linear=False, x_linear=True, tag="_x_linear"),
    dict(xy_linear=True, x_linear=False, tag="_xy_linear"),
    dict(xy_linear=False, x_linear=False, tag=""),
]:
    suffix = setup["tag"]
    jax_crtres = JAXCRT(
        adata=adata,
        xy_linear=setup["xy_linear"],
        x_linear=setup["x_linear"],
        batch_key=None,
        n_latent=N_LATENT,
        n_hidden=128,
        batch_size=128,
        xy_batch_size=128,
        n_epochs_kl_warmup=50,
        x_lr=1e-4,
        xy_lr=1e-4,
        xy_patience=20,
        x_patience=20,
        n_hidden_xy=N_HIDDEN_XY,
        x_dropout_rate=0.0,
        xy_dropout_rate=0.0,
        n_epochs=10000,
        n_mc_samples=N_MC_SAMPLES,
        percent_dev=PERCENT_DEV,
        precision=None,
        split_seed=SPLIT_SEED,
    )
    jax_crtres.train_all()
    eval_adata = adata[(~jax_crtres.adata.obs["is_dev"].values)].copy()
    if not RANDOMFOREST:
        res = jax_crtres.get_importance(
            eval_adata,
            batch_size=256,
        )
    else:
        res = jax_crtres.get_importance_rf(
            eval_adata,
            batch_size=10000,
        )
    _res_df = (
        pd.DataFrame(res["pvalues"])
        .stack()
        .to_frame("pvalue")
        .reset_index()
        .rename(columns={"level_0": "gene_id", "level_1": "protein"})
        .assign(model=f"jax_crt{suffix}")
    )
    res_df.append(_res_df)
res_df = pd.concat(res_df, axis=0)

ols = OLS(adata=adata)
ols_res = ols.compute().assign(gene_id=lambda x: x.gene_group_id)

if NONLINEARITY == "id":
    adata_gt = adata.copy()
    adata_gt.X = adata_gt.layers["X_transformed"]  # right transformation
    ols_valid = OLS(adata=adata_gt, from_raw=True)  # No need to retransform data
    ols_valid_res = ols_valid.compute().assign(
        gene_id=lambda x: x.gene_group_id, model="ols_valid"
    )


ew = ElementWiseOLS(adata=adata)
ew_res = ew.compute().assign(gene_id=lambda x: x.gene_group_id)
# Saving
gene_influences = (
    pd.DataFrame(gene_has_influence)
    .stack()
    .to_frame("has_influence")
    .reset_index()
    .rename(columns={"level_0": "gene_id", "level_1": "protein"})
)

to_concat = [
    res_df,
    ols_res,
    ew_res,
]
if NONLINEARITY == "id":
    to_concat.append(ols_valid_res)
res_ = (
    pd.concat(to_concat)
    .assign(**args)
    .merge(gene_influences, on=["gene_id", "protein"], how="left")
)
dtypes = {key: "category" for key in args.keys()}
res_ = res_.astype(dtypes)
res_.to_pickle(FILENAME)
gene_mapper = pd.Series(adata.var.index)
gene_mapper.to_pickle(FILENAME_GeneInfo)
protein_mapper = pd.Series(
    [
        "protein{}".format(prot_id)
        for prot_id in range(adata.obsm["protein_expression"].shape[-1])
    ]
)
protein_mapper.to_pickle(FILENAME_ProteinInfo)
gene_influences.to_pickle(FILENAME_GeneInfluence)
losses = jax_crtres.losses.to_pickle(FILENAME_Losses)

# %%
all_res = jax_crtres.get_hier_importance(
    n_clusters=N_CLUSTERS,
    eval_adata=eval_adata,
    batch_size=256,
)
all_res["gene_results"].to_netcdf(
    os.path.join(BASE_DIR, "crt_results_{}_{}.nc".format(TAG, UUID))
)
all_res["cluster_results"].to_csv(
    os.path.join(BASE_DIR, "crt_cluster_results_{}_{}.csv".format(TAG, UUID))
)
all_res["cluster_results"].to_pickle(
    os.path.join(BASE_DIR, "crt_cluster_results_{}_{}.pkl".format(TAG, UUID))
)
all_res["gene_to_cluster"].to_csv(
    os.path.join(BASE_DIR, "crt_gene_to_cluster_{}_{}.csv".format(TAG, UUID))
)
all_res["gene_to_cluster"].to_pickle(
    os.path.join(BASE_DIR, "crt_gene_to_cluster_{}_{}.pkl".format(TAG, UUID))
)
np.save(
    os.path.join(BASE_DIR, "crt_linkage_matrix_{}_{}.npy".format(TAG, UUID)),
    all_res["Z"],
)


# # %%
# import matplotlib.pyplot as plt

# plt.hist(res["pvalues"][gene_has_influence], alpha=0.25, bins=25)
# plt.hist(res["pvalues"][~gene_has_influence], alpha=0.25, bins=25)
# # %%
# res_b = jax_crtres.get_importance(
#         eval_adata,
#         batch_size=256,
#     )
# # %%
# plt.hist(res_b["pvalues"][gene_has_influence], alpha=0.25, bins=25)
# plt.hist(res_b["pvalues"][~gene_has_influence], alpha=0.25, bins=25)
# # %%
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestRegressor

# params = {
#     "n_estimators": [50, 100, 200, 300],
#     "max_depth": [10, 20, 30],
#     "max_features": ["auto", "sqrt", "log2"],
# }

# estimator = RandomForestRegressor()
# res = GridSearchCV(
#     estimator,
#     params,
# )
# X = jax_crtres.adata.X
# X = np.log1p(1e6 * X / X.sum(axis=1, keepdims=True))
# y = jax_crtres.adata.obsm["protein_expression_norm"].squeeze()
# res.fit(
#     X, y
# )
# # %%
