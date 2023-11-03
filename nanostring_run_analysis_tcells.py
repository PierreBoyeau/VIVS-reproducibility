# %%
import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc

from cpt import select_architecture
from cpt.benchmark.jax_crt import JAXCRT
from cpt.benchmark.ols import ElementWiseOLS
from statsmodels.stats.multitest import multipletests
import uuid


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".45"


# %%
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--bandwidth", type=str, required=True)
    args.add_argument("--clip_target", type=float, default=3.0)
    args.add_argument("--semisynth", action="store_true", default=False)
    args.add_argument("--savedir", type=str)
    args.add_argument("--xy_include_batch", action="store_true", default=False)
    args.add_argument("--tcellsonly", action="store_true", default=False)
    args.add_argument("--foreground_genes", type=str, default=None)
    return vars(args.parse_args())


args = parse_args()

# %%
EXPERIMENT_ID = str(uuid.uuid4())
SEMISYNTH = args["semisynth"]
BANDWIDTH = args["bandwidth"]
CLIP_TARGET = args["clip_target"]
SAVEDIR = args["savedir"]
TCELLSONLY = args["tcellsonly"]
FOREGROUND_GENES = args["foreground_genes"]
EXPERIMENT_DIR = os.path.join(SAVEDIR, EXPERIMENT_ID)
DATASET = "all"

os.makedirs(EXPERIMENT_DIR)

# %%
# Load data
batch_key = None
adata = sc.read_h5ad(
    "nanostring_finished_hotspot_densities_tumorcells_interpretable.h5ad"
)
adata.X = np.asarray(adata.layers["counts"].todense())
XY_INCLUDE_BATCH = False

# if TCELLSONLY:
adata = adata[adata.obs["celltypes_coarse_hand"] == "Tcell"].copy()

if FOREGROUND_GENES is not None:
    foreground_genes = pd.read_csv(FOREGROUND_GENES, index_col=0).values.squeeze()
    adata = adata[:, foreground_genes].copy()
    sc.pp.filter_cells(adata, min_counts=10)

normalized_counts = adata.X.copy()
if SEMISYNTH:
    wmats = np.random.randn(50, 10)
    synth_prots = adata.X[:, :50].sum(1)
    synth_prots = np.asarray(synth_prots).squeeze()
    synth_prots = normalized_counts[:, :50]
    synth_prots = np.asarray(synth_prots)
    synth_prots = synth_prots @ wmats
    synth_prots = (synth_prots - synth_prots.mean()) / synth_prots.std()
    synth_prots += np.random.normal(0, 1, synth_prots.shape)
    adata.obsm["protein_expression"] = synth_prots
else:
    response_key = BANDWIDTH
    responses = adata.obs[response_key].copy().values
    if CLIP_TARGET is not None:
        responses = np.clip(responses, 0, CLIP_TARGET)
    adata.obsm["protein_expression"] = responses[:, None]

# %%
CRT_KWARGS = dict(
    n_epochs=10000,
    percent_dev=0.5,
    n_mc_samples=1000,
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
    xy_include_batch=XY_INCLUDE_BATCH,
)
GRID_PARAMETERS = dict(
    n_hidden_xy=[8, 16, 32, 64, 128, 256],
)

# %%
# Run CPT
crttool = JAXCRT(
    adata,
    batch_key=batch_key,
    # xy_linear=True,
    # **final_parameters
    n_hidden_xy=8,
    **CRT_KWARGS,
)
crttool.train_all()
crttool.save_params(save_path=EXPERIMENT_DIR, save_adata=False)

# %%
# Compute pvalues for the overall considered dataset
all_res = pd.DataFrame()
is_t = (adata.obs["celltypes_coarse_hand"] == "Tcell").values
mask = is_t
name = "Tcell"
eval_adata = adata[mask].copy()
marginal = ElementWiseOLS(eval_adata)
marginal_res = marginal.compute()
marginal_res.loc[:, "gene"] = adata.var.index.values
marginal_res.loc[:, "padj"] = multipletests(
    marginal_res.loc[:, "pvalue"], method="fdr_bh"
)[1]

eval_adata = adata[(~crttool.adata.obs["is_dev"].values) & mask].copy()
# res = crttool.get_importance(eval_adata)
# res_df = pd.DataFrame(
#     dict(
#         pvalue=res["pvalues"].squeeze(),
#         padj=res["padj"].squeeze(),
#         gene=adata.var.index.values,
#     )
# ).assign(model="CRT")
all_res = crttool.get_hier_importance(
    n_clusters=[25, 50, 100, 150, 200],
    eval_adata=eval_adata,
    batch_size=512,
)
res_hier = all_res["gene_results"]
res_hier.to_netcdf(os.path.join(EXPERIMENT_DIR, "{}_hier_results.nc".format(BANDWIDTH)))
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


_all_res = pd.concat(
    [
        marginal_res,
    ],
    axis=0,
).assign(
    subpopulation=name,
    experiment_id=EXPERIMENT_ID,
    semisynth=SEMISYNTH,
    bandwidth=BANDWIDTH,
    train_data=DATASET,
    eval_data=DATASET,
    clip_target=CLIP_TARGET,
    xy_include_batch=args["xy_include_batch"],
)
all_res = pd.concat([all_res, _all_res], axis=0)
all_res.to_pickle(os.path.join(EXPERIMENT_DIR, "{}_results_all.pkl".format(BANDWIDTH)))


# %%
