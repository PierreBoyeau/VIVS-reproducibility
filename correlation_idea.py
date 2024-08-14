# %%
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.stats import spearmanr
import os
import pickle
from scvi.data import pbmc_seurat_v4_cite_seq
from cpt.benchmark import JAXCRT


def save_obj(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f)



# %%
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lambda", type=float, default=0.01)
args = vars(parser.parse_args())

# %%
CRT_KWARGS = dict(
    n_epochs=200,
    # n_epochs=10,
    batch_key=None,
    percent_dev=0.7,
    n_mc_samples=1000,
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
model_kwargs = {"n_hidden_xy": 8,}

# %%_}"
lambda_ = args["lambda"]
seed = args["seed"]

EXPERIMENT_DIR = f"results/correlation/lambda{lambda_}_seed{seed}"
np.random.seed(seed)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


# %%
adata = sc.read(
    "pbmc_10k_protein_v3.h5ad",
    backup_url="https://github.com/YosefLab/scVI-data/raw/master/pbmc_10k_protein_v3.h5ad",
)

sc.pp.filter_genes(adata, min_cells=100)
sc.pp.highly_variable_genes(adata, n_top_genes=500, flavor="seurat_v3")
adata = adata[:, adata.var["highly_variable"]].copy()
gene_names = adata.var_names
# %%
# n_cells_expressing = (adata.X != 0).mean(0)
# n_cells_expressing.argmax()

# %%
top_expressed_genes = np.argsort((adata.X != 0).sum(0))[::-1][:20]
positive_gene_id = top_expressed_genes[0]
x_ = adata.X[:, positive_gene_id].flatten()




xprime = x_ + np.random.poisson(lambda_, size=x_.shape)

# x_ = np.log1p(x_)
# plt.hist(x_)
# %%
spearmanr(x_, xprime)

# %%
y_target = np.log1p(adata.X[:, top_expressed_genes].sum(1))
y_target = y_target ** 2

# %%
X_all = np.concatenate(
    [
        adata.X,
        xprime[:, None]
    ],
    axis=1
)
spurious_gene_id = X_all.shape[1] - 1
var_all = pd.DataFrame(
    index=np.concatenate([gene_names, ["XPRIME"]])
)
adata_final = sc.AnnData(
    X=X_all,
    var=var_all,
    obsm={"protein_expression": y_target[:, None]}
)
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
    n_clusters=[100, 500],
    eval_adata=eval_adata,
    batch_size=128,
)
all_res["positive_gene"] = positive_gene_id
all_res["spurious_gene"] = spurious_gene_id
save_obj(all_res, os.path.join(EXPERIMENT_DIR, "crt_results.pickle"))

# %%
