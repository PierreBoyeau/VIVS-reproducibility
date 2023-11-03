# %%
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
from cpt._utils import spatial_get_density

# %%
file = "/home/pierre/data/nanostring_bis_finished_hotspot_densities.h5ad"
adata = sc.read_h5ad(file)
adata.obs.loc[:, "density"] = np.zeros(adata.shape[0])
adata.obs.loc[:, ["X", "Y"]] = adata.obsm["X_umap_scvi"]
adata.obs.loc[:, "CD8A"] = np.asarray(adata[:, "CD8A"].X).squeeze()
adata.obs.loc[:, "CD8B"] = np.asarray(adata[:, "CD8B"].X).squeeze()
adata.obs.loc[:, "CD4"] = np.asarray(adata[:, "CD4"].X).squeeze()
adata.obs["celltypes_coarse_hand"].unique()

# %%
tissue_len = np.sqrt(25.0) * 1e-3  # ~~25 mm^2
adata.obs.loc[:, ["x_", "y_"]] = MinMaxScaler().fit_transform(adata.obs[["x", "y"]])
spatial_coor = adata.obs[["x_", "y_"]]

# %%
for bandwidth_microns in tqdm([1, 10, 50, 100, 200, 500, 1000]):
    bandwidth_norm = bandwidth_microns * 1e-6 / tissue_len
    subpopulation_of_interest = adata.obs["celltypes_coarse_hand"] == "Tumor_cell"
    obs_name = "density_bw{}mum".format(bandwidth_microns)

    spatial_get_density(
        adata,
        spatial_coor,
        subpopulation_of_interest,
        bandwidth=bandwidth_norm,
        compute_relative_density=False,
        new_obs_name=obs_name,
    )

adata.write_h5ad(
    "nanostring_finished_hotspot_densities_tumorcells_interpretable.h5ad"
)
