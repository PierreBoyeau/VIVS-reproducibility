import numpy as np
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.dataloaders import AnnDataLoader
from scvi.data.fields import CategoricalObsField, LayerField, ObsmField

import jax

def construct_dataloader(
    adata: AnnData,
    batch_size: int = 128,
    shuffle = False,
    layer = None,
    batch_key = None,
    protein_key = None,
    **kwargs,
):
    anndata_fields = [
        LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
        CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
        ObsmField("protein_expression", protein_key),
    ]
    # register new fields for latent mode if needed
    adata_manager = AnnDataManager(
        fields=anndata_fields
    )
    adata_manager.register_fields(adata, **kwargs)
    train_dl = AnnDataLoader(adata_manager, batch_size=batch_size, shuffle=shuffle, iter_ndarray=True)
    return train_dl