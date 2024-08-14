import os

import fastcluster
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
from flax import serialization
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scvi import REGISTRY_KEYS
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import numpyro.distributions as dist
import xarray as xr
from orbax.checkpoint import PyTreeCheckpointer, Checkpointer
from sklearn.ensemble import RandomForestRegressor

from cpt._utils import one_hot
from cpt.model.models_jax import SCVICRT, ImportanceScorer, ImportanceScorerLinear
from cpt.model.train_utils import train_scvi, train_statistic
from cpt.model.utils_jax import construct_dataloader


class JAXCRT:
    def __init__(
        self,
        adata,
        protein_names=None,
        batch_key=None,
        n_latent=10,
        n_hidden=128,
        batch_size=128,
        xy_batch_size=128,
        n_epochs_kl_warmup=50,
        x_lr=1e-3,
        xy_lr=1e-3,
        x_likelihood="nb",
        xy_patience=20,
        x_dropout_rate=0.0,
        xy_dropout_rate=0.0,
        xy_residual=False,
        xy_activation="relu",
        x_patience=20,
        n_hidden_xy=128,
        n_epochs=100,
        n_epochs_x=None,
        n_mc_samples=300,
        percent_dev=0.5,
        precision=None,
        xy_linear=False,
        x_linear=False,
        x_early_stopping_metric="elbo",
        x_tx=None,
        xy_tx=None,
        compute_pvalue_on="val",
        xy_include_batch=False,
        xy_loss="mse",
        log1p_factor=1e6,
        split_seed=0,
    ):
        self.n_genes = adata.X.shape[1]
        self.n_proteins = adata.obsm["protein_expression"].shape[-1]
        self.n_batch = (
            adata.obs[batch_key].unique().shape[0] if batch_key is not None else 0
        )
        protein_expression = adata.obsm["protein_expression"]
        if xy_loss == "mse":
            adata.obsm["protein_expression_norm"] = (
                protein_expression - protein_expression.mean(0)
            ) / protein_expression.std(0)
        elif xy_loss == "binary":
            adata.obsm["protein_expression_norm"] = protein_expression
        elif xy_loss == "poisson":
            adata.obsm["protein_expression_norm"] = protein_expression
        self.protein_names = (
            adata.obsm["protein_expression"].columns
            if hasattr(adata.obsm["protein_expression"], "columns")
            else np.arange(self.n_proteins)
        )
        self.protein_key = "protein_expression_norm"
        self.protein_field_name = "protein_expression"

        self.init_params = pd.Series(
            dict(
                batch_key=batch_key,
                n_latent=n_latent,
                n_hidden=n_hidden,
                batch_size=batch_size,
                n_epochs_kl_warmup=n_epochs_kl_warmup,
                x_lr=x_lr,
                xy_lr=xy_lr,
                xy_patience=xy_patience,
                x_dropout_rate=x_dropout_rate,
                xy_dropout_rate=xy_dropout_rate,
                x_patience=x_patience,
                n_hidden_xy=n_hidden_xy,
                n_epochs=n_epochs,
                n_mc_samples=n_mc_samples,
                percent_dev=percent_dev,
                precision=precision,
                xy_loss=xy_loss,
            )
        )

        self.compute_pvalue_on = compute_pvalue_on

        self.x_tx = x_tx
        self.xy_tx = xy_tx
        self.scvi_model_kwargs = dict(
            n_input=self.n_genes,
            n_latent=n_latent,
            n_hidden=n_hidden,
            precision=precision,
            likelihood=x_likelihood,
            dropout_rate=x_dropout_rate,
            linear_decoder=x_linear,
        )
        self.xy_include_batch = xy_include_batch
        self.xy_input_size = (
            self.n_genes + self.n_batch if xy_include_batch else self.n_genes
        )
        self.log1p_factor = log1p_factor

        if xy_linear:
            self.importance_cls = ImportanceScorerLinear
            self.importance_kwargs = dict(
                n_features=self.n_proteins,
                loss_type=xy_loss,
            )
        else:
            self.importance_cls = ImportanceScorer
            self.importance_kwargs = dict(
                n_hidden=n_hidden_xy,
                n_features=self.n_proteins,
                dropout_rate=xy_dropout_rate,
                loss_type=xy_loss,
                residual=xy_residual,
                activation=xy_activation,
            )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_epochs_x = n_epochs_x
        self.batch_key = batch_key
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.n_mc_samples = n_mc_samples
        self.xy_batch_size = xy_batch_size
        self.x_lr = x_lr
        self.x_patience = x_patience
        self.x_early_stopping_metric = x_early_stopping_metric
        self.xy_lr = xy_lr
        self.xy_patience = xy_patience

        adata_log = adata.copy()
        sc.pp.normalize_total(adata_log, target_sum=1e6)
        sc.pp.log1p(adata_log)
        self.adata_log = adata_log

        n_obs = adata.X.shape[0]
        np.random.seed(split_seed)
        self.is_dev = np.random.random(n_obs) <= percent_dev
        self.adata = adata
        self.adata.obs.loc[:, "is_dev"] = self.is_dev

        self.is_dev_heldout = np.random.random(n_obs) <= 0.0
        self.adata.obs.loc[:, "is_dev_heldout"] = self.is_dev & (self.is_dev_heldout)
        self.adata.obs.loc[:, "is_dev_train"] = self.is_dev & (~self.is_dev_heldout)

        self.x_params = None
        self.x_batch_stats = None
        self.xy_batch_stats = None
        self.xy_params = None
        self.losses = None

    @property
    def generative_model(self):
        return SCVICRT(**self.scvi_model_kwargs)

    def init_generative_model(self):
        init_rngs = {"params": jax.random.PRNGKey(0), "z": jax.random.PRNGKey(1)}
        init_counts = jnp.ones((self.batch_size, self.n_genes))
        init_batches = jnp.zeros((self.batch_size, 1))

        return self.generative_model.init(
            init_rngs,
            init_counts,
            init_batches,
        )

    @property
    def statistic_model(self):
        return self.importance_cls(**self.importance_kwargs)

    def init_statistic_model(self):
        init_rngs = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}
        init_log_counts = jnp.ones((self.batch_size, self.xy_input_size))
        init_prots = jnp.ones((self.batch_size, self.n_proteins))
        return self.statistic_model.init(init_rngs, init_log_counts, init_prots)

    def train_scvi(self):
        # Construct dataloader
        train_adata = self.adata[self.adata.obs["is_dev"]].copy()
        val_adata = self.adata[~self.adata.obs["is_dev"]].copy()

        n_epochs = self.n_epochs_x if self.n_epochs_x is not None else self.n_epochs
        losses, x_params, x_batch_stats = train_scvi(
            generative_model=self.generative_model,
            train_adata=train_adata,
            val_adata=val_adata,
            batch_size=self.batch_size,
            batch_key=self.batch_key,
            protein_key=self.protein_key,
            lr=self.x_lr,
            n_epochs=n_epochs,
            n_epochs_kl_warmup=self.n_epochs_kl_warmup,
            early_stopping_metric=self.x_early_stopping_metric,
            patience=self.x_patience,
            tx=self.x_tx,
        )
        self.losses = losses
        self.x_params = x_params
        self.x_batch_stats = x_batch_stats

    def train_statistic(self):
        train_adata = self.adata[self.adata.obs["is_dev"]].copy()
        val_adata = self.adata[~self.adata.obs["is_dev"]].copy()

        losses, xy_params, xy_batch_stats = train_statistic(
            statistic_model=self.statistic_model,
            train_adata=train_adata,
            val_adata=val_adata,
            batch_size=self.xy_batch_size,
            batch_key=self.batch_key,
            protein_key=self.protein_key,
            lr=self.xy_lr,
            n_epochs=self.n_epochs,
            patience=self.xy_patience,
            tx=self.xy_tx,
            include_batch_in_input=self.xy_include_batch,
            log1p_factor=self.log1p_factor,
        )
        if self.losses is None:
            self.losses = losses
        else:
            self.losses = pd.concat([self.losses, losses])
        self.xy_params = xy_params
        self.xy_batch_stats = xy_batch_stats

    def train_all(self):
        self.train_scvi()
        self.train_statistic()

    def process_xy_input(self, x, batch_indices):
        return (
            jnp.concatenate(
                [x, jax.nn.one_hot(batch_indices.squeeze(-1), self.n_batch)], axis=-1
            )
            if self.xy_include_batch
            else x
        )

    def get_importance_rf(
        self, eval_adata=None, batch_size=10000,
    ):
        assert self.n_proteins == 1
        if eval_adata is None:
            if self.compute_pvalue_on == "val":
                eval_adata = self.adata[~self.adata.obs["is_dev"]].copy()
            elif self.compute_pvalue_on == "all":
                eval_adata = self.adata.copy()
        n_obs = eval_adata.X.shape[0]

        reg = RandomForestRegressor(
            n_estimators=200, max_features="auto", max_depth=4, n_jobs=24
        )
        train_adata = self.adata[self.adata.obs["is_dev"]].copy()
        y = train_adata.obsm[self.protein_key]
        assert y.shape[1] == 1
        xnorm = np.log1p(1e6 * train_adata.X / train_adata.X.sum(axis=1, keepdims=True))
        reg.fit(xnorm, y)
        self.reg = reg
        
        # ys = self.adata.obsm[self.protein_key]
        # tss = ((ys - ys.mean())**2).sum()

        eval_dl = construct_dataloader(
            eval_adata,
            batch_size=batch_size,
            shuffle=False,
            batch_key=self.batch_key,
            protein_key=self.protein_key,
        )
        total_its = n_obs // batch_size

        rng = jax.random.PRNGKey(0)
        x_rng, z_rng = jax.random.split(rng)

        tilde_ts = np.zeros((self.n_mc_samples, self.n_genes, self.n_proteins))
        observed_ts = np.zeros((self.n_proteins,))

        @jax.jit
        def randomize(inputs, z_rng):
            outs = self.generative_model.apply(
                {
                    "params": self.x_params,
                    "batch_stats": self.x_batch_stats,
                },
                *inputs,
                rngs={"z": z_rng},
                training=False,
            )
            px = outs["px"]
            return px

        def get_xtilde(x, x_tildes, gene_id):
            x_ = x.at[..., gene_id].set(x_tildes[..., gene_id])
            return x_

        def get_t(x, y):
            x = jnp.log1p(self.log1p_factor * x / x.sum(axis=1, keepdims=True))
            x = np.array(x)
            ypred = reg.predict(x)
            loss = (y.squeeze(-1) - ypred) ** 2
            return loss.sum()

        for tensors in tqdm(eval_dl, total=total_its):
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])

            protein_expression = np.array(tensors[self.protein_field_name])
            px = randomize([x, batch_indices], z_rng)

            observed_ts += get_t(np.array(x), protein_expression) / n_obs
            x_ = x.copy()
            for mc_trial in tqdm(range(self.n_mc_samples)):
                x_tildes = px.sample(x_rng)
                x_rng, _ = jax.random.split(x_rng)

                for g_ in range(self.n_genes):
                    x_tilde = np.array(get_xtilde(x_, x_tildes, g_))
                    t = get_t(x_tilde, protein_expression)

                    tilde_ts[mc_trial, g_] += t / n_obs
            z_rng, _ = jax.random.split(z_rng)
        pval = (1.0 + (observed_ts >= tilde_ts).sum(0)) / (1.0 + self.n_mc_samples)
        padj = np.array(
            [multipletests(_pval, method="fdr_bh")[1] for _pval in pval.T]
        ).T

        return dict(
            obs_ts=np.asarray(observed_ts),
            null_ts=np.asarray(tilde_ts),
            pvalues=np.asarray(pval),
            padj=padj,
        )

    def get_importance(
        self, eval_adata=None, batch_size=128, n_mc_per_pass=1, use_vmap=True
    ):
        if eval_adata is None:
            if self.compute_pvalue_on == "val":
                eval_adata = self.adata[~self.adata.obs["is_dev"]].copy()
            elif self.compute_pvalue_on == "all":
                eval_adata = self.adata.copy()
        n_obs = eval_adata.X.shape[0]

        eval_dl = construct_dataloader(
            eval_adata,
            batch_size=batch_size,
            shuffle=False,
            batch_key=self.batch_key,
            protein_key=self.protein_key,
        )
        total_its = n_obs // batch_size

        rng = jax.random.PRNGKey(0)
        x_rng, z_rng = jax.random.split(rng)

        tilde_ts = jnp.zeros((self.n_mc_samples, self.n_genes, self.n_proteins))
        observed_ts = jnp.zeros((self.n_proteins,))
        gene_ids = jnp.arange(self.n_genes)

        @jax.jit
        def get_tilde_t(x, batch_indices, y):
            x_ = jnp.log1p(self.log1p_factor * x / jnp.sum(x, axis=-1, keepdims=True))
            x_ = self.process_xy_input(x_, batch_indices)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x_,
                y,
                training=False,
            )["all_loss"].sum(0)
            return res

        @jax.jit
        def randomize(inputs, z_rng):
            outs = self.generative_model.apply(
                {
                    "params": self.x_params,
                    "batch_stats": self.x_batch_stats,
                },
                *inputs,
                rngs={"z": z_rng},
                training=False,
            )
            px = outs["px"]
            return px

        def _compute_loss(x, xtilde, gene_id, batch_indices, y):
            x_ = x.at[..., gene_id].set(xtilde)
            x__ = jnp.log1p(self.log1p_factor * x_ / jnp.sum(x_, axis=-1, keepdims=True))
            x__ = self.process_xy_input(x__, batch_indices)
            # shape (n_cells, n_proteins)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x__,
                y,
                training=False,
            )["all_loss"].sum(0)
            return res

        @jax.jit
        def compute_tilde_t(x, px, x_rng, batch_indices, y):
            _x_tilde = px.sample(x_rng)
            if use_vmap:
                _tilde_t_k = jax.vmap(_compute_loss, (None, -1, 0, None, None), 0)(
                    x, _x_tilde, gene_ids, batch_indices, y
                )
            else:
                # @jax.jit
                def parallel_fn(xs):
                    tilde_x = xs[:-1]
                    gene_id = xs[-1]
                    return _compute_loss(x, tilde_x, gene_id, batch_indices, y)

                xs = jnp.concatenate([_x_tilde, gene_ids[None]], axis=0)
                _tilde_t_k = jax.lax.map(parallel_fn, jnp.transpose(xs))
            return _tilde_t_k

        @jax.jit
        def double_compute_tilde_t(x, px, x_rng, batch_indices, y):
            _x_tilde = px.sample(x_rng, sample_shape=(n_mc_per_pass,))
            _fn = jax.vmap(_compute_loss, (None, -1, 0, None, None), 0)
            _fn = jax.vmap(_fn, (None, 0, None, None, None), 0)
            return _fn(x, _x_tilde, gene_ids, batch_indices, y)

        n_passes = self.n_mc_samples // n_mc_per_pass

        for tensors in tqdm(eval_dl, total=total_its):
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            protein_expression = jnp.array(tensors[self.protein_field_name])
            observed_ts += get_tilde_t(x, batch_indices, protein_expression) / n_obs
            px = randomize([x, batch_indices], z_rng)

            _tilde_t = []
            if n_mc_per_pass == 1:
                for _ in range(self.n_mc_samples):
                    _tilde_t_k = compute_tilde_t(
                        x, px, x_rng, batch_indices, protein_expression
                    )
                    _tilde_t.append(_tilde_t_k[None])
                    x_rng, _ = jax.random.split(x_rng)
            else:
                for _ in range(n_passes):
                    _tilde_t_k = double_compute_tilde_t(
                        x, px, x_rng, batch_indices, protein_expression
                    )
                    _tilde_t.append(_tilde_t_k)
                    x_rng, _ = jax.random.split(x_rng)
            _tilde_t = jnp.concatenate(_tilde_t, axis=0)
            tilde_ts += _tilde_t / n_obs
            z_rng, _ = jax.random.split(z_rng)
        pval = (1.0 + (observed_ts >= tilde_ts).sum(0)) / (1.0 + self.n_mc_samples)
        padj = np.array(
            [multipletests(_pval, method="fdr_bh")[1] for _pval in pval.T]
        ).T

        return dict(
            obs_ts=np.asarray(observed_ts),
            null_ts=np.asarray(tilde_ts),
            pvalues=np.asarray(pval),
            padj=padj,
        )

    def get_importance_bis(self, eval_adata=None, batch_size=128):
        if eval_adata is None:
            if self.compute_pvalue_on == "val":
                eval_adata = self.adata[~self.adata.obs["is_dev"]].copy()
            elif self.compute_pvalue_on == "all":
                eval_adata = self.adata.copy()
        n_obs = eval_adata.X.shape[0]

        eval_dl = construct_dataloader(
            eval_adata,
            batch_size=batch_size,
            shuffle=False,
            batch_key=self.batch_key,
            protein_key=self.protein_key,
        )
        total_its = n_obs // batch_size

        rng = jax.random.PRNGKey(0)
        x_rng, z_rng = jax.random.split(rng)

        tilde_ts = jnp.zeros((self.n_genes, self.n_mc_samples, self.n_proteins))
        observed_ts = jnp.zeros((self.n_proteins,))

        @jax.jit
        def get_tilde_t(x, batch_indices, y):
            x_ = jnp.log1p(self.log1p_factor * x / jnp.sum(x, axis=-1, keepdims=True))
            x_ = self.process_xy_input(x_, batch_indices)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x_,
                y,
                training=False,
            )["all_loss"].sum(0)
            return res

        @jax.jit
        def randomize(inputs, z_rng):
            outs = self.generative_model.apply(
                {
                    "params": self.x_params,
                    "batch_stats": self.x_batch_stats,
                },
                *inputs,
                rngs={"z": z_rng},
                training=False,
            )
            px = outs["px"]
            return px

        @jax.jit
        def compute_tilde_t(x, px, gene_id, x_rng, batch_indices, y):
            _x_tilde = px.sample(x_rng)
            x_ = x.at[..., gene_id].set(_x_tilde)
            x__ = jnp.log1p(self.log1p_factor * x_ / jnp.sum(x_, axis=-1, keepdims=True))
            x__ = self.process_xy_input(x__, batch_indices)
            # shape (n_cells, n_proteins)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x__,
                y,
                training=False,
            )["all_loss"].sum(1)
            return res[None]

        for tensors in tqdm(eval_dl, total=total_its):
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            protein_expression = jnp.array(tensors[self.protein_field_name])
            observed_ts += get_tilde_t(x, batch_indices, protein_expression) / n_obs
            px = randomize([x, batch_indices], z_rng)

            _tilde_t = []
            mean, conc = px.mean, px.concentration
            x_ = jnp.tile(x[None], (self.n_mc_samples, 1, 1))
            y_ = jnp.tile(protein_expression[None], (self.n_mc_samples, 1, 1))
            x_rngs = jax.random.split(x_rng, self.n_genes)

            # @jax.jit
            def parallel_fn(inputs):
                gene = inputs[0]
                x_rng = inputs[1:].astype(jnp.uint32)
                px = dist.NegativeBinomial2(mean[..., gene], conc[..., gene])
                _tilde_t_k = compute_tilde_t(x_, px, gene, x_rng, batch_indices, y_)
                return _tilde_t_k[None]

            @jax.jit
            def jitted_map_fn(inputs):
                return jax.lax.map(parallel_fn, inputs)

            inputs = jnp.concatenate(
                [jnp.arange(self.n_genes)[:, None], x_rngs], axis=1
            )
            # _tilde_t = jax.lax.map(parallel_fn, inputs).squeeze()
            _tilde_t = jitted_map_fn(inputs).squeeze()
            x_rng, _ = jax.random.split(x_rng)

            tilde_ts += _tilde_t / n_obs
            z_rng, _ = jax.random.split(z_rng)
        pval = (1.0 + (observed_ts >= tilde_ts).sum(1)) / (1.0 + self.n_mc_samples)
        padj = np.array(
            [multipletests(_pval, method="fdr_bh")[1] for _pval in pval.T]
        ).T

        return dict(
            obs_ts=np.asarray(observed_ts),
            null_ts=np.asarray(tilde_ts),
            pvalues=np.asarray(pval),
            padj=padj,
        )

    def get_cell_scores(
        self,
        gene_ids,
        protein_ids=None,
        eval_adata=None,
        batch_size=None,
        n_mc_samples=None,
    ):
        if eval_adata is None:
            eval_adata = self.adata[~self.adata.obs["is_dev"]].copy()
        n_obs = eval_adata.X.shape[0]
        batch_size = batch_size if batch_size is not None else self.batch_size
        n_mc_samples = n_mc_samples if n_mc_samples is not None else self.n_mc_samples

        eval_dl = construct_dataloader(
            eval_adata,
            batch_size=batch_size,
            shuffle=False,
            batch_key=self.batch_key,
            protein_key=self.protein_key,
        )
        total_its = n_obs // batch_size

        rng = jax.random.PRNGKey(0)
        x_rng, z_rng = jax.random.split(rng)

        gene_ids = jnp.array(gene_ids)
        protein_ids = (
            jnp.array(protein_ids)
            if protein_ids is not None
            else jnp.arange(self.n_proteins)
        )
        tilde_t_mean = []
        obs_t = []

        @jax.jit
        def get_tilde_t_nosum(x, batch_indices, y):
            x_ = jnp.log1p(self.log1p_factor * x / jnp.sum(x, axis=-1, keepdims=True))
            x_ = self.process_xy_input(x_, batch_indices)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x_,
                y,
                training=False,
            )["all_loss"][..., protein_ids]
            return res

        def _compute_loss(x, xtilde, gene_id, batch_indices, y):
            x_ = x.at[..., gene_id].set(xtilde)
            x__ = jnp.log1p(self.log1p_factor * x_ / jnp.sum(x_, axis=-1, keepdims=True))
            x__ = self.process_xy_input(x__, batch_indices)
            # shape (n_cells, n_proteins)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x__,
                y,
                training=False,
            )["all_loss"][..., protein_ids]
            return res

        @jax.jit
        def randomize(inputs, z_rng):
            outs = self.generative_model.apply(
                {
                    "params": self.x_params,
                    "batch_stats": self.x_batch_stats,
                },
                *inputs,
                rngs={"z": z_rng},
                training=False,
            )
            px = outs["px"]
            return px

        @jax.jit
        def compute_tilde_t(x, px, x_rng, batch_indices, y):
            _x_tilde = px.sample(x_rng)[..., gene_ids]
            _tilde_t_k = jax.vmap(_compute_loss, (None, -1, 0, None, None), 1)(
                x, _x_tilde, gene_ids, batch_indices, y
            )
            return _tilde_t_k

        for tensors in tqdm(eval_dl, total=total_its):
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            protein_expression = jnp.array(tensors[self.protein_field_name])
            px = randomize([x, batch_indices], z_rng)
            _tilde_t = []
            for _ in range(n_mc_samples):
                _tilde_t_k = compute_tilde_t(
                    x, px, x_rng, batch_indices, protein_expression
                )
                # _tilde_t = _tilde_t.at[mc_sample].set(_tilde_t_k)
                _tilde_t.append(_tilde_t_k[None])
                x_rng, _ = jax.random.split(x_rng)
            _tilde_t = jnp.concatenate(_tilde_t, axis=0).mean(0)
            observed_ts = get_tilde_t_nosum(x, batch_indices, protein_expression)

            z_rng, _ = jax.random.split(z_rng)
            # score = _tilde_t - observed_ts[:, None]
            tilde_t_mean.append(np.asarray(_tilde_t))
            obs_t.append(np.asarray(observed_ts[:, None]))
        tilde_t_mean = np.concatenate(tilde_t_mean)
        obs_t = np.concatenate(obs_t)
        return dict(
            tilde_t_mean=tilde_t_mean,
            obs_t=obs_t,
        )

    @staticmethod
    def _save_params(pytree, save_path):
        checkpointer = PyTreeCheckpointer()
        checkpointer.save(save_path, pytree)

    def save_params(self, save_path, save_adata=False):
        files = [
            (serialization.to_bytes(self.x_params), "x_params_bytes"),
            (serialization.to_bytes(self.x_batch_stats), "x_batch_stats_bytes"),
            (serialization.to_bytes(self.xy_batch_stats), "xy_batch_stats_bytes"),
            (serialization.to_bytes(self.xy_params), "xy_params_bytes"),
        ]
        for file, name in files:
            with open(os.path.join(save_path, name), "wb") as f:
                f.write(file)

        if save_adata:
            self.adata.write(os.path.join(save_path, "adata.h5ad"))

        self.init_params.to_csv(os.path.join(save_path, "init_params.csv"))

    @classmethod
    def load(cls, load_path, adata=None):
        init_params = (
            pd.read_csv(
                os.path.join(load_path, "init_params.csv"), index_col=0, squeeze=True
            )
            .replace({np.nan: None})
            .to_dict()
        )
        init_params["n_latent"] = int(init_params["n_latent"])
        init_params["n_hidden"] = int(init_params["n_hidden"])
        init_params["batch_size"] = int(init_params["batch_size"])
        init_params["n_epochs_kl_warmup"] = int(init_params["n_epochs_kl_warmup"])
        init_params["xy_patience"] = int(init_params["xy_patience"])
        init_params["x_patience"] = int(init_params["x_patience"])
        init_params["n_hidden_xy"] = int(init_params["n_hidden_xy"])
        init_params["n_epochs"] = int(init_params["n_epochs"])
        init_params["n_mc_samples"] = int(init_params["n_mc_samples"])
        init_params["percent_dev"] = float(init_params["percent_dev"])
        # init_params["n_genes"] = int(init_params["n_genes"])
        init_params["x_dropout_rate"] = float(init_params["x_dropout_rate"])
        init_params["xy_dropout_rate"] = float(init_params["xy_dropout_rate"])

        if adata is None:
            adata = sc.read(os.path.join(load_path, "adata.h5ad"))
        obj = cls(adata, **init_params)
        gen_variables = obj.init_generative_model()
        stat_variables = obj.init_statistic_model()

        obj.x_params = gen_variables["params"]
        obj.x_batch_stats = gen_variables["batch_stats"]
        obj.xy_params = stat_variables["params"]
        obj.xy_batch_stats = stat_variables["batch_stats"]

        files = [
            "x_params_bytes",
            "x_batch_stats_bytes",
            "xy_batch_stats_bytes",
            "xy_params_bytes",
        ]

        for name in files:
            with open(os.path.join(load_path, name), "rb") as f:
                file = f.read()
            if name == "x_params_bytes":
                obj.x_params = serialization.from_bytes(obj.x_params, file)
            elif name == "x_batch_stats_bytes":
                obj.x_batch_stats = serialization.from_bytes(obj.x_batch_stats, file)
            elif name == "xy_batch_stats_bytes":
                obj.xy_batch_stats = serialization.from_bytes(obj.xy_batch_stats, file)
            elif name == "xy_params_bytes":
                obj.xy_params = serialization.from_bytes(obj.xy_params, file)
        return obj

    def get_gene_correlations(self, adata=None):
        """Compute G times G gene correlation matrix."""

        cpu_device = jax.devices("cpu")[0]
        scdl = construct_dataloader(
            adata,
            batch_size=self.batch_size,
            shuffle=True,
            batch_key=self.batch_key,
            protein_key=self.protein_key,
        )

        @jax.jit
        def get_scales(inputs, z_rng):
            return self.generative_model.apply(
                {
                    "params": self.x_params,
                    "batch_stats": self.x_batch_stats,
                },
                *inputs,
                rngs={"z": z_rng},
                training=False,
            )["h"]

        xx_est = jax.device_put(
            jnp.zeros((self.n_genes, self.n_genes)), device=cpu_device
        )
        x_est = jax.device_put(jnp.zeros(self.n_genes), device=cpu_device)
        z_rng = jax.random.PRNGKey(0)
        n_obs_total = adata.X.shape[0]
        for tensors in scdl:
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            z_rng = jax.random.PRNGKey(0)
            z_rng, _ = jax.random.split(z_rng)
            scales = get_scales([x, batch_indices], z_rng)
            x_est += jax.device_put(
                jnp.sum(scales, axis=0) / n_obs_total, device=cpu_device
            )
            xx_est += jax.device_put(
                jnp.sum(
                    jax.lax.batch_matmul(scales[..., None], scales[:, None]), axis=0
                )
                / n_obs_total,
                device=cpu_device,
            )
        x_est = x_est[None]
        cov_ = xx_est - jnp.matmul(x_est.T, x_est)
        factor_ = 1.0 / jnp.sqrt(jnp.diag(cov_))
        dmat = jnp.diag(factor_)
        corr_ = dmat @ (cov_ @ dmat)
        return np.asarray(corr_)

    def get_gene_groupings(
        self,
        adata=None,
        method="complete",
        return_z=False,
        n_clusters=None,
    ):
        """Computes gene groupings based on gene correlations.

        Parameters
        ----------
        adata :
            adata used to compute gene correlations.
        method :
            Linkage for hierarchical clustering.
        return_z :
            Whether to return linkage matrix.
        n_clusters :
            Number of desired clusters.
        """
        adata = self.adata[self.adata.obs.is_dev].copy() if adata is None else adata
        corr_ = self.get_gene_correlations(adata=adata)
        pseudo_dist = 1 - corr_
        pseudo_dist = (pseudo_dist + pseudo_dist.T) / 2
        pseudo_dist = np.clip(pseudo_dist, a_min=0.0, a_max=100.0)
        pseudo_dist = pseudo_dist - np.diag(np.diag(pseudo_dist))
        dist_vec = squareform(pseudo_dist, checks=False)
        Z = fastcluster.linkage(dist_vec, method=method)
        Z = hierarchy.optimal_leaf_ordering(Z, dist_vec)
        gene_order = hierarchy.leaves_list(Z)

        assert n_clusters is not None
        if not isinstance(n_clusters, list):
            n_clusters = [n_clusters]
        gene_groupings = []
        for n_cluster in n_clusters:
            if n_cluster >= self.n_genes:
                continue
            cluster_assignments = hierarchy.fcluster(Z, n_cluster, criterion="maxclust")
            cluster_assignments -= 1
            gene_groupings.append(cluster_assignments)
        if return_z:
            return (
                gene_groupings,
                Z,
                gene_order,
            )
        return gene_groupings

    def get_hier_importance(
        self,
        n_clusters,
        eval_adata=None,
        batch_size=128,
        gene_groupings=None,
        gene_order=None,
        group_key=None,
        method="complete",
        use_vmap=True,
    ):
        if eval_adata is None:
            if self.compute_pvalue_on == "val":
                eval_adata = self.adata[~self.adata.obs["is_dev"]].copy()
            elif self.compute_pvalue_on == "all":
                eval_adata = self.adata.copy()

        # compute gene groups
        train_adata = self.adata[self.adata.obs["is_dev"]].copy()
        if gene_groupings is None:
            gene_groupings, Z, gene_order = self.get_gene_groupings(
                adata=train_adata,
                n_clusters=n_clusters,
                return_z=True,
                method=method,
            )
        else:
            assert gene_order is not None
        gene_groupings.append(np.arange(self.n_genes).astype(np.int32))
        gene_groups_oh = [jnp.array(one_hot(grouping)) for grouping in gene_groupings]
        gene_groups_sizes = [gg.shape[1] for gg in gene_groups_oh]

        rng = jax.random.PRNGKey(0)
        x_rng, z_rng = jax.random.split(rng)

        tilde_ts = [
            jnp.zeros((self.n_mc_samples, sz, self.n_proteins))
            for sz in gene_groups_sizes
        ]
        observed_ts = jnp.zeros((self.n_proteins,))

        @jax.jit
        def get_tilde_t(x, y):
            x_ = jnp.log1p(self.log1p_factor * x / jnp.sum(x, axis=-1, keepdims=True))
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x_,
                y,
                training=False,
            )["all_loss"].sum(0)
            return res

        def randomize(inputs, z_rng):
            outs = self.generative_model.apply(
                {
                    "params": self.x_params,
                    "batch_stats": self.x_batch_stats,
                },
                *inputs,
                rngs={"z": z_rng},
                training=False,
            )
            px = outs["px"]
            return px

        def _compute_loss_group(x, xtilde, gene_groups, y):
            x_ = x * (1.0 - gene_groups) + (xtilde * gene_groups)
            x__ = jnp.log1p(self.log1p_factor * x_ / jnp.sum(x_, axis=-1, keepdims=True))
            # shape (n_cells, n_proteins)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x__,
                y,
                training=False,
            )["all_loss"].sum(0)
            return res

        @jax.jit
        def compute_tilde_t_group(x, x_tilde, y, gene_groups):
            if use_vmap:
                _tilde_t_k = jax.vmap(_compute_loss_group, (None, None, -1, None), 0)(
                    x, x_tilde, gene_groups, y
                )
            else:

                def parallel_fn(gene_group):
                    return _compute_loss_group(x, x_tilde, gene_group, y)

                _tilde_t_k = jax.lax.map(parallel_fn, gene_groups)
            return _tilde_t_k

        if group_key is None:
            group_key = "group"
            eval_adata.obs.loc[:, group_key] = "0"

        groups = eval_adata.obs[group_key].unique()
        all_gene_results = []
        all_cluster_results = []
        for group in groups:
            # construct dataloader
            eval_adata_ = eval_adata[eval_adata.obs[group_key] == group].copy()
            n_obs = eval_adata_.n_obs
            print(group)
            eval_dl = construct_dataloader(
                eval_adata_,
                batch_size=batch_size,
                shuffle=False,
                batch_key=self.batch_key,
                protein_key=self.protein_key,
            )
            total_its = n_obs // batch_size
            for tensors in tqdm(eval_dl, total=total_its):
                x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
                batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
                protein_expression = jnp.array(tensors[self.protein_field_name])

                observed_ts += get_tilde_t(x, protein_expression) / n_obs
                px = randomize([x, batch_indices], z_rng)

                for k in range(self.n_mc_samples):
                    x_tilde = px.sample(x_rng)
                    x_rng, _ = jax.random.split(x_rng)
                    for gene_group_idx, gene_group_oh in enumerate(gene_groups_oh):
                        _tilde_t_k = (
                            compute_tilde_t_group(
                                x, x_tilde, protein_expression, gene_group_oh
                            )
                            / n_obs
                        )

                        updated_tilde_t = tilde_ts[gene_group_idx][k] + _tilde_t_k
                        tilde_ts[gene_group_idx] = (
                            tilde_ts[gene_group_idx].at[k].set(updated_tilde_t)
                        )
                z_rng, _ = jax.random.split(z_rng)

            gene_results, cluster_results = self._construct_results(
                observed_ts=observed_ts,
                tilde_ts=tilde_ts,
                gene_groupings=gene_groupings,
                gene_groups_sizes=gene_groups_sizes,
            )
            gene_results = gene_results.assign_coords(group=group)
            all_gene_results.append(gene_results)
            all_cluster_results.append(cluster_results)
        all_gene_results = xr.concat(all_gene_results, dim="group")

        all_cluster_results = pd.concat(all_cluster_results, axis=0)

        # needs to store some specific genes for the visualization
        ordered_genes = self.adata.var_names[gene_order]
        add_gene_info = []
        gene_to_clusters = []
        for resolution_idx, group, resolution in zip(
            np.arange(len(n_clusters) + 1), gene_groupings, n_clusters + [self.n_genes]
        ):
            ordered_groups = group[gene_order]
            gene_to_clusters.append(
                pd.DataFrame(
                    {
                        f"clusterassignment_{resolution}": ordered_groups,
                    },
                    index=ordered_genes,
                )
            )
            for cluster_id in np.unique(ordered_groups):
                cluster_genes = ordered_genes[ordered_groups == cluster_id]
                g1 = cluster_genes[0]
                gf = cluster_genes[-1]
                add_gene_info.append(
                    dict(
                        cluster_id=cluster_id,
                        g1=g1,
                        gf=gf,
                        resolution=resolution,
                        resolution_idx=resolution_idx,
                    )
                )
        gene_to_clusters = pd.concat(gene_to_clusters, axis=1).assign(
            gene_idx=np.arange(len(ordered_genes)),
        )
        add_gene_info = pd.DataFrame(add_gene_info)
        all_cluster_results = all_cluster_results.merge(
            add_gene_info, on=["cluster_id", "resolution"]
        ).assign(
            g1_idx=lambda x: x.g1.map(gene_to_clusters.gene_idx),
            gf_idx=lambda x: x.gf.map(gene_to_clusters.gene_idx),
        )
        return dict(
            gene_results=all_gene_results.squeeze(),
            cluster_results=all_cluster_results,
            gene_to_cluster=gene_to_clusters,
            Z=Z,
        )

    def predict_t(self, adata=None, batch_size=128):
        @jax.jit
        def get_t(x, y):
            x_ = jnp.log1p(self.log1p_factor * x / jnp.sum(x, axis=-1, keepdims=True))
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x_,
                y,
                training=False,
            )["all_loss"]
            return res

        adata = self.adata if adata is None else adata
        eval_dl = construct_dataloader(
            adata,
            batch_size=batch_size,
            shuffle=False,
            batch_key=self.batch_key,
            protein_key=self.protein_key,
        )
        res = []
        for tensors in eval_dl:
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            # batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            protein_expression = jnp.array(tensors[self.protein_field_name])
            res.append(np.array(get_t(x, protein_expression)))
        return np.concatenate(res, axis=0)

    def get_latent(self, adata=None):
        adata = self.adata if adata is None else adata
        dl = construct_dataloader(
            self.adata,
            batch_size=128,
            shuffle=False,
            batch_key=self.batch_key,
            protein_key=self.protein_key,
        )

        @jax.jit
        def _get_latent(inputs, z_rng):
            outs = self.generative_model.apply(
                {
                    "params": self.x_params,
                    "batch_stats": self.x_batch_stats,
                },
                *inputs,
                rngs={"z": z_rng},
                training=False,
            )
            return outs["qz"].loc

        cpu_device = jax.devices("cpu")[0]
        zs = []
        z_rng = jax.random.PRNGKey(0)
        for tensors in dl:
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            qz = _get_latent([x, batch_indices], z_rng)
            qz = jax.device_put(qz, device=cpu_device)
            zs.append(qz)
        zs = jnp.concatenate(zs, axis=0)
        return np.array(zs)

    def predict_f(self, adata=None):
        adata = self.adata if adata is None else adata
        dl = construct_dataloader(
            adata,
            batch_size=128,
            shuffle=False,
            batch_key=self.batch_key,
            protein_key=self.protein_key,
        )

        @jax.jit
        def _get_f(x, y, batch_indices):
            x_ = jnp.log1p(self.log1p_factor * x / jnp.sum(x, axis=-1, keepdims=True))
            x_ = self.process_xy_input(x_, batch_indices)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x,
                y,
                training=False,
            )
            return res["h"]

        cpu_device = jax.devices("cpu")[0]
        hs = []
        for tensors in tqdm(dl):
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            protein_expression = jnp.array(tensors[self.protein_field_name])
            h = _get_f(x, protein_expression, batch_indices)
            h = jax.device_put(h, device=cpu_device)
            hs.append(h)
        hs = jnp.concatenate(hs, axis=0)
        return np.array(hs)

    def _construct_results(
        self, observed_ts, tilde_ts, gene_groupings, gene_groups_sizes
    ):
        gene_results = {}
        cluster_results = []
        for resolution_idx, resolution in enumerate(gene_groups_sizes):
            pvals = (1.0 + (observed_ts >= tilde_ts[resolution_idx]).sum(0)) / (
                1.0 + self.n_mc_samples
            )

            padjs = [
                multipletests(pvals[:, protein_id], method="fdr_bh")[1]
                for protein_id in range(self.n_proteins)
            ]
            padjs = jnp.stack(padjs, axis=-1)  # (n_clusters, n_proteins)

            cluster_result = (
                pd.DataFrame(
                    padjs,
                    columns=self.protein_names,
                )
                .stack()
                .reset_index()
                .rename(
                    columns={
                        "level_0": "cluster_id",
                        "level_1": "protein",
                        0: "padj",
                    }
                )
                .assign(resolution=resolution)
            )
            cluster_results.append(cluster_result)

            gene_clusters = gene_groupings[resolution_idx]
            pvals_ = pvals[gene_clusters]
            padjs_ = padjs[gene_clusters]
            coords_ = dict(
                dims=["gene_name", "protein"],
                coords={
                    "gene_name": self.adata.var_names.values,
                    "protein": self.protein_names,
                },
            )
            pvals_ = xr.DataArray(pvals_, **coords_)
            padjs_ = xr.DataArray(padjs_, **coords_)
            cluster_assignments = xr.DataArray(
                gene_clusters,
                dims=["gene_name"],
                coords={"gene_name": self.adata.var_names.values},
            )

            gene_results[f"pvalue_{resolution}"] = pvals_
            gene_results[f"padj_{resolution}"] = padjs_
            gene_results[f"cluster_assignments_{resolution}"] = cluster_assignments
        gene_results = xr.Dataset(gene_results)
        cluster_results = pd.concat(cluster_results)
        return gene_results, cluster_results
