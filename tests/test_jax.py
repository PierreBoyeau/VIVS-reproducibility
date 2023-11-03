import tempfile

import numpy as np
import scanpy as sc

from cpt import select_architecture, select_genes
from cpt.benchmark.jax_crt import JAXCRT
from cpt.data.protein_synth import (
    generate_totalseq_semisynth,
    generate_totalseq_semisynth2,
)


def test_selection():
    adata = sc.read(
        "pbmc_10k_protein_v3.h5ad",
        backup_url="https://github.com/YosefLab/scVI-data/raw/master/pbmc_10k_protein_v3.h5ad",
    )
    sc.pp.subsample(adata, n_obs=1000, random_state=0)
    adata_ = select_genes(adata, n_top_genes=50)
    assert adata_.shape == (1000, 50)


def test_jax():
    data = generate_totalseq_semisynth2(
        frac=0.1,
        beta_scale=1,
        n_top_genes=50,
        n_proteins=10,
        elementwise_nonlinearity="square",
    )
    data = generate_totalseq_semisynth(frac=0.1, snr=1, n_top_genes=50, n_proteins=10)

    for xy_linear in [True, False]:
        for lkl in ["nb", "zinb", "poisson"]:
            crttool = JAXCRT(
                data["adata"],
                n_epochs=1,
                percent_dev=0.5,
                n_mc_samples=20,
                xy_linear=xy_linear,
                x_likelihood=lkl,
            )
            crttool.train_all()
            crttool.get_importance()
            crttool.get_importance(n_mc_per_pass=10)
            crttool.get_cell_scores(gene_ids=[1, 2, 3])
            crttool.get_latent()

    adata = data["adata"]
    adata.obs.loc[:, "batch_indices"] = np.random.randint(0, 2, adata.shape[0])
    crttool = JAXCRT(
        adata,
        batch_key="batch_indices",
        n_epochs=1,
        percent_dev=0.5,
        n_mc_samples=20,
        xy_linear=xy_linear,
        x_likelihood=lkl,
        xy_include_batch=True,
    )
    crttool.train_all()
    crttool.get_importance()
    crttool.get_importance(n_mc_per_pass=10)
    crttool.get_cell_scores(gene_ids=[1, 2, 3])
    crttool.get_latent()

    adata.obsm["protein_expression"] = np.random.rand(adata.shape[0], 1) >= 0.5
    for xy_linear in [True, False]:
        crttool = JAXCRT(
            adata,
            batch_key="batch_indices",
            n_epochs=1,
            percent_dev=0.5,
            n_mc_samples=20,
            xy_linear=xy_linear,
            x_likelihood=lkl,
            xy_loss="binary",
        )
        crttool.train_all()
        crttool.get_importance()
        crttool.get_importance(n_mc_per_pass=10)
        crttool.get_cell_scores(gene_ids=[1, 2, 3])


def test_select_architecture():
    data = generate_totalseq_semisynth(frac=0.1, snr=1, n_top_genes=50, n_proteins=10)
    base_params = dict(
        n_epochs=1,
        percent_dev=0.5,
    )
    grid = dict(
        n_hidden_xy=[8, 16, 32],
        xy_lr=[1e-3, 1e-2],
    )
    select_architecture(
        JAXCRT,
        data["adata"],
        xy_base_params=base_params,
        xy_grid=grid,
    )


def test_data():
    generate_totalseq_semisynth2(
        frac=0.3,
        seed=1,
        n_top_genes=500,
        n_proteins=5,
        noise="gaussian",
        beta_scale=10000,
        elementwise_nonlinearity="id",
        nonlinearity="id",
    )


def test_hierarchy():
    data = generate_totalseq_semisynth(frac=0.1, snr=1, n_top_genes=50, n_proteins=10)
    adata = data["adata"]
    adata.obs.loc[:, "celltype"] = np.random.randint(0, 2, adata.shape[0])
    crttool = JAXCRT(
        adata,
        n_epochs=1,
        percent_dev=0.5,
        n_mc_samples=20,
        xy_linear=False,
        x_likelihood="nb",
    )
    crttool.train_all()
    res1 = crttool.get_hier_importance(n_clusters=[5, 10])["gene_results"]
    res2 = crttool.get_hier_importance(n_clusters=[5, 10], group_key="celltype")["gene_results"]
    res3 = crttool.get_hier_importance(n_clusters=[5, 10, 25, 40], group_key="celltype")["gene_results"]
    assert res1["padj_10"].shape == (50, 10)
    assert res2["padj_5"].shape == (2, 50, 10)
    crttool.get_latent()
    crttool.predict_f()

    with tempfile.TemporaryDirectory() as tmpdirname:
        crttool.save_params(save_path=tmpdirname, save_adata=False)
        JAXCRT.load(tmpdirname, data["adata"])


def test_samplings():
    data = generate_totalseq_semisynth2(
        frac=0.1,
        beta_scale=1,
        n_top_genes=50,
        n_proteins=10,
        elementwise_nonlinearity="square",
    )
    data = generate_totalseq_semisynth(frac=0.1, snr=1, n_top_genes=50, n_proteins=10)

    crttool = JAXCRT(
        data["adata"],
        n_epochs=1,
        percent_dev=0.5,
        n_mc_samples=20,
        xy_linear=False,
        x_likelihood="nb",
    )
    crttool.train_all()
    res = crttool.get_importance_bis()
    assert res["pvalues"].shape == (50, 10)


def test_rf():
    data = generate_totalseq_semisynth2(
        frac=0.1,
        beta_scale=1,
        n_top_genes=50,
        n_proteins=10,
        elementwise_nonlinearity="square",
    )
    data = generate_totalseq_semisynth(frac=0.1, snr=1, n_top_genes=50, n_proteins=1)
    
    adata = data["adata"]
    sc.pp.filter_cells(adata, min_genes=3)
    crttool = JAXCRT(
        adata,
        n_epochs=1,
        percent_dev=0.5,
        n_mc_samples=20,
        xy_linear=False,
        x_likelihood="nb",
    )
    crttool.train_all()
    res = crttool.get_importance_rf(batch_size=10000)
