# CPT

## Installation

On a clean conda environment, install jax on the GPU and install the following packages

```
pip install fastcluster flax numpyro orbax
pip install "matplotlib<3.7" plotnine xarray scanpy
pip install scvi-tools
```

## Run experiments

```
bash script_semisynth.sh
bash nanostring.sh
bash citeseq.sh
```