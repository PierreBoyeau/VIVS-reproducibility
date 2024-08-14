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


for the perturb-seq experiment, start by downloading the data from Frangieh et al., 2021 from [here](https://github.com/theislab/sc-pert).

The experiment can then be run using
```
python perturbseq_run_analysis.py --sgrna MYSGNRA
```