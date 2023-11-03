NONLINEARITY="id"
ELEMENTWISE_NONLINEARITY="square"
PERCENT_DEV=0.7
BASE_SPARSITY=0.3
BASE_BETA_SCALE=10000.0
N_GENES_BASE=500
n_proteins=5
DATA_NOISE="gaussian"
BASE_DIR="results/semisynth_main"
BASE_BETA_SCALE=10000.0

for i in 1 2 3 4 5
do
    # Figure 1
    python run_semisynthetic_jax.py --n_genes $N_GENES_BASE --data_noise $DATA_NOISE --sparsity $BASE_SPARSITY --beta_scale $BASE_BETA_SCALE  --tag main  --importance_statistic MLP --seed $i --nonlinearity $NONLINEARITY --n_hidden_xy 64 --n_mc_samples 2000 --percent_dev $PERCENT_DEV --elementwise_nonlinearity $ELEMENTWISE_NONLINEARITY --n_proteins $n_proteins --base_dir $BASE_DIR

    # # Fig 2a
    for sparsity in 0.05 0.1 0.2 0.3 0.4 0.5
    do
        python run_semisynthetic_jax.py --n_genes $N_GENES_BASE --data_noise $DATA_NOISE --sparsity $sparsity --beta_scale $BASE_BETA_SCALE  --tag sparsity --torch_seed $i  --importance_statistic MLP --seed $i --nonlinearity $NONLINEARITY --n_hidden_xy 64 --n_mc_samples 2000  --percent_dev $PERCENT_DEV --elementwise_nonlinearity $ELEMENTWISE_NONLINEARITY --n_proteins $n_proteins 
    done
    for n_genes in 100 200 500 1000 2000
    do
        python run_semisynthetic_jax.py --n_genes $n_genes --data_noise $DATA_NOISE --sparsity $BASE_SPARSITY --beta_scale $BASE_BETA_SCALE  --tag n_genes --torch_seed $i  --importance_statistic MLP --seed $i --nonlinearity $NONLINEARITY --n_hidden_xy 64 --n_mc_samples 2000 --percent_dev $PERCENT_DEV --elementwise_nonlinearity $ELEMENTWISE_NONLINEARITY --n_proteins $n_proteins 
    done
    for beta_scale in 10 100 1000 2000 10000
    do
        python run_semisynthetic_jax.py --n_genes $N_GENES_BASE --data_noise $DATA_NOISE --sparsity $BASE_SPARSITY --beta_scale $beta_scale  --tag beta_scale --torch_seed $i  --importance_statistic MLP --seed $i --nonlinearity $NONLINEARITY --n_hidden_xy 64 --n_mc_samples 2000 --percent_dev $PERCENT_DEV --elementwise_nonlinearity $ELEMENTWISE_NONLINEARITY --n_proteins $n_proteins 
    done
    python run_semisynthetic_jax.py --n_genes $N_GENES_BASE --data_noise $DATA_NOISE --sparsity $BASE_SPARSITY --beta_scale $BASE_BETA_SCALE  --tag nn_fig --torch_seed $i  --importance_statistic linear --seed $i --nonlinearity $NONLINEARITY --n_mc_samples 2000 --percent_dev $PERCENT_DEV --n_proteins $n_proteins --elementwise_nonlinearity $ELEMENTWISE_NONLINEARITY
    for n_hidden_xy in 8 16 32 64 128
    do
        python run_semisynthetic_jax.py --n_genes $N_GENES_BASE --data_noise $DATA_NOISE --sparsity $BASE_SPARSITY --beta_scale $BASE_BETA_SCALE  --tag nn_fig --torch_seed $i  --importance_statistic MLP --seed $i --nonlinearity $NONLINEARITY --n_hidden_xy $n_hidden_xy --n_mc_samples 2000 --percent_dev $PERCENT_DEV --elementwise_nonlinearity $ELEMENTWISE_NONLINEARITY --n_proteins $n_proteins 
    done
done

BASE_DIR="results/semisynth_hier_RF"
N_GENES_BASE=100
n_proteins = 1
for i in 1 2 3 4 5
do
#     # Figure 1
    python run_semisynthetic_jax.py --n_genes $N_GENES_BASE --data_noise $DATA_NOISE --sparsity $BASE_SPARSITY --beta_scale $BASE_BETA_SCALE  --tag main  --importance_statistic MLP --seed $i --nonlinearity $NONLINEARITY --n_hidden_xy 64 --n_mc_samples 100 --percent_dev $PERCENT_DEV --elementwise_nonlinearity $ELEMENTWISE_NONLINEARITY --n_proteins $n_proteins --base_dir $BASE_DIR --randomforest
done

# # ### POISSON
NONLINEARITY="id"
ELEMENTWISE_NONLINEARITY="square"
PERCENT_DEV=0.7
BASE_SPARSITY=0.1
N_GENES_BASE=500
DATA_NOISE="poisson"
n_proteins=5
base_beta_scale=1000.0


for i in 1 2 3 4
do
        python run_semisynthetic_jax.py --n_genes $N_GENES_BASE --data_noise $DATA_NOISE --sparsity $BASE_SPARSITY --beta_scale $base_beta_scale  --tag poisson  --importance_statistic MLP --seed $i --nonlinearity $NONLINEARITY --n_hidden_xy 32 --n_mc_samples 2000 --percent_dev $PERCENT_DEV --elementwise_nonlinearity $ELEMENTWISE_NONLINEARITY --n_proteins $n_proteins 
done
