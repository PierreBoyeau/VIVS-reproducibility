"""Credits to GENIE3
https://github.com/vahuynh/GENIE3/blob/master/GENIE3_python/GENIE3.py

"""

from sklearn.tree import BaseDecisionTree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# from sklearn.ensemble import ExtraTreesRegressor
# from cuml import RandomForestRegressor
from numpy import *
import time
from operator import itemgetter
from multiprocessing import Pool


def compute_feature_importances(estimator):
    if isinstance(estimator, BaseDecisionTree):
        return estimator.tree_.compute_feature_importances(normalize=False)
    else:
        importances = [e.tree_.compute_feature_importances(normalize=False)
                       for e in estimator.estimators_]
        importances = array(importances)
        return sum(importances,axis=0) / len(estimator)


def GENIE3(
    expr_data,
    gene_names,
    regulators,
    targets,
    tree_method='RF',
    K='sqrt',
    ntrees=100,
    nthreads=1
):
    """Computation of tree-based scores for all putative regulatory links.
    
    Parameters
    ----------
    
    expr_data: numpy array
        Array containing gene expression values. Each row corresponds to a condition and each column corresponds to a gene.
        
    gene_names: list of strings, optional
        List of length p, where p is the number of columns in expr_data, containing the names of the genes. The i-th item of gene_names must correspond to the i-th column of expr_data.
        default: None
        
    regulators: list of strings, optional
        List containing the names of the candidate regulators. When a list of regulators is provided, the names of all the genes must be provided (in gene_names). When regulators is set to 'all', any gene can be a candidate regulator.
        default: 'all'
        
    tree-method: 'RF' or 'ET', optional
        Specifies which tree-based procedure is used: either Random Forest ('RF') or Extra-Trees ('ET')
        default: 'RF'
        
    K: 'sqrt', 'all' or a positive integer, optional
        Specifies the number of selected attributes at each node of one tree: either the square root of the number of candidate regulators ('sqrt'), the total number of candidate regulators ('all'), or any positive integer.
        default: 'sqrt'
         
    ntrees: positive integer, optional
        Specifies the number of trees grown in an ensemble.
        default: 1000
    
    nthreads: positive integer, optional
        Number of threads used for parallel computing
        default: 1
        
        
    Returns
    -------

    vim:
        An array in which the element (i,j) is the score of the edge directed from the i-th gene to the j-th gene. 
        All diagonal elements are set to zero (auto-regulations are not considered). 
        When a list of candidate regulators is provided, the scores of all the edges directed from a gene that is not a candidate regulator are set to zero.

    trees

    """

    
    # Get the indices of the candidate regulators
    if regulators == 'all':
        raise ValueError("Please provide a list of candidate regulators")
    
    regulator_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    target_idx = [i for i, gene in enumerate(gene_names) if gene in targets]

    
    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
    # VIM = zeros((ngenes, ngenes))
    trees = dict()
    for i_target in target_idx:
        target_name = gene_names[i_target]

        # _, estimator = GENIE3_single(
        estimator = GENIE3_single(
            expr_data,
            output_idx=i_target,
            input_idx=regulator_idx,
            tree_method=tree_method,
            K=K,
            ntrees=ntrees,
        )
        trees[target_name] = estimator
        # VIM[i,:] = vi
    # VIM = transpose(VIM)
    return trees


def GENIE3_single(
    expr_data,
    output_idx,
    input_idx,
    tree_method,
    K,
    ntrees,
    n_threads=16,
):
    
    ngenes = expr_data.shape[1]
    output = expr_data[:,output_idx]  # Expression of target gene
    output = output / std(output)  # Normalize output data
    
    # Remove target gene from candidate regulators
    # input_idx = input_idx[:]
    # if output_idx in input_idx:
    #     input_idx.remove(output_idx)
    expr_data_input = expr_data[:,input_idx]
    
    # Parameter K of the tree-based method
    if (K == 'all') or (isinstance(K,int) and K >= len(input_idx)):
        max_features = "auto"
    else:
        max_features = K
    
    if tree_method == 'RF':
        treeEstimator = RandomForestRegressor(n_estimators=200,max_features=max_features, max_depth=16, n_jobs=24)
        # treeEstimator = RandomForestRegressor(n_estimators=25,max_features=max_features, n_bins=15, n_streams=8)
    elif tree_method == 'ET':
        treeEstimator = ExtraTreesRegressor(n_estimators=ntrees,max_features=max_features)

    # Learn ensemble of trees
    treeEstimator.fit(expr_data_input, output)
    
    # Compute importance scores
    # feature_importances = compute_feature_importances(treeEstimator)
    return treeEstimator
        
        
        