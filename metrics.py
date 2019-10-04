# Here we use the coranking metric for evaluating the quality of embeddings 
# developed by Samuel Jackson available at https://github.com/samueljackson92/coranking
# The code is teh implementation of methods discussed by Lee and Verleysen published in:
#
# Lee, John A., and Michel Verleysen. "Quality assessment of dimensionality 
# reduction: Rank-based criteria." Neurocomputing 72.7 (2009): 1431-1443.
# Requires the following:
# git clone --recursive https://github.com/samueljackson92/coranking.git
# cd coranking/
# python setup.py install --user

import os
#limit the number of threds numpy/scipy are using
#os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing as mp
import numpy as np
import coranking
from coranking.metrics import trustworthiness, continuity, LCMC

from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.utils.graph_shortest_path import graph_shortest_path
# from scipy.sparse.csgraph import minimum_spanning_tree

from utils import beta_to_Pt, distance2

NCORES = 10


def QNK(Q, K):
    N = Q.shape[0]
    QNK = 0
    for k in range(K):
        for l in  range(K):
            QNK += Q[k, l]
    QNK /= (N*K)
    return QNK

def coranking_quality(X, Y, method = 'QNK', min_k=1, max_k = 50):
    Q = coranking.coranking_matrix(X, Y)
    if method == 'QNK':
        score = [QNK(Q, k) for k in range(min_k, max_k+1)]
    elif method == 'trustworthiness':
        score = trustworthiness(Q, min_k, max_k+1)
    elif method == 'continuity':
        score = continuity(Q, min_k, max_k+1)
    elif method == 'lcmc':
        score = LCMC(Q, min_k, max_k+1)
    else:
        raise ValueError("Unsupported method.")
    return score


def gDist(X, metric = 'euclidean', 
          perplexity = 30, thresh = 1e-10,
          path_method='auto', directed=False):
    if metric == 'euclidean':
        DX = distance2(X)
    else:
        DX = pairwise_distances(X, metric=metric)
    resKernel = beta_to_Pt(X, beta=None, time_step=1, 
                           perplexity=perplexity, thresh=thresh,
                           save=False, metric=metric)
    P = resKernel['Pt'].toarray()
    DX[P < thresh] = 0
    DX = csr_matrix(DX)
    GX  = graph_shortest_path(DX, method=path_method, 
                              directed=directed)    
    return GX

def geo_distance(X, metric = 'euclidean', 
                 perplexity = 30, thresh = 1e-10,
                 path_method='auto', directed=False):
    def perpGDist(perp):
        gD = gDist(X, metric= metric, perplexity=perp,
                   thresh=thresh, path_method=path_method, 
                   directed=directed)
        return gD
    if type(perplexity) == type(list([])):    
        results = {}
        pool = mp.Pool(processes = min(len(perplexity), mp.cpu_count()))
        for perp in perplexity:
            trial = 'perp' + str(perp) 
            results[trial] = pool.apply_async(gDist, args = (X, metric, perp))
        pool.close()
        pool.join()
        return {name : result.get() for name, result in results.items()}
    else:
        return perpGDist(perplexity)

    
def geo_rho(X, Y, GX = None, metric = 'euclidean', perplexity = 30, 
            thresh = 1e-10, path_method='auto', directed=False, 
            mean = False, parallel = False):
    MAX_NCORES = 50
    n = X.shape[0]; ny = Y.shape[0]
    if n != ny:
        raise ValueError('X and Y have inconsistent dimensions')
    if metric == 'euclidean':
        DY = distance2(Y)
    else:
        DY = pairwise_distances(Y, metric=metric)
    if GX is None:
        print('Estimating geodesic distance')
        GX  = geo_distance(X, metric=metric, perplexity=perplexity, 
                           thresh=thresh, path_method=path_method, 
                           directed=directed)
    if not parallel:
        rho_lst = []
        for i in range(n):
            rho, _ = spearmanr(GX[i, :], DY[i, :])  
            rho_lst.append(rho) 
    else:
        results = {}
        pool = mp.Pool(processes = min(n, MAX_NCORES, mp.cpu_count()))
        for i in range(n):
            results[i] = pool.apply_async(spearmanr, args = ((GX[i, :], DY[i, :])))
        pool.close()
        pool.join()
        rho_lst = {name : result.get() for name, result in results.items()}
        rho_lst = [rho for (rho, p) in rho_lst.values()]
    if mean:
        return np.mean(rho_lst)
    return rho_lst
    

# Tcsr = minimum_spanning_tree(X)
# Tcsr.toarray().astype(int)


