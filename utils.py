import os
#limit the number of threds numpy/scipy are using
#os.environ["OMP_NUM_THREADS"] = "1"
import scipy.sparse
import numpy as np
from sklearn.metrics import pairwise_distances

from plotting import *

MACHINE_EPSILON = np.finfo(np.double).eps


def entropy_and_prob(d2=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """
    # Compute P-row and corresponding perplexity
    w = np.exp(-d2.copy() * beta)
    sum_w = sum(w)
    sum_w = np.maximum(sum_w, MACHINE_EPSILON)
    entropy = np.log(sum_w) + beta * np.sum(d2 * w) / sum_w
    p = w / sum_w
    return entropy, p, w

def tsne_condP(D2=np.array([]), perplexity=30.0,
               tol=1e-5,max_iter=50, verbose=0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """
    # Initialize some variables
    (n, n1) = D2.shape
    if n != n1:
        print("Error: D2 argument must be a square 2D array.")
        return -1
    condP = np.zeros((n, n))
    W = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):
        # Print progress
        if i % 500 == 0 and verbose >= 2:
            print("Computing conditional P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        D2i = D2[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (Hi, condPi, Wi) = entropy_and_prob(D2i, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = Hi - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < max_iter:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (Hi, condPi, Wi) = entropy_and_prob(D2i, beta[i])
            Hdiff = Hi - logU
            tries += 1

        # Set the final row of conditional P
        W[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = Wi
        condP[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = condPi
    bandwidth = np.sqrt(0.5 / beta)
    if verbose:
        print("Mean value of sigma: %f" % np.mean(bandwidth))
    return bandwidth, W, condP

def distance2(X=np.array([])):
    sum_X = np.sum(np.square(X), 1)
    D2 = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    D2 = np.maximum(D2, MACHINE_EPSILON)
    return(D2)

def beta_to_Pt(X, beta = None, time_step = 1, perplexity=30.0,
               thresh = 1e-10, save=False, save_dir = "./", 
               metric = 'euclidean', scaled = False, 
               from_file = False, file_dir = None, 
               svd_comp = 100, sparse_svd = False):
    n, p = X.shape
    if not from_file:
        if metric == 'euclidean':
            D = distance2(X)
        else:
            D = pairwise_distances(X, metric = metric)
        if beta is None:
            bandwidth, W, condP = tsne_condP(D, perplexity=perplexity)
            bandwidth2 = bandwidth ** 2
        else:
            bandwidth2 = 0.5/beta
            condP = np.exp(-D * beta.reshape(n, 1))
            np.fill_diagonal(condP, 0)  # VERY important setp
            condP = condP / np.sum(condP, axis = 1)
    else:
        data = np.fromfile(file_dir + '/condP_val.dat', dtype=np.dtype('d'))
        indices = np.fromfile(file_dir + '/condP_col.dat', dtype=np.dtype('uint32'))
        indptr = np.fromfile(file_dir + '/condP_row.dat', dtype=np.dtype('uint32'))
        beta = np.fromfile(file_dir + '/beta.dat', dtype=np.dtype('d'))
        bandwidth = np.sqrt(0.5 / beta)
        condP = scipy.sparse.csr_matrix((data, indices, indptr))
    
    if(time_step > 1):
        if not sparse_svd:
            condP = np.linalg.matrix_power(condP, time_step)
        else:
            u, s, vt = scipy.sparse.linalg.svds(condP, k = svd_comp)
            condP = u.dot(np.diag(s ** time_step).dot(vt))

    if scaled:
        mask = condP < (1/n)
        scale = bandwidth2 / np.max(bandwidth2)
        condP = condP / scale.reshape((n, 1))
        condP[mask] = 0  # this makes the between cluster probabilities still small
        
    if isinstance(condP, scipy.sparse.csr.csr_matrix):
        condP = condP.toarray()
    Pt = (condP + condP.T)
    np.fill_diagonal(Pt, 0) 
    Pt = Pt / np.sum(Pt)
    Pt[Pt < thresh] = 0
    print("Frac non-zero %f" %(np.sum(Pt >= thresh)/(n*n)))
    Pt = scipy.sparse.csr_matrix(Pt)
    
    if save:
        #cwd = os.getcwd()  
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(save_dir):
            raise ValueError('Directory does not exist')
        Pt.data.tofile(save_dir + '/P_val.dat')
        Pt.indices.tofile(save_dir + '/P_col.dat')
        Pt.indptr.tofile(save_dir + '/P_row.dat')
        #os.chdir(cwd)
    return {'Pt' : Pt, 'bandwidth' : bandwidth}


def maptpow_by_squaring(mat, t, thresh=1e-16):
    n1, n2 = mat.shape
    t = int(t)
    mat = mat.multiply(mat >= thresh)
    print("time step %d" %t)
    if n1 != n2:
        return -1
    if (t < 0):
        return -1
    elif (t == 0):
        if isinstance(condP, scipy.sparse.csr.csr_matrix):
            return  scipy.sparse.eye(n1)
        else:
            return  np.identity(n1)
    elif (t == 1):
        return  mat;
    else:
        square_prod = mat.dot(mat)
        if (t % 2 == 0):
            return maptpow_by_squaring(square_prod,  t / 2);
        else:
            return mat.dot(maptpow_by_squaring(square_prod, (t - 1) / 2))
    
# t = 10
# condP_t = exp_by_squaring(condP, t)
      
        
def filter_csr_cols(m, thresh):
    """
    Filter each sparse matrix so that the cumulative sum 
    for each row is not greater than the threshold.

    m must be a csr_matrix. m is modified in-place.
    """
    seq = np.arange(m.shape[0])
    for k in range(m.indptr.size - 1):
        start, end = m.indptr[k:k + 2]
        idx = np.argsort(-m.data[start:end])
        vals_ord = m.data[start:end][idx]
        idx_ord = m.indices[start:end][idx]
        csum = np.cumsum(vals_ord) 
        if(np.sum(csum >= thresh) < 1):
            continue
        last_nnz = idx_ord[csum >= thresh][0]
        outer_idx = range(start, end)[idx_ord]
        m.data[outer_idx[last_nnz:len(outer_idx)]] = 0
        
        
def get_res(res_dict, label, perps, tsteps=None, figsize = (15, 15), idx = 0, **kwargs):
    embd_time = {name : out['time'] for name, out in res_dict.items()}
    embd = {name : out['embedding'] for name, out in res_dict.items()}     
    # Plot examples:
    if tsteps is None:  
        ncopies = int(len(res_dict.keys())/len(perps))
        n = min(5, ncopies)
        items2plot = ['perp%d_it%d' %(p, i) for i in range(n) for p in perps]
        ncol = len(perps)
    else:
        items2plot = ['perp%d_tstep%d_it%d' %(p, t, idx) for p in perps for t in tsteps]
        ncol = len(tsteps)
    emd2plot = [embd[name] for name in items2plot]
    names2plot = [x.split('_it')[0] for x in items2plot]
    names2plot = [x.replace("_", ", ") for x in names2plot]    
    p = plot_embdeddings(emd2plot, color=label, 
                         name_lst=names2plot, figsize=figsize, 
                         s = 10, edgecolor='black', linewidth=0.1, 
                         **kwargs)
    return embd, embd_time

        