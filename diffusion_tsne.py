# This is a really basic function that does not do almost any sanity checks
#
# Usage example:
#	import sys; sys.path.append('../')
#   from fast_tsne import fast_tsne
#   import numpy as np
#	X = np.random.randn(1000, 50)
#	Z = fast_tsne(X, perplexity = 30)
#
# Originally written by Dmitry Kobak
# adapted by Lan Huong Nguyen


import os
import sys
import subprocess
import struct
import numpy as np

def diffusion_tsne(X, map_dims=2, distance_metric = "Euclidean",
                  theta=.5, nbody_algo='FFT', knn_algo='annoy',
                  perplexity=30, time_steps=1, scale_probs=False,
                  sigma=-1, K=-1, df=1, load_affinities=None,
                  max_iter=1000, learning_rate=200, early_exag_coeff=12,
                  stop_early_exag_iter=250,  late_exag_coeff=-1,
                  start_late_exag_iter=-1, momentum=.5, final_momentum=.8,
                  mom_switch_iter=250, no_momentum_during_exag=False,
                  nterms=3, intervals_per_integer=1, min_num_intervals=50,
                  n_trees=50, search_k=None, seed=-1, initialization=None,
                  return_loss=False, nthreads=None, data_path=None,
                  result_path=None, affinities_dir=None, save_files=False,
                  n_iter_check=50, row_thresh = -1.0, verbose=True):
    """Run t-SNE. This implementation supports exact t-SNE, Barnes-Hut t-SNE and FFT-accelerated
    interpolation-based t-SNE (FIt-SNE). This is a Python wrapper to a C++ executable.

    Parameters
    ----------
    X: 2D numpy array
        Array of observations (n times p)
    map_dims: int
        Number of embedding dimensions. Default 2. FIt-SNE supports only 1 or 2
        dimensions.
    distance_metric: string
        Distance metric to be used by knn_algo = 'annoy'
        (supported: "Euclidean", "Angular", or "Manhattan") or
        knn_algo = 'vp-tree' (supported "Euclidean", "Precomputed") algorithms.
    theta: double
        Set to 0 for exact t-SNE. If non-zero, then the code will use either
        Barnes Hut or FIt-SNE based on `nbody_algo`. If Barnes Hut, then theta
        determins the accuracy of BH approximation. Default 0.5.
    nbody_algo: {'Barnes-Hut', 'FFT'}
        If theta is nonzero, this determins whether to use FIt-SNE or Barnes
        Hut approximation. Default is 'FFT'.
    knn_algo: {'vp-tree', 'annoy'}
        Use exact nearest neighbours with VP trees (as in BH t-SNE) or
        approximate nearest neighbors with Annoy. Default is 'annoy'.
    perplexity: double
        Perplexity is used to determine the bandwidth of the Gaussian kernel in
        the input  space.  Default 30.
    time_steps: int
        Controls the bumber of time steps for diffusion process, equivaletntly
        the power to which the conditional probability matrix is raised to.
        Default is 1.
    scale_probs: boolean
        If True, the conditional probabilities are scaled by beta to reflect
        difference between local similarities between neighbourhoods due to
        varying density of data sampling.
        Default is False.
    sigma: boolean
        The standard deviation of the Gaussian kernel to be used for all points
        instead of choosing it adaptively via perplexity. Set to -1 to use
        perplexity. Default is -1.
    K: int
        The number of nearest neighbours to use when using fixed sigma instead
        of perplexity calibration. Set to -1 when perplexity is used.
        Default is -1.
    df: double
        Controls the degree of freedom of t-distribution. Must be positive.
        The actual degree of freedom is 2*df-1. The standard t-SNE choice of 1
        degree of freedom corresponds to df=1. Large df approximates Gaussian
        kernel. df<1 corresponds to heavier tails, which can often resolve
        substructure in the embedding. See Kobak et al. (2019) for details.
        Default is 1.0.
    load_affinities: {'load', 'save', 'save only', None}
        If 'save' and 'save only', input similarities (p_ij) are saved into
        a file. If 'save only' optimization and embeddinng will not be
        computed, only the input similarities are saved. If 'load', they are
        loaded from a file and not recomputed. If None, they are not saved and
        not loaded. Default is None.
    max_iter: int
            Number of gradient descent iterations. Default 1000.
    learning_rate: double
        Learning rate. Default 200.
    early_exag_coeff: double
        Coefficient for early exaggeration. Default 12.
    stop_early_exag_iter: int
        When to switch off early exaggeration. Default 250.
    late_exag_coeff: double
        Coefficient for late exaggeration. Set to -1 in order not to use late
        exaggeration. Default -1.
    start_late_exag_iter:
        When to start late exaggeration. Set to -1 in order not to use late
        exaggeration.  Default -1.
    momentum: double
        Initial value of momentum. Default 0.5.
    final_momentum: double
        The value of momentum to use later in the optimisation. Default 0.8.
    mom_switch_iter: int
        Iteration number to switch from momentum to final_momentum. Default 250.
    no_mometum_during_exag: boolean
        Whether to switch off momentum during the early exaggeration phase (can
        be useful for experiments with large exaggeration coefficients).
        Default is False.
    n_trees: int
        When using Annoy, the number of search trees to use. Default is 50.
    search_k: int
        When using Annoy, the number of nodes to inspect during search. Default
        is 3*perplexity*n_trees (or K*n_trees when using fixed sigma).
    nterms: int
        If using FIt-SNE, this is the number of interpolation points per
        sub-interval
    intervals_per_integer: double
        See min_num_intervals
    min_num_intervals: int
        The interpolation grid is chosen on each step of the gradient descent.
        If Y is the current embedding, let maxloc = ceiling(max(Y.flatten)) and
        minloc = floor(min(Y.flatten)), i.e. the points are contained in a
        [minloc, maxloc]^no_dims box. The number of intervals in each
        dimension is either min_num_intervals or ceiling((maxloc-minloc)/
        intervals_per_integer), whichever is larger. min_num_intervals must be
        a positive integer and intervals_per_integer must be positive real
        value. Defaults: min_num_intervals=50, intervals_per_integer = 1.
    seed: int
        Seed for random initialisation. Use -1 to initialise random number
        generator with current time. Default -1.
    initialization: numpy aray
         N x no_dims array to intialize the solution. Default: None.
    nthreads: int
        Number of threads to use. Default is None (i.e. use all available
        threads).
    return_loss: boolean
        If True, the function returns the loss values computed during
        optimisation together with the final embedding. If False, only the
        embedding is returned. Default is False.
    data_path: string
        full path and name to the data and parameter file. If not set,
        the data is saved automatically in a './data.dat' file.
    result_path: string
        full path and name to the file. If not set, the data is saves
        automatically in a './result.dat' file.
    affinities_dir: string
        full path to directory where affinity matrices are saved if
        load_affinity is 'save' or 'load'.
    n_iter_check: int
        check cost function every n_iter_check iteration
    verbose: boolean
        whether log messages should be printed.

    Returns
    -------
    Y: numpy array
        The embedding.
    loss: numpy array
        Loss values computed during optimisation. Only returned if return_loss is True.
    """

    cwd = os.getcwd()
    if data_path is None:
        data_path = os.getcwd() + '/data.dat'
    if result_path is None:
        result_path = os.getcwd() + '/result.dat'

    if (load_affinities is not None) and (affinities_dir is not None):
        if not os.path.exists(affinities_dir):
            print("Making directory %s" %affinities_dir)
            os.makedirs(affinities_dir)
        if os.path.isdir(affinities_dir):
            os.chdir(affinities_dir)
            data_path = 'data.dat'
            result_path = 'result.dat'
        else:
            raise ValueError("'affinities_dir' directory, %s, does not exist"
                             %affinities_dir)


    # X should be a numpy array of 64-bit doubles
    X = np.array(X).astype(float)
    n, d = X.shape

    if sigma > 0 and K > 0:
        perplexity = -1       # C++ requires perplexity=-1 in order to use sigma

    if search_k is None:
        if perplexity > 0:
            search_k = 3 * perplexity * n_trees
        elif perplexity == 0:
            search_k = 3 * np.max(perplexity_list) * n_trees
        else:
            search_k = K * n_trees

    if nbody_algo == 'Barnes-Hut':
        nbody_algo = 1
    elif nbody_algo == 'FFT': 
        nbody_algo = 2
    else:
        raise ValueError("'nbody_algo' must be one of {'Barnes-Hut', 'FFT'}")
        
    if knn_algo == 'vp-tree':
        knn_algo = 2
    elif knn_algo == "annoy": # ANNOY
        knn_algo = 1
    else:
        raise ValueError("'knn_algo' must be one of {'vp-tree', 'annoy'}")

    if load_affinities == 'load':
        load_affinities = 1
    elif load_affinities == 'save':
        load_affinities = 2
    elif load_affinities == 'save only':
        load_affinities = 3
    else:
        load_affinities = 0

    if distance_metric == "Euclidean":
        distance_metric = 0
    elif distance_metric=="Angular":
        distance_metric = 1
    elif distance_metric=="Manhattan":
        distance_metric = 2
    elif distance_metric=="Precomputed":
        distance_metric = -1
    else:
        print("Unsupported distance metric, %s, chosen.\n"  %distance_metric)

    if distance_metric > 0 and knn_algo == 2:
        print("Distance metric, %s, is not supported with 'vp-tree'" +
            "algorithm.\n" % distance_metric)
        return -1

    if distance_metric < 0 and knn_algo == 1:
        print("Distance metric, %s, is not supported with 'annoy'" +
            "algorithm.\n" % distance_metric)
        return -1

    if no_momentum_during_exag:
        no_momentum_during_exag = 1
    else:
        no_momentum_during_exag = 0

    if scale_probs:
        scale_probs = 1
    else:
        scale_probs = 0

    if verbose:
        verbose = 1
    else:
        verbose = 0
        
    if nthreads is None:
        nthreads = 1

#     if verbose:
#         print("Current directory %s" %os.getcwd())
#         print("data_path: %s" %data_path)
    # write data file
    with open(data_path, 'wb') as f:
        f.write(struct.pack('=i', nthreads))
        f.write(struct.pack('=i', distance_metric))
        f.write(struct.pack('=i', verbose))
        f.write(struct.pack('=i', n))
        f.write(struct.pack('=i', d))
        f.write(struct.pack('=i', map_dims))
        f.write(struct.pack('=d', theta))
        f.write(struct.pack('=i', knn_algo))
        f.write(struct.pack('=i', nbody_algo))
        f.write(struct.pack('=d', perplexity))
        f.write(struct.pack('=i', time_steps))
        f.write(struct.pack('=i', scale_probs))
        f.write(struct.pack('=d', sigma))
        f.write(struct.pack('=i', K))
        f.write(struct.pack('=d', df))
        f.write(struct.pack('=i', load_affinities))
        f.write(struct.pack('=i', max_iter))
        f.write(struct.pack('=i', n_iter_check))
        f.write(struct.pack('=d', learning_rate))
        f.write(struct.pack('=d', early_exag_coeff))
        f.write(struct.pack('=i', stop_early_exag_iter))
        f.write(struct.pack('=d', late_exag_coeff))
        f.write(struct.pack('=i', start_late_exag_iter))
        f.write(struct.pack('=d', momentum))
        f.write(struct.pack('=d', final_momentum))
        f.write(struct.pack('=i', mom_switch_iter))
        f.write(struct.pack('=i', no_momentum_during_exag))
        f.write(struct.pack('=i', n_trees))
        f.write(struct.pack('=i', search_k))
        f.write(struct.pack('=i', nterms))
        f.write(struct.pack('=d', intervals_per_integer))
        f.write(struct.pack('=i', min_num_intervals))
        f.write(struct.pack('=d', row_thresh))
        f.write(X.tobytes())
        f.write(struct.pack('=i', seed))

        if initialization is not None:
            initialization = np.array(initialization).astype(float)
            f.write(initialization.tobytes())

    # run t-sne
    if sys.platform == 'darwin': # 'darwin' denotes Mac OS X
        binary_filepath = '/bin/diffusion_tsne_mac'
    else:
        binary_filepath = '/bin/diffusion_tsne'

#     if verbose:
#         print("Using compiled binary file: %s" %binary_filepath)

    flag = subprocess.call([os.path.dirname(os.path.realpath(__file__)) +
        binary_filepath, data_path, result_path])

    if load_affinities > 2:
        return 
    
    if (flag != 0) :
        print('Diffusion tsne call failed in ' + str(os.getcwd()))
        os.chdir(cwd)
        return -1

    # read results file
    with open(result_path, 'rb') as f:
        n, = struct.unpack('=i', f.read(4))
        md, = struct.unpack('=i', f.read(4))
        sz = struct.calcsize('=d')
        buf = f.read(sz*n*md)
        x_tsne = [struct.unpack_from('=d', buf, sz*offset) for offset in range(n*md)]
        x_tsne = np.array(x_tsne).reshape((n,md))
        _, = struct.unpack('=i', f.read(4))
        buf = f.read(sz*max_iter)
        loss = [struct.unpack_from('=d', buf, sz*offset) for offset in range(max_iter)]
        loss = np.array(loss).squeeze()
        idx = ((np.arange(max_iter) + 1) % 50) > 0
        loss[idx] = np.nan

    if not save_files:
        os.remove(data_path)
        os.remove(result_path)

    os.chdir(cwd)
    if return_loss:
        return (x_tsne, loss)
    else:
        return x_tsne
