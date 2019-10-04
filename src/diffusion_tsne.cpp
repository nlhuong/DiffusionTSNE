/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "../lib/winlibs/stdafx.h"
#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <iostream>
#include <thread>
#include <functional>
#include <vector>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>                        /* clock_t, clock, CLOCKS_PER_SEC */
//#include <Rcpp.h>

#include "../include/vptree.h"
#include "../include/sptree.h"
#include "../include/diffusion_tsne.h"
#include "../include/helper.h"
#include "../include/random_walks.h"

#include "../include/nbodyfft.h"
#include "../include/annoylib.h"
#include "../include/kissrandom.h"
#include "../include/parallel_for.h"


#ifdef _WIN32
#include "../lib/winlibs/unistd.h"
#else
#include <unistd.h>
#endif

#define _CRT_SECURE_NO_WARNINGS

#ifdef _OPENMP
  #include <omp.h>
#endif

bool SAVE_INTERMEDIATE = false;

// extern "C" {
//   #include <R_ext/BLAS.h>
// }

using namespace std;


DIFFTSNE::DIFFTSNE() {
  load_affinities=0;
  map_dims=2; distance_metric="Euclidean";
  theta=.5, exact=(theta == .0);
  // Use 'annoy' for fiding neighbours and 'FFT' Interpolation for optimization
  knn_algo=1; nbody_algo=2;
  // Affinity parameters
  perplexity=30; sigma=-1; kneigh=-1; df=1; time_steps=1;
  scale_probs=false;

  rand_seed=-1; skip_random_init=false;
  max_iter=1000; niter_check=50; learning_rate=200;
  early_exag_coeff=12; stop_lying_iter=250;
  late_exag_coeff=-1; start_late_exag_iter=-1;
  momentum=.5; final_momentum=.8; mom_switch_iter=250;
  no_momentum_during_exag=false;
  nterms=3; intervals_per_integer=1; min_num_intervals=50;   // 'FFT' parameters
  n_trees=50; search_k=3 * perplexity * n_trees;              // K-NN Parameters
  row_thresh = 0.99;            //row sum threshold for pruning of matrix power
  nthreads=std::thread::hardware_concurrency();

  #ifdef _OPENMP
    int threads = nthreads;
    if (nthreads==0) {
      threads = omp_get_max_threads();
    }

    // Print notice whether OpenMP is used
    if (verbose) {
      printf("OpenMP is working. %d threads.\n", threads);
    }
  #endif
}

DIFFTSNE::DIFFTSNE(double NDims, double Theta, string DistanceMetric,
  double Perplexity, int TimeSteps, bool ScaleProbs, int DF, int KnnAlgo,
  int NbodyAlgo, int Max_iter, double Learning_rate, double Early_exag_coeff,
  int Stop_lying_iter, double Late_exag_coeff, int Start_late_exag_iter,
  double Momentum, double Final_momentum,  int Mom_switch_iter,
  bool NoMomentumDuringExag, int Nterms, int Intervals_per_integer,
  int Min_num_intervals, int N_trees, int Search_k, bool Init, int Seed,
  int Num_threads, int Niter_check, double Row_Thresh, bool Verbose) :
    map_dims(NDims), theta(Theta), distance_metric(DistanceMetric),
    perplexity(Perplexity), time_steps(TimeSteps), scale_probs(ScaleProbs),
    df(DF), knn_algo(KnnAlgo), nbody_algo(NbodyAlgo), max_iter(Max_iter),
    learning_rate(Learning_rate), early_exag_coeff(Early_exag_coeff),
    stop_lying_iter(Stop_lying_iter), late_exag_coeff(Learning_rate),
    start_late_exag_iter(Start_late_exag_iter), momentum(Momentum),
    final_momentum(Final_momentum), no_momentum_during_exag(NoMomentumDuringExag),
    mom_switch_iter(Mom_switch_iter), nterms(Nterms),
    intervals_per_integer(Intervals_per_integer),
    min_num_intervals(Min_num_intervals),
    n_trees(N_trees), search_k(Search_k), skip_random_init(Init),
    rand_seed(Seed), nthreads(Num_threads), niter_check(Niter_check),
    row_thresh(Row_Thresh), verbose(Verbose), exact(theta==.0){

    #ifdef _OPENMP
      int threads = nthreads;
      if (nthreads==0) {
        threads = omp_get_max_threads();
      }

      // Print notice whether OpenMP is used
      if (verbose) {
        printf("OpenMP is working. %d threads.\n", threads);
      }
    #endif
    sigma = -1;
    load_affinities = 0;
    return;
}

DIFFTSNE::DIFFTSNE(double NDims, double Theta, string DistanceMetric,
  double Sigma, int Kneigh, int TimeSteps, bool ScaleProbs, int DF,
  int KnnAlgo, int NbodyAlgo, int Max_iter, double Learning_rate,
  double Early_exag_coeff, int Stop_lying_iter, double Late_exag_coeff,
  int Start_late_exag_iter, double Momentum, double Final_momentum,
  int Mom_switch_iter, bool NoMomentumDuringExag,  int Nterms,
  int Intervals_per_integer, int Min_num_intervals, int N_trees, int Search_k,
  bool Init, int Seed, int Num_threads, int Niter_check, double Row_Thresh,
  bool Verbose) :
    map_dims(NDims), theta(Theta), distance_metric(DistanceMetric),
    sigma(Sigma), kneigh(Kneigh), time_steps(TimeSteps),
    scale_probs(ScaleProbs), df(DF), knn_algo(KnnAlgo), nbody_algo(NbodyAlgo),
    max_iter(Max_iter), learning_rate(Learning_rate),
    early_exag_coeff(Early_exag_coeff), stop_lying_iter(Stop_lying_iter),
    late_exag_coeff(Learning_rate), start_late_exag_iter(Start_late_exag_iter),
    momentum(Momentum), final_momentum(Final_momentum),
    mom_switch_iter(Mom_switch_iter),
    no_momentum_during_exag(NoMomentumDuringExag),
    nterms(Nterms), intervals_per_integer(Intervals_per_integer),
    min_num_intervals(Min_num_intervals),
    n_trees(N_trees), search_k(Search_k), skip_random_init(Init),
    rand_seed(Seed), nthreads(Num_threads), niter_check(Niter_check), row_thresh(Row_Thresh), verbose(Verbose), exact(theta==.0){

    #ifdef _OPENMP
      int threads = nthreads;
      if (nthreads==0) {
        threads = omp_get_max_threads();
      }

      // Print notice whether OpenMP is used
      if (verbose) {
        printf("OpenMP is working. %d threads.\n", threads);
      }
    #endif
    perplexity = -1;
    load_affinities = 0;
    return;
}



// Perform t-SNE on input matrix data
int DIFFTSNE::run(double* X, const unsigned int N, const int D, double* Y,
  double * itercost) {

  // Obtaining affinities ---------------------------
  if (load_affinities == 1) {  // Load affinities
    int err;
    if (exact) {
      err = loadDenseAffinities(P, N, verbose);
    } else {
      err = loadSparseAffinities(row_P, col_P, val_P, N, verbose);
    }
    if (err < 0) return err;
  } else {
    if (verbose) {
      printf("Computing input similarities...\n");
      std::cout << "Using 'distance_metric': " << distance_metric <<std::endl;
      if (perplexity < 0) {
        printf("Using no_dim = %d, fixed bandwidth = %f, "
          "k-nn = %d, and theta = %f.\n",
          map_dims, sigma, kneigh, theta);
      } else {
        // Some logging messages
        if (N - 1 < 3 * perplexity) {
          printf("Error: Perplexity too large for the number of data points!\n");
          exit(EXIT_FAILURE);
          // Rcpp::stop("Perplexity too large for the number of data points!\n");
        }
        printf("Using no_dims = %d, perplexity = %f, and theta = %f\n",
          map_dims, perplexity, theta);
      }
    }
    // TODO Add normalization steps for the input data?
    zeroMean(X, N, D);

    clock_t start = clock();

    if(exact) {
      // Compute similarities
      computeGaussianPerplexityExactly(X, N, D);

    } else {

      int K = 0;
      if (perplexity > 0) {
        K = 3*perplexity;
      } else if (sigma <= 0) {
        printf("Wrong sigma, %f, value.", sigma);
        exit(EXIT_FAILURE);
      } else if (kneigh <= 0){
        printf("Wrong kneigh, %d, value while using fixed sigma.", kneigh);
        exit(EXIT_FAILURE);
      } else {
        K = kneigh;
      }
      // Compute asymmetric pairwise input similarities
      if (knn_algo == 1) {
        int error_code = 0;
        printf("Using ANNOY for knn search, with parameters: n_trees %d and search_k %d\n", n_trees, search_k);
        if(distance_metric == "Euclidean") {
          error_code = computeGaussianPerplexity<Euclidean>(X, N, D, K);
        } else if(distance_metric == "Angular") {
          error_code = computeGaussianPerplexity<Angular>(X, N, D, K);
        }else if(distance_metric == "Manhattan") {
          error_code = computeGaussianPerplexity<Manhattan>(X, N, D, K);
        } else {
          std::cout << "'distance_metric', " << distance_metric
                   << " is not supported when using ANNOY." <<std::endl;
        }
        if (error_code < 0) return error_code;
      } else if (knn_algo == 2){
        printf("Using VP trees for nearest neighbor search\n");
        if (distance_metric == "Precomputed") {
         computeGaussianPerplexity<precomputed_distance>(X, N, D, K);
        } else if (distance_metric == "Euclidean") {
         computeGaussianPerplexity<euclidean_distance>(X, N, D, K);
        } else {
          std::cout<< "'distance_metric', " << distance_metric
                   << " is not supported when using VP-Tree." <<std::endl;
        }
      } else {
          printf("Unsupported knn_algo algorithm. Exiting");
          exit(EXIT_FAILURE);
      }

      if (verbose) {
        int numel = row_P[N];
        printf("Sparsity (fraction of nnzs) in asymmetric similarities: %d "
          "(%f)!\n", row_P[N], (double) row_P[N] / ((double) N * (double) N));
      }
    }
    clock_t end = clock();
    printf("Input similarities computed in %4.2f seconds!\n",
      (float) (end - start) / CLOCKS_PER_SEC);
    printf("-------------------------------------------------------------\n");
  }

  if (load_affinities >= 2) {
    int err = saveBetas(betas.data(), N, verbose);
    err = saveRowSums(sums_condP.data(), N, verbose);
    err = saveMeanDist(mean_knn_dist.data(), N, verbose);
    if(exact) {
      err = saveDenseAffinities(P.data(), N, verbose);
    } else {
      err = saveSparseAffinities(row_P.data(), col_P.data(), val_P.data(),
        N, verbose);
    }
    if (err < 0) return err;
    if (load_affinities > 2) {
      if (verbose) {
        printf("Exiting without computing an embedding. "
         "Error code is %d. \n", err);
       }
      return err;
    }
  }

  // Symmetrize input similarities
  if (verbose) printf("Symmetrizing...\n");
  if (exact) {
    symmetrizeWithEigen(P, N);
    // Normalize input similarities
    double sum_P = 0;
    for(size_t i = 0; i < P.size(); i++) sum_P += P[i];
    for(size_t i = 0; i < P.size(); i++) P[i] /= sum_P;
  } else {
    symmetrizeWithEigen(row_P, col_P, val_P, N);
    // Normalize input similarities
    double sum_P = .0;
    for(unsigned int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
    for(unsigned int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;

    if (verbose) {
      int numel = row_P[N];
      printf("Sparsity (fraction of nnzs) in symmetric similarities: %d "
        "(%f)!\n", row_P[N], (double) row_P[N] / ((double) N * (double) N));
    }
  }

  // Coputing embedding ----------------
  trainIterations(N, Y, itercost);

  if (exact) {
    P.clear();
  } else {
    row_P.clear();
    col_P.clear();
    val_P.clear();
  }
  betas.clear();
  sums_condP.clear();
  return 0;
}


// Perform t-SNE on nearest-neighbor input data
void DIFFTSNE::run(const int* nn_index, double* nn_dist,
  const unsigned int N, double* Y, double* itercost) {

  if (N - 1 < 3 * perplexity) {
    printf("Perplexity too large for the number of data points!\n");
    exit(EXIT_FAILURE);
    // Rcpp::stop("Perplexity too large for the number of data points!\n");
  }
  if (verbose) {
    printf("Computing input similarities...\n");
    printf("Using map_dims = %d, perplexity = %f, and theta = %f\n",
      map_dims, perplexity, theta);
  }
  clock_t start = clock();

  // Compute asymmetric pairwise input similarities
  computeGaussianPerplexity(nn_index, nn_dist, N, kneigh);

  // Symmetrize input similarities
  if (verbose) printf("Symmetrizing...\n");
  symmetrizeWithEigen(row_P, col_P, val_P, N);
  double sum_P = .0;
  for(unsigned int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
  for(unsigned int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;

  if (verbose) {
    if(exact) {
      printf("Sparsity (fraction of nnzs) in input similarities: %f)!\n",
       (double) row_P[N] / ((double) N * (double) N));
    }
    clock_t end = clock();
    printf("Input similarities computed in %4.2f seconds!",
      (float) (end - start) / CLOCKS_PER_SEC);
  }

  trainIterations(N, Y, itercost);
  return;
}


// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool DIFFTSNE::load_data(const char *data_path, double **data,
  int *n, int *d, double **Y) {

	FILE *h;
	if((h = fopen(data_path, "r+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}
  int DistanceMetric, Verbose, ScaleProbs, NoMomentumDuringExag;
	size_t result;       // need this to get rid of warnings that otherwise appear
  result = fread(&nthreads, sizeof(int), 1, h);
  result = fread(&DistanceMetric, sizeof(int), 1, h);
  result = fread(&Verbose, sizeof(int), 1, h);
	result = fread(n, sizeof(int), 1, h);     		      // number of datapoints
	result = fread(d, sizeof(int), 1, h);	  		        // original dimensionality
  result = fread(&map_dims, sizeof(int), 1, h);       // output dimensionality
	result = fread(&theta, sizeof(double), 1, h);		    // gradient accuracy
  result = fread(&knn_algo, sizeof(int),1,h);         // VP-trees or Annoy
  result = fread(&nbody_algo, sizeof(int),1,h);       // Barnes-Hut or FFT
	result = fread(&perplexity, sizeof(double), 1, h);	// perplexity
  result = fread(&time_steps, sizeof(int),1,h);
  result = fread(&ScaleProbs, sizeof(int),1,h);
	result = fread(&sigma, sizeof(double),1,h);              // input kernel width
  result = fread(&kneigh, sizeof(int),1,h);                 // no. of neighbours
  result = fread(&df, sizeof(double),1,h);      // no. of kernel deg. of freedom
  result = fread(&load_affinities, sizeof(int),1,h);
	result = fread(&max_iter, sizeof(int), 1, h);     // maximum no. of iterations
  result = fread(&niter_check, sizeof(int), 1, h);
  result = fread(&learning_rate, sizeof(double),1,h);           // learning rate
  result = fread(&early_exag_coeff, sizeof(double),1,h);
	result = fread(&stop_lying_iter, sizeof(int), 1, h);
  result = fread(&late_exag_coeff, sizeof(double),1,h);
  result = fread(&start_late_exag_iter, sizeof(int),1,h);
	result = fread(&momentum, sizeof(double),1,h);            // initial momentum
	result = fread(&final_momentum, sizeof(double),1,h);      // final momentum
  result = fread(&mom_switch_iter, sizeof(int),1,h);
  result = fread(&NoMomentumDuringExag, sizeof(int),1,h);
	result = fread(&n_trees, sizeof(int),1,h);           // no. of trees for Annoy
	result = fread(&search_k, sizeof(int),1,h);              // search_k for Annoy
	result = fread(&nterms, sizeof(int),1,h);                    // FFT parameter
	result = fread(&intervals_per_integer, sizeof(double),1,h);  // FFT parameter
	result = fread(&min_num_intervals, sizeof(int),1,h);         // FFT parameter
  result = fread(&row_thresh, sizeof(double),1,h);      // thresh for mat power

  if((nbody_algo == 2) && (map_dims > 2)){
    printf("FFT interpolation scheme supports only 1 or 2 output dimensions, not %d\n", map_dims);
    exit(EXIT_FAILURE);
  }

  exact = (theta == 0.0);

  if (DistanceMetric == 0) distance_metric="Euclidean";
  else if (DistanceMetric == 1) distance_metric="Angular";
  else if (DistanceMetric == 2) distance_metric="Manhattan";
  else if (DistanceMetric == -1) distance_metric="Precomputed";
  else{
    std::cout<< "'distance_metric', " << distance_metric
             << " is not supported." <<std::endl;
    exit(EXIT_FAILURE);
  }

  no_momentum_during_exag = true;
  if (NoMomentumDuringExag == 0) no_momentum_during_exag = false;
  scale_probs = false;
  if (ScaleProbs == 1) scale_probs = true;
  verbose = false;
  if (Verbose == 1) verbose = true;

  if(verbose) {
  	printf("Reading the following parameters:\n"
        "\t (%d x %d) input dataset, map_dims %d,\n"
        "\t theta %lf, knn_algo %d, nbody_algo %d,\n"
        "\t perplexity %lf, sigma %lf,\n"
        "\t time_steps %d, scale_probs %d, t-dist df %lf,\n"
        "\t max_iter %d, niter_check %d, learning_rate %lf,\n"
        "\t early_exag_coeff %lf, stop_lying_iter %d,\n"
        "\t late_exag_coeff %lf, start_late_exag_iter %d,\n"
        "\t momentum %lf, final_momentum %lf,\n"
  			"\t mom_switch_iter %d, no_momentum_during_exag %d,\n"
        "\t n_trees %d, search_k %d, K %d, nterms %d,\n"
  			"\t interval_per_integer %lf, min_num_intervals %d,\n",
  			*n, *d, map_dims, theta, knn_algo, nbody_algo,
        perplexity, sigma, time_steps, ScaleProbs, df,
        max_iter, niter_check, learning_rate,
        early_exag_coeff, stop_lying_iter,
        late_exag_coeff, start_late_exag_iter,
        momentum, final_momentum, mom_switch_iter, NoMomentumDuringExag,
        n_trees, search_k, kneigh, nterms, intervals_per_integer,
        min_num_intervals);
  }

  if(!feof(h)) {
  	*data = (double*) malloc(*d * *n * sizeof(double));
  	if(*data == NULL) {
      throw std::bad_alloc();
      // Rcpp::stop("Memory allocation failed!\n");
    }
  	result = fread(*data, sizeof(double), *n * *d, h);           // the data
    if(verbose) {
      printf("Read the %i x %i data matrix successfully. X[0,0] = %lf\n",
        *n, *d, *data[0]);
    }
  }
  if(!feof(h)) {
		result = fread(&rand_seed, sizeof(int), 1, h);             // random seed
	}

	// allocating space for the t-sne solution
	*Y = (double*) malloc(*n * map_dims * sizeof(double));
	if(*Y == NULL) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }
	// if the file has not ended, the remaining part is the initialization
	if(!feof(h)){
		result = fread(*Y, sizeof(double), *n * map_dims, h);
		if(result < *n * map_dims){
			skip_random_init = false;
		} else {
			skip_random_init = true;
		}
	} else{
		skip_random_init = false;
	}
  fclose(h);

  if(skip_random_init && verbose){
    printf("Read the initialization successfully.\n");
  }
  if(verbose) {
    printf("-------------------------------------------------------------\n");
    printf("Using %d threads...\n", nthreads);
  }

	return true;
}


// Function that saves map to a t-SNE file
void DIFFTSNE::save_data(const char *result_path, const double* data,
  int n, const std::vector<double> itercost) {
	// Open file, write first 2 integers and then the data
	FILE *h;
	if((h = fopen(result_path, "w+b")) == NULL) {
  	printf("Error: could not open data file.\n");
  	return;
	}
  printf("Itercost size is: %ld\n", itercost.size());
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&map_dims, sizeof(int), 1, h);
	fwrite(data, sizeof(double), n * map_dims, h);
	fwrite(&max_iter, sizeof(int), 1, h);
	fwrite(itercost.data(), sizeof(double), itercost.size(), h);
	fclose(h);
	printf("Wrote the %i x %i data matrix successfully.\n", n, map_dims);
}


// =============================================================================
// Private Functions
// =============================================================================

// Main optimization iterator
void DIFFTSNE::trainIterations(const unsigned int N, double* Y,
  double* itercost) {

  if(!exact && (nbody_algo != 2) && (nbody_algo != 1)) {
    printf("Error: Undefined gradient algorithm");
    exit(EXIT_FAILURE);
    //Rcpp::stop("Undefined gradient algorithm!\n");
  }

  if (exact && verbose){
    printf("Using exact algorithm.\n");
  } else if (!exact && (nbody_algo == 2) && verbose) {
    printf("Using FIt-SNE approximation.\n");
  } else if (!exact && (nbody_algo == 1) && verbose) {
    printf("Using the Barnes-Hut approximation.\n");
  }

  // Allocate some memory
  double* dY    = (double*) malloc(N * map_dims * sizeof(double));
  double* uY    = (double*) malloc(N * map_dims * sizeof(double));
  double* gains = (double*) malloc(N * map_dims * sizeof(double));

  if(dY == NULL || uY == NULL || gains == NULL) {
    //Rcpp::stop("Memory allocation failed!\n");
    throw std::bad_alloc();
  }
  for(unsigned int i = 0; i < N * map_dims; i++)    uY[i] =  .0;
  for(unsigned int i = 0; i < N * map_dims; i++) gains[i] = 1.0;

	// Initialize solution (randomly), if not already done
	if (!skip_random_init) {
    if (verbose) printf("Randomly initializing the solution.\n");
    if (rand_seed >= 0) {
      if (verbose) printf("Using random seed: %d\n", rand_seed);
      srand((unsigned int) rand_seed);
    } else {
      if (verbose) printf("Using current time as random seed...\n");
      srand(time(NULL));
    }
    for(unsigned int i = 0; i < N * map_dims; i++) {
      Y[i] = randn() * .0001;
    }
    if (verbose) printf("Y[0] = %lf\n", Y[0]);
  } else {
	  if (verbose) printf("Using the given initialization.\n");
  }

  // Lie about the P-values
  if (verbose) printf("Exaggerating Ps by %f\n", early_exag_coeff);
  if(exact) {
    for(unsigned long i = 0; i < (unsigned long)N * N; i++)
        P[i] *= early_exag_coeff;
  } else {
    for(unsigned long i = 0; i < row_P[N]; i++)
        val_P[i] *= early_exag_coeff;
  }

  if(SAVE_INTERMEDIATE) save_intermediate_y(0, Y, N, map_dims);
  if (no_momentum_during_exag && verbose) {
    printf("No momentum during the exaggeration phase.\n");
  } else {
    printf("Will use momentum during exaggeration phase\n");
  }

  if (verbose) printf("Learning embedding...\n");
  clock_t start = clock(), end;
  float block_time=0, total_time=0;
  for(int iter = 0; iter < max_iter; iter++) {
    // Stop lying about the P-values after a while, and switch momentum
    if(iter == stop_lying_iter) {
      if (verbose) printf("Unexaggerating Ps by %f\n", early_exag_coeff);
      if(exact) {
        for(unsigned long i = 0; i < (unsigned long)N * N; i++)
          P[i] /= early_exag_coeff;
      } else {
        for(unsigned int i = 0; i < row_P[N]; i++)
          val_P[i] /= early_exag_coeff;
      }
    }
    // Start lying again about the P-values after a while
    if (iter == start_late_exag_iter) {
      if (verbose) printf("Exaggerating Ps by %f\n", late_exag_coeff);
      if (exact) {
        for(unsigned long i = 0; i < (unsigned long)N * N; i++)
          P[i] *= late_exag_coeff;
      } else {
        for(unsigned int i = 0; i < row_P[N]; i++)
          val_P[i] *= late_exag_coeff;
      }
    }
    // Switch momentum
    if(iter == mom_switch_iter) momentum = final_momentum;

    // Compute (approximate) gradient
    if(exact) { computeExactGradient(P.data(), Y, N, dY); }
    else {
      if (nbody_algo == 1) {                    // Barnes-Hut approximation
        computeGradient(P.data(), row_P.data(), col_P.data(), val_P.data(),
          Y, N, dY, theta);
      } else {        // FFT accelerated interpolation
        if (map_dims == 1) {
          computeFftGradientOneD(P.data(), row_P.data(), col_P.data(),
            val_P.data(), Y, N, dY);
        } else {
          if (df == 1.0) {
            computeFftGradient(P.data(), row_P.data(), col_P.data(),
              val_P.data(), Y, N, dY);
          } else {
            computeFftGradientVariableDf(P.data(), row_P.data(), col_P.data(),
              val_P.data(), Y, N, dY);
          }
        }
      }
    }

    // Update iterates
    if (no_momentum_during_exag && (iter < stop_lying_iter)) {
      // During early exaggeration or compression, no trickery (i.e. no
      // momentum, or gains). Just good old fashion gradient descent
      for (int i = 0; i < N * map_dims; i++) { Y[i] = Y[i] - dY[i]; }
    } else {
      // Update gains
      for(unsigned int i = 0; i < N * map_dims; i++) {
        gains[i] = (sign(dY[i]) != sign(uY[i]))
                    ? (gains[i] + .2)
                    : (gains[i] * .8);
      }

      for(unsigned int i = 0; i < N * map_dims; i++) {
        if(gains[i] < .01) gains[i] = .01;
      }
      // Perform gradient update (with momentum and gains)
      for(unsigned int i = 0; i < N * map_dims; i++)
        uY[i] = momentum * uY[i] - learning_rate * gains[i] * dY[i];

      for(unsigned int i = 0; i < N * map_dims; i++)  Y[i] = Y[i] + uY[i];
    }

    // Make solution zero-mean
    zeroMean(Y, N, map_dims);

    // Print out progress
    if((iter+1) % niter_check == 0 || (iter +1) == max_iter) {
      end = clock();
      double C = .0;
      if(exact) { C = evaluateError(P.data(), Y, N); }
      else { // doing approximate computation here!
        if (nbody_algo == 1) {
          C = evaluateError(row_P.data(), col_P.data(), val_P.data(), Y, N);
        } else {
          C = evaluateErrorFft(row_P.data(), col_P.data(), val_P.data(), Y, N);
        }
      }
      // Adjusting the KL divergence if exaggeration is currently turned
      // on. See https://github.com/pavlin-policar/fastTSNE/blob/master/notes/notes.pdf, Section 3.2
      if (iter < stop_lying_iter && stop_lying_iter != -1) {
        C = C/early_exag_coeff - log(early_exag_coeff);
      }
      if (iter >= start_late_exag_iter && start_late_exag_iter != -1) {
        C = C/late_exag_coeff - log(late_exag_coeff);
      }
      itercost[iter] = C;
      if (verbose) {
        block_time = (float) (end - start) / CLOCKS_PER_SEC;
        total_time += block_time;
        printf("Iteration %d (50 iterations in %4.2f seconds), cost %f\n", iter+1, block_time, C);
      }
      start = clock();
    }
    if(SAVE_INTERMEDIATE) save_intermediate_y(iter, Y, N, map_dims);
  } // End iteration forloop

  if (verbose) {
    end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;
    printf("Fitting performed in %4.2f seconds.\n",total_time);
  }

  // Clean up memory
  free(dY);
  free(uY);
  free(gains);
  return;
}

// Exact gradient of the t-SNE cost function -----------------------------------
void DIFFTSNE::computeExactGradient(const double *P, double *Y,
  const unsigned int N, double *dC) {

  // Make sure the current gradient contains zeros
  for (int i = 0; i < N * map_dims; i++) dC[i] = 0.0;

  // Compute the squared Euclidean distance matrix
  auto *DDy = (double *) malloc(N * N * sizeof(double));
  if (DDy == NULL) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n")
  }
  computeSquaredEuclideanDistance(Y, N, map_dims, DDy);

  // Compute Q-matrix and normalization sum
  auto *Q = (double *) malloc(N * N * sizeof(double));
  if (Q == NULL){
    throw std::bad_alloc();throw std::bad_alloc();
    //Rcpp::stop("Memory allocation failed!\n")
  }
  auto *Qpow = (double *) malloc(N * N * sizeof(double));
  if (Qpow == NULL) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n")
  }
  double sum_Q = .0;
  int nN = 0;
  for (unsigned int n = 0; n < N; n++) {
    for (unsigned int m = 0; m < N; m++) {
      if (n != m) {
        //Q[nN + m] = 1.0 / pow(1.0 + DDy[nN + m]/(double)df, df);
        Q[nN + m] = 1.0 / (1.0 + DDy[nN + m]/(double)df);
        Qpow[nN + m] = pow(Q[nN + m], df);
        sum_Q += Qpow[nN + m];
      }
    }
    nN += N;
  }

  // Perform the computation of the gradient
  nN = 0;
  int nD = 0;
  for (unsigned int n = 0; n < N; n++) {
    int mD = 0;
    for (unsigned int m = 0; m < N; m++) {
      if (n != m) {
        double mult = (P[nN + m] - (Qpow[nN + m] / sum_Q)) * (Q[nN + m]);
        for (int d = 0; d < map_dims; d++) {
          dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
        }
      }
      mD += map_dims;
    }
    nN += N;
    nD += map_dims;
  }
  free(Q);      Q  = NULL;
  free(Qpow); Qpow = NULL;
  free(DDy);  DDy  = NULL;
}


// Barnes-Hut gradient of the t-SNE cost function ------------------------------
void DIFFTSNE::computeGradient(const double *P, unsigned int *inp_row_P,
    unsigned int *inp_col_P, double *inp_val_P, double *Y,
    const unsigned int N, double *dC, const double theta) {

  // Construct space-partitioning tree on current map
  SPTree *tree = new SPTree(map_dims, Y, N);

  // Compute all terms required for t-SNE gradient
  double *pos_f = (double *) calloc(N * map_dims, sizeof(double));
  double *neg_f = (double *) calloc(N * map_dims, sizeof(double));
  if (pos_f == NULL || neg_f == NULL) {
    throw std::bad_alloc();
    //Rcpp::stop("Memory allocation failed!\n");
  }
  tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f, nthreads);
  // Storing the output to sum in single-threaded mode; avoid randomness in
  // rounding errors.
  std::vector<double> output(N);
  PARALLEL_FOR(nthreads, N, {
    output[loop_i] = tree->computeNonEdgeForces(loop_i, theta,
      neg_f + loop_i * map_dims);
  });

  double sum_Q = .0;
  for (unsigned int n=0; n<N; ++n) {
    sum_Q += output[n];
  }
  // Compute final t-SNE gradient
  for(unsigned int i = 0; i < N * map_dims; i++) {
     dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
  }

  free(pos_f);
  free(neg_f);
  delete tree;
}


// FFT interpolation gradient of 1D t-SNE cost function ------------------------
void DIFFTSNE::computeFftGradientOneD(const double *P,
  unsigned int *inp_row_P, unsigned int *inp_col_P,
  double *inp_val_P, double *Y, const unsigned int N, double *dC) {

  // Zero out the gradient
  for (int i = 0; i < N * map_dims; i++) dC[i] = 0.0;

  // Push all the points at which we will evaluate
  // Y is stored row major, with a row corresponding to a single point
  // Find the min and max values of Ys
  double y_min = INFINITY;
  double y_max = -INFINITY;
  for (unsigned long i = 0; i < N; i++) {
      if (Y[i] < y_min) y_min = Y[i];
      if (Y[i] > y_max) y_max = Y[i];
  }

  auto n_boxes = static_cast<int>(fmax(min_num_intervals,
    (y_max - y_min) / intervals_per_integer));

  // The number of "charges" or s+2 sums i.e. number of kernel sums
  int n_terms = 3;

  auto *chargesQij = new double[N * n_terms];
  auto *potentialsQij = new double[N * n_terms]();

  // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
  for (unsigned long j = 0; j < N; j++) {
    chargesQij[j * n_terms + 0] = 1;
    chargesQij[j * n_terms + 1] = Y[j];
    chargesQij[j * n_terms + 2] = Y[j] * Y[j];
  }

  auto *box_lower_bounds = new double[n_boxes];
  auto *box_upper_bounds = new double[n_boxes];
  auto *y_tilde_spacings = new double[nterms];
  auto *y_tilde = new double[nterms * n_boxes]();
  auto *fft_kernel_vector = new complex<double>[
    2 * nterms * n_boxes];

  precompute(y_min, y_max, n_boxes, nterms, &squared_cauchy,
      box_lower_bounds, box_upper_bounds, y_tilde_spacings, y_tilde,
      fft_kernel_vector);
  nbodyfft(N, n_terms, Y, chargesQij, n_boxes, nterms,
      box_lower_bounds, box_upper_bounds, y_tilde_spacings, y_tilde,
      fft_kernel_vector, potentialsQij);

  delete[] box_lower_bounds;
  delete[] box_upper_bounds;
  delete[] y_tilde_spacings;
  delete[] y_tilde;
  delete[] fft_kernel_vector;

  // Compute the normalization constant Z or sum of q_{ij}. This expression is different from the one in the original
  // paper, but equivalent. This is done so we need only use a single kernel (K_2 in the paper) instead of two
  // different ones. We subtract N at the end because the following sums over all i, j, whereas Z contains i \neq j
  double sum_Q = 0;
  for (unsigned long i = 0; i < N; i++) {
    double phi1 = potentialsQij[i * n_terms + 0];
    double phi2 = potentialsQij[i * n_terms + 1];
    double phi3 = potentialsQij[i * n_terms + 2];

    sum_Q += (1 + Y[i] * Y[i]) * phi1 - 2 * (Y[i] * phi2) + phi3;
  }
  sum_Q -= N;
  this->current_sum_Q = sum_Q;

  // Now, figure out the Gaussian component of the gradient. This corresponds
  //to the "attraction" term of the gradient. It was calculated using a fast
  // KNN approach, so here we just use the results that were passed to this
  // function  unsigned int ind2 = 0;
  double *pos_f = new double[N];

  PARALLEL_FOR(nthreads, N, {
      double dim1 = 0;
      for (unsigned int i = inp_row_P[loop_i]; i < inp_row_P[loop_i + 1]; i++) {
          // Compute pairwise distance and Q-value
          unsigned int ind3 = inp_col_P[i];
          double d_ij = Y[loop_i] - Y[ind3];
          double q_ij = 1 / (1 + d_ij * d_ij);
          dim1 += inp_val_P[i] * q_ij * d_ij;
      }
          pos_f[loop_i] = dim1;

  });

  // Make the negative term, or F_rep in the equation 3 of the paper
  double *neg_f = new double[N];
  for (unsigned int n = 0; n < N; n++) {
    neg_f[n] = (Y[n] * potentialsQij[n * n_terms] -
      potentialsQij[n * n_terms + 1]) / sum_Q;

    dC[n] = pos_f[n] - neg_f[n];
  }

  delete[] chargesQij;
  delete[] potentialsQij;
  delete[] pos_f;
  delete[] neg_f;
}


// FFT interpolation gradient of variable degree of freedom --------------------
void DIFFTSNE::computeFftGradientVariableDf(const double *P,
  unsigned int *inp_row_P, unsigned int *inp_col_P,
  double *inp_val_P, double *Y, const unsigned int N, double *dC) {

  // Zero out the gradient
  for (int i = 0; i < N * map_dims; i++) dC[i] = 0.0;

  // For convenience, split the x and y coordinate values
  auto *xs = new double[N];
  auto *ys = new double[N];

  double min_coord = INFINITY;
  double max_coord = -INFINITY;
  // Find the min/max values of the x and y coordinates
  for (unsigned long i = 0; i < N; i++) {
    xs[i] = Y[i * 2 + 0];
    ys[i] = Y[i * 2 + 1];
    if (xs[i] > max_coord) max_coord = xs[i];
    else if (xs[i] < min_coord) min_coord = xs[i];
    if (ys[i] > max_coord) max_coord = ys[i];
    else if (ys[i] < min_coord) min_coord = ys[i];
  }
  // Compute the number of boxes in a single dimension and the total number
  // of boxes in 2d
  auto n_boxes_per_dim = static_cast<int>(fmax(min_num_intervals,
    (max_coord - min_coord) / intervals_per_integer));

  // FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d
  // 11^e 13^f, where e+f is either 0 or 1, and the other exponents are
  // arbitrary
  int allowed_n_boxes_per_dim[20] = {25,36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140,150, 175, 200};
  if ( n_boxes_per_dim < allowed_n_boxes_per_dim[19] ) {
      //Round up to nearest grid point
      int chosen_i;
      for (chosen_i =0; allowed_n_boxes_per_dim[chosen_i]< n_boxes_per_dim; chosen_i++);
      n_boxes_per_dim = allowed_n_boxes_per_dim[chosen_i];
  }

  // The number of "charges" or s+2 sums i.e. number of kernel sums
  int squared_n_terms = 3;
  auto *SquaredChargesQij = new double[N * squared_n_terms];
  auto *SquaredPotentialsQij = new double[N * squared_n_terms]();

  // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
  for (unsigned long j = 0; j < N; j++) {
    SquaredChargesQij[j * squared_n_terms + 0] = xs[j];
    SquaredChargesQij[j * squared_n_terms + 1] = ys[j];
    SquaredChargesQij[j * squared_n_terms + 2] = 1;
  }

  // Compute the number of boxes in a single dimension and the total number of boxes in 2d
  int n_boxes = n_boxes_per_dim * n_boxes_per_dim;

  auto *box_lower_bounds = new double[2 * n_boxes];
  auto *box_upper_bounds = new double[2 * n_boxes];
  auto *y_tilde_spacings = new double[nterms];
  int nterms_1d = nterms * n_boxes_per_dim;
  auto *x_tilde = new double[nterms_1d]();
  auto *y_tilde = new double[nterms_1d]();
  auto *fft_kernel_tilde = new complex<double>[
    2 * nterms_1d * 2 * nterms_1d];

  precompute_2d(max_coord, min_coord, max_coord, min_coord, n_boxes_per_dim,
    nterms, &squared_general_kernel_2d, box_lower_bounds,
    box_upper_bounds, y_tilde_spacings, x_tilde, y_tilde, fft_kernel_tilde,
    df);
  n_body_fft_2d(N, squared_n_terms, xs, ys, SquaredChargesQij,
    n_boxes_per_dim, nterms, box_lower_bounds,
    box_upper_bounds, y_tilde_spacings, fft_kernel_tilde,
    SquaredPotentialsQij, nthreads);

  int not_squared_n_terms = 1;
  auto *NotSquaredChargesQij = new double[N * not_squared_n_terms];
  auto *NotSquaredPotentialsQij = new double[N * not_squared_n_terms]();

  // Prepare the terms that we'll use for the sum i.e. the repulsive forces
  for (unsigned long j = 0; j < N; j++) {
      NotSquaredChargesQij[j * not_squared_n_terms + 0] = 1;
  }

  precompute_2d(max_coord, min_coord, max_coord, min_coord, n_boxes_per_dim,
    nterms, &general_kernel_2d, box_lower_bounds,
    box_upper_bounds, y_tilde_spacings, x_tilde, y_tilde,
    fft_kernel_tilde,df);
  n_body_fft_2d(N, not_squared_n_terms, xs, ys, NotSquaredChargesQij,
    n_boxes_per_dim, nterms, box_lower_bounds,
    box_upper_bounds, y_tilde_spacings, fft_kernel_tilde,
    NotSquaredPotentialsQij, nthreads);

  // Compute the normalization constant Z or sum of q_{ij}.
  double sum_Q = 0;
  for (unsigned long i = 0; i < N; i++) {
      double h1 = NotSquaredPotentialsQij[i * not_squared_n_terms+ 0];
      sum_Q += h1;
  }
  sum_Q -= N;

    // Now, figure out the "attraction" (Gaussian) terms of the gradient.
    // This was calculated using a fast KNN approach, so here we just use
    // the results that were passed to this function
    unsigned int ind2 = 0;
    double *pos_f = new double[N * 2];
    // Loop over all edges in the graph
    PARALLEL_FOR(nthreads, N, {
      double dim1 = 0;
      double dim2 = 0;

      for (unsigned int i = inp_row_P[loop_i]; i < inp_row_P[loop_i + 1]; i++) {
      // Compute pairwise distance and Q-value
          unsigned int ind3 = inp_col_P[i];
          double d_ij = (xs[loop_i] - xs[ind3]) * (xs[loop_i] - xs[ind3]) + (ys[loop_i] - ys[ind3]) * (ys[loop_i] - ys[ind3]);
          double q_ij = 1 / (1 + d_ij/df);

          dim1 += inp_val_P[i] * q_ij * (xs[loop_i] - xs[ind3]);
          dim2 += inp_val_P[i] * q_ij * (ys[loop_i] - ys[ind3]);
      }
      pos_f[loop_i * 2 + 0] = dim1;
      pos_f[loop_i * 2 + 1] = dim2;
    });

    // Make the negative term, or F_rep in the equation 3 of the paper
    double *neg_f = new double[N * 2];
    for (unsigned int i = 0; i < N; i++) {
        double h2 = SquaredPotentialsQij[i * squared_n_terms];
        double h3 = SquaredPotentialsQij[i * squared_n_terms + 1];
        double h4 = SquaredPotentialsQij[i * squared_n_terms + 2];
        neg_f[i * 2 + 0] = ( xs[i] *h4 - h2 ) / sum_Q;
        neg_f[i * 2 + 1] = (ys[i] *h4 - h3 ) / sum_Q;

        dC[i * 2 + 0] = (pos_f[i * 2] - neg_f[i * 2]);
        dC[i * 2 + 1] = (pos_f[i * 2 + 1] - neg_f[i * 2 + 1]);
    }

    this->current_sum_Q = sum_Q;

/*  FILE *fp = nullptr;
    char buffer[500];
    sprintf(buffer, "temp/fft_gradient%d.txt", itTest);
    fp = fopen(buffer, "w"); // Open file for writing
    for (int i = 0; i < N; i++) {
            fprintf(fp, "%d,%.12e,%.12e\n", i, neg_f[i * 2] , neg_f[i * 2 + 1]);
    }
    fclose(fp);*/

    delete[] pos_f;
    delete[] neg_f;
    delete[] SquaredPotentialsQij;
    delete[] NotSquaredPotentialsQij;
    delete[] SquaredChargesQij;
    delete[] NotSquaredChargesQij;
    delete[] xs;
    delete[] ys;
    delete[] box_lower_bounds;
    delete[] box_upper_bounds;
    delete[] y_tilde_spacings;
    delete[] y_tilde;
    delete[] x_tilde;
    delete[] fft_kernel_tilde;
}

// Compute the gradient of the t-SNE cost function using the FFT interpolation
// based approximation
void DIFFTSNE::computeFftGradient(const double *P,
  unsigned int *inp_row_P, unsigned int *inp_col_P,
  double *inp_val_P, double *Y, const unsigned int N,
  double *dC) {

  // Zero out the gradient
  for (int i = 0; i < N * map_dims; i++) dC[i] = 0.0;

  // For convenience, split the x and y coordinate values
  auto *xs = new double[N];
  auto *ys = new double[N];

  double min_coord = INFINITY;
  double max_coord = -INFINITY;
  // Find the min/max values of the x and y coordinates
  for (unsigned long i = 0; i < N; i++) {
    xs[i] = Y[i * 2 + 0];
    ys[i] = Y[i * 2 + 1];
    if (xs[i] > max_coord) max_coord = xs[i];
    else if (xs[i] < min_coord) min_coord = xs[i];
    if (ys[i] > max_coord) max_coord = ys[i];
    else if (ys[i] < min_coord) min_coord = ys[i];
  }

  // The number of "charges" or s+2 sums i.e. number of kernel sums
  int n_terms = 4;
  auto *chargesQij = new double[N * n_terms];
  auto *potentialsQij = new double[N * n_terms]();

  // Prepare the terms that we'll use to compute the sum i.e.
  // the repulsive forces
  for (unsigned long j = 0; j < N; j++) {
    chargesQij[j * n_terms + 0] = 1;
    chargesQij[j * n_terms + 1] = xs[j];
    chargesQij[j * n_terms + 2] = ys[j];
    chargesQij[j * n_terms + 3] = xs[j] * xs[j] + ys[j] * ys[j];
  }

  // Compute the number of boxes in a single dimension and the total number
  //of boxes in 2d
  auto n_boxes_per_dim = static_cast<int>(fmax(min_num_intervals,
    (max_coord - min_coord) / intervals_per_integer));

  // FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d
  // 11^e 13^f, where e+f is either 0 or 1, and the other exponents are
  // arbitrary
  int allowed_n_boxes_per_dim[20] = {25,36, 50, 55, 60, 65, 70, 75, 80, 85, 90,
    96, 100, 110, 120, 130, 140,150, 175, 200};
  if ( n_boxes_per_dim < allowed_n_boxes_per_dim[19] ) {
    //Round up to nearest grid point
    int chosen_i;
    for (chosen_i =0; allowed_n_boxes_per_dim[chosen_i]< n_boxes_per_dim; chosen_i++);
    n_boxes_per_dim = allowed_n_boxes_per_dim[chosen_i];
  }

  int n_boxes = n_boxes_per_dim * n_boxes_per_dim;

  auto *box_lower_bounds = new double[2 * n_boxes];
  auto *box_upper_bounds = new double[2 * n_boxes];
  auto *y_tilde_spacings = new double[nterms];
  int nterms_1d = nterms * n_boxes_per_dim;
  auto *x_tilde = new double[nterms_1d]();
  auto *y_tilde = new double[nterms_1d]();
  auto *fft_kernel_tilde = new complex<double>[
    2 * nterms_1d * 2 * nterms_1d];

  precompute_2d(max_coord, min_coord, max_coord, min_coord, n_boxes_per_dim,
      nterms, &squared_cauchy_2d, box_lower_bounds,
      box_upper_bounds, y_tilde_spacings, x_tilde, y_tilde,
      fft_kernel_tilde,1.0);
  n_body_fft_2d(N, n_terms, xs, ys, chargesQij, n_boxes_per_dim,
      nterms, box_lower_bounds, box_upper_bounds,
      y_tilde_spacings, fft_kernel_tilde, potentialsQij,nthreads);

  // Compute the normalization constant Z or sum of q_{ij}. This expression
  // is different from the one in the original paper, but equivalent. This is
  // done so we need only use a single kernel (K_2 in the paper) instead of
  //two different ones. We subtract N at the end because the following sums
  // over all i, j, whereas Z contains i \neq j
  double sum_Q = 0;
  for (unsigned long i = 0; i < N; i++) {
    double phi1 = potentialsQij[i * n_terms + 0];
    double phi2 = potentialsQij[i * n_terms + 1];
    double phi3 = potentialsQij[i * n_terms + 2];
    double phi4 = potentialsQij[i * n_terms + 3];

    sum_Q += (1 + xs[i] * xs[i] + ys[i] * ys[i]) * phi1 - 2
      * (xs[i] * phi2 + ys[i] * phi3) + phi4;
  }
  sum_Q -= N;

  this->current_sum_Q = sum_Q;

  double *pos_f = new double[N * 2];

  // Now, figure out the Gaussian component of the gradient. This corresponds
  // to the "attraction" term of the gradient. It was calculated using a fast
  // KNN approach, so here we just use the results that were passed to this
  // function
  PARALLEL_FOR(nthreads, N,
    {
     double dim1 = 0;
     double dim2 = 0;

     for (unsigned int i = inp_row_P[loop_i];
       i < inp_row_P[loop_i + 1]; i++) {
       // Compute pairwise distance and Q-value
       unsigned int ind3 = inp_col_P[i];
       double d_ij = (xs[loop_i] - xs[ind3]) * (xs[loop_i]
         - xs[ind3]) + (ys[loop_i] - ys[ind3])
         * (ys[loop_i] - ys[ind3]);
       double q_ij = 1 / (1 + d_ij);

       dim1 += inp_val_P[i] * q_ij * (xs[loop_i] - xs[ind3]);
       dim2 += inp_val_P[i] * q_ij * (ys[loop_i] - ys[ind3]);
     }
     pos_f[loop_i * 2 + 0] = dim1;
     pos_f[loop_i * 2 + 1] = dim2;
  });

  // Make the negative term, or F_rep in the equation 3 of the paper
  double *neg_f = new double[N * 2];
  for (unsigned int i = 0; i < N; i++) {
      neg_f[i * 2 + 0] = (xs[i] * potentialsQij[i * n_terms]
        - potentialsQij[i * n_terms + 1]) / sum_Q;
      neg_f[i * 2 + 1] = (ys[i] * potentialsQij[i * n_terms]
        - potentialsQij[i * n_terms + 2]) / sum_Q;
      dC[i * 2 + 0] = pos_f[i * 2] - neg_f[i * 2];
      dC[i * 2 + 1] = pos_f[i * 2 + 1] - neg_f[i * 2 + 1];
  }

  delete[] pos_f;
  delete[] neg_f;
  delete[] potentialsQij;
  delete[] chargesQij;
  delete[] xs;
  delete[] ys;
  delete[] box_lower_bounds;
  delete[] box_upper_bounds;
  delete[] y_tilde_spacings;
  delete[] y_tilde;
  delete[] x_tilde;
  delete[] fft_kernel_tilde;
}


// Evaluate t-SNE cost function (exactly)
double DIFFTSNE::evaluateError(const double *P, double *Y,
  const unsigned int N) {
  // Compute the squared Euclidean distance matrix
  double *DDy = (double *) malloc(N * N * sizeof(double));
  double *Q = (double *) malloc(N * N * sizeof(double));
  if (DDy == NULL || Q == NULL) {
    throw std::bad_alloc();
    //Rcpp::stop("Memory allocation failed!\n");
  }
  computeSquaredEuclideanDistance(Y, N, map_dims, DDy);

  // Compute Q-matrix and normalization sum
  int nN = 0;
  double sum_Q = DBL_MIN;
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < N; m++) {
      if (n != m) {
        //Q[nN + m] = 1.0 / pow(1.0 + DDy[nN + m]/(double)df, df);
        Q[nN + m] = 1.0 / (1.0 + DDy[nN + m]/(double)df);
        Q[nN + m ] = pow(Q[nN +m ], df);
        sum_Q += Q[nN + m];
      } else Q[nN + m] = DBL_MIN;
    }
    nN += N;
  }
  for (unsigned int i = 0; i < N * N; i++) Q[i] /= sum_Q;

  // Sum t-SNE error
  double C = .0;
  for (unsigned int n = 0; n < N * N; n++) {
      C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
  }

  // Clean up memory
  free(DDy);
  free(Q);
  return C;
}

// Evaluate t-SNE cost function (approximately)

double DIFFTSNE::evaluateError(const unsigned int *row_P,
  const unsigned int *col_P, const double *val_P, double *Y,
  const unsigned int N) {

  // Get estimate of normalization term
  SPTree* tree = new SPTree(map_dims, Y, N);
  double *buff = (double *) calloc(map_dims, sizeof(double));
  double sum_Q = .0;
  for (unsigned int n = 0; n < N; n++) {
    sum_Q += tree->computeNonEdgeForces(n, theta, buff);
  }
  double C = .0;
  PARALLEL_FOR(nthreads, N, {
    double *buff = (double *) calloc(map_dims, sizeof(double));
    int ind1 = loop_i * map_dims;
    double temp = 0;
    for (int i = row_P[loop_i]; i < row_P[loop_i + 1]; i++) {
      double Q = .0;
      int ind2 = col_P[i] * map_dims;
      for (int d = 0; d < map_dims; d++) buff[d] = Y[ind1 + d];
      for (int d = 0; d < map_dims; d++) buff[d] -= Y[ind2 + d];
      for (int d = 0; d < map_dims; d++) Q += buff[d] * buff[d];
      Q = (1.0 / (1.0 + Q)) / sum_Q;
      temp += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
    }
    C += temp;
    free(buff);
  });

  // Clean up memory
  free(buff);
  delete tree;
  return C;
}

// Evaluate t-SNE cost function (approximately) using FFT

double DIFFTSNE::evaluateErrorFft(const unsigned int *row_P,
  const unsigned int *col_P, const double *val_P, double *Y,
  const unsigned int N) {

    // Get estimate of normalization term
    double sum_Q = this->current_sum_Q;

    // Loop over all edges to compute t-SNE error
    double C = .0;

    PARALLEL_FOR(nthreads, N, {
      double *buff = (double *) calloc(map_dims, sizeof(double));
      int ind1 = loop_i * map_dims;
      double temp = 0;
      for (int i = row_P[loop_i]; i < row_P[loop_i + 1]; i++) {
        double Q = .0;
        int ind2 = col_P[i] * map_dims;
        for (int d = 0; d < map_dims; d++) buff[d] = Y[ind1 + d];
        for (int d = 0; d < map_dims; d++) buff[d] -= Y[ind2 + d];
        for (int d = 0; d < map_dims; d++) Q += buff[d] * buff[d];
        Q = pow(1.0 / (1.0 + Q/df),  df) / sum_Q;
        temp += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
      }
      C += temp;
      free(buff);
    });

    return C;
}

// Computing Affinities ========================================================


// Converts an array of [squared] Euclidean distances into similarities
// aka affinities using a specified perplexity value
void DIFFTSNE::computeSimilarities(double *cur_condP,
  double *cur_sum_condP, double *cur_beta, const double *cur_Dist,
  const int K, const bool ifSquared, const int zero_idx){

  double beta;
  if(sigma > 0) {                    // fixed kernel bandwidth: no binary search
    double sum_P = 0;
    beta = 1/(2*sigma*sigma);
    for(int m = 0; m < K; m++) {
      cur_condP[m] = exp(-beta *
        (ifSquared ? cur_Dist[m] : cur_Dist[m]*cur_Dist[m]));
    }
    if (zero_idx >= 0) cur_condP[zero_idx] = DBL_MIN;      // set P{i|i}' s to 0
    sum_P = DBL_MIN;
    for(int m = 0; m < K; m++) { sum_P += cur_condP[m]; }
    *cur_sum_condP = sum_P;
  } else {
    // findBeta function in helpers.h
    beta = findBeta(cur_condP, cur_sum_condP, cur_Dist,
      perplexity, K, zero_idx, ifSquared, false, 200);
  }
  // Normalize conditional transition probabilities
  for(int m = 0; m < K; m++) cur_condP[m] /= *cur_sum_condP;
  *cur_beta = beta;
  return;
}


//Scaling the similarities to preserve input variance differences across regions
void DIFFTSNE::scaleCondProbsSparse(const unsigned int N){
  if(verbose){
    printf("Scaling cond. probabilities to preserve input variances...\n");
  }
  double eps = 0.01;
  double sum_betas = 0;
  for(int i=0; i < N; i++) {sum_betas += betas[i];} //*betas[i];}
  PARALLEL_FOR(nthreads, N, { // inner loop
    int n  = loop_i;
    double new_sum_condP = 0;
    double scale = eps + (1 - eps)*(N * betas[n] / sum_betas); // * betas[n]; // * sums_condP[n];
    //betas[n] / (1 + betas[n]);
    //double scale = 1/(0.01 + mean_knn_dist[n]);
    for (int k=row_P[n]; k < row_P[n + 1]; k++){
      // double beta_j = betas[col_P[k]];
      // double scale = 2.0 * fmin(beta_i, beta_j);
      val_P[k] = scale * val_P[k];
      new_sum_condP += val_P[k];
    }
    sums_condP[n] = new_sum_condP;
  }); // end inner loop
}


void DIFFTSNE::scaleCondProbsDense(const unsigned int N){
  if(verbose){
    printf("Scaling cond. probabilities to preserve input variances...\n");
  }
  double eps = 0.01;
  double sum_betas = 0;
  for(int i=0; i < N; i++) {sum_betas += betas[i];}
  PARALLEL_FOR(nthreads, N, { // inner loop
    int n  = loop_i;
    double new_sum_condP = 0;
    double scale = eps + (1 - eps)*(N * betas[n] / sum_betas);
    for (int m=0; m < N; m++){
      P[n*N + m] *= scale;
      new_sum_condP += P[n*N + m];
    }
    sums_condP[n] = new_sum_condP;
  }); // end inner loop
}


// Input similarities using exact algorithm ------------------------------------
void DIFFTSNE::computeGaussianPerplexityExactly(const double *X,
  const unsigned int N, const int D) {

  if (perplexity < 0) {
    printf("Using a fixed kernel bandwidth, manually set.\n");
  } else {
    printf("Using perplexity, to find kernel bandwidths.\n");
  }

  // Compute the squared Euclidean distance matrix
  size_t N2=N; N2*=N;
  std::vector<double> DD;
  try {
    P.resize(N2); DD.resize(N2);
    mean_knn_dist.resize(N);
    betas.resize(N); sums_condP.resize(N);
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }

  if (distance_metric == "Precomputed") {
    const double *XnD = X;
    for (unsigned int n = 0; n < N2; n++) { DD[n] = XnD[n];}
	} else if (distance_metric == "Euclidean") {
    computeSquaredEuclideanDistance(X, N, D, DD.data());
  } else {
    std::cout << "'distance_metric', " << distance_metric
              << " is not supported when using exact algorithm." <<std::endl;
  }

  // Convert distances to similarities using Gaussian kernel row by row
  int rowStart = 0;
  for(int n = 0; n < (int) N; n++) {
    computeSimilarities(P.data() + rowStart, sums_condP.data() + n,
      betas.data() + n, DD.data() + rowStart, N, true, n);
    rowStart += N;
  }

  for (int n = 0; n < N; n++) {
    mean_knn_dist[n] = 0;
    for(int m = 0; m < (int) N; m++) {
      mean_knn_dist[n] += DD[n*N + m] * P[n*N + m];
    }
  };


  if(time_steps > 1) {// raise P to the 'time_steps' power
    matrixPower(P, N, time_steps, verbose);
  }

  if(scale_probs) {
    scaleCondProbsDense(N);
  }
  if(verbose) {
    printf("Computed a dense conditional probability matrix.\n");
  }
}


// Input similarities using ANNOY ----------------------------------------------
template<typename Distance>
int DIFFTSNE::computeGaussianPerplexity(const double* X,
  const unsigned int N, const int D, const int K){

  if(perplexity > 0 && perplexity > K)
    printf("Perplexity should be lower than K!\n");

  // Allocate the memory we need
  setupApproximateMemory(N, K);

  printf("Building Annoy tree...\n");
  AnnoyIndex<int, double, Distance, Kiss32Random> tree =
    AnnoyIndex<int, double, Distance, Kiss32Random>(D);

  if (rand_seed > 0) { tree.set_seed(rand_seed); }

  for(int i=0; i<N; ++i){
    double *vec = (double *) malloc( D * sizeof(double) );
    for(int z=0; z<D; ++z){
      vec[z] = X[i*D+z];
    }
    tree.add_item(i, vec);
  }
  tree.build(n_trees);

  printf("Done building tree. Beginning nearest neighbor search... \n");
  std::vector<int> nneighs(N, 0);
  PARALLEL_FOR(nthreads, N, { // inner loop
    int n = loop_i;
    // Find nearest neighbors
    std::vector<int> closest_indices;
    std::vector<double> closest_distances;
    closest_indices.reserve(K+1);
    closest_distances.reserve(K+1);
    tree.get_nns_by_item(n, K+1, search_k, &closest_indices,
      &closest_distances);
    //Check if it returns enough neighbors
    if (closest_indices.size() < K+1 ) { nneighs[n] ++; }

    double *cur_beta = betas.data() + n;
    double *cur_sum_P = sums_condP.data() + n;
    double *cur_P = val_P.data() + row_P[n];
    computeSimilarities(cur_P, cur_sum_P, cur_beta,
      closest_distances.data()+1, K, false, -1);
    //unsigned int *cur_col_P = col_P.data() + row_P[n];
    for(unsigned int m = 0; m < K; m++) {
      //cur_col_P[m] = (unsigned int) closest_indices[m + 1];
      col_P[row_P[n] + m] = (unsigned int) closest_indices[m + 1];
    }
    mean_knn_dist[n] = 0;
    for (unsigned int m = 0; m < K; m++) {
      mean_knn_dist[n] += closest_distances[m + 1] *
        closest_distances[m + 1] * cur_P[m];
    };
    if (verbose) {
      if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);
    }
    closest_indices.clear();
    closest_distances.clear();
  }); // end inner loop

  int insufficient_neighs = 0;
  for(int j=0; j < N; j++) {
    if(nneighs[j] > 0) insufficient_neighs++;
  }

  if(insufficient_neighs > 0){
    printf("ERROR: Requesting perplexity*3 = %d neighbors, but ANNOY found "
           "fewer neighbors for %d nodes.\n Please set search_k larger "
           " than %d.\n", K, insufficient_neighs, search_k);
    return -1;
  }

  if(time_steps > 1) {// raise P to the 'time_steps' power
    if(N <= 20000) {
      matrixPower(row_P, col_P, val_P, N, time_steps, row_thresh,
        nthreads, verbose);
    } else {
      // precision depends on this
      int n_walks = 100000;
      tstep_trans_probs(row_P, col_P, val_P, time_steps, n_walks, row_thresh,
        nthreads, rand_seed, verbose);
    }
  }

  if(scale_probs) {
    scaleCondProbsSparse(N);
  }

  // Clean up memory
  //free(vec);
  return 0;
}


// Compute input similarities with a fixed perplexity using ball trees
template<double (*distance)(const DataPoint&, const DataPoint& )>
void DIFFTSNE::computeGaussianPerplexity(double *X, const unsigned int N,
  const int D, const int K) {

  if(perplexity > K) printf("Perplexity should be lower than K!\n");

  // Allocate the memory we need
  setupApproximateMemory(N, K);

  // Build ball tree on data set
  printf("Building VP tree...\n");
  VpTree<DataPoint, distance> *tree = new VpTree<DataPoint, distance>();
  vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
  for (unsigned int n = 0; n < N; n++) {obj_X[n] = DataPoint(D, n, X + n * D);}
  tree->create(obj_X);

  printf("Done building tree. Beginning nearest neighbor search... \n");
  PARALLEL_FOR(nthreads, N, { // inner loop
    int n  = loop_i;
    vector<DataPoint> closest_indices;
    vector<double> closest_distances;
    closest_indices.reserve(K+1);
    closest_distances.reserve(K+1);
    // Find nearest neighbors
    tree->search(obj_X[n], K + 1, &closest_indices,
      &closest_distances);

    double *cur_beta = betas.data() + n;
    double *cur_sum_P = sums_condP.data() + n;
    double *cur_P = val_P.data() + row_P[n];
    computeSimilarities(cur_P, cur_sum_P, cur_beta,
      closest_distances.data()+1, K, false, -1);
    //unsigned int *cur_col_P = col_P.data() + row_P[n];
    for(unsigned int m = 0; m < K; m++) {
      col_P[row_P[n] + m] = (unsigned int) closest_indices[m + 1].index();
      //cur_col_P[m] = (unsigned int) closest_indices[m + 1].index();
    }
    mean_knn_dist[n] = 0;
    for (unsigned int m = 0; m < K; m++) {
      mean_knn_dist[n] += closest_distances[m + 1] *
        closest_distances[m + 1] * cur_P[m];
    };
    if (verbose) {
      if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);
    }
  });
  // Clean up memory
  obj_X.clear();
  delete tree;

  if(time_steps > 1) {// raise P to the 'time_steps' power
    // threshold for cumulative row-sum pruning
    if(N <= 20000) {
      matrixPower(row_P, col_P, val_P, N, time_steps, row_thresh,
        nthreads, verbose);
    } else {
      // precision depends on this
      int n_walks = 100000;
      tstep_trans_probs(row_P, col_P, val_P, time_steps, n_walks, row_thresh,
        nthreads, rand_seed, verbose);
    }

    // if(N <= 5000) {
    //   matrixPower(row_P, col_P, val_P, sums_condP, N, time_steps, 0.99,
    //     nthreads, verbose);
    // } else {
    //    // Use landmark approximation or symmetric matrix Eigendecomposition
    // }
  }

  if(scale_probs) {
    scaleCondProbsSparse(N);
  }
}

// Compute input similarities w/ fixed perplexity from nearest-neighbour res.
void DIFFTSNE::computeGaussianPerplexity(const int* nn_idx,
  double* nn_dist, const unsigned int N, const int K) {

  if(perplexity > K) printf("Perplexity should be lower than K!\n");

  // Allocate the memory we need
  setupApproximateMemory(N, K);

  // Loop over all points to find nearest neighbors
  int steps_completed = 0;
  //#pragma omp parallel for schedule(guided) num_threads(nthreads)
  PARALLEL_FOR(nthreads, N, {
  //for(unsigned int n = 0; n < N; n++) {
    int n = loop_i;
    double *cur_P = val_P.data() + row_P[n];
    double *cur_sum_condP = sums_condP.data() + n;
    double *cur_beta = betas.data() + n;
    computeSimilarities(cur_P, cur_sum_condP, cur_beta,
       nn_dist + row_P[n], K, false, -1);

   mean_knn_dist[n] = 0;
   double * dist_ptr = nn_dist + row_P[n];
   for (unsigned int m = 0; m < K; m++) {
     mean_knn_dist[n] += dist_ptr[m]*dist_ptr[m] * cur_P[m];
   };

    const int * cur_idx = nn_idx + row_P[n];
    unsigned int * cur_col_P = col_P.data() + row_P[n];
    for (int m=0; m<K; ++m) {
        cur_col_P[m] = cur_idx[m];
    }

    //#pragma omp atomic
    ++steps_completed;

    if (verbose && steps_completed % 10000 == 0) {
      printf(" - point %d of %d\n", steps_completed, N);
    }
  });

  if(time_steps > 1) {// raise P to the 'time_steps' power
    if(N <= 20000) {
      matrixPower(row_P, col_P, val_P, N, time_steps, row_thresh,
        nthreads, verbose);
    } else {
      // precision depends on this
      int n_walks = 100000;
      tstep_trans_probs(row_P, col_P, val_P, time_steps, n_walks, row_thresh,
        nthreads, rand_seed, verbose);
    }
  }

  if(scale_probs) {
    scaleCondProbsSparse(N);
  }

}

// Approximately allocate the memory for input similarities
void DIFFTSNE::setupApproximateMemory(const unsigned int N, const int K) {
  if (verbose) {
    printf("Allocating memory needed. N: %d, K: %d, N*K = %d\n", N, K, N*K);
  }
  try {
    row_P.resize(N+1);
    col_P.resize(N*K);
    val_P.resize(N*K);
    betas.resize(N);
    mean_knn_dist.resize(N);
    sums_condP.resize(N);
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }
  row_P[0] = 0;
  for(unsigned int n = 0; n < N; n++) {row_P[n + 1] = row_P[n] + K;}
  return;
}



// =============================================================================
