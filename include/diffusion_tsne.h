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

#include <vector>
#include "datapoint.h"

#ifndef DIFFTSNE_H
#define DIFFTSNE_H


static double sign(double x) {
  return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

class DIFFTSNE {

public:

  // main user param settings:
  int map_dims, max_iter, nthreads;

  DIFFTSNE();
  //~DIFFTSNE();

  DIFFTSNE(double map_dims, double theta, std::string distance_metric,
    double sigma, int kneigh, int time_steps, bool scale_probs, int df,
    int knn_algo, int nbody_algo, int max_iter, double learning_rate,
    double early_exag_coeff, int stop_lying_iter, double late_exag_coeff,
    int start_late_exag_iter, double momentum, double final_momentum,
    int mom_switch_iter, bool no_momentum_during_exag, int nterms,
    int intervals_per_integer, int min_num_intervals, int n_trees,
    int search_k, bool skip_random_init, int rand_seed, int nthreads,
    int niter_check, double row_thresh, bool verbose);

  DIFFTSNE(double map_dims, double theta, std::string distance_metric,
    double perplexity, int time_steps, bool scale_probs, int df,
    int knn_algo, int nbody_algo, int max_iter, double learning_rate,
    double early_exag_coeff, int stop_lying_iter, double late_exag_coeff,
    int start_late_exag_iter, double momentum, double final_momentum,
    int mom_switch_iter, bool no_momentum_during_exag, int nterms,
    int intervals_per_integer, int min_num_intervals,  int n_trees,
    int search_k, bool skip_random_init, int rand_seed, int nthreads,
    int niter_check,double row_thresh, bool verbose);

    int run(double* X, const unsigned int N, const int D, double* Y,
      double * itercost);
    void run(const int* nn_index, double* nn_dist,
      const unsigned int N, double* Y, double* itercost);

    bool load_data(const char *data_path, double **data, int *n, int *d,
      double **Y);

    void save_data(const char *result_path, const double* data,
      int n, const std::vector<double> itercost);


private:

    // Member variables ========================================================
    std::string distance_metric;
    double theta, perplexity, sigma, df, row_thresh;
    int time_steps, rand_seed, load_affinities, niter_check;
    bool exact, scale_probs, skip_random_init, no_momentum_during_exag, verbose;
    // alg params
    int knn_algo, n_trees, search_k, kneigh;
    int nbody_algo, nterms, min_num_intervals;
    // optimization parameters:
    int stop_lying_iter, start_late_exag_iter, mom_switch_iter;
    double learning_rate, early_exag_coeff, late_exag_coeff;
    double momentum, final_momentum, intervals_per_integer;
    // Internal storage
    std::vector<unsigned int> row_P, col_P;
    std::vector<double> P, val_P, betas, sums_condP, mean_knn_dist, itercost;
    double current_sum_Q;

    // Member functions ========================================================
    void setupApproximateMemory(const unsigned int N, const int K);
    void trainIterations(const unsigned int N, double* Y, double* itercost);

    // Input Affinity Computatios  --------------------------------------------
    void computeSimilarities(double *cur_condP, double *cur_sum_condP,
      double *cur_beta, const double *cur_Dist, const int K,
      const bool ifSquared, const int zero_idx);

    void scaleCondProbsDense(const unsigned int N);

    void scaleCondProbsSparse(const unsigned int N);

    void computeGaussianPerplexityExactly(const double *X,
      const unsigned int N, const int D);

    template<typename Distance>
    int computeGaussianPerplexity(const double* X, const unsigned int N,
        const int D, const int K);

    template<double (*distance)(const DataPoint&, const DataPoint& )>
    void computeGaussianPerplexity(double *X, const unsigned int N,
        const int D, const int K);

    void computeGaussianPerplexity(const int* nn_dex, double* nn_dist,
      const unsigned int N, const int K);

    // Gradients Computatios  --------------------------------------------------

    double evaluateError(const double *P, double *Y, const unsigned int N);

    double evaluateError(const unsigned int *row_P,
      const unsigned int *col_P, const double *val_P,
      double *Y, const unsigned int N);

    double evaluateErrorFft(const unsigned int *row_P,
      const unsigned int *col_P, const double *val_P, double *Y,
      const unsigned int N);

    void computeExactGradient(const double *P, double *Y,
      const unsigned int N, double *dC);

    void computeGradient(const double *P, unsigned int *inp_row_P,
        unsigned int *inp_col_P, double *inp_val_P, double *Y,
        const unsigned int N, double *dC, const double theta);

    void computeFftGradientOneD(const double *P, unsigned int *inp_row_P,
        unsigned int *inp_col_P, double *inp_val_P, double *Y,
        const unsigned int N, double *dC);

    void computeFftGradientVariableDf(const double *P,
      unsigned int *inp_row_P, unsigned int *inp_col_P,
      double *inp_val_P, double *Y, const unsigned int N, double *dC);

    void computeFftGradient(const double *P, unsigned int *inp_row_P,
        unsigned int *inp_col_P, double *inp_val_P, double *Y,
        const unsigned int N, double *dC);
};

#endif
