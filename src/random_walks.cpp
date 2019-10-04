/*
  This code is adapted from the python implementation of node2vec:
  https://github.com/aditya-grover/node2vec/blob/master/src/node2vec.py
*/

#include <iostream>
#include <map>
#include <random>
#include <vector>
#include <time.h>

#include "../include/parallel_for.h"
#include "../include/pcg_random.hpp"


typedef std::pair<unsigned int, int> PairUintInt;
typedef std::pair<int, double> PairIntDbl;
typedef std::map<unsigned int, int> FreqMap;


/*
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
*/
void get_alias(std::vector<PairIntDbl> &alias_table,
  const std::vector<double> &probs) {

  int K = probs.size();
  double sum_p = 0;
  for (int i = 0; i < K; i++)
    sum_p += probs[i];

  alias_table.resize(K);
  for (int i = 0; i < K; i++)
    alias_table[i].second = probs[i] * K / sum_p;

  // smaller and larger are underflow and overflow with respect to uniform prob
  std::vector<int> smaller, larger;
  for(int kk = 0; kk < K; kk++) {
    if (alias_table[kk].second < 1.0) {
      smaller.push_back(kk);
    } else {
      larger.push_back(kk);
    }
  }

  int sm, lg;
  while (smaller.size() > 0 && larger.size() > 0) {
    sm = smaller.back();
    lg = larger.back();

    smaller.pop_back();
    larger.pop_back();

    // asign lg as complement idx to sm
    alias_table[sm].first = lg;
    // compute remainder from merging lg and sm
    alias_table[lg].second = alias_table[lg].second +
      alias_table[sm].second - 1.0;

    if (alias_table[lg].second < 1.0) {
      smaller.push_back(lg);  // if remainder < uniform seek a complement for lg
    } else {
      larger.push_back(lg);   // else push it back to the overflow stack
    }
  }

  while (smaller.size() > 0) {
    sm = smaller.back();
    smaller.pop_back();
    alias_table[sm].first = sm;
    alias_table[sm].second = 1.0;
  }

  while (larger.size() > 0) {
    lg = larger.back();
    larger.pop_back();
    alias_table[lg].first = lg;
    alias_table[lg].second = 1.0;
  }
}

/* Sample from an alias table */
int alias_draw(const std::vector<PairIntDbl> &alias_table,
  double rnd1, double rnd2, const int seed) {

  if (rnd1 < 0 && rnd2 < 0) {
    pcg32 pcg_gen(seed);
    // Seed and  make a random number engine
    if(seed < 0) {
      pcg_gen.seed(pcg_extras::seed_seq_from<std::random_device>());
    }
    std::uniform_real_distribution<double> runi(0, 1.0);
    rnd1 = runi(pcg_gen);
    rnd2 = runi(pcg_gen);
  }
  int kk = int(floor(rnd1 * alias_table.size()));
  return (rnd2 < alias_table[kk].second) ? kk : alias_table[kk].first;
}


// Transition probability matrix is stored in a CSR format as:
// (row_P, col_P, val_P): row pointer, column indices, and values of nnz terms.

/*
 Convert the CSR transition probability data into alias table
 for each data point
*/
void preprocess_transition_probs(
  std::vector<std::vector<PairIntDbl>> &alias_probs,
  const std::vector<unsigned int> &row_P,
  const std::vector<double> &val_P,
  const int nthreads, const bool verbose) {

  clock_t t0 = clock();
  int N = row_P.size() - 1;
  alias_probs.resize(N);

  PARALLEL_FOR(nthreads, N, {
    int n = loop_i;
    std::vector<double> probs(
      val_P.begin() + row_P[n],
      val_P.begin() + row_P[n + 1]);
    get_alias(alias_probs[n], probs);
  });

  if(verbose) {
    printf("Processed P to get alias table in %4.2f seconds.\n",
      (float) (clock() - t0) / CLOCKS_PER_SEC);
  }
}

/*
  Starting at 'start', simulate n_walks (t_step)-random walks over a graph of N
  nodes using alias sampling method. Transition probabilities are stored in
  an alias table format for each data point. The indices of transition however
  are stored in col_P vector.
*/
void simulate_walks(
  FreqMap &freq_counts,
  const std::vector<unsigned int> &row_P,
  const std::vector<unsigned int> &col_P,
  const std::vector<std::vector<PairIntDbl>> &alias_probs,
  const unsigned int start, const int t_step, const int n_walks,
  const int nthreads, const int seed){

  // Seed with a real random value, if available
  pcg32 pcg_gen(seed);
  // Seed and  make a random number engine
  if(seed < 0) {
    pcg_gen.seed(pcg_extras::seed_seq_from<std::random_device>());
  }
  std::uniform_real_distribution<double> runi(0, 1.0);

  // // Generate all random numbers first
  // std::vector<double> rnd1_transition, rnd2_transition;
  // std::vector<unsigned int> end_points;
  //
  // try {
  //   clock_t t0 = clock();
  //   rnd1_transition.resize(n_walks*t_step);
  //   rnd2_transition.resize(n_walks*t_step);
  //   end_points.resize(n_walks);
  //
  //   for(int kk = 0; kk < n_walks*t_step; ++kk) {
  //     rnd1_transition[kk] = runi(pcg_gen);
  //     rnd2_transition[kk] = runi(pcg_gen);
  //   }
  // } catch (std::bad_alloc const&) {
  //   throw std::bad_alloc();
  //   // Rcpp::stop("Memory allocation failed!\n");
  // }

  // Simulate walks
  // We parallelize in here for each random walks not at the data point level
  // for memory limitations and saving the random numbers
  //

  // pcg32 pcg_gen(seed);
  // std::uniform_real_distribution<double> runi(0, 1.0);
  unsigned int idx_end, end, row_startIDX;
  //PARALLEL_FOR(nthreads, n_walks, {
  for(int n = 0; n < n_walks; n++) {
    //int n  = loop_i;
    end = start; //unsigned int
    row_startIDX = row_P[start]; //unsigned int
    for(int t = 0; t < t_step; t++) {
       idx_end = alias_draw( //unsigned int
         alias_probs[end],
         runi(pcg_gen), //rnd1_transition[n*t_step + t],
         runi(pcg_gen), -1);//rnd2_transition[n*t_step + t], -1);
      end = col_P[row_startIDX + idx_end];
      row_startIDX = row_P[end];
    }
    //end_points[n] = end;
    ++freq_counts[end];
  //});
  }
  // for(int n = 0; n < end_points.size(); n++) {
  //   ++freq_counts[end_points[n]];
  // }
}


/*
 Process frequencies of end-point from random walks;
 output a vector of states and a vector of associated normalized frequencies
 (probabilities). Filter out states with very small probabilities leaving
 frequencies summing up to at least tot_thresh.
*/
void compress_frequencies(
  std::vector<unsigned int> &state_names,
  std::vector<double> &state_freqs,
  const FreqMap &freq_counts,
  const double tot_thresh){

  std::vector<PairUintInt> freq_vec(freq_counts.begin(), freq_counts.end());

  int sum_counts = 0, len_kept = freq_vec.size();
  for(int i = 0; i < freq_vec.size(); i++) {
    sum_counts += freq_vec[i].second;
  }
  if(tot_thresh > 0) {
    // sort descending order of frequencies
    std::sort(freq_vec.begin(), freq_vec.end(),
      [](PairUintInt elem1, PairUintInt elem2){
        return elem1.second > elem2.second;});
    int count_thresh = tot_thresh * sum_counts;
    len_kept = 0; sum_counts = 0;
    while (sum_counts <= count_thresh && len_kept <= freq_vec.size()) {
      sum_counts += freq_vec[len_kept].second;
      len_kept ++;
    }
  }
  if(len_kept == (freq_vec.size() + 1)) { len_kept--;}
  state_names.resize(len_kept);
  state_freqs.resize(len_kept);
  for(int i = 0; i < len_kept; i++) {
    state_names[i] = freq_vec[i].first;
    state_freqs[i] = (double) freq_vec[i].second / sum_counts;
  }
}


/*
  Approximate the t-step transition probabilities using Monte Carlo
  random walk method and alias sampling.
*/
void tstep_trans_probs(
    std::vector<unsigned int> &row_P,
    std::vector<unsigned int> &col_P,
    std::vector<double> &val_P,
    const int t_step, const int n_walks, const double thresh,
    const int nthreads, const int seed, const bool verbose){

  int N = row_P.size() - 1;

  // Build the alias tables
  std::vector<std::vector<PairIntDbl>> alias_probs;
  preprocess_transition_probs(alias_probs,
    row_P, val_P, nthreads, verbose);

  clock_t t0 = clock();
  std::vector<std::vector<double>> matrix_val_P(N);
  std::vector<std::vector<unsigned int>> matrix_col_P(N);
  std::vector<float> rwalk_time(N), compfreq_time(N);

  //for(unsigned int i_point = 0; i_point < N; i_point++) {
  PARALLEL_FOR(nthreads, N, {
    unsigned int i_point  = (unsigned int) loop_i;
    FreqMap freq_counts;
    clock_t time0 = clock();
    simulate_walks(freq_counts, row_P, col_P, alias_probs,
      i_point, t_step, n_walks, nthreads, seed + (int) i_point);
    rwalk_time[i_point] = (float) (clock() - time0)/CLOCKS_PER_SEC;

    time0 = clock();
    compress_frequencies(matrix_col_P[i_point], matrix_val_P[i_point],
      freq_counts, thresh);
    compfreq_time[i_point] = (float) (clock() - time0)/CLOCKS_PER_SEC;
    freq_counts.clear();
  });
  //}
  if(verbose) {
    printf("Generated %d %d-step random walks for all %d data points "
      "in %4.2f seconds!\n", n_walks, t_step, N,
      (float) (clock() - t0) / CLOCKS_PER_SEC);
      float sim_rndwalk_time = 0, comp_freq_Time = 0;
      for(int i = 0; i < rwalk_time.size(); i++) {
        sim_rndwalk_time += rwalk_time[i];
      }
      for(int i = 0; i < compfreq_time.size(); i++) {
        comp_freq_Time += compfreq_time[i];
      }
      printf("Total random walk time: %4.2f seconds.\n", sim_rndwalk_time);
      printf("Total freq computation time: %4.2f seconds.\n",
        comp_freq_Time);
  }

  // Update values of transition probabilities in the input CSR matrix
  int nnz = 0;
  for(int i=0; i < N; i++)
    nnz += (int) matrix_col_P[i].size();

  try {
    if(verbose) {
      printf("Resizing CSR matrix' input vectors to length %d...\n", nnz);
    }
    col_P.resize(nnz); val_P.resize(nnz);
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }

  unsigned int kk = 0;
  for (int row = 0; row < N; ++row) {
    row_P[row] = kk;
    for (int col = 0; col < matrix_col_P[row].size(); col++) {
      col_P[kk] = matrix_col_P[row][col];
      val_P[kk] = matrix_val_P[row][col];
      kk++;
    }
  }
  row_P[N] = kk;
}
