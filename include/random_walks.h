#include <map>
#include <vector>


void get_alias(
  std::vector<std::pair<int, double>> &alias_table,
  const std::vector<double> &probs);


int alias_draw(
  const std::vector<std::pair<int, double>> &alias_table,
  double rnd1, double rnd2, const int seed);


void preprocess_transition_probs(
  std::vector<std::vector<std::pair<int, double>>> &alias_probs,
  const std::vector<unsigned int> &row_P, const std::vector<double> &val_P,
  const int nthreads, const bool verbose);

void simulate_walks(
  std::map<unsigned int, int> &freq_counts,
  const std::vector<unsigned int> &row_P,
  const std::vector<unsigned int> &col_P,
  const std::vector<std::vector<std::pair<int, double>>> &alias_probs,
  const unsigned int start, const int t_step, const int n_walks,
  const int nthreads, const int seed);

void compress_frequencies(
    std::vector<unsigned int> &state_names,
    std::vector<double> &state_freqs,
    const std::map<unsigned int, int> &freq_counts,
    const double tot_thresh);

void tstep_trans_probs(
    std::vector<unsigned int> &row_P,
    std::vector<unsigned int> &col_P,
    std::vector<double> &val_P,
    const int t_step, const int n_walks, const double thresh,
    const int nthreads, const int seed, const bool verbose);
