#include <iostream>
#include <fstream>
#include <thread>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <time.h>                        /* clock_t, clock, CLOCKS_PER_SEC */

#include "../include/parallel_for.h"
#include "../include/helper.h"

// extern "C" {
//   #include <R_ext/BLAS.h>
// }

using namespace Eigen;
typedef std::pair<int, double> pairINTDBL;
typedef SparseMatrix<double,RowMajor> CsrMat;


double squared_cauchy(double x, double y) {
    return pow(1.0 + pow(x - y, 2), -2);
}

double squared_cauchy_2d(double x1, double x2, double y1, double y2,double df) {
    return pow(1.0 + pow(x1 - y1, 2) + pow(x2 - y2, 2), -2);
}

double general_kernel_2d(double x1, double x2, double y1, double y2, double df) {
    return pow(1.0 + ((x1 - y1)*(x1-y1) + (x2 - y2)*(x2-y2))/df, -(df));
}

double squared_general_kernel_2d(double x1, double x2, double y1, double y2,
    double df) {
    return pow(1.0 + ((x1 - y1)*(x1-y1) + (x2 - y2)*(x2-y2))/df, -(df+1.0));
}


// Generates a Gaussian random number
double randn() {
  double x, y, radius;
  do {
    x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
    y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
    radius = (x * x) + (y * y);
  } while ((radius >= 1.0) || (radius == 0.0));
  radius = sqrt(-2 * log(radius) / radius);
  x *= radius;
  return x;
}


// Makes data zero-mean
void zeroMean(double *X, const unsigned int N, const int D) {
  // Compute data mean
  double *mean = (double *) calloc(D, sizeof(double));
  if (mean == NULL) throw std::bad_alloc();

  int nD = 0;
  for (unsigned int n = 0; n < N; n++) {
    for (int d = 0; d < D; d++) {
      mean[d] += X[nD + d];
    }
    nD += D;
  }
  for (int d = 0; d < D; d++) {
    mean[d] /= (double) N;
  }

  // Subtract data mean
  nD = 0;
  for (unsigned int n = 0; n < N; n++) {
    for (int d = 0; d < D; d++) {
        X[nD + d] -= mean[d];
    }
    nD += D;
  }
  free(mean);
}


// Compute squared Euclidean distance matrix (No BLAS)
void computeSquaredEuclideanDistance(const double* X,
  const unsigned int N, const int D, double* DD) {
  const double* XnD = X;
  for(unsigned int n = 0; n < N; ++n, XnD += D) {
    const double* XmD = XnD + D;
    double* curr_elem = &DD[n*N + n];
    *curr_elem = 0.0;
    double* curr_elem_sym = curr_elem + N;
    for(unsigned int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
      *(++curr_elem) = 0.0;
      for(int d = 0; d < D; ++d) {
        *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
      }
      *curr_elem_sym = *curr_elem;
    }
  }
}


// // Compute squared Euclidean distance matrix (using BLAS)
// void computeSquaredEuclideanDistanceWithBLAS(
//   double* X, unsigned int N, int D, double* DD) {
//     double* dataSums = (double*) calloc(N, sizeof(double));
//     if(dataSums == NULL) {
//       throw std::bad_alloc();
//       //Rcpp::stop("Memory allocation failed!\n");
//     }
//     for(unsigned int n = 0; n < N; n++) {
//       for(int d = 0; d < D; d++) {
//         dataSums[n] += (X[n * D + d] * X[n * D + d]);
//       }
//     }
//     for(unsigned long n = 0; n < N; n++) {
//       for(unsigned long m = 0; m < N; m++) {
//         DD[n * N + m] = dataSums[n] + dataSums[m];
//       }
//     }
//     double a1 = -2.0;
//     double a2 = 1.0;
//     int Nsigned = N;
//     dgemm_("T", "N", &Nsigned, &Nsigned, &D, &a1, X, &D, X, &D, &a2, DD,
//       &Nsigned);
//     free(dataSums); dataSums = NULL;
// }


//Find betas (inverse smooth bandwidths)
double findBeta(double *asym_W, double * sum_asym_W,
  const double *distances, const double perplexity, const int K,
  const int zero_idx, const bool ifSquared,
  const bool shift_by_1NN, const int maxITER) {

  double dists[K];
  for(int k=0; k < K; k++) {dists[k] = distances[k];}

  // [ADDED code] shift dists by min dist like in UMAP
  if (shift_by_1NN) {
    double min_dist = INFINITY;
    for (int k=0; k < K; k++) {
      if ((k!= zero_idx) && (dists[k] < min_dist)){
        min_dist = dists[k];
      }
    }
    for (int k=0; k < K; k++) {
      if (k!= zero_idx) dists[k] -= min_dist;
    }
  }
  // Using binary search to find the appropriate kernel width
  double beta  = 1.0;
  double sum_P = 0;
  bool found      = false;
  double min_beta = -DBL_MAX;
  double max_beta =  DBL_MAX;
  double tol      = 1e-5;
  double logU     = log(perplexity);
  double H;

  // Iterate until we found a good kernel width
  int iter = 0;
  while(!found && iter < maxITER) {
    // Compute Gaussian kernel row
    for(int m = 0; m < K; m++) {
      asym_W[m] = exp(-beta *
        (ifSquared ? dists[m] : dists[m]*dists[m]));
    }
    // for P{i|i}' s
    if (zero_idx >= 0) { asym_W[zero_idx] = DBL_MIN; }

    // Compute entropy
    sum_P = DBL_MIN;
    for(int m = 0; m < K; m++) { sum_P += asym_W[m];}
    H = 0.0;
    for(int m = 0; m < K; m++) {
      H += ((ifSquared ? dists[m] : dists[m] * dists[m]) *
        asym_W[m]);
    }
    H = beta * (H / sum_P) + log(sum_P);

    // Evaluate whether the entropy is within the tolerance level
    double Hdiff = H - logU;
    if(Hdiff < tol && -Hdiff < tol) {
        found = true;
    } else {
      if(Hdiff > 0) {
        min_beta = beta;
        if(max_beta == DBL_MAX || max_beta == -DBL_MAX) beta *= 2.0;
        else beta = (beta + max_beta) / 2.0;
      } else {
        max_beta = beta;
        if(min_beta == -DBL_MAX || min_beta == DBL_MAX) beta /= 2.0;
        else beta = (beta + min_beta) / 2.0;
      }
    }
    // Update iteration counter
    iter++;
  }
  *sum_asym_W = sum_P;
  return beta;
}

// Compputing matrix power with Eigen ==========================================

// Create an Eigen dense matrix from pointer to the matrix elements' values
MatrixXd convertToEigenMatrix(const std::vector<double> &data,
  const int nrows, const int ncols) {
  //Dense Eigen matrix format
  Matrix<double, Dynamic, Dynamic> eMat(nrows, ncols);
  // Populate matrix with numbers
  int rowStart;
  for (int row = 0; row < nrows; row++){
    rowStart = row*ncols;
    for (int col = 0; col < ncols; col++){
      eMat(row, col) = (float) data[ rowStart + col];
    }
  }
  return eMat;
}

// Create an Eigen CSR matrix from pointer to the matrix elements' values
CsrMat convertToEigenCSR(
  const std::vector<unsigned int> &row_ptr,
  const std::vector<unsigned int> &col_ptr,
  const std::vector<double> &val_ptr,
  const int nrows, const int ncols) {

  int numele = row_ptr[nrows];
  std::vector<Triplet<double>> tripletList;
  tripletList.reserve(numele);
  int curRowIdx = 0;
  for (int k=0; k < numele; k++){
    if(row_ptr[curRowIdx + 1] <= k) {
      curRowIdx ++;
    }
    tripletList.push_back(Triplet<double>(curRowIdx, col_ptr[k], val_ptr[k]));
  }
  CsrMat csrM(nrows,ncols);
  csrM.setFromTriplets(tripletList.begin(), tripletList.end());
  return csrM;
}


// Driver function to sort the vector elements by
// first element of pair in descending order
bool sortinrev(const pairINTDBL &pair1, const pairINTDBL &pair2)
{ return (pair1.second > pair2.second); }

void pruneRowSums(CsrMat &spmat, const double thresh, const int nthreads,
  const bool verbose) {

  int N = spmat.rows();
  if(verbose) printf("Pruning elements of the matrix power...\n");

  //PARALLEL_FOR(nthreads, N, { // inner loop
    //int row = loop_i;
  for(unsigned int row = 0; row < N; row++) {
    int row_len = spmat.innerVector(row).size();
    // Using pair vector to keep track of indexes
    std::vector<pairINTDBL> vp; //(row_len);
    //int k = 0;
    for (CsrMat::InnerIterator it(spmat,row); it; ++it) {
      vp.push_back(std::make_pair(it.index(), it.value()));
      //vp[k] = std::make_pair(it.value(), it.index());
      //k++;
    }
    // Sorting pair vector
    std::sort(vp.begin(), vp.end(), sortinrev);
    double cumsum = 0;
    int j_thresh = 0;
    while (cumsum <= thresh && j_thresh < vp.size()) {
      cumsum += vp[j_thresh].second;
      j_thresh++;
    }

    // keep at least 2 terms:
    j_thresh = j_thresh < 2 ? 2 : j_thresh;
//    if(verbose) printf("j_thresh for row %d is %d.\n", row, j_thresh);
    for(int j=j_thresh; j < vp.size(); j++) {
      spmat.coeffRef(row, vp[j].first) = 0;
    }
  //});
  }
  if(verbose) printf("Finished zeroing out elements.\n");

  spmat = spmat.pruned(DBL_MIN, 1.0);
  // PARALLEL_FOR(nthreads, N, { // inner loop
  //   int row  = loop_i;
  for(unsigned int row = 0; row < N; row++) {
    double rowsum = spmat.innerVector(row).sum();
    for (CsrMat::InnerIterator it(spmat,row); it; ++it) {
      it.valueRef() /= rowsum;
    }
  }
  //});
  if(verbose) printf("Finished row scaling.\n");
}


void matPowBySq(MatrixXd &mat, MatrixXd &res, int &power, const bool verbose) {
  if(verbose) printf("Power by squaring.\n");
  // Complexity: 2*floor(log2(power))n^3
  clock_t start = clock();
  while (true) {
    if(verbose) printf("Power %d.\n", power);
    if (std::fmod(power, 2) >= 1) {
      res = mat * res;
    }
    power /= 2;
    if (power < 1) { break;}
    mat = mat * mat;
  }
  if(verbose) {
    clock_t end = clock();
    printf("Matrix power by squaring computed in %4.2f seconds!\n",
      (float) (end - start) / CLOCKS_PER_SEC);
  }
}

// DO NOT USE THIS BECAUSE NOT PARALLEL AND TAKES FOREVER
void matPowBySq(CsrMat &mat, CsrMat &res, int &power, const double ref,
  const double thresh, const int nthreads, const bool verbose) {

  int N = mat.rows();
  if(verbose) printf("Power by squaring.\n");
  clock_t start = clock();
  while (true) {
    if(verbose) printf("Power %d.\n", power);
    if (std::fmod(power, 2) >= 1) {
      res = (mat * res).pruned(ref, 1.0);
      if(thresh > 0) {
        pruneRowSums(res, thresh, nthreads, verbose);
      }
    }
    power /= 2;
    if (power < 1){
      break;
    }
    mat = (mat * mat).pruned(ref, 1.0);
    // keep top elements of each row or res such that their sum is >= thresh
    if(thresh > 0) {
      pruneRowSums(mat, thresh, nthreads, verbose);
    }
    if(verbose) {
      double frac = (double) mat.nonZeros(); frac /= (N*N);
      printf("Number of nonzeros in cur. mat: %ld (%f).\n",
         mat.nonZeros(), frac);
      printf("Number of nonzeros in cur. res: %ld (%f).\n",
         res.nonZeros(), frac);
    }
  }

  if(verbose) {
    clock_t end = clock();
    double frac = (double) res.nonZeros() / (N*N);
    printf("Number of nonzeros for matrix power %d is %ld (%f).\n",
      power, res.nonZeros(), frac);
    printf("Matrix power by squaring computed in %4.2f seconds!\n",
      (float) (end - start) / CLOCKS_PER_SEC);
  }
}



void matrixPower(std::vector<double> &Mat, const unsigned int N,
  const int exponent, const bool verbose){

  if (exponent <= 0){
    printf("exponent parameter must be a positive integer.\n");
    exit(EXIT_FAILURE);
  }

  double frac;
  int nnz, power = exponent;
  MatrixXd m_tmp, res;
  try{
    m_tmp = convertToEigenMatrix(Mat, N, N);
    res = m_tmp;
    power--;
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }

  if(verbose) {
    printf("Raising matrix to the power %d.\n", exponent);
    nnz = (m_tmp.array() > 0).count();
    frac = (double) nnz / (N*N);
    printf("Starting number of nonzeros: %d (%f).\n", nnz, frac);
  }

  // Main part computing powers
  matPowBySq(m_tmp, res, power, verbose);

  nnz = 0;
  for (int i=0; i < N; i++){
    for (int j=0; j < N; j++){
       Mat[i*N + j] = res(i, j);
       if (res(i, j) > 0 ) {
         nnz++;
       }
    }
  }

  if(verbose) {
    frac = (double) nnz / (N*N);
    printf("Final number of nonzeros for matrix power %d is %d (%f).\n",
      exponent, nnz, frac);
  }
}


void matrixPower(std::vector<unsigned int> &row_M,
    std::vector<unsigned int> &col_M, std::vector<double> &val_M,
    const unsigned int N, const int exponent, const double thresh,
    const int nthreads, const bool verbose){

  Eigen::setNbThreads(nthreads);
  if (exponent < 0){
    printf("'exponent' parameter must be a non-negative integer.");
    exit(EXIT_FAILURE);
  }

  double frac;
  int nnz, power = exponent;
  CsrMat m_tmp, res;
  try {
    m_tmp = convertToEigenCSR(row_M, col_M, val_M, N, N);
    res = m_tmp;
    power--;
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }

  if(verbose) {
    frac = (double) m_tmp.nonZeros(); frac /= (N*N);
    printf("Raising matrix to the power %d.\n", exponent);
    printf("Using %d threads for Eigen computations.\n", Eigen::nbThreads());
    printf("Starting number of nonzeros: %ld (%f).\n", m_tmp.nonZeros(), frac);
  }

  // Main part computing powers
  double ref = 1e-10;
  if(N <= 1e5) {
    if(verbose) printf("Using dense Eigen matrices.\n");
    // Convert to dense for faster matrix multiplication
    MatrixXd dm_tmp = MatrixXd(m_tmp);
    MatrixXd dres = MatrixXd(res);
    matPowBySq(dm_tmp, dres, power, verbose);
    res = dres.sparseView(1.0, ref);
  } else {
    printf("Matrix too big to multiply. Exiting.\n");
    exit(EXIT_FAILURE);
    //matPowBySq(m_tmp, res, power, ref, thresh, nthreads, verbose);
  }

  if(verbose) {
    frac = (double) res.nonZeros() / (N*N);
    printf("Number of nonzeros for matrix power %d is %ld (%f).\n",
      exponent, res.nonZeros(), frac);
  }

  if(thresh > 0) {
    pruneRowSums(res, thresh, nthreads, verbose);
  }

  // Update the values of the input data for CSR sparse matrix
  // resizing the data vectors:
  try {
    if(verbose) printf("Resizing CSR matrix' input vectors...\n");
    nnz = (int) res.nonZeros();
    col_M.resize(nnz); val_M.resize(nnz);
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }
  unsigned int k = 0;
  for (int row=0; row < res.outerSize(); ++row) {
    row_M[row] = k;
    for (CsrMat::InnerIterator it(res,row); it; ++it) {
      col_M[k] = it.index();
      val_M[k] = it.value();
      k++;
    }
  }
  row_M[res.outerSize()] = k;

  if(verbose) {
    frac = (double) k / (N*N);
    printf("Final number of nonzeros for matrix power %d is %d (%f).\n",
      exponent, k, frac);
  }
}


// Symmetrize dense matrix
void symmetrizeWithEigen(std::vector<double> &Mat, const unsigned int N) {
  MatrixXd emat, symMat;
  try{
    emat = convertToEigenMatrix(Mat, N, N);
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }
  symMat = MatrixXd (emat.transpose()) + emat;
  printf("Computed symmetrized sum...\n");

  for (int i=0; i < N; i++){
    for (int j=0; j < N; j++){
       Mat[i*N + j] = symMat(i, j);
    }
  }
}


void symmetrizeWithEigen(std::vector<unsigned int> &row_M,
  std::vector<unsigned int> &col_M, std::vector<double> &val_M,
  const unsigned int N) {
  CsrMat spMat, symMat;
  try {
    spMat = convertToEigenCSR(row_M, col_M, val_M, N, N);
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }
  symMat = CsrMat(spMat.transpose()) + spMat;
  symMat = symMat.pruned(DBL_MIN, 1.0);
  printf("Computed symmetrized sum...\n");
  try {
    int nnz = (int) symMat.nonZeros();
    col_M.resize(nnz); val_M.resize(nnz);
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }
  unsigned int k = 0;
  for (int row=0; row < symMat.outerSize(); ++row) {
    row_M[row] = k;
    for (CsrMat::InnerIterator it(symMat,row); it; ++it) {
      col_M[k] = it.index();
      val_M[k] = it.value();
      k++;
    }
  }
  row_M[symMat.outerSize()] = k;
}


// Symmetrize dense matrix
void symmetrizeDenseMatrix(double * Mat, const unsigned int N) {
  for(unsigned long n = 0; n < N; n++) {
    for(unsigned long m = n + 1; m < N; m++) {
      Mat[n * N + m] += Mat[m * N + n];
      Mat[m * N + n]  = Mat[n * N + m];
    }
  }
}


// Need to change very slow and not parallel
// Symmetrize Compressed Sparse Row format patrix
void symmetrizeSparseMatrix(std::vector<unsigned int> &row_M,
  std::vector<unsigned int> &col_M, std::vector<double> &val_M,
  const unsigned int N) {
  // Count number of elements and row counts of symmetric matrix
  int* row_counts = (int*) calloc(N, sizeof(int));
  if(row_counts == NULL) {
    //Rcpp::stop("Memory allocation failed!\n");
    throw std::bad_alloc();
  }

  for(unsigned int n = 0; n < N; n++) {
    for(unsigned int i = row_M[n]; i < row_M[n + 1]; i++) {
      // Check whether element (col_M[i], n) is present
      bool present = false;
      for(unsigned int m = row_M[col_M[i]]; m < row_M[col_M[i] + 1]; m++) {
        if(col_M[m] == n) present = true;
      }
      if(present) row_counts[n]++;
      else {
        row_counts[n]++;
        row_counts[col_M[i]]++;
      }
    }
  }
  int no_elem = 0;
  for(unsigned int n = 0; n < N; n++) no_elem += row_counts[n];

  // Allocate memory for symmetrized matrix
  std::vector<unsigned int> sym_row_M, sym_col_M(no_elem);
  std::vector<double> sym_val_M(no_elem);
  try {
    sym_row_M.resize(N+1);
    sym_col_M.resize(no_elem);
    sym_val_M.resize(no_elem);
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }

  // Construct new row indices for symmetric matrix
  sym_row_M[0] = 0;
  for(unsigned int n = 0; n < N; n++)
    sym_row_M[n + 1] = sym_row_M[n] + row_counts[n];

  // Fill the result matrix
  int* offset = (int*) calloc(N, sizeof(int));
  if(offset == NULL) {
    throw std::bad_alloc();
    //Rcpp::stop("Memory allocation failed!\n");
  }
  for(unsigned int n = 0; n < N; n++) {
    // considering element(n, col_M[i])
    for(unsigned int i = row_M[n]; i < row_M[n + 1]; i++){
      // Check whether element (col_M[i], n) is present
      bool present = false;
      for(unsigned int m = row_M[col_M[i]]; m < row_M[col_M[i] + 1]; m++) {
        if(col_M[m] == n) {
          present = true;
          if(n <= col_M[i]) { // make sure we do not add elements twice
            sym_col_M[sym_row_M[n]        + offset[n]]        = col_M[i];
            sym_col_M[sym_row_M[col_M[i]] + offset[col_M[i]]] = n;
            sym_val_M[sym_row_M[n]        + offset[n]]        = val_M[i] + val_M[m];
            sym_val_M[sym_row_M[col_M[i]] + offset[col_M[i]]] = val_M[i] + val_M[m];
          }
        }
      }

      // If (col_M[i], n) is not present, there is no addition involved
      if(!present) {
        sym_col_M[sym_row_M[n]        + offset[n]]        = col_M[i];
        sym_col_M[sym_row_M[col_M[i]] + offset[col_M[i]]] = n;
        sym_val_M[sym_row_M[n]        + offset[n]]        = val_M[i];
        sym_val_M[sym_row_M[col_M[i]] + offset[col_M[i]]] = val_M[i];
      }
      // Update offsets
      if(!present || (present && n <= col_M[i])) {
        offset[n]++;
        if(col_M[i] != n) offset[col_M[i]]++;
      }
    }
  }
  // Divide the result by two
  for(int i = 0; i < no_elem; i++) sym_val_M[i] /= 2.0;

  // Return symmetrized matrices
  row_M.swap(sym_row_M);
  col_M.swap(sym_col_M);
  val_M.swap(sym_val_M);

  // Free up some memery
  free(offset); offset = NULL;
  free(row_counts); row_counts  = NULL;
}


// Looading and saving affinity data ===========================================

// Load affinities from files in the current directory
int loadDenseAffinities(std::vector<double> &Aff,
  const unsigned int N, const bool verbose) {
  FILE *h;
  size_t result, N2=N; N2*=N;
  if(verbose){
    printf("Loading exact input similarities from file...\n");
  }
  try {
    Aff.resize(N2);
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }
  if ((h = fopen("affinity_dense.dat", "w+b")) == NULL) {
    printf("Error: could not open 'affinity_dense.dat' file.\n");
    return -2;
  }
  result = fread(Aff.data(), sizeof(double), N*N, h);
  fclose(h);
  return 0;
}


// Load sparse affinities saved ad CSR format in 3 files
int loadSparseAffinities(std::vector<unsigned int> &row_Aff,
  std::vector<unsigned int> &col_Aff, std::vector<double> &val_Aff,
  const unsigned int N, const bool verbose){
  FILE *h;
  size_t result;
  if(verbose) {
    printf("Loading approximate input similarities from files...\n");
  }
  try {
    row_Aff.resize(N + 1);
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }
  if ((h = fopen("affinity_row.dat", "rb")) == NULL) {
    printf("Error: could not open 'affinity_val.dat' file.\n");
    return -2;
  }
  result = fread(row_Aff.data(), sizeof(unsigned int), N + 1, h);
  fclose(h);

  int numel = row_Aff[N];
  try {
    col_Aff.resize(numel); val_Aff.resize(numel);
  } catch (std::bad_alloc const&) {
    throw std::bad_alloc();
    // Rcpp::stop("Memory allocation failed!\n");
  }
  if ((h = fopen("affinity_val.dat", "rb")) == NULL) {
    printf("Error: could not open 'affinity_val.dat' file.\n");
    return -2;
  }
  result = fread(val_Aff.data(), sizeof(double), numel, h);
  fclose(h);

  if ((h = fopen("affinity_col.dat", "rb")) == NULL) {
    printf("Error: could not open 'affinity_col.dat' file.\n");
    return -2;
  }
  result = fread(col_Aff.data(), sizeof(unsigned int), numel, h);
  fclose(h);
  if(verbose){
    printf("   number of nonzero elements is %d (%f)\n", numel,
           (double) numel / ((double) N * (double) N));
    printf("   val_Aff: %f %f %f ... %f %f %f\n", val_Aff[0], val_Aff[1],
      val_Aff[2], val_Aff[numel - 3], val_Aff[numel - 2], val_Aff[numel - 1]);
    printf("   col_Aff: %d %d %d ... %d %d %d\n", col_Aff[0], col_Aff[1],
      col_Aff[2], col_Aff[numel - 3], col_Aff[numel - 2], col_Aff[numel - 1]);
    printf("   row_Aff: %d %d %d ... %d %d %d\n", row_Aff[0], row_Aff[1],
      row_Aff[2], row_Aff[N - 2], row_Aff[N - 1], row_Aff[N]);
  }
  return 0;
}


// Save all affinity data to files in current directory
int saveDenseAffinities(double * A, const unsigned int N,
  const bool verbose){
  if (verbose) {
    printf("Saving dense similarities and distances to file...\n");
  }
  FILE *h;
  if ((h = fopen("affinity_dense.dat", "w+b")) == NULL) {
    printf("Error: could not open data file to write.\n");
    return -2;
  }
  fwrite(A, sizeof(double), N*N, h);
  fclose(h);
  return 0;
}


// Save all affinity data to files in current directory
int saveSparseAffinities(const unsigned int * row_Aff,
  const unsigned int * col_Aff, const double * val_Aff,
  const unsigned int N, const bool verbose){
  FILE *h;
  if(verbose){
    printf("Saving similarities and distances (CSR format) to file...\n");
  }
  if ((h = fopen("affinity_row.dat", "w+b")) == NULL) {
    printf("Error: could not open data file to write.\n");
    return -2;
  }
  fwrite(row_Aff, sizeof(unsigned int), N + 1, h);
  fclose(h);

  int numel = row_Aff[N];
  if ((h = fopen("affinity_val.dat", "w+b")) == NULL) {
    printf("Error: could not open data file to write.\n");
    return -2;
  }
  fwrite(val_Aff, sizeof(double), numel, h);
  fclose(h);

  if ((h = fopen("affinity_col.dat", "w+b")) == NULL) {
    printf("Error: could not open data file to write.\n");
    return -2;
  }
  fwrite(col_Aff, sizeof(unsigned int), numel, h);
  fclose(h);

  if(verbose){
    printf("   Number of nonzero elements is %d.\n", numel);
    printf("   val_Aff: %f %f %f ... %f %f %f\n", val_Aff[0], val_Aff[1],
      val_Aff[2], val_Aff[numel - 3], val_Aff[numel - 2], val_Aff[numel - 1]);
    printf("   col_Aff: %d %d %d ... %d %d %d\n", col_Aff[0], col_Aff[1],
      col_Aff[2], col_Aff[numel - 3], col_Aff[numel - 2], col_Aff[numel - 1]);
    printf("   row_Aff: %d %d %d ... %d %d %d\n", row_Aff[0], row_Aff[1],
      row_Aff[2], row_Aff[N - 2], row_Aff[N - 1], row_Aff[N]);
  }
  return 0;
}

// save betas
int saveBetas(const double * betas, const unsigned int N, const bool verbose){
  FILE *h;
  if(verbose){
    printf("Saving betas to file...\n");
  }
  if ((h = fopen("betas.dat", "w+b")) == NULL) {
    printf("Error: could not open data file to write.\n");
    return -2;
  }
  fwrite(betas, sizeof(double), N, h);
  fclose(h);
  if(verbose){
    printf("   betas: %f %f %f ... %f %f %f\n", betas[0], betas[1],
      betas[2], betas[N - 3], betas[N - 2], betas[N - 1]);
  }
  return 0;
}

int saveMeanDist(const double * meanDists, const unsigned int N,
  const bool verbose){
  FILE *h;
  if(verbose){
    printf("Saving mean_knn_dist to file...\n");
  }
  if ((h = fopen("mean_dists.dat", "w+b")) == NULL) {
    printf("Error: could not open data file to write.\n");
    return -2;
  }
  fwrite(meanDists, sizeof(double), N, h);
  fclose(h);
  if(verbose){
    printf("   meanDists: %f %f %f ... %f %f %f\n",
      meanDists[0], meanDists[1], meanDists[2],
      meanDists[N - 3], meanDists[N - 2], meanDists[N - 1]);
  }
  return 0;
}

int saveRowSums(const double * rowSums, const unsigned int N,
  const bool verbose){
  FILE *h;
  if(verbose){
    printf("Saving affinities row sums to file...\n");
  }
  if ((h = fopen("affinity_rowsums.dat", "w+b")) == NULL) {
    printf("Error: could not open data file to write.\n");
    return -2;
  }
  fwrite(rowSums, sizeof(double), N, h);
  fclose(h);
  if(verbose) {
    printf("   row sums: %f %f %f ... %f %f %f\n", rowSums[0],
      rowSums[1], rowSums[2], rowSums[N - 3], rowSums[N - 2],
      rowSums[N - 1]);
  }
  return 0;
}


//Helper function for printing Y at each iteration. Useful for debugging
void save_intermediate_y(int iter, double *Y, int N, int no_dims) {
  std::ofstream fileStream;
  std::string file = "dat/intermediate" + std::to_string(iter) + ".txt";
  const char* filename = file.c_str();
  fileStream.open(filename);
  if (fileStream.fail()) {
    // file could not be opened
    printf("Error: could not open %s file to write intermediate results.\n",
      filename);
    exit(EXIT_FAILURE);
  }
  printf("Saving intermediate results to: %s", filename);
  for (int j = 0; j < N; j++) {
      for (int i = 0; i < no_dims; i++) {
          fileStream << Y[j * no_dims + i] << " ";
      }
      fileStream << "\n";
  }
  fileStream.close();
}
