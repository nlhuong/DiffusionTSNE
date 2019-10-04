#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

double squared_cauchy(double x, double y);

double squared_cauchy_2d(double x1, double x2, double y1,
  double y2,double df);

double general_kernel_2d(double x1, double x2, double y1,
  double y2, double df);

double squared_general_kernel_2d(double x1, double x2, double y1,
  double y2, double df);

// Generates random Gaussian numbers
double randn();

// Makes data zero-mean
void zeroMean(double *X, const unsigned int N, const int D);

// Compute squared Euclidean distance matrix (No BLAS)
void computeSquaredEuclideanDistance(const double* X,
   const unsigned int N, const int D, double* DD);

// // Compute squared Euclidean distance matrix (using BLAS)
// void computeSquaredEuclideanDistanceWithBLAS(
//   double* X, unsigned int N, int D, double* DD);

//Find betas (inverse smooth bandwidths)
double findBeta(double *asym_W, double * sum_asym_W,
  const double *distances, const double perplexity, const int K,
  const int zero_idx, const bool ifSquared,
  const bool shift_by_1NN, const int maxITER);


// Create an Eigen dense matrix from pointer to the matrix elements' values
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>convertToEigenMatrix(
  const std::vector<double> &data, const int nrows, const int ncols);

// Create an Eigen CSR matrix from pointer to the matrix elements
Eigen::SparseMatrix<double,Eigen::RowMajor> convertToEigenCSR(
  const std::vector<unsigned int> &row_ptr,
  const std::vector<unsigned int> &col_ptr,
  const std::vector<double> &val_ptr,
  const int nrows, const int ncols);

bool sortinrev(const std::pair<int, double> &a,
  const std::pair<int,double> &b);

void pruneRowSums(Eigen::SparseMatrix<double,Eigen::RowMajor> &spmat,
  const double thresh, const int nthreads, const bool verbose);

void matPowBySq(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &res, int &power,
   const bool verbose);

void matPowBySq(Eigen::SparseMatrix<double,Eigen::RowMajor> &mat,
  Eigen::SparseMatrix<double,Eigen::RowMajor> &res, int &power,
  const double ref, const double thresh, const int nthreads,
  const bool verbose);

// Matrix power for dense matrix using pointer to matrix elements
void matrixPower(std::vector<double> &Mat,
  const unsigned int N, const int exponent, const bool verbose);

// Matrix power for CSR matrix  using pointer to matrix elements
void matrixPower(std::vector<unsigned int> &row_M,
    std::vector<unsigned int> &col_M, std::vector<double> &val_M,
    const unsigned int N, const int exponent, const double thresh,
    const int nthreads, const bool verbose);

// Symmetrize dense matrix
void symmetrizeDenseMatrix(std::vector<double> &Mat, const unsigned int N);

// Symmetrize dense matrix using Eigen
void symmetrizeWithEigen(std::vector<double> &Mat, const unsigned int N);

// Symmetrize Compressed Sparse Row format patrix
void symmetrizeSparseMatrix(std::vector<unsigned int> &row_M,
  std::vector<unsigned int> &col_M, std::vector<double> &val_M,
  const unsigned int N);

// Symmetrize Compressed Sparse Row format patrix with Eigen
void symmetrizeWithEigen(std::vector<unsigned int> &row_M,
  std::vector<unsigned int> &col_M, std::vector<double> &val_M,
  const unsigned int N);


// Load affinities from files in the current directory
int loadDenseAffinities(std::vector<double> &Aff,
  const unsigned int N, const bool verbose);

// Load sparse affinities saved ad CSR format in 3 files
int loadSparseAffinities(std::vector<unsigned int> &row_Aff,
  std::vector<unsigned int> &col_Aff, std::vector<double> &val_Aff,
  const unsigned int N, const bool verbose);

// Save all affinity data to files in current directory
int saveDenseAffinities(double * A, const unsigned int N, const bool verbose);

// Save all affinity data to files in current directory
int saveSparseAffinities(const unsigned int * row_Aff,
  const unsigned int * col_Aff, const double * val_Aff,
  const unsigned int N, const bool verbose);

// save betas
int saveBetas(const double * betas, const unsigned int N,
  const bool verbose);

// save sum_j d_{ij}^2p_{j|i} for each i
int saveMeanDist(const double * meanDists, const unsigned int N,
  const bool verbose);

// save row sums of affinities
int saveRowSums(const double * rowSums, const unsigned int N,
  const bool verbose);

void save_intermediate_y(int iter, double *Y, int N, int no_dims);
