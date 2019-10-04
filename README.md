# Diffusion t-SNE

This repository contains c++ code for performing Diffusion t-SNE. We adapt the standard t-SNE to incorporate powers of conditional probability matrix, $P^t$ to achieve multiscale views of the input data.

Our code is build on top of FIt-SNE (https://github.com/KlugerLab/FIt-SNE) which is currently the fastest, most scalable approximation method for t-SNE.

Diffusion t-SNE depends on a few libraries:

* Eigen http://eigen.tuxfamily.org/ header-only library for matrix operations, and needs to be downloaded and linked when compiling Diffusion t-SNE
* FFTW http://www.fftw.org/, which is required by FIt-SNE and need to be installed.

To compile the code into an executable, run the following from the root directory:

```
g++ -std=c++11 -O3 -I /path/to/eigen/ src/* -o bin/diffusion_tsne -pthread -fopenmp -lfftw3 -lm -march=native
```

A `python` wrapper `diffusion_tsne.py` can be used to run the compiled code.