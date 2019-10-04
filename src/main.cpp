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
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include <iostream>
#include <vector>
#include <string.h>
#include <time.h>                        /* clock_t, clock, CLOCKS_PER_SEC */
#include "../include/diffusion_tsne.h"


int main(int argc, char *argv[]) {
  clock_t start = clock();
  const char version_number[] =  "1.0";
	printf("==================== Diffusion t-SNE v%s ====================\n",
    version_number);

	// Define some variables:
  // num data points, orig dim, embd. dim, threads
  int N, D, K,  n_threads = -1;
  double *data, *initial_data, *Y;
  const char *data_path, *result_path;
	data_path = "data.dat"; result_path = "result.dat";

	if(argc >= 2) {
		data_path = argv[1];
	}
	if(argc >= 3) {
		result_path = argv[2];
	}
  std::cout<<"diffusion_difftsne data_path: "<< data_path <<std::endl;
  std::cout<<"diffusion_difftsne result_path: "<< result_path <<std::endl;
	if(argc >= 4) {
		n_threads = (unsigned int) strtoul(argv[3], (char **) NULL, 10);
    std::cout<<"diffusion_difftsne nthreads: "<< n_threads <<std::endl;
	}

  // Initialize the DIFFTSNE object
  DIFFTSNE* difftsne = new DIFFTSNE();

  // Read the parameters and the dataset
	if(difftsne->load_data(data_path, &data, &N, &D, &Y)) {

		// Now fire up the SNE implementation
    if(n_threads > 0) {
      difftsne->nthreads = n_threads;
      printf("Updating number of threads to %d.\n", n_threads);
    };
    int max_iter = difftsne->max_iter;
    int map_dims = difftsne->map_dims;
    std::vector<double> itercost;
    try {
      itercost.resize(max_iter);
    } catch (std::bad_alloc const&){
      throw std::bad_alloc();
      //printf("Memory allocation failed!\n"); exit(EXIT_FAILURE);
      // Rcpp::stop("Memory allocation failed!\n");
    }
		int error_code = 0;
		error_code = difftsne->run(data, N, D, Y, itercost.data());

		if (error_code < 0) {
      printf("Diffusion t-SNE failed.");
      exit(error_code);
    }
		// Save the results
		difftsne->save_data(result_path, Y, N, itercost);

		// Clean up the memory
		free(data); data = NULL;
		free(Y); Y = NULL;
		itercost.clear();
	}
	delete(difftsne);
  clock_t end = clock();
	printf("Finished Diffusion t-SNE in total time: %4.2f.\n\n",
    (float) (end - start) / CLOCKS_PER_SEC);
}
