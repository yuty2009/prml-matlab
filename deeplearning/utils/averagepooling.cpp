/*
 * averagepooling.cpp
 *
 * Implements the average-pooling transfer function. Takes a 4D tensor shaped as
 * (rows, cols, nchannels, nsamples) and a pooling shape as (prows, pcols) and
 * returns a set of stochastic sampled values with the corresponding indices in the input
 * matrix.
 *
 * e.g.
 *      [m] = averagepooling(IM, [2 2])
 *
 *  Created on: July 6, 2017
 *      Author: yuty <yuty2009@gmail.com>
 *  Based on the max-pooling code provided by:
 *      Jonathan Masci <jonathan@idsia.ch>
 * 
 * This file is available under the terms of the GNU GPLv2.
 */

#include "mex.h"

#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int debug = 0;

/**
 * Computes the average-pooling for the given 2D map, and no, the name is not a typo.
 * All pointers are passed already offset so to avoid cumbersome indexing.
 *
 * @param ptr_data pointer set to the begin of this map
 * @param DATA_DIMS data dimensions
 * @param ptr_pool pooling sizes
 * @param ptr_out pointer to the output average-values set to the right position
 */
template <typename T>
inline void compute_average_pooling(T *ptr_data, const mwSize *DATA_DIMS, T *ptr_pool,
    T *ptr_out, int tile_start)
{
  T m;
  int count = 0;

  for (int col = 0; col < DATA_DIMS[1]; col += ptr_pool[1]) {
      for (int row = 0; row < DATA_DIMS[0]; row += ptr_pool[0]) {
          if (debug)
            fprintf(stderr, "r = %i, c = %i \n", row, col);

          m = 0;
          for (int pcol = 0; (pcol < ptr_pool[1] && col + pcol < DATA_DIMS[1]); ++pcol) {
              for (int prow = 0; (prow < ptr_pool[0] && row + prow < DATA_DIMS[0]); ++prow) {
				  m += ptr_data[IDX2C(row + prow, col + pcol, DATA_DIMS[0])];
              }
          }
		  m /= ptr_pool[0]*ptr_pool[1];

          if (debug)
            fprintf(stderr, "count = %i\n",count);

          ptr_out[count] = m;
          count++;
      }
  }
}

/**
 * This is the wrapper for the actual computation.
 * It is a template so that multiple types can be handled.
 */
template <typename T>
void mexaveragepooling(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], mxClassID classID)
{

  /***************************************************************************/
  /** Variables */
  /***************************************************************************/
  mwSize IDX_DIMS[1];
  mwSize DATA_DIMS[4];
  mwSize M_DIMS[4];
  const mwSize *POOL_DIMS;
  int DATA_NUMEL;
  int POOL_NUMEL;

  /**
   * Pointers to data
   */
  T *ptr_data = NULL;
  T *ptr_pool = NULL;
  T *ptr_out = NULL;

  /***************************************************************************/
  /** Setting input pointers *************************************************/
  /***************************************************************************/
  ptr_data = (T *)mxGetData(prhs[0]);
  ptr_pool = (T *)mxGetData(prhs[1]);
  if (debug)
    fprintf(stderr,"Pooling size: h=%f, w=%f\n", ptr_pool[0], ptr_pool[1]);

  /***************************************************************************/
  /** Setting parameters *****************************************************/
  /***************************************************************************/
  /* Data dimensions. As also a 2D tensor can be used I fill empty dimensions
   * with 1 */
  const mwSize *tmp = mxGetDimensions(prhs[0]);
  DATA_DIMS[0] = tmp[0];
  DATA_DIMS[1] = tmp[1];
  if (mxGetNumberOfDimensions(prhs[0]) == 2) {
      DATA_DIMS[2] = 1;
      DATA_DIMS[3] = 1;
  } else if (mxGetNumberOfDimensions(prhs[0]) == 3) {
      DATA_DIMS[2] = tmp[2];
      DATA_DIMS[3] = 1;
  } else {
      DATA_DIMS[2] = tmp[2];
      DATA_DIMS[3] = tmp[3];
  }

  DATA_NUMEL = DATA_DIMS[0] * DATA_DIMS[1] * DATA_DIMS[2] * DATA_DIMS[3];
  if (debug)
    fprintf(stderr,"Data size: h=%d, w=%d, z=%d, n=%d (%d)\n", DATA_DIMS[0], DATA_DIMS[1], DATA_DIMS[2], DATA_DIMS[3], DATA_NUMEL);

  /* Output dimensions: the first output argument is of size equals to the input
   * whereas the second is of size equals to the number of pooled values.
   * Below there is ceil because also non complete tiles are considered when
   * input dims are not multiples of pooling dims. */
  M_DIMS[0] = ceil(float(DATA_DIMS[0]) / float(ptr_pool[0]));
  M_DIMS[1] = ceil(float(DATA_DIMS[1]) / float(ptr_pool[1]));
  M_DIMS[2] = DATA_DIMS[2];
  M_DIMS[3] = DATA_DIMS[3];
  IDX_DIMS[0] = M_DIMS[0] * M_DIMS[1] * M_DIMS[2] * M_DIMS[3];
  if (debug){
      fprintf(stderr,"Each output image has (%d, %d) pooled values, "
          "IDXs size: h=%d \n", M_DIMS[0], M_DIMS[1], IDX_DIMS[0]);
      fprintf(stderr, "M size: h=%d, w=%d, z=%d, n=%d\n", M_DIMS[0], M_DIMS[1], M_DIMS[2], M_DIMS[3]);
  }

  /***************************************************************************/
  /** Variables allocation ***************************************************/
  /***************************************************************************/
  /* OUTPUTS: average-values */
  plhs[0] = mxCreateNumericArray(4, M_DIMS, classID, mxREAL);
  ptr_out = (T *)mxGetData(plhs[0]);

  /***************************************************************************/
  /** Compute average-pooling ****************************************************/
  /***************************************************************************/
  int tile_start = 0;
  int ptr_offset = 0;
  int M_sample_size = M_DIMS[0] * M_DIMS[1] * M_DIMS[2];
  int D_sample_size = DATA_DIMS[0] * DATA_DIMS[1] * DATA_DIMS[2];

  for (int n = 0; n < DATA_DIMS[3]; ++n) {
#pragma omp parallel for
      for (int k = 0; k < DATA_DIMS[2]; ++k) {
          tile_start = n * M_sample_size + k * M_DIMS[0] * M_DIMS[1];
          ptr_offset = n * D_sample_size + k * DATA_DIMS[0] * DATA_DIMS[1];

          compute_average_pooling (&ptr_data[ptr_offset], DATA_DIMS, ptr_pool, &ptr_out[tile_start], ptr_offset);

          if (debug)
            fprintf(stderr, "tile_start: %i, ptr_offset: %i\n", tile_start, ptr_offset);
      }
  }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /***************************************************************************/
  /** Check input ************************************************************/
  /***************************************************************************/
  if (nrhs !=2)
    mexErrMsgTxt("Must have 2 input arguments: x, pooling_shape");

  if (nlhs !=1)
    mexErrMsgTxt("Must have 1 output arguments ([average_value])");

  if (mxIsComplex(prhs[0]) || !(mxIsClass(prhs[0],"single") || mxIsClass(prhs[0],"double")))
    mexErrMsgTxt("Input data must be real, single/double type");

  if (mxIsComplex(prhs[1]) || !(mxIsClass(prhs[1],"single") || mxIsClass(prhs[1],"double")))
    mexErrMsgTxt("Pooling dimensions (rows, cols) must be real, single/double type");

  if (mxGetNumberOfDimensions(prhs[0]) < 2)
    mexErrMsgTxt("Input data must have at least 2-dimensions (rows, cols, nchannels, nsamples) "
        "\nThe last two dimensions will be considered to be 1.");

  if (mxGetNumberOfDimensions(prhs[1]) != 2)
    mexErrMsgTxt("Pooling data must have 2-dimensions (prows, pcols)");

  mxClassID classID = mxGetClassID(prhs[0]);

  /** This is mainly to avoid two typenames. Should not be a big usability issue. */
  if (mxGetClassID(prhs[1]) != classID)
    mexErrMsgTxt("Input data and pooling need to be of the same type");

  /***************************************************************************/
  /** Switch for the supported data types */
  /***************************************************************************/
  if (classID == mxSINGLE_CLASS) {
      if (debug)
        fprintf(stderr, "Executing the single version\n");

      mexaveragepooling<float>(nlhs, plhs, nrhs, prhs, classID);
  }  else if (classID == mxDOUBLE_CLASS) {
      if (debug)
        fprintf(stderr, "Executing the double version\n");

      mexaveragepooling<double>(nlhs, plhs, nrhs, prhs, classID);
  }
}
