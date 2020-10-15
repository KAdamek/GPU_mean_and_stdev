#ifndef MSD_GPU_FINAL_CU
#define MSD_GPU_FINAL_CU

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MSD_params.h"
#include "MSD_GPU_kernels_shared.cu"

//----------------------------------------------------------------------------------------
//--------> Kernels
template<typename nelements_accumulator>
__global__ void MSD_GPU_final_regular(float *d_partial_MSD, int *d_partial_nElements, float *d_output_MSD, nelements_accumulator *d_output_nElements, int size) {
	__shared__ float s_par_MSD[2*MSD_WARP*MSD_WARP];
	__shared__ nelements_accumulator s_par_nElements[MSD_WARP*MSD_WARP];

	float M, S;
	nelements_accumulator j;
	
	Sum_partials_regular( &M, &S, &j, &d_partial_MSD[blockIdx.x*size*MSD_PARTIAL_SIZE], &d_partial_nElements[blockIdx.x*size], s_par_MSD, s_par_nElements, size);

	//----------------------------------------------
	//---- Writing data
	if (threadIdx.x == 0) {
		d_output_MSD[MSD_RESULTS_SIZE*blockIdx.x] = M / (double) j;
		d_output_MSD[MSD_RESULTS_SIZE*blockIdx.x + 1] = sqrt(S / (double) j);
		d_output_nElements[blockIdx.x] = j;
	}
}


template<typename nelements_accumulator>
__global__ void MSD_GPU_final_regular(float *d_partial_MSD, int *d_partial_nElements, float *d_output_MSD, nelements_accumulator *d_output_nElements, float *d_previous_MSD, nelements_accumulator *d_previous_nElements, int size) {
	__shared__ float s_par_MSD[2*MSD_WARP*MSD_WARP];
	__shared__ nelements_accumulator s_par_nElements[MSD_WARP*MSD_WARP];

	float M, S;
	nelements_accumulator j;
	
	Sum_partials_regular( &M, &S, &j, &d_partial_MSD[blockIdx.x*size*MSD_PARTIAL_SIZE], &d_partial_nElements[blockIdx.x*size], s_par_MSD, s_par_nElements, size);

	if((*d_previous_nElements)>0){
		Merge(&M, &S, &j, d_previous_MSD[0], d_previous_MSD[1], (*d_previous_nElements));
	}
	
	//----------------------------------------------
	//---- Writing data
	if (threadIdx.x == 0) {
		d_output_MSD[MSD_RESULTS_SIZE*blockIdx.x] = M / (double) j;
		d_output_MSD[MSD_RESULTS_SIZE*blockIdx.x + 1] = sqrt(S / (double) j);
		d_output_nElements[blockIdx.x] = j;
		d_previous_MSD[MSD_RESULTS_SIZE*blockIdx.x] = M;
		d_previous_MSD[MSD_RESULTS_SIZE*blockIdx.x + 1] = S;
		d_previous_nElements[blockIdx.x] = j;
	}
}


template<typename nelements_accumulator>
__global__ void MSD_GPU_final_nonregular(float *d_partial_MSD, int *d_partial_nElements, float *d_output_MSD, nelements_accumulator *d_output_nElements, int size) {
	__shared__ float s_par_MSD[2*MSD_WARP*MSD_WARP];
	__shared__ nelements_accumulator s_par_nElements[MSD_WARP*MSD_WARP];
	
	float M, S;
	nelements_accumulator j;

	Sum_partials_nonregular( &M, &S, &j, &d_partial_MSD[blockIdx.x*size*MSD_PARTIAL_SIZE], &d_partial_nElements[blockIdx.x*size], s_par_MSD, s_par_nElements, size);
	
	//----------------------------------------------
	//---- Writing data
	if (threadIdx.x == 0) {
		d_output_MSD[MSD_RESULTS_SIZE*blockIdx.x]     = M / (double) j;
		d_output_MSD[MSD_RESULTS_SIZE*blockIdx.x + 1] = sqrt(S / (double) j);
		d_output_nElements[blockIdx.x] = j;
		//printf("Mean=%f; Stdev=%f; j=%e;\n", d_output_MSD[0], d_output_MSD[1], (double) j);
	}
}


template<typename nelements_accumulator>
__global__ void MSD_GPU_final_nonregular(float *d_partial_MSD, int *d_partial_nElements, float *d_output_MSD, nelements_accumulator *d_output_nElements, float *d_previous_MSD, nelements_accumulator *d_previous_nElements, int size) {
	__shared__ float s_par_MSD[2*MSD_WARP*MSD_WARP];
	__shared__ nelements_accumulator s_par_nElements[MSD_WARP*MSD_WARP];
	
	float M, S;
	nelements_accumulator j;

	Sum_partials_nonregular( &M, &S, &j, &d_partial_MSD[blockIdx.x*size*MSD_PARTIAL_SIZE], &d_partial_nElements[blockIdx.x*size], s_par_MSD, s_par_nElements, size);
	
	if((*d_previous_nElements)>0){
		Merge(&M, &S, &j, d_previous_MSD[0], d_previous_MSD[1], (*d_previous_nElements));
	}
	
	//----------------------------------------------
	//---- Writing data
	if (threadIdx.x == 0) {
		d_output_MSD[MSD_RESULTS_SIZE*blockIdx.x] = M / (double) j;
		d_output_MSD[MSD_RESULTS_SIZE*blockIdx.x + 1] = sqrt(S / (double) j);
		d_output_nElements[blockIdx.x] = j;
		d_previous_MSD[MSD_RESULTS_SIZE*blockIdx.x] = M;
		d_previous_MSD[MSD_RESULTS_SIZE*blockIdx.x + 1] = S;
		d_previous_nElements[blockIdx.x] = j;
	}
}

//----------------------------------------------------------------------------------------<



//----------------------------------------------------------------------------
//---------------> Simple C wrappers
// NOTE: for the moment this will work only for fp32
void call_MSD_GPU_final_regular(const dim3 &grid_size, const dim3 &block_size, int shared_memory_bytes, cudaStream_t streams, float *d_partial_MSD, int *d_partial_nElements, float *d_output_MSD, size_t *d_output_nElements, int size){
	MSD_GPU_final_regular<<< grid_size, block_size, shared_memory_bytes, streams>>>(d_partial_MSD, d_partial_nElements, d_output_MSD, d_output_nElements, size);
}

void call_MSD_GPU_final_regular(const dim3 &grid_size, const dim3 &block_size, int shared_memory_bytes, cudaStream_t streams, float *d_partial_MSD, int *d_partial_nElements, float *d_output_MSD, size_t *d_output_nElements, float *d_previous_MSD, size_t *d_previous_nElements, int size){
	MSD_GPU_final_regular<<< grid_size, block_size, shared_memory_bytes, streams>>>(d_partial_MSD, d_partial_nElements, d_output_MSD, d_output_nElements, d_previous_MSD, d_previous_nElements, size);
}

void call_MSD_GPU_final_nonregular(const dim3 &grid_size, const dim3 &block_size, int shared_memory_bytes, cudaStream_t streams, float *d_partial_MSD, int *d_partial_nElements, float *d_output_MSD, size_t *d_output_nElements, int size){
	MSD_GPU_final_nonregular<<< grid_size, block_size, shared_memory_bytes, streams>>>(d_partial_MSD, d_partial_nElements, d_output_MSD, d_output_nElements, size);
}

void call_MSD_GPU_final_nonregular(const dim3 &grid_size, const dim3 &block_size, int shared_memory_bytes, cudaStream_t streams, float *d_partial_MSD, int *d_partial_nElements, float *d_output_MSD, size_t *d_output_nElements, float *d_previous_MSD, size_t *d_previous_nElements, int size){
	MSD_GPU_final_nonregular<<< grid_size, block_size, shared_memory_bytes, streams>>>(d_partial_MSD, d_partial_nElements, d_output_MSD, d_output_nElements, d_previous_MSD, d_previous_nElements, size);
}

//----------------------------------------------------------------------------<

#endif
