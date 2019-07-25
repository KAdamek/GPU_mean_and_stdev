#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MSD_GPU_kernels_shared.cu"

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//------------------- Kernels



//-----------------------------------------------------------------------
//---------------> Computes partials for mean and standard deviation
template<typename input_type>
__global__ void MSD_GPU_calculate_partials_1d(input_type const* __restrict__ d_input, float *d_output_partial_MSD, int *d_output_partial_nElements, int x_steps, size_t nTimesamples, int offset) {
	__shared__ float s_par_MSD[2*MSD_NTHREADS];
	__shared__ int s_par_nElements[MSD_NTHREADS];
	float M, S, ftemp;
	int j;
	
	size_t gpos = blockIdx.z*nTimesamples;
	size_t spos = blockIdx.x*x_steps*MSD_NTHREADS + threadIdx.x;
	M=0;	S=0;	j=0;	ftemp=0;
	if( spos<(nTimesamples-offset) ){
		
		ftemp = (float) d_input[gpos + spos];
		Initiate( &M, &S, &j, ftemp);
		
		spos = spos + MSD_NTHREADS;
		for (int xf = 1; xf < x_steps; xf++) {
			if( spos<(nTimesamples-offset) ){
				ftemp = (float) d_input[gpos + spos];
				Add_one( &M, &S, &j, ftemp);
				spos = spos + MSD_NTHREADS;
			}
		}
	}
	
	s_par_MSD[threadIdx.x] = M;
	s_par_MSD[MSD_NTHREADS + threadIdx.x] = S;
	s_par_nElements[threadIdx.x] = j;
	
	__syncthreads();
	
	Reduce_SM( &M, &S, &j, s_par_MSD, s_par_nElements );
	Reduce_WARP( &M, &S, &j);
	
	//----------------------------------------------
	//---- Writing data
	if (threadIdx.x == 0) {
		gpos = blockIdx.z*gridDim.x + blockIdx.x;
		d_output_partial_MSD[MSD_PARTIAL_SIZE*gpos]     = M;
		d_output_partial_MSD[MSD_PARTIAL_SIZE*gpos + 1] = S;
		d_output_partial_nElements[gpos]                = j;
		//if(blockIdx.x<100) printf("result: b:[%d;%d;%d] M=%f; S=%f; j=%e\n", blockIdx.x, blockIdx.y, blockIdx.z, M, S, (double) j);
	}
}


template<typename input_type>
__global__ void MSD_GPU_calculate_partials_1d_and_minmax(input_type const* __restrict__ d_input, float *d_output_partial_MSD, int *d_output_partial_nElements, int x_steps, size_t nTimesamples, int offset) {
	__shared__ float s_par_MSD[MSD_PARTIAL_SIZE*MSD_NTHREADS];
	__shared__ int s_par_nElements[MSD_NTHREADS];
	float M, S, max, min, ftemp;
	int j;
	
	size_t gpos = blockIdx.z*nTimesamples;
	size_t spos = blockIdx.x*x_steps*MSD_NTHREADS + threadIdx.x;
	M=0;	S=0;	j=0;	ftemp=0;
	if( spos<(nTimesamples-offset) ){
		
		ftemp = (float) d_input[gpos + spos];
		Initiate( &M, &S, &j, ftemp);
		max = ftemp;
		min = ftemp;
		
		spos = spos + MSD_NTHREADS;
		for (int xf = 1; xf < x_steps; xf++) {
			if( spos<(nTimesamples-offset) ){
				ftemp = (float) d_input[gpos + spos];
				max = (fmaxf(max,ftemp));
				min = (fminf(min,ftemp));
				Add_one( &M, &S, &j, ftemp);
				spos = spos + MSD_NTHREADS;
			}
		}
	}
	
	s_par_MSD[threadIdx.x] = M;
	s_par_MSD[blockDim.x + threadIdx.x] = S;
	s_par_MSD[2*blockDim.x + threadIdx.x] = max;
	s_par_MSD[3*blockDim.x + threadIdx.x] = min;
	s_par_nElements[threadIdx.x] = j;
	
	__syncthreads();
	
	Reduce_SM_max( &M, &S, &max, &min, &j, s_par_MSD, s_par_nElements );
	Reduce_WARP_max( &M, &S, &max, &min, &j);
	
	//----------------------------------------------
	//---- Writing data
	if (threadIdx.x == 0) {
		gpos = blockIdx.z*gridDim.x + blockIdx.x;
		d_output_partial_MSD[MSD_PARTIAL_SIZE*gpos] = M;
		d_output_partial_MSD[MSD_PARTIAL_SIZE*gpos + 1] = S;
		d_output_partial_MSD[MSD_PARTIAL_SIZE*gpos + 2] = max;
		d_output_partial_MSD[MSD_PARTIAL_SIZE*gpos + 3] = min;
		d_output_partial_nElements[gpos] = j;
		//if(blockIdx.x<5 && blockIdx.y<5) printf("M=%f; S=%f; max=%f; min=%f; j=%e\n", M, S, max, min, (double) j);
	}
}


template<typename input_type>
__global__ void MSD_BLN_calculate_partials_1d_and_minmax_with_outlier_rejection(input_type const* __restrict__ d_input, float *d_output_partial_MSD, int *d_output_partial_nElements, float *d_MSD, int x_steps, size_t nTimesamples, int offset, float bln_sigma_constant) {
	__shared__ float s_par_MSD[MSD_PARTIAL_SIZE*MSD_NTHREADS];
	__shared__ int s_par_nElements[MSD_NTHREADS];
	
	float M, S, ftemp, max, min;
	int j;
	float limit_down = d_MSD[MSD_RESULTS_SIZE*blockIdx.z] - bln_sigma_constant*d_MSD[MSD_RESULTS_SIZE*blockIdx.z + 1];
	float limit_up = d_MSD[MSD_RESULTS_SIZE*blockIdx.z] + bln_sigma_constant*d_MSD[MSD_RESULTS_SIZE*blockIdx.z + 1];

	size_t temp_gpos = blockIdx.z*gridDim.x + blockIdx.x;
	max = d_output_partial_MSD[MSD_PARTIAL_SIZE*temp_gpos + 2];
	min = d_output_partial_MSD[MSD_PARTIAL_SIZE*temp_gpos + 3];
	if( (min>limit_down) && (max < limit_up) ) return;
	
	size_t gpos = blockIdx.z*nTimesamples;
	size_t spos = blockIdx.x*x_steps*MSD_NTHREADS + threadIdx.x;
	M=0;	S=0;	j=0;	ftemp=0;	max=0;	min=0;
	if( spos<(nTimesamples-offset) ){
		for (int xf = 0; xf < x_steps; xf++) {
			if( spos<(nTimesamples-offset) ){
				ftemp = (float) d_input[gpos + spos];
				if( (ftemp>limit_down) && (ftemp < limit_up) ){
					if(j==0){
						Initiate( &M, &S, &j, ftemp);
						max = ftemp;
						min = ftemp;
					}
					else{
						Add_one( &M, &S, &j, ftemp);
						max = fmaxf(max, ftemp);
						min = fminf(min, ftemp);
					}			
				}
				spos = spos + MSD_NTHREADS;
			}
		}
		
	}
	
	s_par_MSD[threadIdx.x] = M;
	s_par_MSD[blockDim.x + threadIdx.x] = S;
	s_par_MSD[2*blockDim.x + threadIdx.x] = max;
	s_par_MSD[3*blockDim.x + threadIdx.x] = min;
	s_par_nElements[threadIdx.x] = j;
	
	__syncthreads();
	
	Reduce_SM_max( &M, &S, &max, &min, &j, s_par_MSD, s_par_nElements );
	Reduce_WARP_max( &M, &S, &max, &min, &j);
	
	//----------------------------------------------
	//---- Writing data
	if (threadIdx.x == 0) {
		gpos = blockIdx.z*gridDim.x + blockIdx.x;
		d_output_partial_MSD[MSD_PARTIAL_SIZE*gpos] = M;
		d_output_partial_MSD[MSD_PARTIAL_SIZE*gpos + 1] = S;
		d_output_partial_MSD[MSD_PARTIAL_SIZE*gpos + 2] = max;
		d_output_partial_MSD[MSD_PARTIAL_SIZE*gpos + 3] = min;
		d_output_partial_nElements[gpos] = j;
		//if(blockIdx.x<5 && blockIdx.y<5) printf("b:[%d;%d;%d] M=%f; S=%f; max=%f; min=%f; j=%e\n", blockIdx.x, blockIdx.y, blockIdx.z, M, S, max, min, (double) j);
	}
}
//------------------- Kernels with functions
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------



template<typename input_type>
void call_MSD_GPU_calculate_partials_1d(const dim3 &grid_size, const dim3 &block_size, int shared_memory_bytes, cudaStream_t streams,
			input_type *d_input, float *d_output_partial_MSD, int *d_output_partial_nElements, int y_steps, size_t nTimesamples, int offset){
	MSD_GPU_calculate_partials_1d<<< grid_size, block_size, shared_memory_bytes, streams>>>(d_input, d_output_partial_MSD, d_output_partial_nElements, y_steps, nTimesamples, offset);
}

template<typename input_type>
void call_MSD_GPU_calculate_partials_1d_and_minmax(const dim3 &grid_size, const dim3 &block_size, int shared_memory_bytes, cudaStream_t streams,
			input_type *d_input, float *d_output_partial_MSD, int *d_output_partial_nElements, int y_steps, size_t nTimesamples, int offset) {
	MSD_GPU_calculate_partials_1d_and_minmax<<< grid_size, block_size, shared_memory_bytes, streams>>>(d_input, d_output_partial_MSD, d_output_partial_nElements, y_steps, nTimesamples, offset);
}

template<typename input_type>
void call_MSD_BLN_calculate_partials_1d_and_minmax_with_outlier_rejection(const dim3 &grid_size, const dim3 &block_size, int shared_memory_bytes, cudaStream_t streams,
			input_type *d_input, float *d_output_partial_MSD, int *d_output_partial_nElements, float *d_MSD, int y_steps, size_t nTimesamples, int offset, float bln_sigma_constant){
	MSD_BLN_calculate_partials_1d_and_minmax_with_outlier_rejection<<< grid_size, block_size, shared_memory_bytes, streams>>>(d_input, d_output_partial_MSD, d_output_partial_nElements, d_MSD, y_steps, nTimesamples, offset, bln_sigma_constant);
}
//---------------------------------------------------------------------------<




