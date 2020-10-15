#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MSD_params.h"
#include "MSD_Configuration.h"

#include "timer.h"
#include "MSD_GPU_kernels_shared.cu"
#include "MSD_GPU_kernels_final.cu"
#include "MSD_GPU_kernels_1d.cu"
#include "MSD_GPU_kernels_2d.cu"
#include "MSD_GPU_kernels_3d.cu"

//#define MSD_DEBUG
//#define MSD_DEBUG_BLOCKS

//----------------------------------------------------------------------------------------------
//--------------------------> Invisible from outside
void MSD_init(){
	//---------> Specific nVidia stuff
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
}

template<typename input_type>
int MSD_outlier_rejection(float *d_MSD, size_t *d_MSD_nElements, input_type *d_input, MSD_Configuration *MSD_conf){
	cudaError_t CUDA_error;
	CUDA_error = cudaGetLastError();
	if(CUDA_error != cudaSuccess) return(3);
	
	GpuTimer timer;
	double MSD_time = 0;
	
	timer.Start();
	//-------- Timed --->
	MSD_init();
	
	dim3 partials_gridSize  = MSD_conf->get_partial_gridSize();
	dim3 partials_blockSize = MSD_conf->get_partial_blockSize();
	dim3 final_gridSize     = MSD_conf->get_final_gridSize();
	dim3 final_blockSize    = MSD_conf->get_final_blockSize();
	cudaStream_t cuda_stream    = MSD_conf->get_CUDA_stream();
	
	float *d_partial_MSD       = MSD_conf->get_pointer_partial_MSD();
	int *d_partial_nElements   = MSD_conf->get_pointer_partial_nElements();
	int3 nSteps = MSD_conf->get_nSteps();
	int nDim     = MSD_conf->get_nDim(); 
	size_t dim_x = MSD_conf->get_dim_x();
	size_t dim_y = MSD_conf->get_dim_y();
	size_t dim_z = MSD_conf->get_dim_z();
	int offset = MSD_conf->get_offset();
	float sigma_threshold = MSD_conf->get_sigma_threshold();
	int nBlocks_total = MSD_conf->get_nBlocks_total();
	
	if(nDim==1){
		call_MSD_GPU_calculate_partials_1d_and_minmax(partials_gridSize, partials_blockSize, 0, cuda_stream, d_input, d_partial_MSD, d_partial_nElements, nSteps.x, dim_x, offset);
	}
	else if(nDim==2){
		call_MSD_GPU_calculate_partials_2d_and_minmax(partials_gridSize, partials_blockSize, 0, cuda_stream, d_input, d_partial_MSD, d_partial_nElements, dim_x, dim_y, offset);
	}
	else if(nDim==3){
		call_MSD_GPU_calculate_partials_3d_and_minmax(partials_gridSize, partials_blockSize, 0, cuda_stream, d_input, d_partial_MSD, d_partial_nElements, dim_x, dim_y, offset);
	}
	call_MSD_GPU_final_regular(final_gridSize, final_blockSize, 0, cuda_stream, d_partial_MSD, d_partial_nElements, d_MSD, d_MSD_nElements, nBlocks_total);
	//-------- Timed ---<
	timer.Stop();
	MSD_time += timer.Elapsed();
	//printf("Initial step done in	\033[1;32m%g\033[0m ms.\n", MSD_time);
	cudaStreamSynchronize(cuda_stream);
	
	#ifdef MSD_DEBUG
	float h_MSD[MSD_RESULTS_SIZE];
	size_t h_MSD_elements;
	cudaMemcpy( h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy( &h_MSD_elements, d_MSD_nElements, sizeof(size_t), cudaMemcpyDeviceToHost);
	printf("d_MSD=[%f;%f]; d_MSD_nElements=%d\n", h_MSD[0], h_MSD[1], (size_t) h_MSD_elements);
	#endif
	
	#ifdef MSD_DEBUG_BLOCKS
	float4 *h_blocks;
	h_blocks = new float4[nBlocks_total];
	cudaMemcpy( h_blocks, d_partial_MSD, nBlocks_total*sizeof(float4), cudaMemcpyDeviceToHost);
	int nBlocks_x = partials_gridSize.x;
	int nBlocks_y = partials_gridSize.y;
	for(int x=0; x<nBlocks_x; x++){
		for(int y=0; y<nBlocks_y; y++){
			printf("[%f;%f;%f;%f] ", h_blocks[y*nBlocks_x + x].x, h_blocks[y*nBlocks_x + x].y, h_blocks[y*nBlocks_x + x].z, h_blocks[y*nBlocks_x + x].w);
		}
		printf("\n");
	}
	delete [] h_blocks;
	#endif
	
	//TODO Criteria should be in the MSD_Config
	for(int i=0; i<5; i++){
		timer.Start();
		//-------- Timed --->
		if(nDim==1){
			call_MSD_BLN_calculate_partials_1d_and_minmax_with_outlier_rejection(partials_gridSize, partials_blockSize, 0, cuda_stream, d_input, d_partial_MSD, d_partial_nElements, d_MSD, nSteps.x, dim_x, offset, sigma_threshold);
		}
		else if(nDim==2){
			call_MSD_BLN_calculate_partials_2d_and_minmax_with_outlier_rejection(partials_gridSize, partials_blockSize, 0, cuda_stream, d_input, d_partial_MSD, d_partial_nElements, d_MSD, dim_x, dim_y, offset, sigma_threshold);
		}
		else if(nDim==3){
			call_MSD_BLN_calculate_partials_3d_and_minmax_with_outlier_rejection(partials_gridSize, partials_blockSize, 0, cuda_stream, d_input, d_partial_MSD, d_partial_nElements, d_MSD, dim_x, dim_y, offset, sigma_threshold);
		}
		call_MSD_GPU_final_nonregular(final_gridSize, final_blockSize, 0, cuda_stream, d_partial_MSD, d_partial_nElements, d_MSD, d_MSD_nElements, nBlocks_total);
		//-------- Timed ---<
		timer.Stop();
		MSD_time += timer.Elapsed();
		cudaStreamSynchronize(cuda_stream);
		
		#ifdef MSD_DEBUG
		float h_MSD[MSD_RESULTS_SIZE];
		size_t h_MSD_elements;
		cudaMemcpy( h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy( &h_MSD_elements, d_MSD_nElements, sizeof(size_t), cudaMemcpyDeviceToHost);
		printf("d_MSD=[%f;%f]; d_MSD_nElements=%d\n", h_MSD[0], h_MSD[1], (size_t) h_MSD_elements);
		#endif
	}
	
	MSD_conf->MSD_time = MSD_time;

	CUDA_error = cudaPeekAtLastError();
	if(CUDA_error != cudaSuccess) return(4);
	else return(0);
}



template<typename input_type>
int MSD_normal(float *d_MSD, size_t *d_MSD_nElements, input_type *d_input, MSD_Configuration *MSD_conf){
	cudaError_t CUDA_error;
	CUDA_error = cudaGetLastError();
	if(CUDA_error != cudaSuccess) return(3);
	
	GpuTimer timer;
	double MSD_time = 0;
	
	timer.Start();	
	//-------- Timed --->
	MSD_init();
	
	dim3 partials_gridSize  = MSD_conf->get_partial_gridSize();
	dim3 partials_blockSize = MSD_conf->get_partial_blockSize();
	dim3 final_gridSize    = MSD_conf->get_final_gridSize();
	dim3 final_blockSize   = MSD_conf->get_final_blockSize();
	cudaStream_t cuda_stream    = MSD_conf->get_CUDA_stream();
	
	float *d_partial_MSD       = MSD_conf->get_pointer_partial_MSD();
	int *d_partial_nElements   = MSD_conf->get_pointer_partial_nElements();
	int3 nSteps = MSD_conf->get_nSteps();
	int nDim     = MSD_conf->get_nDim(); 
	size_t dim_x = MSD_conf->get_dim_x();
	size_t dim_y = MSD_conf->get_dim_y();
	size_t dim_z = MSD_conf->get_dim_z();
	int offset = MSD_conf->get_offset();
	float sigma_threshold = MSD_conf->get_sigma_threshold();
	int nBlocks_total = MSD_conf->get_nBlocks_total();
	
	if(nDim==1){
		call_MSD_GPU_calculate_partials_1d(partials_gridSize, partials_blockSize, 0, cuda_stream, d_input, d_partial_MSD, d_partial_nElements, nSteps.x, dim_x, offset);
	}
	else if(nDim==2){
		call_MSD_GPU_calculate_partials_2d(partials_gridSize, partials_blockSize, 0, cuda_stream, d_input, d_partial_MSD, d_partial_nElements, dim_x, dim_y, offset);
	}
	else if(nDim==3){
		call_MSD_GPU_calculate_partials_3d(partials_gridSize, partials_blockSize, 0, cuda_stream, d_input, d_partial_MSD, d_partial_nElements, dim_x, dim_y, offset);
	}
	//cudaStreamSynchronize(cuda_stream);
	call_MSD_GPU_final_regular(final_gridSize, final_blockSize, 0, cuda_stream, d_partial_MSD, d_partial_nElements, d_MSD, d_MSD_nElements, nBlocks_total);
	//-------- Timed ---<
	timer.Stop();
	MSD_time += timer.Elapsed();
	cudaStreamSynchronize(cuda_stream);
	
	#ifdef MSD_DEBUG
	float h_MSD[MSD_RESULTS_SIZE];
	cudaMemcpy( h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy( h_MSD, d_MSD, sizeof(float), cudaMemcpyDeviceToHost);
	printf("d_MSD=[%f;%f];\n", h_MSD[0], h_MSD[1]);
	#endif
	
	MSD_conf->MSD_time = MSD_time;
	CUDA_error = cudaPeekAtLastError();
	if(CUDA_error != cudaSuccess) return(4);
	else return(0);
}

//----------------------------------------------------------------------------------------------




//----------------------------------------------------------------------------------------------
//--------------------------> Visible from outside

// higher dimension would be done by configuring MSD_config and this will then launch the kernels.
// Kernels must be capable of processing 1D, 1D batched in direction of primary coordinate, 1D batched in direction of secondary coordinate, 2D, 3D
// Kernels should be capable of windowing the results.
// HOW TO IMPLEMENT:
// 1D: this would require that kernels would go in all directions x,y,z so we could sum in the kernel any size we want.
// 1D batched: For this I need to modify the 'final' kernels so they can produce multiple outputs. Problem is kernels themselves...
// 1D batched transposed: I'm not sure.
// 2D: fine 
// 2D batched: multiple outputs for 'final' kernel
// 3D: fine
// 3D batched: multiple kernels for 'final' kernel
// Direct mean and stdev for small batched jobs
// TODO list: 
MSD_Error MSD_GetMeanStdev(float *d_MSD, size_t *d_MSD_nElements, float *d_input, MSD_Configuration &MSD_conf){
	MSD_Error MSD_error;
	if( !MSD_conf.MSD_ready() ) return(5);
	
	//--------> MSD
	if( MSD_conf.MSD_outlier_rejection() ){
		MSD_error = MSD_outlier_rejection(d_MSD, d_MSD_nElements, d_input, &MSD_conf);
	}
	else {
		MSD_error = MSD_normal(d_MSD, d_MSD_nElements, d_input, &MSD_conf);
	}
	
	return(MSD_error);
}

//-----------------------------------------------------------------------------------------<

