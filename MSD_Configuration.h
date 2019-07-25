#ifndef __MSD_CONFIGURATION__
#define __MSD_CONFIGURATION__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <iostream>
#include <vector>

#include "MSD_params.h"
typedef int MSD_Error;


class MSD_window {
public:
	size_t x_start;
	size_t x_end;
	size_t y_start;
	size_t y_end;	
	size_t z_start;
	size_t z_end;
	
	MSD_window(){
		x_start = 0;
		x_end = 0;
		y_start = 0;
		y_end = 0;	
		z_start = 0;
		z_end = 0;
	}
};

class MSD_Configuration {
public:
	int3 nSteps;
	dim3 partials_gridSize;
	dim3 partials_blockSize;
	dim3 final_gridSize;
	dim3 final_blockSize;
	int nBlocks_total;
	size_t address;
	int nDim;
	int nBatches;
	std::vector<size_t> data_dim;
	size_t dim_x;
	size_t dim_y;
	size_t dim_z;
	int offset;
	bool outlier_rejection;
	float OR_sigma_range;
	double MSD_time;
	
	float *d_partial_MSD;
	int   *d_partial_nElements;
	
	cudaStream_t cuda_stream;
	
	MSD_Error MSD_error;
	
	bool ready;
	

	int Choose_Divider(size_t number, size_t max_divider) {
		int seive[12]={2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31};
		int f, nRest, nBlocks, N, N_accepted;

		N=1; N_accepted=1;
		do {
			N=1;
			for(f=0; f<12; f++) {
				nBlocks=number/seive[f];
				nRest=number - nBlocks*seive[f];
				if(nRest==0) {
					N=seive[f];
					N_accepted=N_accepted*N;
					break;
				}
			}
			number=number/N;
		} while(((size_t) N_accepted)<=max_divider && N>1);

		return(N_accepted/N);
	}

	int ChooseNumberOfThreads() {
		int nThreads=2048;
		int itemp=0;

		while(itemp==0 && nThreads>32) {
			nThreads=(nThreads>>1);
			itemp=(int)(nBlocks_total/(nThreads*32));
		}
		if(nThreads<32) nThreads=32;

		return(nThreads);
	}

	void Calculate_Kernel_Parameters_1d(int nBatches) {
		int nThreads;
		
		//--> Grid and block for partial calculation
		nSteps.x  = 8;
		partials_gridSize.x = (int) (dim_x-offset + nSteps.x*MSD_NTHREADS - 1)/(nSteps.x*MSD_NTHREADS);
		partials_gridSize.y = 1;
		partials_gridSize.z = nBatches;

		partials_blockSize.x = MSD_NTHREADS;
		partials_blockSize.y = 1;
		partials_blockSize.z = 1;
		
		nBlocks_total = partials_gridSize.x*partials_gridSize.y;
		//-----------------------------------------------<
		
		//--> Grid and block for final calculation
		nThreads = ChooseNumberOfThreads();

		final_gridSize.x = nBatches;
		final_gridSize.y = 1;
		final_gridSize.z = 1;

		final_blockSize.x = nThreads;
		final_blockSize.y = 1;
		final_blockSize.z = 1;
		//-----------------------------------------------<
	}
	
	void Calculate_Kernel_Parameters_2d(int nBatches) {
		int nThreads;
		
		//--> Grid and block for partial calculation
		nSteps.x  = 1;
		partials_gridSize.x = (int)((dim_x-offset + MSD_NTHREADS - 1)/MSD_NTHREADS);

		nSteps.y  = Choose_Divider(dim_y, 64);
		partials_gridSize.y = dim_y/nSteps.y; // we can do this because nSteps.y divides dim_y without remainder

		partials_gridSize.z = nBatches;
		
		nBlocks_total = partials_gridSize.x*partials_gridSize.y;

		partials_blockSize.x = MSD_NTHREADS;
		partials_blockSize.y = 1;
		partials_blockSize.z = 1;
		//-----------------------------------------------<
		
		//--> Grid and block for final calculation
		nThreads = ChooseNumberOfThreads();

		final_gridSize.x = nBatches;
		final_gridSize.y = 1;
		final_gridSize.z = 1;

		final_blockSize.x = nThreads;
		final_blockSize.y = 1;
		final_blockSize.z = 1;
		//-----------------------------------------------<
	}

	void Calculate_Kernel_Parameters_3d(int nBatches) {
		int nThreads;
		
		//--> Grid and block for partial calculation
		nSteps.x  = 1;
		partials_gridSize.x = (int)((dim_x-offset + MSD_NTHREADS - 1)/MSD_NTHREADS);

		nSteps.y  = Choose_Divider(dim_y, 64);
		partials_gridSize.y = dim_y/nSteps.y; // we can do this because nSteps.y divides dim_y without remainder
		
		nSteps.z = 1;
		partials_gridSize.z = dim_z;
		
		nBlocks_total = partials_gridSize.x*partials_gridSize.y*partials_gridSize.z;

		partials_blockSize.x = MSD_NTHREADS;
		partials_blockSize.y = 1;
		partials_blockSize.z = 1;
		//-----------------------------------------------<
		
		//--> Grid and block for final calculation
		nThreads = ChooseNumberOfThreads();

		final_gridSize.x = 1;
		final_gridSize.y = 1;
		final_gridSize.z = 1;

		final_blockSize.x = nThreads;
		final_blockSize.y = 1;
		final_blockSize.z = 1;
		//-----------------------------------------------<
		
		if(nBatches>1) MSD_error = 8;
	}
	
	void Allocate_temporary_workarea(){
		cudaError_t CUDA_error;
		if(nBlocks_total>0){
			CUDA_error = cudaMalloc((void **) &d_partial_MSD, nBatches*nBlocks_total*MSD_PARTIAL_SIZE*sizeof(float));
			if(CUDA_error != cudaSuccess) MSD_error = 1;
			CUDA_error = cudaMalloc((void **) &d_partial_nElements, nBatches*nBlocks_total*sizeof(int));
			if(CUDA_error != cudaSuccess) MSD_error = 1;
		}
	}
	
	void Reset() {
		nSteps.y = 0; nSteps.y = 0; nSteps.z = 0;
		partials_gridSize.x = 1; partials_gridSize.y = 1; partials_gridSize.z = 1;
		partials_blockSize.x = 1; partials_blockSize.y = 1; partials_blockSize.z = 1;
		final_gridSize.x = 1; final_gridSize.y = 1; final_gridSize.z = 1;
		final_blockSize.x = 1; final_blockSize.y = 1; final_blockSize.z = 1;
		nBlocks_total = 0;
		address = 0;
		dim_z = 0;
		dim_y = 0;
		dim_x = 0;
		data_dim.clear();
		offset = 0;
		MSD_time = 0;
		
		d_partial_MSD = NULL;
		d_partial_nElements = NULL;
		
		MSD_error = 0;
		ready = false;
		
		cuda_stream = NULL;
	}

	void PrintDebug() {
		printf("MSD-library --> Data dimensions: %zu x %zu x %zu; offset:%d;\n", dim_x, dim_y, dim_z, offset);
		printf("MSD-library --> nSteps:[%d;%d;%d]; nBlocks_total:%d; address:%zu;\n", nSteps.x, nSteps.y, nSteps.z, nBlocks_total, address);
		printf("MSD-library --> partials_gridSize=[%d;%d;%d]; partials_blockSize=[%d;%d;%d]\n", partials_gridSize.x, partials_gridSize.y, partials_gridSize.z, partials_blockSize.x, partials_blockSize.y, partials_blockSize.z);
		printf("MSD-library --> final_gridSize=[%d;%d;%d]; final_blockSize=[%d;%d;%d]\n", final_gridSize.x, final_gridSize.y, final_gridSize.z, final_blockSize.x, final_blockSize.y, final_blockSize.z);
	}
	
	void PrintInfo(){
		printf("Mean and standard deviation calculated in %g ms.\n", MSD_time);
	}
	
	void Bind_cuda_stream(cudaStream_t t_cuda_stream){
		cuda_stream = t_cuda_stream;
	}
	
	MSD_Error Create_MSD_Plan(std::vector<size_t> t_data_dimensions, int t_offset, bool enable_outlier_rejection, float t_OR_sigma_range, int t_nBatches=1){
		data_dim = t_data_dimensions;
		nDim = (int) data_dim.size();
		nBatches = t_nBatches;
		if(nBatches<1) {
			MSD_error = 7;
			return MSD_error;
		}
		
		if(data_dim.size()==0 || data_dim.size()>3) {
			MSD_error = 6;
			return(MSD_error);
		}
		if(data_dim.size()==1) {
			dim_x = data_dim[0];
		}
		if(data_dim.size()==2) {
			dim_y = data_dim[0];
			dim_x = data_dim[1];
		}
		if(data_dim.size()==3) {
			dim_z = data_dim[0];
			dim_y = data_dim[1];
			dim_x = data_dim[2];
		}
		
		offset   = t_offset;
		
		outlier_rejection = enable_outlier_rejection;
		OR_sigma_range = t_OR_sigma_range;
		
		if(nDim==1) Calculate_Kernel_Parameters_1d(nBatches);
		if(nDim==2) Calculate_Kernel_Parameters_2d(nBatches);
		if(nDim==3) Calculate_Kernel_Parameters_3d(nBatches);
		Allocate_temporary_workarea();
		
		
		//Debug
		//PrintDebug();
		if(MSD_error==0) ready=true;
		return(MSD_error);
	}
	
	MSD_Error Destroy_MSD_Plan(){
		cudaError_t CUDA_error;
		data_dim.clear();
		
		if(d_partial_MSD!=NULL){
			CUDA_error = cudaFree(d_partial_MSD);
			if(CUDA_error != cudaSuccess) MSD_error = 2;
			d_partial_MSD = NULL;
		}
		
		if(d_partial_nElements!=NULL){
			CUDA_error = cudaFree(d_partial_nElements);
			if(CUDA_error != cudaSuccess) MSD_error = 2;
			d_partial_nElements = NULL;
		}
		
		return(MSD_error);
	}
	
	MSD_Configuration(void) {
		Reset();
	}
	
	~MSD_Configuration(){
		cudaError_t CUDA_error;
		data_dim.clear();

		if(d_partial_MSD!=NULL){
			CUDA_error = cudaFree(d_partial_MSD);
			if(CUDA_error != cudaSuccess) MSD_error = 2;
		}
		
		if(d_partial_nElements!=NULL){
			CUDA_error = cudaFree(d_partial_nElements);
			if(CUDA_error != cudaSuccess) MSD_error = 2;
		}
	}
};

#endif

