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
private:
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
	size_t offset;
	bool outlier_rejection;
	float sigma_threshold;
	
	float *d_partial_MSD;
	int   *d_partial_nElements;
	
	cudaStream_t cuda_stream;
	
	bool ready;

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
		partials_gridSize.x = (int) (((long int) (dim_x-offset + nSteps.x*MSD_NTHREADS - 1))/(nSteps.x*MSD_NTHREADS));
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

		nSteps.y  = MSD_Y_STEPS;
		partials_gridSize.y = ((double) (dim_y + MSD_Y_STEPS - 1))/((double) MSD_Y_STEPS);

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

		nSteps.y  = MSD_Y_STEPS;
		partials_gridSize.y = ((double) (dim_y + MSD_Y_STEPS - 1))/((double) MSD_Y_STEPS);
		
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
			size_t free_memory, total_memory;
			cudaMemGetInfo(&free_memory,&total_memory);
			size_t partial_MSD_size = nBatches*nBlocks_total*MSD_PARTIAL_SIZE*sizeof(float);
			size_t partial_nElements_size = nBatches*nBlocks_total*sizeof(int);
			if( (partial_MSD_size+partial_nElements_size)<free_memory) {
				CUDA_error = cudaMalloc((void **) &d_partial_MSD, partial_MSD_size);
				if(CUDA_error != cudaSuccess) MSD_error = 1;
				CUDA_error = cudaMalloc((void **) &d_partial_nElements, partial_nElements_size);
				if(CUDA_error != cudaSuccess) MSD_error = 1;
			}
			else MSD_error = 13;
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
		outlier_rejection = false;
		sigma_threshold = 0;
		
		d_partial_MSD = NULL;
		d_partial_nElements = NULL;
		
		MSD_error = 0;
		ready = false;
		
		cuda_stream = NULL;
	}

	
public:
	MSD_Error MSD_error;
	double MSD_time;
	
	//----> Getters
	bool MSD_ready(void) {
		return(ready);
	}
	bool MSD_outlier_rejection(void) {
		return(outlier_rejection);
	}
	dim3 get_partial_gridSize(){
		return(partials_gridSize);
	}
	dim3 get_partial_blockSize(){
		return(partials_blockSize);
	}
	dim3 get_final_gridSize(){
		return(final_gridSize);
	}
	dim3 get_final_blockSize(){
		return(final_blockSize);
	}
	cudaStream_t get_CUDA_stream(){
		return(cuda_stream);
	}
	float* get_pointer_partial_MSD(){
		return(d_partial_MSD);
	}
	int* get_pointer_partial_nElements(){
		return(d_partial_nElements);
	}
	int3 get_nSteps(){
		return(nSteps);
	}
	int get_nDim(){
		return(nDim);
	}
	size_t get_dim_x(){
		return(dim_x);
	}
	size_t get_dim_y(){
		return(dim_y);
	}
	size_t get_dim_z(){
		return(dim_z);
	}
	size_t get_offset(){
		return(offset);
	}
	float get_sigma_threshold(){
		return(sigma_threshold);
	}
	int get_nBlocks_total(){
		return(nBlocks_total);
	}
	
	
	//----> User functions	
	void PrintDebug() {
		printf("MSD-library --> Data dimensions: %zu x %zu x %zu; offset:%zu;\n", dim_x, dim_y, dim_z, offset);
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
	
	MSD_Error Create_MSD_Plan(std::vector<size_t> t_data_dimensions, size_t t_offset, bool enable_outlier_rejection, float t_sigma_threshold, int t_nBatches=1){
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
			if(dim_x<=0) MSD_error = 10;
		}
		if(data_dim.size()==2) {
			dim_y = data_dim[0];
			if(dim_y<=0) MSD_error = 11;
			dim_x = data_dim[1];
			if(dim_x<=0) MSD_error = 10;
		}
		if(data_dim.size()==3) {
			dim_z = data_dim[0];
			if(dim_z<=0) MSD_error = 12;
			dim_y = data_dim[1];
			if(dim_y<=0) MSD_error = 11;
			dim_x = data_dim[2];
			if(dim_x<=0) MSD_error = 10;
		}
		if(MSD_error>0) return(MSD_error);
		
		offset   = t_offset;
		if(offset>(dim_x-2)) {
			MSD_error = 9;
			return(MSD_error);
		}
		
		outlier_rejection = enable_outlier_rejection;
		sigma_threshold = t_sigma_threshold;
		
		if(nDim==1) Calculate_Kernel_Parameters_1d(nBatches);
		if(nDim==2) Calculate_Kernel_Parameters_2d(nBatches);
		if(nDim==3) Calculate_Kernel_Parameters_3d(nBatches);
		if(MSD_error>0) return(MSD_error);
		Allocate_temporary_workarea();
		if(MSD_error>0) return(MSD_error);
		
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

