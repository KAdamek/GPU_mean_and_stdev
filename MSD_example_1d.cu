#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <vector>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "timer.h"
#include "MSD_GPU_library.h"

void Generate_dataset(float *h_input, size_t dim_x, int offset){
	for(size_t x=0; x<dim_x; x++){
		h_input[x]=rand() / (float)RAND_MAX;
		if(x>(dim_x-offset)) h_input[x] = 10000;
	}
}

//---------------------------------------------------------------------------------
//-------> Kahan MSD
void d_kahan_summation(float *signal, size_t dim_y, size_t dim_x, size_t offset, float *result, float *error){
	double sum;
	double sum_error;
	double a,b;
	
	sum=0;
	sum_error=0;
	for(size_t d=0;d<dim_y; d++){
		for(size_t s=0; s<(dim_x-offset); s++){
			a=signal[(size_t) (d*dim_x + s)]-sum_error;
			b=sum+a;
			sum_error=(b-sum);
			sum_error=sum_error-a;
			sum=b;
		}
	}
	*result=sum;
	*error=sum_error;
}

void d_kahan_sd(float *signal, size_t dim_y, size_t dim_x, size_t offset, double mean, float *result, float *error){
	double sum;
	double sum_error;
	double a,b,dtemp;
	
	sum=0;
	sum_error=0;
	for(size_t d=0;d<dim_y; d++){
		for(size_t s=0; s<(dim_x-offset); s++){
			dtemp=(signal[(size_t) (d*dim_x + s)]-sum_error - mean);
			a=dtemp*dtemp;
			b=sum+a;
			sum_error=(b-sum);
			sum_error=sum_error-a;
			sum=b;
		}
	}
	*result=sum;
	*error=sum_error;
}

void MSD_Kahan(float *h_input, size_t dim_y, size_t dim_x, size_t offset, double *mean, double *sd){
	float error, signal_mean, signal_sd;
	size_t nElements=dim_y*(dim_x-offset);
	
	d_kahan_summation(h_input, dim_y, dim_x, offset, &signal_mean, &error);
	signal_mean=signal_mean/nElements;
	
	d_kahan_sd(h_input, dim_y, dim_x, offset, signal_mean, &signal_sd, &error);
	signal_sd=sqrt(signal_sd/nElements);

	*mean=signal_mean;
	*sd=signal_sd;
}
//-------> Kahan MSD
//---------------------------------------------------------------------------------


int main(int argc, char* argv[]) {
	size_t dim_x;
	int offset;
	int device_id;

	// Check!
	char * pEnd;
	if (argc==4) {
		dim_x        = strtol(argv[1],&pEnd,10);
		offset       = strtol(argv[2],&pEnd,10);
		device_id    = strtol(argv[3],&pEnd,10);
	}
	else {
		printf("Argument error!\n");
		printf(" 1) x dimension of the data\n");
		printf(" 2) offset\n");
		printf(" 3) device id\n");
        return(1);
	}
	
	size_t input_size = dim_x;
	size_t MSD_size = 2;

	float *h_input;
	float *h_MSD;
	size_t h_MSD_nElements;

	h_input		 = (float *)malloc(input_size*sizeof(float));
	h_MSD 		 = (float *)malloc(MSD_size*sizeof(float));
	memset(h_MSD, 0.0, MSD_size*sizeof(float));

	srand(time(NULL));
	Generate_dataset(h_input, dim_x, offset);
	
	//----------------------------------------------------->
	//------------------------ DEVICE --------------------->
	int deviceCount;
	cudaError_t error_id;
	error_id = cudaGetDeviceCount(&deviceCount);
	if(error_id != cudaSuccess) {
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error_id) );
		return(1);
	}
	if(device_id>=deviceCount) {
		printf("Selected device is not available! Device id is %d;\n", device_id);
		return(1);
	}
	if (cudaSetDevice(device_id) != cudaSuccess) {
		printf("ERROR! unable to set the device with id %d.\n", device_id);
		return(1);
	}
	
	//---------> Checking memory
	size_t free_mem,total_mem;
	cudaMemGetInfo(&free_mem,&total_mem);
	float free_memory = (float) free_mem/(1024.0*1024.0);
	float memory_required = (input_size*sizeof(float))/(1024.0*1024.0);
	if(memory_required>free_memory) {
		printf("Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", (float) total_mem/(1024.0*1024.0), free_memory ,memory_required);
		printf("\n \n Array is too big for the device! \n \n"); 
		return(1);
	}
	
	//---------> Measurements
	double transfer_in, transfer_out;
	transfer_in=0.0; transfer_out=0.0;
	GpuTimer timer;
	
	//---------> Memory allocation
	float *d_input;
	float *d_MSD;
	size_t *d_MSD_nElements;
	if ( cudaSuccess != cudaMalloc((void **) &d_input,  sizeof(float)*input_size)) {printf("CUDA API error\n"); return(0);}
	if ( cudaSuccess != cudaMalloc((void **) &d_MSD, sizeof(float)*MSD_RESULTS_SIZE)) {printf("CUDA API error\n"); return(0);}
	if ( cudaSuccess != cudaMalloc((void **) &d_MSD_nElements, sizeof(size_t))) {printf("CUDA API error\n"); return(0);}
	
	//---------> Copy data to the device
	printf("Data transfer to the device memory...: \t");
	timer.Start();
	cudaMemcpy(d_input, h_input, input_size*sizeof(float), cudaMemcpyHostToDevice);
	timer.Stop();
	transfer_in+=timer.Elapsed();
	printf("done in %g ms.\n", timer.Elapsed());
	
	//---------> Create MSD plan
	MSD_Error MSD_error;
	bool outlier_rejection = false;
	MSD_Configuration MSD_conf;
	std::vector<size_t> dimensions={dim_x}; // dimensions of the data. Fastest moving coordinate is at the end.
	MSD_error = MSD_conf.Create_MSD_Plan(dimensions, offset, outlier_rejection, 3.0);
	if(MSD_error!=MSDSuccess) Get_MSD_Error(MSD_error);
	
	//---------> Get mean and stdev through library
	timer.Start();
	MSD_error = MSD_GetMeanStdev(d_MSD, d_MSD_nElements, d_input, MSD_conf);
	timer.Stop();
	printf("Calculation of mean and standard deviation took %g ms\n", timer.Elapsed());
	printf("MSD GPU library says it took: %g ms\n", MSD_conf.MSD_time);
	if(MSD_error!=MSDSuccess) Get_MSD_Error(MSD_error);
	MSD_conf.PrintInfo();
	
	//---------> Copy data to the host
	printf("Data transfer to the host...: \t");
	timer.Start();
	cudaMemcpy( h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy( &h_MSD_nElements, d_MSD_nElements, sizeof(h_MSD_nElements), cudaMemcpyDeviceToHost);
	timer.Stop();
	transfer_out+=timer.Elapsed();
	printf("done in %g ms.\n", timer.Elapsed());
	
	printf("\n\nMSD GPU library outputs one float array (for example d_MSD)\n which contains mean as d_MSD[0] and standard deviation as d_MSD[1].\n Values calculated by MSD GPU library are mean = %f; stdev = %f\n\n", h_MSD[0], h_MSD[1]);
	
	//---------> Feeing allocated resources
	if ( cudaSuccess != cudaFree(d_input)) {printf("CUDA API error\n"); return(0);}
	if ( cudaSuccess != cudaFree(d_MSD)) {printf("CUDA API error\n"); return(0);}
	if ( cudaSuccess != cudaFree(d_MSD_nElements)) {printf("CUDA API error\n"); return(0);}
	MSD_error = MSD_conf.Destroy_MSD_Plan();
	if(MSD_error!=MSDSuccess) Get_MSD_Error(MSD_error);
	//------------------------ DEVICE ---------------------<
	//-----------------------------------------------------<
	
	//---> Checks
	double signal_mean, signal_sd, merror, sderror;
	MSD_Kahan(h_input, 1, dim_x, offset, &signal_mean, &signal_sd);
	merror  = sqrt((signal_mean-h_MSD[0])*(signal_mean-h_MSD[0]));
	sderror = sqrt((signal_sd-h_MSD[1])*(signal_sd-h_MSD[1]));
	
	printf("GPU results: Mean: %e, Standard deviation: %e; Number of elements:%zu;\n", h_MSD[0], h_MSD[1], h_MSD_nElements);
	printf("MSD_kahan:   Mean: %e, Standard deviation: %e;\n",signal_mean, signal_sd);
	printf("Difference Kahan-GPU Mean:%e; Standard deviation:%e;\n", merror, sderror);
	//-------<
	
	free(h_input);
	free(h_MSD);

	cudaDeviceReset();
	
	printf("Finished!\n");

	return (0);
}

