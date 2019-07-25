#include "debug.h"

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

void Generate_dataset(float *h_input, size_t dim_x, int offset, int nBatches){
	for(int b=0; b<nBatches; b++){
		for(size_t x=0; x<dim_x; x++){
			h_input[b*dim_x + x]= (rand() / (float)RAND_MAX)*((float) (b+1));
			if(x>(dim_x-offset)) h_input[b*dim_x + x] = 10000;
		}
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
	int nBatches;
	int offset;
	int device_id;
	int nRuns;

	// Check!
	char * pEnd;
	if (argc==6) {
		dim_x        = strtol(argv[1],&pEnd,10);
		nBatches     = strtol(argv[2],&pEnd,10);
		offset       = strtol(argv[3],&pEnd,10);
		device_id    = strtol(argv[4],&pEnd,10);
		nRuns        = strtol(argv[5],&pEnd,10);
	}
	else {
		printf("Argument error!\n");
		printf(" 1) x dimension of the data\n");
		printf(" 2) Number of batches\n");
		printf(" 3) offset\n");
		printf(" 4) device id\n");
		printf(" 5) number of GPU kernel runs (optional)\n");
        return(1);
	}
	
	if(DEBUG) {
		printf("dim_x:        %zu\n",dim_x);
		printf("nBatches      %d\n",nBatches);
		printf("offset:       %d\n",offset);
		printf("device id:    %d\n",device_id);
		printf("nRuns:        %d\n",nRuns);
	}
	
	//----------------> GSL stuff 
	//const gsl_rng_type *rndType;
	//gsl_rng *rnd_handle;
	//gsl_rng_env_setup();
	//long int seed=(long int) time(NULL);
	//rndType = gsl_rng_default;
	//rnd_handle = gsl_rng_alloc (rndType);
	//gsl_rng_set(rnd_handle,seed);
	//----------------> GSL stuff 
	
	size_t input_size = dim_x*nBatches*sizeof(float);
	size_t MSD_size = MSD_RESULTS_SIZE*nBatches*sizeof(float);
	size_t MSD_elements_size = nBatches*sizeof(size_t);
	
	if(VERBOSE) printf("Input:%0.3f MB;\n",input_size/(1024.0*1024.0));
	if(VERBOSE) printf("\t\tWelcome\n");

	float *h_input;
	float *h_MSD;
	size_t *h_MSD_nElements;

	h_input		    = (float *)malloc(input_size);
	h_MSD 		    = (float *)malloc(MSD_size);
	h_MSD_nElements = (size_t *)malloc(MSD_elements_size);
	memset(h_MSD, 0.0, MSD_size);
	memset(h_MSD_nElements, 0.0, MSD_elements_size);

	srand(time(NULL));
	Generate_dataset(h_input, dim_x, offset, nBatches);
	
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
	float memory_required = (input_size)/(1024.0*1024.0);
	printf("\n");
	printf("Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", (float) total_mem/(1024.0*1024.0), free_memory ,memory_required);
	if(memory_required>free_memory) {
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
	if(cudaSuccess!=cudaMalloc((void **) &d_input, input_size) ) printf("Error!\n");
	if(cudaSuccess!=cudaMalloc((void **) &d_MSD, MSD_size) ) printf("Error!\n");
	if(cudaSuccess!=cudaMalloc((void **) &d_MSD_nElements, MSD_elements_size) ) printf("Error!\n");
	
	//---------> Copy data to the device
	printf("Data transfer to the device memory...: \t");
	timer.Start();
	cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
	timer.Stop();
	transfer_in+=timer.Elapsed();
	printf("done in %g ms.\n", timer.Elapsed());
	
	//---------> Create MSD plan
	MSD_Error MSD_error;
	bool outlier_rejection = false;
	MSD_Configuration MSD_conf;
	std::vector<size_t> dimensions={dim_x}; // dimensions of the data. Fastest moving coordinate is at the end.
	MSD_error = MSD_conf.Create_MSD_Plan(dimensions, offset, outlier_rejection, 3.0, nBatches);
	if(MSD_error!=MSDSuccess) Get_MSD_Error(MSD_error);
	if(DEBUG) MSD_conf.PrintDebug();
	
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
	cudaMemcpy( h_MSD, d_MSD, MSD_size, cudaMemcpyDeviceToHost);
	cudaMemcpy( h_MSD_nElements, d_MSD_nElements, MSD_elements_size, cudaMemcpyDeviceToHost);
	timer.Stop();
	transfer_out+=timer.Elapsed();
	printf("done in %g ms.\n", timer.Elapsed());
	
	printf("\nMSD GPU library outputs one float array (for example d_MSD)\n which contains mean as d_MSD[0] and standard deviation as d_MSD[1].\n Values calculated by MSD GPU library are mean = %f; stdev = %f\n\n", h_MSD[0], h_MSD[1]);
	printf("When processing multiple batches it creates an array with elements of the batches stored mean as MSD[batch*MSD_RESULTS_SIZE] and standard deviation as MSD[batch*MSD_RESULTS_SIZE + 1]\n");
	for(int b=0; b<nBatches; b++){
		printf("Batch %d mean=%f; stdev=%f; number of elements=%zu\n", b, h_MSD[b*MSD_RESULTS_SIZE], h_MSD[b*MSD_RESULTS_SIZE + 1], h_MSD_nElements[b]);
	}
	
	//---------> Feeing allocated resources
	cudaFree(d_input);
	cudaFree(d_MSD);
	cudaFree(d_MSD_nElements);
	MSD_conf.Destroy_MSD_Plan();
	//------------------------ DEVICE ---------------------<
	//-----------------------------------------------------<
	
	if (CHECK){
		double signal_mean, signal_sd, merror, sderror;
		for(int b=0; b<nBatches; b++){
			MSD_Kahan(&h_input[b*dim_x], dim_x, 1, offset, &signal_mean, &signal_sd);
			merror  = sqrt((signal_mean-h_MSD[b*MSD_RESULTS_SIZE])*(signal_mean-h_MSD[b*MSD_RESULTS_SIZE]));
			sderror = sqrt((signal_sd-h_MSD[b*MSD_RESULTS_SIZE + 1])*(signal_sd-h_MSD[b*MSD_RESULTS_SIZE + 1]));
			if(merror<1e-3 && sderror<1e-2) printf("     Test:\033[1;32mPASSED\033[0m\n");
			else printf("     Test:\033[1;31mFAILED\033[0m\n     Difference Kahan-GPU Mean:%e; Standard deviation:%e;\n", merror, sderror);
			
			printf("GPU results: Mean: %e, Standard deviation: %e; Number of elements:%zu;\n", h_MSD[b*MSD_RESULTS_SIZE], h_MSD[b*MSD_RESULTS_SIZE + 1], h_MSD_nElements[b]);
			printf("MSD_kahan:   Mean: %e, Standard deviation: %e;\n",signal_mean, signal_sd);
			printf("Difference Kahan-GPU Mean:%e; Standard deviation:%e;\n", merror, sderror);
		}
	}
	
	free(h_input);
	free(h_MSD);
	free(h_MSD_nElements);

	cudaDeviceReset();
	
	if (VERBOSE) printf("Finished!\n");

	return (0);
}

