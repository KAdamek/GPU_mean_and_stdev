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

void Generate_dataset(float *h_input, size_t dim_x, size_t dim_y, size_t dim_z, int offset){
	for(size_t z=0; z<dim_z; z++){
		for(size_t y=0; y<dim_y; y++){
			for(size_t x=0; x<dim_x; x++){
				h_input[z*dim_x*dim_y + y*dim_x + x]=rand() / (float)RAND_MAX;
				if(x>(dim_x-offset)) h_input[z*dim_x*dim_y + y*dim_x + x] = 10000;
			}
		}
	}
	
	long int nSpikes = ( ((float) dim_z)*((float) dim_y)*((float) dim_x))*0.05;
	for(long int f=0; f<nSpikes; f++){
		size_t x = (size_t) (((double) dim_x)*((double) rand() / (double) RAND_MAX));
		size_t y = (size_t) (((double) dim_y)*((double) rand() / (double) RAND_MAX));
		size_t z = (size_t) (((double) dim_z)*((double) rand() / (double) RAND_MAX));
		h_input[z*dim_x*dim_y + y*dim_x + x]=2;
	}
}

//---------------------------------------------------------------------------------
//-------> Kahan MSD
void d_kahan_summation(float *signal, size_t dim_x, size_t dim_y, size_t dim_z, size_t offset, float *result, float *error){
	double sum;
	double sum_error;
	double a,b;
	
	sum=0;
	sum_error=0;
	for(size_t z=0; z<dim_z; z++){
		for(size_t y=0; y<dim_y; y++){
			for(size_t x=0; x<(dim_x-offset); x++){
				a=signal[(size_t) (z*dim_x*dim_y + y*dim_x + x)]-sum_error;
				b=sum+a;
				sum_error=(b-sum);
				sum_error=sum_error-a;
				sum=b;
			}
		}
	}
	*result=sum;
	*error=sum_error;
}

void d_kahan_sd(float *signal, size_t dim_x, size_t dim_y, size_t dim_z, size_t offset, double mean, float *result, float *error){
	double sum;
	double sum_error;
	double a,b,dtemp;
	
	sum=0;
	sum_error=0;
	for(size_t z=0; z<dim_z; z++){
		for(size_t y=0; y<dim_y; y++){
			for(size_t x=0; x<(dim_x-offset); x++){
				dtemp=(signal[(size_t) (z*dim_x*dim_y + y*dim_x + x)]-sum_error - mean);
				a=dtemp*dtemp;
				b=sum+a;
				sum_error=(b-sum);
				sum_error=sum_error-a;
				sum=b;
			}
		}
	}
	*result=sum;
	*error=sum_error;
}

void MSD_Kahan(float *h_input, size_t dim_x, size_t dim_y, size_t dim_z, size_t offset, double *mean, double *sd){
	float error, signal_mean, signal_sd;
	size_t nElements=dim_z*dim_y*(dim_x-offset);
	
	d_kahan_summation(h_input, dim_x, dim_y, dim_z, offset, &signal_mean, &error);
	signal_mean=signal_mean/nElements;
	
	d_kahan_sd(h_input, dim_x, dim_y, dim_z, offset, signal_mean, &signal_sd, &error);
	signal_sd=sqrt(signal_sd/nElements);

	*mean=signal_mean;
	*sd=signal_sd;
}
//-------> Kahan MSD
//---------------------------------------------------------------------------------


int main(int argc, char* argv[]) {
	size_t dim_x;
	size_t dim_y;
	size_t dim_z;
	int offset;
	int device_id;

	// Check!
	char * pEnd;
	if (argc==6) {
		dim_x        = strtol(argv[1],&pEnd,10); // this with CONV_SIZE gives signal size
		dim_y        = strtol(argv[2],&pEnd,10);
		dim_z        = strtol(argv[3],&pEnd,10);
		offset       = strtol(argv[4],&pEnd,10);
		device_id    = strtol(argv[5],&pEnd,10);
	}
	else {
		printf("Argument error!\n");
		printf(" 1) dimensions x\n");
		printf(" 2) dimensions y\n");
		printf(" 3) dimensions z\n");
		printf(" 4) offset\n");
		printf(" 5) device id\n");
        return(1);
	}
	
	size_t input_size = dim_x*dim_y*dim_z;
	size_t MSD_size = 2;

	float *h_input;
	float *h_MSD;
	size_t h_MSD_nElements;

	h_input		 = (float *)malloc(input_size*sizeof(float));
	h_MSD 		 = (float *)malloc(MSD_size*sizeof(float));
	memset(h_MSD, 0.0, MSD_size*sizeof(float));

	srand(time(NULL));
	Generate_dataset(h_input, dim_x, dim_y, dim_z, offset);
	
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
	cudaMalloc((void **) &d_input,  sizeof(float)*input_size);
	cudaMalloc((void **) &d_MSD, sizeof(float)*MSD_RESULTS_SIZE);
	cudaMalloc((void **) &d_MSD_nElements, sizeof(size_t));
	
	//---------> Copy data to the device
	printf("Data transfer to the device memory...: \t");
	timer.Start();
	cudaMemcpy(d_input, h_input, input_size*sizeof(float), cudaMemcpyHostToDevice);
	timer.Stop();
	transfer_in+=timer.Elapsed();
	printf("done in %g ms.\n", timer.Elapsed());
	
	//---------> Create MSD plan
	MSD_Error MSD_error;
	bool outlier_rejection = true;
	MSD_Configuration MSD_conf;
	std::vector<size_t> dimensions={dim_z, dim_y, dim_x}; // dimensions of the data. Fastest moving coordinate is at the end.
	MSD_error = MSD_conf.Create_MSD_Plan(dimensions, offset, outlier_rejection, 2.5);
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
	
	printf("\nMSD GPU library outputs one float array (for example d_MSD)\n which contains mean as d_MSD[0] and standard deviation as d_MSD[1].\n Values calculated by MSD GPU library are mean = %f; stdev = %f\n\n", h_MSD[0], h_MSD[1]);
	
	//---------> Feeing allocated resources
	cudaFree(d_input);
	cudaFree(d_MSD);
	cudaFree(d_MSD_nElements);
	MSD_conf.Destroy_MSD_Plan();
	//------------------------ DEVICE ---------------------<
	//-----------------------------------------------------<
	
	double signal_mean, signal_sd;
	MSD_Kahan(h_input, dim_x, dim_y, dim_z, offset, &signal_mean, &signal_sd);
	
	printf("GPU results with outlier rejection:      Mean: %e, Standard deviation: %e; Number of elements:%zu;\n", h_MSD[0], h_MSD[1], h_MSD_nElements);
	printf("CPU results without outlier rejection:   Mean: %e, Standard deviation: %e;\n",signal_mean, signal_sd);
	
	free(h_input);
	free(h_MSD);

	cudaDeviceReset();

	return (0);
}

