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

float max_error = 1.0e-4;

void Generate_dataset(float *h_input, size_t dim_x, size_t dim_y, size_t dim_z, size_t offset, float scale, float spike_ratio){
	for(size_t z=0; z<dim_z; z++){
		for(size_t y=0; y<dim_y; y++){
			for(size_t x=0; x<dim_x; x++){
				size_t pos = z*dim_x*dim_y + y*dim_x + x;
				h_input[pos] = ( rand() / ((float) RAND_MAX) ) * scale;
				if(x>(dim_x-offset)) h_input[pos] = 10000;
			}
		}
	}
	
	long int nSpikes = ( ((float) dim_z)*((float) dim_y)*((float) dim_x))*spike_ratio;
	for(long int f=0; f<nSpikes; f++){
		size_t x = (size_t) (((double) dim_x)*((double) rand() / (double) RAND_MAX));
		size_t y = (size_t) (((double) dim_y)*((double) rand() / (double) RAND_MAX));
		size_t z = (size_t) (((double) dim_z)*((double) rand() / (double) RAND_MAX));
		size_t pos = z*dim_x*dim_y + y*dim_x + x;
		h_input[pos] = 2.0*scale;
	}
}

void Generate_dataset_for_offset_test(float *h_input, size_t dim_x, size_t dim_y, size_t dim_z, size_t offset){
	for(size_t z=0; z<dim_z; z++){
		for(size_t y=0; y<dim_y; y++){
			for(size_t x=0; x<dim_x; x++){
				size_t pos = z*dim_x*dim_y + y*dim_x + x;
				h_input[pos] = ( rand() / ((float) RAND_MAX) ) * (10000.0/((double) dim_x))*((double) x/((double) dim_x));
				if(x>(dim_x-offset)) h_input[pos] = 10000;
			}
		}
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

bool Check_memory(size_t dim_x, size_t dim_y, size_t dim_z){
	size_t free_memory, total_memory, required_memory;
	cudaMemGetInfo(&free_memory,&total_memory);
	required_memory = dim_x*dim_y*dim_z*sizeof(float) + 1 + MSD_RESULTS_SIZE;
	if(required_memory>free_memory) {
		printf("Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", (float) total_memory/(1024.0*1024.0), (float) free_memory/(1024.0*1024.0) , (float) required_memory/(1024.0*1024.0));
		printf("\n \n Array is too big for the device! \n \n"); 
		return(1);
	}
	else return(0);
}

int MSD(float *h_input, size_t dim_x, size_t dim_y, size_t dim_z, size_t offset, bool outlier_rejection, float outlier_rejection_sigma, MSD_Error *error, int verbose = 0){
	GpuTimer timer;
	
	int MSD_size = MSD_RESULTS_SIZE*sizeof(float);
	int MSD_elements_size = sizeof(size_t);
	size_t input_size = dim_x*dim_y*dim_z*sizeof(float);
	
	//---------> CPU Memory allocation
	float *h_MSD;
	size_t *h_MSD_nElements;
	h_MSD 		    = (float *)malloc(MSD_size);
	h_MSD_nElements = (size_t *)malloc(MSD_elements_size);
	memset(h_MSD, 0.0, MSD_size);
	memset(h_MSD_nElements, 0.0, MSD_elements_size);
	
	//---------> GPU Memory allocation
	float *d_input;
	float *d_MSD;
	size_t *d_MSD_nElements;
	if ( cudaSuccess != cudaMalloc((void **) &d_input, input_size)) {
		printf("CUDA API error while allocating GPU memory\n");
	}
	if ( cudaSuccess != cudaMalloc((void **) &d_MSD, MSD_size)) {
		printf("CUDA API error while allocating GPU memory\n");
	}
	if ( cudaSuccess != cudaMalloc((void **) &d_MSD_nElements, MSD_elements_size)) {
		printf("CUDA API error while allocating GPU memory\n");
	}
	
	//---------> Copy data to the device
	cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
	
	//---------> Create MSD plan
	MSD_Error MSD_error;
	MSD_Configuration MSD_conf;
	std::vector<size_t> dimensions={dim_z, dim_y, dim_x}; // dimensions of the data. Fastest moving coordinate is at the end.
	MSD_error = MSD_conf.Create_MSD_Plan(dimensions, offset, outlier_rejection, outlier_rejection_sigma);
	if(MSD_error!=MSDSuccess) Get_MSD_Error(MSD_error);
	*error = MSD_error;
	
	//---------> Get mean and stdev through library
	timer.Start();
	MSD_error = MSD_GetMeanStdev(d_MSD, d_MSD_nElements, d_input, MSD_conf);
	timer.Stop();
	if(MSD_error!=MSDSuccess) Get_MSD_Error(MSD_error);
	if(verbose) printf("Calculation of mean and standard deviation took %g ms\n", timer.Elapsed());
	
	//---------> Copy data to the host
	cudaMemcpy( h_MSD, d_MSD, MSD_size, cudaMemcpyDeviceToHost);
	cudaMemcpy( h_MSD_nElements, d_MSD_nElements, MSD_elements_size, cudaMemcpyDeviceToHost);
	
	//---> Checks
	int no_check_error = 1;
	double signal_mean, signal_sd, merror, sderror;
	MSD_Kahan(h_input, dim_x, dim_y, dim_z, offset, &signal_mean, &signal_sd);
	merror  = sqrt((signal_mean-h_MSD[0])*(signal_mean-h_MSD[0]));
	sderror = sqrt((signal_sd-h_MSD[1])*(signal_sd-h_MSD[1]));
	if(merror>max_error && sderror>max_error) no_check_error = no_check_error*0;
	
	if(verbose) {
		printf("GPU results: Mean: %e, Standard deviation: %e; Number of elements:%zu;\n", h_MSD[0], h_MSD[1], h_MSD_nElements[0]);
		printf("CPU results: Mean: %e, Standard deviation: %e;\n",signal_mean, signal_sd);
		if(!outlier_rejection) printf("Difference CPU-GPU Mean:%e; Standard deviation:%e;\n", merror, sderror);
	}
	
	free(h_MSD);
	free(h_MSD_nElements);
	if ( cudaSuccess != cudaFree(d_input)) {
		printf("CUDA API error while deallocating GPU memory\n");
	}
	if ( cudaSuccess != cudaFree(d_MSD)) {
		printf("CUDA API error while deallocating GPU memory\n");
	}
	if ( cudaSuccess != cudaFree(d_MSD_nElements)) {
		printf("CUDA API error while deallocating GPU memory\n");
	}
	MSD_error = MSD_conf.Destroy_MSD_Plan();
	if(MSD_error!=MSDSuccess) Get_MSD_Error(MSD_error);
	
	if(no_check_error==1) return(1);
	else return(0);
}

int main(int argc, char* argv[]) {
	size_t dim_x;
	size_t dim_y;
	size_t dim_z;
	int offset;
	int device_id;

	// Check!
	char * pEnd;
	if (argc==6) {
		dim_x        = strtol(argv[1],&pEnd,10);
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
		printf("Example: MSD_example_3d.exe 1000 1000 1000 15 0\n");
        return(1);
	}
	
	size_t input_size = dim_x*dim_y*dim_z;
	
	//---------> Device initialization
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
	
	size_t free_memory, total_memory;
	cudaMemGetInfo(&free_memory,&total_memory);
	offset = 0;
	if(Check_memory(dim_x, dim_y, dim_z)) return(1);
	
	//---------> CPU Memory allocation
	float *h_input;
	srand(time(NULL));
	MSD_Error MSD_error;

	//============================== Mean and standard deviation ==========================o
	printf("Allocating host memory\n");
	h_input = (float *)malloc(input_size*sizeof(float));
	printf("Generating data\n");
	Generate_dataset(h_input, dim_x, dim_y, dim_z, offset, 1.0, 0.05);
	
	printf("\nMean and standard deviation:\n"); 
	MSD(h_input, dim_x, dim_y, dim_z, offset, false, 0.0, &MSD_error, 1);
	if(MSD_error!=MSDSuccess) Get_MSD_Error(MSD_error);
	
	printf("\nMean and standard deviation with outlier rejection:\n"); 
	MSD(h_input, dim_x, dim_y, dim_z, offset, true, 3.0, &MSD_error, 1);
	if(MSD_error!=MSDSuccess) Get_MSD_Error(MSD_error);
	
	free(h_input);
	//--------------------<


	//============================== Unit tests ==========================o
	printf("\n\n====================================\n");
	printf("Unit tests:\n");
	int test_passed = 1;
	
	printf("Different data size:");
	test_passed = 1;
	int sizes_size = 9;
	size_t sizes[] = {2, 3, 7, 50, 100, 1000, (size_t) pow((free_memory/(sizeof(float)*4)),0.333333), (size_t) pow((free_memory/(sizeof(float)*2)),0.333333), (size_t) pow(((free_memory/sizeof(float))*(3.0/4.0)),0.333333)};
	for(int f=0; f<sizes_size; f++){
		offset = 0;
		dim_x = sizes[f];
		dim_y = sizes[f];
		dim_z = sizes[f];
		if(!Check_memory(dim_x, dim_y, dim_z)) {
			h_input = (float *)malloc(dim_x*dim_y*dim_z*sizeof(float));
			Generate_dataset(h_input, dim_x, dim_y, dim_z, offset, 1.0, 0.05);
			int error = MSD(h_input, dim_x, dim_y, dim_z, offset, false, 0.0, &MSD_error, 0);
			test_passed = test_passed*error;
			free(h_input);
		}
		printf(".");
		fflush(stdout);
	}
	if(test_passed) printf("PASSED\n");
	else printf("FAILED\n");
	printf("\n");	
	
	printf("Different offset values:");
	test_passed = 1;
	dim_x = 1000;
	dim_y = 1000;
	dim_z = 1000;
	std::vector<size_t> offs{0, dim_x/4, dim_x/2, (size_t) ((3.0/4.0)*dim_x)};
	for(size_t f=0; f<offs.size(); f++){
		offset = offs[f];
		h_input = (float *)malloc(dim_x*dim_y*dim_z*sizeof(float));
		Generate_dataset_for_offset_test(h_input, dim_x, dim_y, dim_z, offset);
		int error = MSD(h_input, dim_x, dim_y, dim_z, offset, false, 0.0, &MSD_error, 0);
		test_passed = test_passed*error;
		free(h_input);
		printf(".");
		fflush(stdout);
	}
	if(test_passed) printf("PASSED\n");
	else printf("FAILED\n");
	printf("\n");
	

	
	//----------------------------------------------------------->
	printf("Check individual blocks: ");
	{
		dim_x = 1000;
		dim_y = 1000;
		dim_z = 1000;
		offset = 0;
		
		MSD_Error MSD_error;
		MSD_Configuration MSD_conf;
		std::vector<size_t> dimensions={dim_z, dim_y, dim_x};
		MSD_error = MSD_conf.Create_MSD_Plan(dimensions, offset, false, 0.0, 1);
		if(MSD_error!=MSDSuccess) Get_MSD_Error(MSD_error);
		
		dim3 partial_gridsize = MSD_conf.get_partial_gridSize();
		int3 nSteps = MSD_conf.get_nSteps();
		int nBlocks = partial_gridsize.x*partial_gridsize.y*partial_gridsize.z;
		int block_size_x = MSD_NTHREADS;
		int block_size_y = nSteps.y;
		int block_size_z = nSteps.z;
		size_t partial_MSD_size = nBlocks*MSD_PARTIAL_SIZE*sizeof(float);
		size_t partial_MSD_nElements_size = nBlocks*sizeof(int);
		size_t MSD_size = MSD_RESULTS_SIZE*sizeof(float);
		size_t MSD_elements_size = sizeof(size_t);
		size_t input_size = dim_x*dim_y*dim_z*sizeof(float);
		float *h_MSD;
		size_t *h_MSD_nElements;
		float *h_partial_MSD;
		int *h_partial_MSD_nElements;
		h_input         = (float *)malloc(input_size);
		h_MSD 		    = (float *)malloc(MSD_size);
		h_MSD_nElements = (size_t *)malloc(MSD_elements_size);
		h_partial_MSD   = (float *)malloc(partial_MSD_size);
		h_partial_MSD_nElements = (int *)malloc(partial_MSD_nElements_size);
		memset(h_MSD, 0.0, MSD_size);
		memset(h_MSD_nElements, 0.0, MSD_elements_size);
		
		float *d_input;
		float *d_MSD;
		size_t *d_MSD_nElements;
		if ( cudaSuccess != cudaMalloc((void **) &d_input, input_size)) {
			printf("CUDA API error while allocating GPU memory\n");
		}
		if ( cudaSuccess != cudaMalloc((void **) &d_MSD, MSD_size)) {
			printf("CUDA API error while allocating GPU memory\n");
		}
		if ( cudaSuccess != cudaMalloc((void **) &d_MSD_nElements, MSD_elements_size)) {
			printf("CUDA API error while allocating GPU memory\n");
		}
		
		
		Generate_dataset(h_input, dim_x, dim_y, dim_z, offset, 1.0, 0.05);
		cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
		MSD_error = MSD_GetMeanStdev(d_MSD, d_MSD_nElements, d_input, MSD_conf);
		cudaMemcpy( h_MSD, d_MSD, MSD_size, cudaMemcpyDeviceToHost);
		cudaMemcpy( h_MSD_nElements, d_MSD_nElements, MSD_elements_size, cudaMemcpyDeviceToHost);
		cudaMemcpy( h_partial_MSD, MSD_conf.get_pointer_partial_MSD(), partial_MSD_size, cudaMemcpyDeviceToHost);
		cudaMemcpy( h_partial_MSD_nElements, MSD_conf.get_pointer_partial_nElements(), partial_MSD_nElements_size, cudaMemcpyDeviceToHost);

		int no_check_error = 1;
		
		int nBlocks_x = (int) ((dim_x + block_size_x - 1)/block_size_x);
		int nBlocks_y = (int) ((dim_y + block_size_y - 1)/block_size_y);
		int nBlocks_z = (int) (dim_z/block_size_z);
		float *tempdata;
		for(int bz = 0; bz<nBlocks_z; bz++){
			for(int by = 0; by<nBlocks_y; by++){
				for(int bx = 0; bx<nBlocks_x; bx++){
					double signal_mean, signal_sd, merror, sderror;
					int limit_x = (bx==(nBlocks_x-1)?(dim_x-bx*block_size_x):block_size_x);
					int limit_y = (by==(nBlocks_y-1)?(dim_y-by*block_size_y):block_size_y);
					tempdata = new float[limit_x*limit_y];
					for(int ty = 0; ty<limit_y; ty++){
						for(int tx = 0; tx<limit_x; tx++){
							size_t pos = bz*dim_x*dim_y + (by*block_size_y + ty)*dim_x + bx*block_size_x + tx;
							tempdata[ty*limit_x + tx]=h_input[pos];
						}
					}
					MSD_Kahan(tempdata, limit_x, limit_y, 1, 0, &signal_mean, &signal_sd);
					size_t respos = bz*nBlocks_x*nBlocks_y + by*nBlocks_x + bx;
					float GPU_mean = h_partial_MSD[MSD_PARTIAL_SIZE*respos]/((double) h_partial_MSD_nElements[respos]);
					float GPU_sd = sqrt(h_partial_MSD[MSD_PARTIAL_SIZE*respos + 1]/((double) h_partial_MSD_nElements[respos]));
					merror  = sqrt((signal_mean-GPU_mean)*(signal_mean-GPU_mean));
					sderror = sqrt((signal_sd-GPU_sd)*(signal_sd-GPU_sd));
					if(merror>max_error && sderror>max_error) no_check_error = no_check_error*0;
					delete [] tempdata;
				}
			}
		}
		
		
		if ( cudaSuccess != cudaFree(d_input)) {
			printf("CUDA API error while deallocating GPU memory\n");
		}
		if ( cudaSuccess != cudaFree(d_MSD)) {
			printf("CUDA API error while deallocating GPU memory\n");
		}
		if ( cudaSuccess != cudaFree(d_MSD_nElements)) {
			printf("CUDA API error while deallocating GPU memory\n");
		}
		
		free(h_input);
		free(h_MSD);
		free(h_MSD_nElements);
		free(h_partial_MSD);
		free(h_partial_MSD_nElements);
		
		if(no_check_error==1) printf("PASSED\n");
		else printf("FAILED\n");
	}
	
	









	return (0);
}

