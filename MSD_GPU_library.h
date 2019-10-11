#ifndef __MSD_GPU_LIBRARY__
#define __MSD_GPU_LIBRARY__

#include <stdio.h>

#define MSDSuccess 0

#include "MSD_params.h"
#include "MSD_Configuration.h"


void Get_MSD_Error(MSD_Error error){
	switch(error) {
		case 0 :
			printf("No error.\n");
			break;
		case 1 :
			printf("CUDA API error! Cannot allocate memory for temporary work area!\n" );
			break;
		case 2 :
			printf("CUDA API error! Cannot deallocate temporary work area.\n" );
			break;
		case 3 :
			printf("CUDA API error! There is existing CUDA API error! cudaGetLastError() did not return cudaSuccess before MSD kernels were launched.\n" );
			break;
		case 4 :
			printf("CUDA API error! Error occurred during MSD kernels.\n" );
			break;
		case 5 :
			printf("MSD Plan was not configured correctly.\n" );
			break;
		case 6 :
			printf("MSD Plan can only work up to and including three dimensions. Number of dimensions is 0 or more then 3.\n" );
			break;
		case 7 :
			printf("Number of batches must be greater then zero.\n" );
			break;
		case 8 :
			printf("MSD Plan cannot perform more then one batch for 3d data.\n" );
			break;
		default :
			printf("Invalid error\n" );
	}
}

MSD_Error MSD_GetMeanStdev(float *d_MSD, size_t *d_MSD_nElements, float *d_input, MSD_Configuration &MSD_conf);



#endif

