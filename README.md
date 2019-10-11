README v0.1 / 14 AUGUST 2019

# MSDGPU
GPU accelerated library for calculation of mean and standard deviation using CUDA

## Introduction
The MSDGPU calculates the mean and standard deviation of the data using NVIDIA CUDA. The library can calculate mean and standard deviation of one-dimensional, two-dimensional and three-dimensional data. It is also possible to calculate mean and standard deviation in batched mode producing independent mean and standard deviations for each one-dimensional or two-dimensional array.

The library also allows calculating the mean and standard deviation with outlier rejection. The element is classified as an outlier if its z-value is higher then used a defined threshold.

The library uses parallel implementation of the algorithm by T. F. Chan 1983 (Tony F. Chan et al.; Algorithms for Computing the Sample Variance: Analysis and Recommendations; The American Statistician, Vol. 37, No. 3 (Aug., 1983), pp. 242-247; url:http://www.jstor.org/stable/2683386).


## Usage

The mean and standard deviation library (MSDGPU) adopts the following approach to the user interface. First, the user instantiates the library's configuration class, then creates where parameters of the input data are set, the user also specifies whenever outlier rejection is requested. The user can pass stream id (cudaStream_t) in which all GPU kernels of the MSDGPU library will be executed. Without specifying the stream MSDGPU library will execute in default stream. After the MSDGPU plan is created user can execute library function to get the mean and standard deviation of the input data. The values of mean and standard deviation will be stored on the device. At the end MSDGPU plan could be destroyed by the user otherwise it will be destroyed when the configuration class goes out of scope. The MSDGPU plan can be reused provided that the pointer to the array and its size remains the same.

### MSD_GPU_library "MSD_GPU_library.h"
	void Get_MSD_Error(MSD_Error error);
	displays the error message associated with the error code
	
	MSD_Error MSD_GetMeanStdev(float *d_MSD, size_t *d_MSD_nElements, float *d_input, MSD_Configuration &MSD_conf);
	executes the MSDGPU library and calculates the mean and standard deviation.
	
### Configuration Class `MSD_Configuration.h`
	The configuration class `MSD_Configuration` takes care of creating the MSD plan for execution of the MSDGPU library. It also provides the MSD_error variable type.
	
	
	`bool MSD_ready(void);` - returns true if plan is ready, false otherwise.
	
	`bool MSD_outlier_rejection(void);` - returns true if outlier rejection is enabled, false otherwise.
	
	Configuration of the GPU kernels
	`dim3 get_partial_gridSize();`
	`dim3 get_partial_blockSize();`
	`dim3 get_final_gridSize();`
	`dim3 get_final_blockSize();`
	
	Simple functions returning given variable
	`cudaStream_t get_CUDA_stream();`
	`float* get_pointer_partial_MSD();`
	`int* get_pointer_partial_nElements();`
	`int3 get_nSteps();`
	`int get_nDim();`
	`size_t get_dim_x();`
	`size_t get_dim_y();`
	`size_t get_dim_z();`
	`int get_offset();`
	`float get_OR_sigma_range();`
	`int get_nBlocks_total();`

	MSD plan
	`MSD_Error Create_MSD_Plan(std::vector<size_t> data_dimensions, int offset, bool enable_outlier_rejection, float OR_sigma_range, int nBatches=1);` - creates MSD plan. 
	+ `data_dimensions` - must contain dimensions of the data where fastest moving coordinate is last.
	+ `offset` - is the number of elements which will be excluded from the calculation of the mean and stdev measured from the end of fastest moving coordinate. For example if one-dimensional array of size 1000 with offset 10 then only first 990 elements are used by MSD. In two-dimensional case a strip of width 10 is excluded.
	+ `enable_outlier_rejection`
	+ `OR_sigma_range` - defines threshold in multiples of stdev for outlier rejection. Any element with absolute value of z-value (sigma, SNR) greater then this threshold is not used by the MSDGPU library.
	+ `nBatches` - the number of batches to be processes. It is assumed that next batch is at index b*(dim_z*dim_y*dimx) where b is batch starting from zero.
	
	`void Bind_cuda_stream(cudaStream_t t_cuda_stream);` - allows to set the cuda stream in which the kernels will be executed. If the stream is not set default stream is used instead.
	
	`MSD_Error Destroy_MSD_Plan();` - destroys MSDD plan.
	
	
##Examples:
Examples are part of the depository. 

## Contributing

Karel Adamek;
Wes Armour;

## Contact

In case you have problems you can contact me using email listed in my github repository.



## Installation

### Requirements

NVIDIA CUDA.

### Installation

When library is compiled you are free to install it as you wish. I'm aware that this is not ideal and I would be grateful for help with this.

### Configuration

Environmental variable `CUDA_HOME` must be set to the directory of the CUDA toolkit (for example `\usr\local\cuda`);
