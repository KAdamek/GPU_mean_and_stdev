###############################################################
# CUDA_HOME are supposed to be on default position
# and set it in your PATH .bashrc
###############################################################
INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda
LIBMSD := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda -L. -lMSDGPU

# use this compilers
# g++ just because the file write
GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc


###############################################################
# Basic flags for compilers, one for debug options
# fmad flags used for reason of floating point operation
###############################################################
NVCCFLAGS = -O3 -arch=sm_70 -std=c++11 --ptxas-options=-v --use_fast_math -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

EXECUTABLE = MSD_example_2d.exe


ifdef reglim
NVCCFLAGS += --maxrregcount=$(reglim)
endif

all: clean msdlib example1d example2d example3d

example2d: timer.h MSD_GPU_library.h MSD_Configuration.h Makefile
	$(NVCC) -o MSD_example_2d.exe MSD_example_2d.cu $(LIBMSD) $(NVCCFLAGS) 

example1d: timer.h MSD_GPU_library.h MSD_Configuration.h Makefile
	$(NVCC) -o MSD_example_1d.exe MSD_example_1d.cu $(LIBMSD) $(NVCCFLAGS) 
	
example3d: timer.h MSD_GPU_library.h MSD_Configuration.h Makefile
	$(NVCC) -o MSD_example_3d.exe MSD_example_3d.cu $(LIBMSD) $(NVCCFLAGS) 

msdlib: MSD-library.o 
	
MSD-library.o: timer.h MSD_Configuration.h MSD_GPU_kernels_2d.cu
	$(NVCC) -c MSD_GPU_host_code.cu $(NVCCFLAGS)
	ar rsv libMSDGPU.a MSD_GPU_host_code.o
	rm *.o

#MSD_GPU_kernels.o: MSD_params.h
#	$(NVCC) -c MSD_GPU_kernels.cu $(NVCCFLAGS)

clean:	
	rm -f *.o *.a *.exe *.~ $(ANALYZE)


