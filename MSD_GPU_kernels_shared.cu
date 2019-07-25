#ifndef MSD_GPU_SHARED_CU
#define MSD_GPU_SHARED_CU

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MSD_params.h"

template<typename T, typename V>
__device__ __inline__ T shfl_down(const T &XX, const V &YY) {
	#if(CUDART_VERSION >= 9000)
		return(__shfl_down_sync(0xffffffff, XX, YY));
	#else
		return(__shfl_down(XX, YY));
	#endif
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//------------- Device functions

//------------------------------------------------------------------------------------------------>
//--------------- Elementary operations
template<typename nelements_accumulator>
__device__ __inline__ void Initiate(float *M, float *S, nelements_accumulator *j, float element){
	*M = element;
	*S = 0;
	*j = 1;
}

template<typename nelements_accumulator>
__device__ __inline__ void Add_one(float *M, float *S, nelements_accumulator *j, float element){
	float ftemp;
	*j = (*j) + 1;
	double r_j = (double) (*j);
	*M = (*M) + element;
	ftemp = ( r_j*element - (*M) );
	*S = (*S) + 1.0f / ( r_j*( r_j - 1.0f ) )*ftemp*ftemp;
}

template<typename nelements_accumulator, typename partial_accumulator>
__device__ __inline__ void Merge(float *A_M, float *A_S, nelements_accumulator *A_j, float B_M, float B_S, partial_accumulator B_j){
	float ftemp;
	double r_B_j = (double) B_j;
	double r_A_j = (double) (*A_j);
	
	ftemp = ( r_B_j / r_A_j)*(*A_M) - B_M;
	(*A_S) = (*A_S) + B_S + ( r_A_j/( r_B_j*(r_A_j + r_B_j) ) )*ftemp*ftemp;
	(*A_M) = (*A_M) + B_M;
	(*A_j) = (*A_j) + B_j;
}

//--------------------------- Elementary operations ----------------------------------------------<


//------------------------------------------------------------------------------------------------>
//--------------- Reductions
template<typename nelements_accumulator>
__device__ __inline__ void Reduce_SM(float *M, float *S, nelements_accumulator *j, float *s_par_MSD, nelements_accumulator *s_par_nElements){
	nelements_accumulator jv;
	
	(*M)=s_par_MSD[threadIdx.x];
	(*S)=s_par_MSD[blockDim.x + threadIdx.x];
	(*j)=s_par_nElements[threadIdx.x];
	
	for (int i = ( blockDim.x >> 1 ); i > MSD_HALF_WARP; i = i >> 1) {
		if (threadIdx.x < i) {
			jv = s_par_nElements[i + threadIdx.x];
			if( jv!=0){
				if( (*j)==0 ){
					(*M) = s_par_MSD[i + threadIdx.x];
					(*S) = s_par_MSD[blockDim.x + i + threadIdx.x];
					(*j) = jv;
				}
				else {
					Merge(M, S, j, s_par_MSD[i + threadIdx.x], s_par_MSD[blockDim.x + i + threadIdx.x], jv);
				}
			}
			
			s_par_MSD[threadIdx.x] = (*M);
			s_par_MSD[blockDim.x + threadIdx.x] = (*S);
			s_par_nElements[threadIdx.x] = (*j);
		}
		__syncthreads();
	}
}

template<typename nelements_accumulator>
__device__ __inline__ void Reduce_SM_max(float *M, float *S, float *max, float *min, nelements_accumulator *j, float *s_par_MSD, nelements_accumulator *s_par_nElements){
	nelements_accumulator jv;
	
	(*M)   = s_par_MSD[threadIdx.x];
	(*S)   = s_par_MSD[blockDim.x + threadIdx.x];
	(*max) = s_par_MSD[2*blockDim.x + threadIdx.x];
	(*min) = s_par_MSD[3*blockDim.x + threadIdx.x];
	(*j)=s_par_nElements[threadIdx.x];
	
	
	for (int i = ( blockDim.x >> 1 ); i > MSD_HALF_WARP; i = i >> 1) {
		if (threadIdx.x < i) {
			jv = s_par_nElements[i + threadIdx.x];
			if( jv!=0){
				if( (*j)==0 ){
					(*M) = s_par_MSD[i + threadIdx.x];
					(*S) = s_par_MSD[blockDim.x + i + threadIdx.x];
					(*max) = s_par_MSD[2*blockDim.x + i + threadIdx.x];
					(*min) = s_par_MSD[3*blockDim.x + i + threadIdx.x];
					(*j) = jv;
				}
				else {
					Merge(M, S, j, s_par_MSD[i + threadIdx.x], s_par_MSD[blockDim.x + i + threadIdx.x], jv);
					(*max) = fmaxf((*max), s_par_MSD[2*blockDim.x + i + threadIdx.x]);
					(*min) = fminf((*min), s_par_MSD[3*blockDim.x + i + threadIdx.x]);
				}
			}
			
			s_par_MSD[threadIdx.x] = (*M);
			s_par_MSD[blockDim.x + threadIdx.x] = (*S);
			s_par_MSD[2*blockDim.x + threadIdx.x] = (*max);
			s_par_MSD[3*blockDim.x + threadIdx.x] = (*min);
			s_par_nElements[threadIdx.x] = (*j);
		}
		__syncthreads();
	}
}


template<typename nelements_accumulator>
__device__ __inline__ void Reduce_SM_regular(float *M, float *S, nelements_accumulator *j, float *s_par_MSD, nelements_accumulator *s_par_nElements){
	(*M)=s_par_MSD[threadIdx.x];
	(*S)=s_par_MSD[blockDim.x + threadIdx.x];
	(*j)=s_par_nElements[threadIdx.x];
	
	for (int i = ( blockDim.x >> 1 ); i > MSD_HALF_WARP; i = i >> 1) {
		if (threadIdx.x < i) {
			Merge(M, S, j, s_par_MSD[i + threadIdx.x], s_par_MSD[blockDim.x + i + threadIdx.x], s_par_nElements[i + threadIdx.x]);
			
			s_par_MSD[threadIdx.x] = (*M);
			s_par_MSD[blockDim.x + threadIdx.x] = (*S);
			s_par_nElements[threadIdx.x] = (*j);
		}
		__syncthreads();
	}
}

template<typename nelements_accumulator>
__device__ __inline__ void Reduce_WARP(float *M, float *S, nelements_accumulator *j){
	float B_M, B_S;
	nelements_accumulator B_j;
	
	for (int q = MSD_HALF_WARP; q > 0; q = q >> 1) {
		B_M = shfl_down((*M), q);
		B_S = shfl_down((*S), q);
		B_j = shfl_down((*j), q);
		
		if(B_j>0){
			if( (*j)==0 ) {
				(*S) = B_S;
				(*M) = B_M;
				(*j) = B_j;
			}
			else {
				Merge(M, S, j, B_M, B_S, B_j);
			}
		}
	}
}

template<typename nelements_accumulator>
__device__ __inline__ void Reduce_WARP_max(float *M, float *S, float *max, float *min, nelements_accumulator *j){
	float B_M, B_S, B_max, B_min;
	nelements_accumulator B_j;
	
	for (int q = MSD_HALF_WARP; q > 0; q = q >> 1) {
		B_M   = shfl_down((*M), q);
		B_S   = shfl_down((*S), q);
		B_max = shfl_down((*max), q);
		B_min = shfl_down((*min), q);
		B_j   = shfl_down((*j), q);
		
		if(B_j>0){
			if( (*j)==0 ) {
				(*S) = B_S;
				(*M) = B_M;
				(*j) = B_j;
				(*max) = B_max;
				(*min) = B_min;
			}
			else {
				Merge(M, S, j, B_M, B_S, B_j);
				(*max) = fmaxf((*max), B_max);
				(*min) = fmaxf((*min), B_min);
			}
		}
	}
}

template<typename nelements_accumulator>
__device__ __inline__ void Reduce_WARP_regular(float *M, float *S, nelements_accumulator *j){
	for (int q = MSD_HALF_WARP; q > 0; q = q >> 1) {
		Merge(M, S, j, shfl_down((*M), q), shfl_down((*S), q), shfl_down((*j), q));
	}
}

//--------------------------- Reductions end -----------------------------------------------------<



template<typename nelements_accumulator>
__device__ void Sum_partials_regular(float *M, float *S, nelements_accumulator *j, float *d_input_MSD, int *d_input_nElements, float *s_par_MSD, nelements_accumulator *s_par_nElements, int size){
	int pos;
	
	//----------------------------------------------
	//---- Summing partials
	pos = threadIdx.x;
	if (size > blockDim.x) {
		(*M) = d_input_MSD[MSD_PARTIAL_SIZE*pos];
		(*S) = d_input_MSD[MSD_PARTIAL_SIZE*pos + 1];
		(*j) = d_input_nElements[pos];
		
		pos = pos + blockDim.x;
		while (pos < size) {
			Merge( M, S, j, d_input_MSD[MSD_PARTIAL_SIZE*pos], d_input_MSD[MSD_PARTIAL_SIZE*pos + 1], d_input_nElements[pos]);
			pos = pos + blockDim.x;
		}

		s_par_MSD[threadIdx.x] = (*M);
		s_par_MSD[blockDim.x + threadIdx.x] = (*S);
		s_par_nElements[threadIdx.x] = (*j);
		
		__syncthreads();

		Reduce_SM_regular( M, S, j, s_par_MSD, s_par_nElements);
		Reduce_WARP_regular(M, S, j);
	}
	else {
		if (threadIdx.x == 0) {
			pos = 0;
			(*M) = d_input_MSD[MSD_PARTIAL_SIZE*pos];
			(*S) = d_input_MSD[MSD_PARTIAL_SIZE*pos + 1];
			(*j) = d_input_nElements[pos];
			
			for (pos = 1; pos < size; pos++) {
				Merge( M, S, j, d_input_MSD[MSD_PARTIAL_SIZE*pos], d_input_MSD[MSD_PARTIAL_SIZE*pos + 1], d_input_nElements[pos]);
			}
		}
	}
	//---- Summing partials
	//----------------------------------------------
}

template<typename nelements_accumulator>
__device__ void Sum_partials_nonregular(float *M, float *S, nelements_accumulator *j, float *d_partial_MSD, int *d_partial_nElements, float *s_par_MSD, nelements_accumulator *s_par_nElements, int size){
	int pos;
	nelements_accumulator jv;
	
	//----------------------------------------------
	//---- Summing partials
	pos = threadIdx.x;
	if (size > blockDim.x) {
		(*M) = 0;	(*S) = 0;	(*j) = 0;
		while (pos < size) {
			jv = d_partial_nElements[pos];
			if( jv>0 ){
				if( (*j)==0 ){
					(*M) = d_partial_MSD[MSD_PARTIAL_SIZE*pos]; 
					(*S) = d_partial_MSD[MSD_PARTIAL_SIZE*pos + 1];
					(*j) = jv;
				}
				else {
					Merge( M, S, j, d_partial_MSD[MSD_PARTIAL_SIZE*pos], d_partial_MSD[MSD_PARTIAL_SIZE*pos + 1], jv);
				}
			}
			pos = pos + blockDim.x;
		}

		s_par_MSD[threadIdx.x] = (*M);
		s_par_MSD[blockDim.x + threadIdx.x] = (*S);
		s_par_nElements[threadIdx.x] = (*j);
		
		__syncthreads();

		Reduce_SM( M, S, j, s_par_MSD, s_par_nElements);
		Reduce_WARP(M, S, j);
	}
	else {
		if (threadIdx.x == 0) {
			pos = 0;
			(*M) = 0;	(*S) = 0;	(*j) = 0;
			for (pos = 1; pos < size; pos++) {
				jv = d_partial_nElements[pos];
				if( jv!=0 ){
					if( (*j)==0 ){
						(*M) = d_partial_MSD[MSD_PARTIAL_SIZE*pos]; 
						(*S) = d_partial_MSD[MSD_PARTIAL_SIZE*pos + 1];
						(*j) = jv;
					}
					else {
						Merge( M, S, j, d_partial_MSD[MSD_PARTIAL_SIZE*pos], d_partial_MSD[MSD_PARTIAL_SIZE*pos + 1], jv);
					}
				}
			}
		}
	}
	//---- Summing partials
	//----------------------------------------------
}

//------------- Device functions
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------


#endif
